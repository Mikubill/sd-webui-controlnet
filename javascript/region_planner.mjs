import { A1111Context } from "./a1111_context.mjs";
import { ControlNetUnit } from "./controlnet_unit.mjs";
import { setImageOnInput, b64toBlob } from "./image_util.mjs";

const COLORS = ["red", "green", "blue", "yellow", "purple"]
const GENERATION_GRID_SIZE = 8;

function snapToMultipleOf(input, size) {
  return Math.round(input / size) * size;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function removePrefix(str, prefix) {
  if (str.startsWith(prefix)) {
    return str.slice(prefix.length);
  }
  return str;
}

class CanvasControlNetUnit {
  /**
   * ControlNetUnit on canvas
   * @param {ControlNetUnit} unit
   * @param {RegionPlanner} planner
   * @param {Number} x
   * @param {Number} y
   * @param {Number} [sendMaskTimeout] Optional.
   */
  constructor(unit, planner, x, y, sendMaskTimeout) {
    this.unit = unit;
    this.planner = planner;
    this.sendMaskTimeout = sendMaskTimeout || 1000;

    this.canvasLayer = new Konva.Layer({
      visible: unit.isEnabled(),
    });

    this.canvasObject = new Konva.Rect({
      x, y,
      width: 50,
      height: 50,
      fill: this.getColor(),
      opacity: this.getOpacity(),
      id: unit.index,
      draggable: this.unit.isActive(),
      dragBoundFunc: (pos) => {
        // pos contains the position the node is being dragged to
        // Get the stage dimensions
        const stageWidth = this.planner.stage.width();
        const stageHeight = this.planner.stage.height();

        // Get the shape dimensions
        const shapeWidth = this.canvasObject.width() * this.canvasObject.scaleX();
        const shapeHeight = this.canvasObject.height() * this.canvasObject.scaleY();

        return {
          x: clamp(pos.x, 0, stageWidth - shapeWidth),
          y: clamp(pos.y, 0, stageHeight - shapeHeight),
        };
      },
    });

    this.transformer = new Konva.Transformer({
      nodes: [this.canvasObject],
      keepRatio: false,
      enabledAnchors: [
        'top-left',
        'top-right',
        'bottom-left',
        'bottom-right',
        'middle-left',
        'middle-right',
        'top-center',
        'bottom-center',
      ],
      rotateEnabled: false,
      borderEnabled: true,
      visible: this.unit.isActive(),
      boundBoxFunc: (oldBox, newBox) => {
        const ratio = this.planner.getCanvasMappingRatio()
        const gridSize = GENERATION_GRID_SIZE * ratio;
        const minCanvas = 64 * ratio;
        const maxCanvas = 2048 * ratio;
        const dimConstraint = (value) => {
          return clamp(snapToMultipleOf(value, gridSize), minCanvas, maxCanvas);
        };
        const boundConstraintX = (value) => {
          return clamp(value, 0, this.planner.stage.width());
        };
        const boundConstraintY = (value) => {
          return clamp(value, 0, this.planner.stage.height());
        };
        // Make sure box snapped to grid.
        const adjustedWidth = dimConstraint(newBox.width);
        const adjustedHeight = dimConstraint(newBox.height);
        const dw = adjustedWidth - newBox.width;
        const dh = adjustedHeight - newBox.height;
        const dx = newBox.x - oldBox.x;
        const dy = newBox.y - oldBox.y;

        newBox.x = newBox.x - (dx !== 0 ? dw : 0);
        newBox.y = newBox.y - (dy !== 0 ? dh : 0);
        newBox.width = adjustedWidth;
        newBox.height = adjustedHeight;

        newBox.x = boundConstraintX(newBox.x);
        newBox.y = boundConstraintY(newBox.y);
        newBox.width = boundConstraintX(newBox.x + newBox.width) - newBox.x;
        newBox.height = boundConstraintY(newBox.y + newBox.height) - newBox.y;
        return newBox;
      }
    });

    this.text = new Konva.Text({
      x: this.canvasObject.x() + 4,
      y: this.canvasObject.y() + 4,
      text: this.getDisplayText(),
      fontSize: 14,
      fill: "white",
      visible: this.unit.isActive(),
    });

    this.canvasLayer.add(this.canvasObject);
    this.canvasLayer.add(this.transformer);
    this.canvasLayer.add(this.text);

    this.unit.onActiveStateChange(() => {
      this.canvasObject.opacity(this.getOpacity());
      this.canvasObject.draggable(this.unit.isActive());
      this.transformer.visible(this.unit.isActive());
      this.text.visible(this.unit.isActive());
    });

    this.unit.onEnabledStateChange(() => {
      this.canvasLayer.visible(unit.isEnabled());
    });

    this.canvasObject.on('dragmove transform', () => {
      this.text.setAttrs({
        x: this.canvasObject.x() + 4,
        y: this.canvasObject.y() + 4,
        text: this.getDisplayText(),
      });
      this.debouncedWriteMask();
    });

    this.debouncedWriteMaskTimeout = null;
  }

  getColor() {
    return COLORS[this.unit.index];
  }

  getOpacity() {
    return this.unit.isActive() ? 1.0 : 0.3;
  }

  getDisplayText() {
    return `Unit${this.unit.index}(${this.getGenerationWidth()} x ${this.getGenerationHeight()})`;
  }

  getGenerationWidth() {
    return snapToMultipleOf(
      this.canvasObject.width() * this.canvasObject.scaleX() / this.planner.getCanvasMappingRatio(),
      GENERATION_GRID_SIZE,
    );
  }

  getGenerationHeight() {
    return snapToMultipleOf(
      this.canvasObject.height() * this.canvasObject.scaleY() / this.planner.getCanvasMappingRatio(),
      GENERATION_GRID_SIZE,
    );
  }

  getGenerationX() {
    return snapToMultipleOf(
      this.canvasObject.x() / this.planner.getCanvasMappingRatio(),
      GENERATION_GRID_SIZE,
    )
  }

  getGenerationY() {
    return snapToMultipleOf(
      this.canvasObject.y() / this.planner.getCanvasMappingRatio(),
      GENERATION_GRID_SIZE,
    )
  }

  toBase64Mask() {
    return this.planner.snapshotTaker.snapshotUnit(this);
  }

  /**
   * Write mask to ControlNetUnit.
   */
  debouncedWriteMask() {
    clearTimeout(this.debouncedWriteMaskTimeout);
    this.debouncedWriteMaskTimeout = setTimeout(() => {
      const base64Mask = this.toBase64Mask();
      const imageUpload = this.unit.effectiveRegionMaskImage.querySelector('input[type="file"]');
      setImageOnInput(
        imageUpload,
        new File([
          b64toBlob(removePrefix(base64Mask, "data:image/png;base64,"), "image/png")
        ], "region_planner.png"),
      );
    }, this.sendMaskTimeout);
  }
}

export class SnapshotTaker {
  constructor(container) {
    this.stage = new Konva.Stage({
      container: container,
    });
  }

  /**
   * Convert canvas unit to a base64 black/white mask.
   * @param {CanvasControlNetUnit} canvasUnit
   */
  snapshotUnit(canvasUnit) {
    this.stage.width(canvasUnit.planner.getGenerationWidth());
    this.stage.height(canvasUnit.planner.getGenerationHeight());
    const layer = new Konva.Layer();
    const rect = new Konva.Rect({
      fill: "white",
      x: canvasUnit.getGenerationX(),
      y: canvasUnit.getGenerationY(),
      height: canvasUnit.getGenerationHeight(),
      width: canvasUnit.getGenerationWidth(),
    });
    const background = new Konva.Rect({
      fill: "black",
      x: 0,
      y: 0,
      height: this.stage.height(),
      width: this.stage.width(),
    });
    layer.add(background);
    layer.add(rect);
    this.stage.add(layer);

    const dataURL = layer.toDataURL({
      mimeType: 'image/png',
      quality: 0.7,
      pixelRatio: 1,
    });

    layer.remove();
    return dataURL;
  }
}

export class RegionPlanner {
  /**
   * Region planner
   * @param {HTMLElement} container
   * @param {Array<ControlNetUnit>} units
   * @param {A1111Context} context
   * @param {SnapshotTaker} snapshotTaker
   * @param {Number} [size] - Optional.
   */
  constructor(container, units, context, snapshotTaker, size) {
    this.container = container;
    this.units = units;
    this.context = context;
    this.snapshotTaker = snapshotTaker;
    this.size = size || 512;

    this.stage = new Konva.Stage({
      container: this.container,
      width: this.getCanvasWidth(),
      height: this.getCanvasHeight(),
    });
    this.context.height_slider.onChange(this.updateCanvasSize.bind(this));
    this.context.width_slider.onChange(this.updateCanvasSize.bind(this));

    this.canvasUnits = this.units.map((unit) => {
      const canvasUnit = new CanvasControlNetUnit(unit, this, 0, 0);
      this.stage.add(canvasUnit.canvasLayer);
      return canvasUnit;
    });
  }

  getGenerationWidth() {
    return this.context.width_slider.getValue();
  }

  getGenerationHeight() {
    return this.context.height_slider.getValue();
  }

  getAspectRatio() {
    return this.getGenerationWidth() / this.getGenerationHeight();
  }

  getCanvasWidth() {
    const aspectRatio = this.getAspectRatio();
    return Math.round(aspectRatio <= 1.0 ? this.size : this.size * aspectRatio);
  }

  getCanvasHeight() {
    const aspectRatio = this.getAspectRatio();
    return Math.round(aspectRatio >= 1.0 ? this.size : this.size / aspectRatio);
  }

  // canvas dim = generation dim * mapping_ratio
  getCanvasMappingRatio() {
    return this.getCanvasHeight() / this.getGenerationHeight();
  }

  updateCanvasSize() {
    this.stage.width(this.getCanvasWidth());
    this.stage.height(this.getCanvasHeight());
  }
}
