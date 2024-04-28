import { A1111Context } from "./a1111_context.mjs";
import { ControlNetUnit } from "./controlnet_unit.mjs";
import { initControlNetModals } from "./modal.mjs";
import { OpenposeEditor } from "./openpose_editor.mjs";
import { loadPhotopea } from "./photopea.mjs";
import { RegionPlanner, SnapshotTaker } from "./region_planner.mjs";

(function () {
  const cnetAllAccordions = new Set();

  function init(accordion) {
    const isImg2Img = accordion.querySelector('.cnet-unit-enabled').id.includes('img2img');
    const generationType = isImg2Img ? "img2img" : "txt2img";
    const a1111Context = new A1111Context(gradioApp(), generationType);

    const units = [...accordion.querySelectorAll('.cnet-unit-tab')].map((tab, i) => {
      const unit = new ControlNetUnit(i, tab, accordion);
      const openposeEditor = new OpenposeEditor(unit);
      unit.openposeEditor = openposeEditor;
      return unit;
    });
    const snapshotTaker = new SnapshotTaker(accordion.querySelector('.cnet-region-planner-snapshot-canvas'));
    new RegionPlanner(
      accordion.querySelector('.cnet-region-planner'),
      units,
      a1111Context,
      snapshotTaker,
    );
    initControlNetModals(accordion);
    loadPhotopea(accordion);
  }

  onUiUpdate(() => {
    gradioApp().querySelectorAll('#controlnet').forEach(accordion => {
      if (cnetAllAccordions.has(accordion)) return;
      try {
        init(accordion);
      } catch (e) {
        console.error(`Failed to init ControlNet front-end: ${e}`)
      }
      cnetAllAccordions.add(accordion);
    });
  });
})();