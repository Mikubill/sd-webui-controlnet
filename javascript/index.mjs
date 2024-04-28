import { ControlNetUnit } from "./controlnet_unit.mjs";
import { initControlNetModals } from "./modal.mjs";
import { OpenposeEditor } from "./openpose_editor.mjs";

(function () {
  const cnetAllAccordions = new Set();
  onUiUpdate(() => {
    gradioApp().querySelectorAll('#controlnet').forEach(accordion => {
      if (cnetAllAccordions.has(accordion)) return;

      accordion.querySelectorAll('.cnet-unit-tab')
        .forEach(tab => {
          const unit = new ControlNetUnit(tab, accordion);
          const openposeEditor = new OpenposeEditor(unit);
        });

      initControlNetModals(accordion);

      cnetAllAccordions.add(accordion);
    });
  });
})();