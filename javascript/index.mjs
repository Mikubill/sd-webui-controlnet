import { ControlNetUnit } from "./controlnet_unit.mjs";
import { initControlNetModals } from "./modal.mjs";

(function () {
  const cnetAllAccordions = new Set();
  onUiUpdate(() => {
    gradioApp().querySelectorAll('#controlnet').forEach(accordion => {
      if (cnetAllAccordions.has(accordion)) return;

      accordion.querySelectorAll('.cnet-unit-tab')
        .forEach(tab => new ControlNetUnit(tab, accordion));

      initControlNetModals(accordion);

      cnetAllAccordions.add(accordion);
    });
  });
})();