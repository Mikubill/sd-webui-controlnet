(function () {
  const cnetRegisteredAccordions = new Set();
  function loadPhotopea() {
    // Simulate an `input` DOM event for Gradio Textbox component. Needed after 
    // you edit its contents in javascript, otherwise your edits
    // will only visible on web page and not sent to python.
    function updateInput(target) {
      let e = new Event("input", { bubbles: true })
      Object.defineProperty(e, "target", { value: target })
      target.dispatchEvent(e);
    }

    function registerChildTriggers(accordion) {
      const photopeaMainTrigger = accordion.querySelector('.cnet-photopea-main-trigger');
      const tabs = accordion.querySelectorAll('.cnet-unit-tab');
      tabs.forEach(tab => {
        const photopeaChildTrigger = tab.querySelector('.cnet-photopea-child-trigger');
        photopeaChildTrigger.addEventListener('click', () => {
          photopeaMainTrigger.click();
        });
      });
    }

    const accordions = gradioApp().querySelectorAll('#controlnet');
    accordions.forEach(accordion => {
      if (cnetRegisteredAccordions.has(accordion)) return;
      registerChildTriggers(accordion);
      cnetRegisteredAccordions.add(accordion);
    });
  }

  onUiUpdate(loadPhotopea);
})();