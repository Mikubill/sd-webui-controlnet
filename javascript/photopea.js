(function () {
  const MESSAGE_END_ACK = "done";
  const MESSAGE_ERROR = "error";

  // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
  function postMessageToPhotopea(message, photopeaWindow) {
    return new Promise((resolve, reject) => {
      const responseDataPieces = [];
      let hasError = false;
      const photopeaMessageHandle = (event) => {
        if (event.source !== photopeaWindow) {
          return;
        }
        // Filter out the ping messages
        if (typeof event.data === 'string' && event.data.includes('MSFAPI#')) {
          return;
        }
        // Ignore "done" when no data has been received. The "done" can come from
        // MSFAPI ping.
        if (event.data === MESSAGE_END_ACK && responseDataPieces.length === 0) {
          return;
        }
        if (event.data === MESSAGE_END_ACK) {
          window.removeEventListener("message", photopeaMessageHandle);
          if (hasError) {
            reject('Photopea Error.');
          } else {
            resolve(responseDataPieces.length === 1 ? responseDataPieces[0] : responseDataPieces);
          }
        } else if (event.data === MESSAGE_ERROR) {
          responseDataPieces.push(event.data);
          hasError = true;
        } else {
          responseDataPieces.push(event.data);
        }
      };

      window.addEventListener("message", photopeaMessageHandle);
      setTimeout(() => reject("Photopea message timeout"), 5000);
      photopeaWindow.postMessage(message, "*");
    });
  }

  // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
  async function invoke(photopeaWindow, func, ...args) {
    const message = `${func.toString()} ${func.name}(${args.map(arg => JSON.stringify(arg)).join(',')});`;
    try {
      return await postMessageToPhotopea(message, photopeaWindow);
    } catch (e) {
      throw `Failed to invoke ${func.name}. ${e}.`;
    }
  }

  // Functions to be called within photopea context.
  // Start of photopea functions
  function pasteImage(base64image) {
    app.open(base64image, null, /* asSmart */ true);
    app.echoToOE('success');
  }

  function setLayerNames(names) {
    const layers = app.activeDocument.layers;
    if (layers.length !== names.length) {
      console.error("layer length does not match names length");
      echoToOE("error");
      return;
    }

    for (let i = 0; i < names.length; i++) {
      const layer = layers[i];
      layer.name = names[i];
    }
  }
  // End of photopea functions

  /**
   * Fetch detected maps from each ControlNet units. 
   * Create a new photopea document.
   * Add those detected maps to the created document.
   */
  async function fetchFromControlNet(tabs, photopeaWindow) {
    const layerNames = [];
    for (const [i, tab] of tabs.entries()) {
      const generatedImage = tab.querySelector('.cnet-generated-image-group .cnet-image img');
      if (!generatedImage) continue;
      await invoke(photopeaWindow, pasteImage, generatedImage.src);
      // Wait 200ms for pasting to fully complete so that we do not ended up with 2 separate
      // documents.
      await new Promise(r => setTimeout(r, 200));
      layerNames.push(`unit-${i}`);
    }
    await invoke(photopeaWindow, setLayerNames, layerNames.reverse());
  }

  /**
   * Send the images in the active photopea document back to each ControlNet units.
   */
  async function sendToControlNet() {

  }

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

    function registerCallbacks(accordion) {
      const photopeaMainTrigger = accordion.querySelector('.cnet-photopea-main-trigger');
      const tabs = accordion.querySelectorAll('.cnet-unit-tab');
      tabs.forEach(tab => {
        const photopeaChildTrigger = tab.querySelector('.cnet-photopea-child-trigger');
        photopeaChildTrigger.addEventListener('click', () => {
          photopeaMainTrigger.click();
        });
      });

      const photopeaWindow = accordion.querySelector('.photopea-iframe').contentWindow;
      accordion.querySelector('.photopea-fetch').addEventListener('click', () => fetchFromControlNet(tabs, photopeaWindow));
      accordion.querySelector('.photopea-send').addEventListener('click', sendToControlNet);
    }

    const accordions = gradioApp().querySelectorAll('#controlnet');
    accordions.forEach(accordion => {
      if (cnetRegisteredAccordions.has(accordion)) return;
      registerCallbacks(accordion);
      cnetRegisteredAccordions.add(accordion);
    });
  }

  onUiUpdate(loadPhotopea);
})();