(function () {
  /*
  MIT LICENSE
  Copyright 2011 Jon Leighton
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
  associated documentation files (the "Software"), to deal in the Software without restriction,
  including without limitation the rights to use, copy, modify, merge, publish, distribute, 
  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial
  portions of the Software. 
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  */
  // From: https://gist.github.com/jonleighton/958841
  function base64ArrayBuffer(arrayBuffer) {
    var base64 = ''
    var encodings = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

    var bytes = new Uint8Array(arrayBuffer)
    var byteLength = bytes.byteLength
    var byteRemainder = byteLength % 3
    var mainLength = byteLength - byteRemainder

    var a, b, c, d
    var chunk

    // Main loop deals with bytes in chunks of 3
    for (var i = 0; i < mainLength; i = i + 3) {
      // Combine the three bytes into a single integer
      chunk = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2]

      // Use bitmasks to extract 6-bit segments from the triplet
      a = (chunk & 16515072) >> 18 // 16515072 = (2^6 - 1) << 18
      b = (chunk & 258048) >> 12 // 258048   = (2^6 - 1) << 12
      c = (chunk & 4032) >> 6 // 4032     = (2^6 - 1) << 6
      d = chunk & 63               // 63       = 2^6 - 1

      // Convert the raw binary segments to the appropriate ASCII encoding
      base64 += encodings[a] + encodings[b] + encodings[c] + encodings[d]
    }

    // Deal with the remaining bytes and padding
    if (byteRemainder == 1) {
      chunk = bytes[mainLength]

      a = (chunk & 252) >> 2 // 252 = (2^6 - 1) << 2

      // Set the 4 least significant bits to zero
      b = (chunk & 3) << 4 // 3   = 2^2 - 1

      base64 += encodings[a] + encodings[b] + '=='
    } else if (byteRemainder == 2) {
      chunk = (bytes[mainLength] << 8) | bytes[mainLength + 1]

      a = (chunk & 64512) >> 10 // 64512 = (2^6 - 1) << 10
      b = (chunk & 1008) >> 4 // 1008  = (2^6 - 1) << 4

      // Set the 2 least significant bits to zero
      c = (chunk & 15) << 2 // 15    = 2^4 - 1

      base64 += encodings[a] + encodings[b] + encodings[c] + '='
    }

    return base64
  }

  // Functions to be called within photopea context.
  // Start of photopea functions
  function pasteImage(base64image) {
    app.open(base64image, null, /* asSmart */ true);
    app.echoToOE("success");
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
    app.echoToOE("success");
  }

  function removeLayersWithNames(names) {
    const layers = app.activeDocument.layers;
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      if (names.includes(layer.name)) {
        layer.remove();
      }
    }
    app.echoToOE("success");
  }

  function getAllLayerNames() {
    const layers = app.activeDocument.layers;
    const names = [];
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      names.push(layer.name);
    }
    app.echoToOE(JSON.stringify(names));
  }

  // Hides all layers except the current one, outputs the whole image, then restores the previous
  // layers state.
  function exportSelectedLayerOnly(format, layerName) {
    // Gets all layers recursively, including the ones inside folders.
    function getAllArtLayers(document) {
      let allArtLayers = [];

      for (let i = 0; i < document.layers.length; i++) {
        const currentLayer = document.layers[i];
        allArtLayers.push(currentLayer);
        if (currentLayer.typename === "LayerSet") {
          allArtLayers = allArtLayers.concat(getAllArtLayers(currentLayer));
        }
      }
      return allArtLayers;
    }

    function makeLayerVisible(layer) {
      let currentLayer = layer;
      while (currentLayer != app.activeDocument) {
        currentLayer.visible = true;
        if (currentLayer.parent.typename != 'Document') {
          currentLayer = currentLayer.parent;
        } else {
          break;
        }
      }
    }


    const allLayers = getAllArtLayers(app.activeDocument);
    // Make all layers except the currently selected one invisible, and store
    // their initial state.
    const layerStates = [];
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      layerStates.push(layer.visible);
    }
    // Hide all layers to begin with
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      layer.visible = false;
    }
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      const selected = layer.name === layerName;
      if (selected) {
        makeLayerVisible(layer);
      }
    }
    app.activeDocument.saveToOE(format);

    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      layer.visible = layerStates[i];
    }
  }
  // End of photopea functions

  const MESSAGE_END_ACK = "done";
  const MESSAGE_ERROR = "error";
  class PhotopeaContext {
    constructor(photopeaWindow) {
      this.photopeaWindow = photopeaWindow;
    }

    // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
    postMessageToPhotopea(message) {
      return new Promise((resolve, reject) => {
        const responseDataPieces = [];
        let hasError = false;
        const photopeaMessageHandle = (event) => {
          if (event.source !== this.photopeaWindow) {
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
        this.photopeaWindow.postMessage(message, "*");
      });
    }

    // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
    async invoke(func, ...args) {
      const message = `${func.toString()} ${func.name}(${args.map(arg => JSON.stringify(arg)).join(',')});`;
      try {
        return await this.postMessageToPhotopea(message);
      } catch (e) {
        throw `Failed to invoke ${func.name}. ${e}.`;
      }
    }

    /**
     * Fetch detected maps from each ControlNet units. 
     * Create a new photopea document.
     * Add those detected maps to the created document.
     */
    async fetchFromControlNet(tabs) {
      const layerNames = [];
      for (const [i, tab] of tabs.entries()) {
        const generatedImage = tab.querySelector('.cnet-generated-image-group .cnet-image img');
        if (!generatedImage) continue;
        await this.invoke(pasteImage, generatedImage.src);
        // Wait 200ms for pasting to fully complete so that we do not ended up with 2 separate
        // documents.
        await new Promise(r => setTimeout(r, 200));
        layerNames.push(`unit-${i}`);
      }
      await this.invoke(removeLayersWithNames, layerNames);
      await this.invoke(setLayerNames, layerNames.reverse());
    }

    /**
     * Send the images in the active photopea document back to each ControlNet units.
     */
    async sendToControlNet(tabs) {
      function sendToControlNetUnit(imageURL, index) {
        const tab = tabs[index];
        const generatedImage = tab.querySelector('.cnet-generated-image-group .cnet-image img');
        generatedImage.src = imageURL;
        const checkbox = tab.querySelector('.cnet-preview-as-input input[type="checkbox"]');
        if (!checkbox.checked) {
          checkbox.click();
        }
      }

      const layerNames =
        JSON.parse(await this.invoke(getAllLayerNames))
          .filter(name => /unit-\d+/.test(name));

      for (const layerName of layerNames) {
        const arrayBuffer = await this.invoke(exportSelectedLayerOnly, 'PNG', layerName);
        const imageURL = 'data:image/png;base64,' + base64ArrayBuffer(arrayBuffer);
        const layerIndex = Number.parseInt(layerName.split('-')[1]);
        sendToControlNetUnit(imageURL, layerIndex);
      }
    }
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
      const photopeaContext = new PhotopeaContext(photopeaWindow, tabs);
      accordion.querySelector('.photopea-fetch').addEventListener('click', () => photopeaContext.fetchFromControlNet(tabs));
      accordion.querySelector('.photopea-send').addEventListener('click', () => photopeaContext.sendToControlNet(tabs));
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