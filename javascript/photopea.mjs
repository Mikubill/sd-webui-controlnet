import { base64ArrayBuffer, b64toBlob, createBlackImageBase64, setImageOnInput } from "./image_util.mjs";

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

function hasActiveDocument() {
  app.echoToOE(app.documents.length > 0 ? "true" : "false");
}
// End of photopea functions

const MESSAGE_END_ACK = "done";
const MESSAGE_ERROR = "error";
const PHOTOPEA_URL = "https://www.photopea.com/";
class PhotopeaContext {
  constructor(photopeaIframe) {
    this.photopeaIframe = photopeaIframe;
    this.timeout = 1000;
  }

  navigateIframe() {
    const iframe = this.photopeaIframe;
    const editorURL = PHOTOPEA_URL;

    return new Promise(async (resolve) => {
      if (iframe.src !== editorURL) {
        iframe.src = editorURL;
        // Stop waiting after 10s.
        setTimeout(resolve, 10000);

        // Testing whether photopea is able to accept message.
        while (true) {
          try {
            await this.invoke(hasActiveDocument);
            break;
          } catch (e) {
            console.log("Keep waiting for photopea to accept message.");
          }
        }
        this.timeout = 5000; // Restore to a longer timeout in normal messaging.
      }
      resolve();
    });
  }

  // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
  postMessageToPhotopea(message) {
    return new Promise((resolve, reject) => {
      const responseDataPieces = [];
      let hasError = false;
      const photopeaMessageHandle = (event) => {
        if (event.source !== this.photopeaIframe.contentWindow) {
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
      setTimeout(() => reject("Photopea message timeout"), this.timeout);
      this.photopeaIframe.contentWindow.postMessage(message, "*");
    });
  }

  // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
  async invoke(func, ...args) {
    await this.navigateIframe();
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
    if (tabs.length === 0) return;
    const isImg2Img = tabs[0].querySelector('.cnet-unit-enabled').id.includes('img2img');
    const generationType = isImg2Img ? 'img2img' : 'txt2img';
    const width = gradioApp().querySelector(`#${generationType}_width input[type=number]`).value;
    const height = gradioApp().querySelector(`#${generationType}_height input[type=number]`).value;

    const layerNames = ["background"];
    await this.invoke(pasteImage, createBlackImageBase64(width, height));
    await new Promise(r => setTimeout(r, 200));
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
    function sendToControlNetUnit(b64Image, index) {
      const tab = tabs[index];
      // Upload image to output image element.
      const outputImage = tab.querySelector('.cnet-photopea-output');
      const outputImageUpload = outputImage.querySelector('input[type="file"]');
      setImageOnInput(outputImageUpload, new File([b64toBlob(b64Image, "image/png")], "photopea_output.png"));

      // Make sure `UsePreviewAsInput` checkbox is checked.
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
      const b64Image = base64ArrayBuffer(arrayBuffer);
      const layerIndex = Number.parseInt(layerName.split('-')[1]);
      sendToControlNetUnit(b64Image, layerIndex);
    }
  }
}

let photopeaWarningShown = false;

function firstTimeUserPrompt() {
  if (opts.controlnet_photopea_warning) {
    const photopeaPopupMsg = "you are about to connect to https://photopea.com\n" +
      "- Click OK: proceed.\n" +
      "- Click Cancel: abort.\n" +
      "Photopea integration can be disabled in Settings > ControlNet > Disable photopea edit.\n" +
      "This popup can be disabled in Settings > ControlNet > Photopea popup warning.";
    if (photopeaWarningShown || confirm(photopeaPopupMsg)) photopeaWarningShown = true;
    else return false;
  }
  return true;
}

export function loadPhotopea(accordion) {
  const photopeaMainTrigger = accordion.querySelector('.cnet-photopea-main-trigger');
  // Photopea edit feature disabled.
  if (!photopeaMainTrigger) {
    console.log("ControlNet photopea edit disabled.");
    return;
  }

  const closeModalButton = accordion.querySelector('.cnet-photopea-edit .cnet-modal-close');
  const tabs = accordion.querySelectorAll('.cnet-unit-tab');
  const photopeaIframe = accordion.querySelector('.photopea-iframe');
  const photopeaContext = new PhotopeaContext(photopeaIframe, tabs);

  tabs.forEach(tab => {
    const photopeaChildTrigger = tab.querySelector('.cnet-photopea-child-trigger');
    photopeaChildTrigger.addEventListener('click', async () => {
      if (!firstTimeUserPrompt()) return;

      photopeaMainTrigger.click();
      if (await photopeaContext.invoke(hasActiveDocument) === "false") {
        await photopeaContext.fetchFromControlNet(tabs);
      }
    });
  });
  accordion.querySelector('.photopea-fetch').addEventListener('click', () => photopeaContext.fetchFromControlNet(tabs));
  accordion.querySelector('.photopea-send').addEventListener('click', () => {
    photopeaContext.sendToControlNet(tabs)
    closeModalButton.click();
  });
}
