import { ControlNetUnit } from "./controlnet_unit.mjs";

export class OpenposeEditor {
    /**
     * OpenposeEditor
     * @param {ControlNetUnit} unit
     */
    constructor(unit) {
        this.unit = unit;
        this.iframe = this.unit.generatedImageGroup.querySelector('.cnet-modal iframe');
        this.closeModalButton = this.unit.generatedImageGroup.querySelector('.cnet-modal .cnet-modal-close');
        this.uploadButton = this.unit.inputImageGroup.querySelector('.cnet-upload-pose input');
        this.editorURL = null;
        this.unit.poseEditButton.addEventListener('click', this.trigger.bind(this));
        // Updates preview image when edit is done.
        window.addEventListener('message', ((event) => {
            const message = event.data;
            const modalId = this.unit.poseEditButton.id.replace('cnet-modal-open-', '');
            if (message.modalId !== modalId) return;
            this.updatePreviewPose(message.poseURL);
            this.closeModalButton.click();
        }).bind(this));
        // Updates preview image when JSON file is uploaded.
        this.uploadButton.addEventListener('change', ((event) => {
            const file = event.target.files[0];
            if (!file)
                return;

            const reader = new FileReader();
            reader.onload = (e) => {
                const contents = e.target.result;
                const poseURL = `data:application/json;base64,${btoa(contents)}`;
                this.updatePreviewPose(poseURL);
            };
            reader.readAsText(file);
            // Reset the file input value so that uploading the same file still triggers callback.
            event.target.value = '';
        }).bind(this));
    }

    async checkEditorAvailable() {
        const LOCAL_EDITOR_PATH = '/openpose_editor_index';
        const REMOTE_EDITOR_PATH = 'https://huchenlei.github.io/sd-webui-openpose-editor/';

        async function testEditorPath(path) {
            const res = await fetch(path);
            return res.status === 200 ? path : null;
        }
        // Use local editor if the user has the extension installed. Fallback
        // onto remote editor if the local editor is not ready yet.
        // See https://github.com/huchenlei/sd-webui-openpose-editor/issues/53
        // for more details.
        return await testEditorPath(LOCAL_EDITOR_PATH) || await testEditorPath(REMOTE_EDITOR_PATH);
    }

    navigateIframe() {
        const iframe = this.iframe;
        const editorURL = this.editorURL;

        function getPathname(rawURL) {
            try {
                return new URL(rawURL).pathname;
            } catch (e) {
                return rawURL;
            }
        }

        return new Promise((resolve) => {
            const darkThemeParam = document.body.classList.contains('dark') ?
                new URLSearchParams({ theme: 'dark' }).toString() :
                '';

            window.addEventListener('message', (event) => {
                const message = event.data;
                if (message['ready']) resolve();
            }, { once: true });

            if ((editorURL.startsWith("http") ? iframe.src : getPathname(iframe.src)) !== editorURL) {
                iframe.src = `${editorURL}?${darkThemeParam}`;
                // By default assume 5 second is enough for the openpose editor
                // to load.
                setTimeout(resolve, 5000);
            } else {
                // If no navigation is required, immediately return.
                resolve();
            }
        });
    }

    // When edit button is clicked.
    async trigger() {
        const inputImageGroup = this.unit.tab.querySelector('.cnet-input-image-group');
        const inputImage = inputImageGroup.querySelector('.cnet-image img');
        const downloadLink = this.unit.generatedImageGroup.querySelector('.cnet-download-pose a');
        const modalId = this.unit.poseEditButton.id.replace('cnet-modal-open-', '');

        if (!this.editorURL) {
            this.editorURL = await this.checkEditorAvailable();
            if (!this.editorURL) {
                alert("No openpose editor available.")
            }
        }

        await this.navigateIframe();
        this.iframe.contentWindow.postMessage({
            modalId,
            imageURL: inputImage ? inputImage.src : undefined,
            poseURL: downloadLink.href,
        }, '*');
        // Focus the iframe so that the focus is no longer on the `Edit` button.
        // Pressing space when the focus is on `Edit` button will trigger
        // the click again to resend the frame message.
        this.iframe.contentWindow.focus();
    }

    /*
    * Writes the pose data URL to an link element on input image group.
    * Click a hidden button to trigger a backend rendering of the pose JSON.
    *
    * The backend should:
    * - Set the rendered pose image as preprocessor generated image.
    */
    updatePreviewPose(poseURL) {
        const downloadLink = this.unit.generatedImageGroup.querySelector('.cnet-download-pose a');
        const renderButton = this.unit.generatedImageGroup.querySelector('.cnet-render-pose');
        const poseTextbox = this.unit.generatedImageGroup.querySelector('.cnet-pose-json textarea');
        const allowPreviewCheckbox = this.unit.allowPreviewCheckbox;

        if (!allowPreviewCheckbox.checked)
            allowPreviewCheckbox.click();

        // Only set href when download link exists and needs an update. `downloadLink`
        // can be null when user closes preview and click `Upload JSON` button again.
        // https://github.com/Mikubill/sd-webui-controlnet/issues/2308
        if (downloadLink !== null)
            downloadLink.href = poseURL;

        poseTextbox.value = poseURL;
        // Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
        // will only visible on web page and not sent to python.
        function updateInput(target) {
            let e = new Event("input", { bubbles: true })
            Object.defineProperty(e, "target", { value: target })
            target.dispatchEvent(e);
        }
        updateInput(poseTextbox);
        renderButton.click();
    }
}
