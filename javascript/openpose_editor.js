(function () {
    async function checkEditorAvailable() {
        const EDITOR_PATH = '/openpose_editor_index';
        const res = await fetch(EDITOR_PATH);
        return res.status === 200;
    }

    const cnetOpenposeEditorRegisteredElements = new Set();
    function loadOpenposeEditor() {
        // Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
        // will only visible on web page and not sent to python.
        function updateInput(target) {
            let e = new Event("input", { bubbles: true })
            Object.defineProperty(e, "target", { value: target })
            target.dispatchEvent(e);
        }

        function navigateIframe(iframe) {
            function getPathname(rawURL) {
                try {
                    return new URL(rawURL).pathname;
                } catch (e) {
                    return rawURL;
                }
            }

            return new Promise((resolve) => {
                const EDITOR_PATH = '/openpose_editor_index';
                const darkThemeParam = document.body.classList.contains('dark') ?
                    new URLSearchParams({ theme: 'dark' }).toString() :
                    '';

                window.addEventListener('message', (event) => {
                    const message = event.data;
                    if (message['ready']) resolve();
                }, { once: true });

                if (getPathname(iframe.src) !== EDITOR_PATH) {
                    iframe.src = `${EDITOR_PATH}?${darkThemeParam}`;
                    // By default assume 5 second is enough for the openpose editor
                    // to load.
                    setTimeout(resolve, 5000);
                } else {
                    // If no navigation is required, immediately return.
                    resolve();
                }
            });
        }

        const imageRows = gradioApp().querySelectorAll('.cnet-image-row');
        imageRows.forEach(imageRow => {
            if (cnetOpenposeEditorRegisteredElements.has(imageRow)) return;
            cnetOpenposeEditorRegisteredElements.add(imageRow);

            const generatedImageGroup = imageRow.querySelector('.cnet-generated-image-group');
            const editButton = generatedImageGroup.querySelector('.cnet-edit-pose');

            editButton.addEventListener('click', async () => {
                const inputImageGroup = imageRow.querySelector('.cnet-input-image-group');
                const inputImage = inputImageGroup.querySelector('.cnet-image img');
                const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
                const modalId = editButton.id.replace('cnet-modal-open-', '');
                const modalIframe = generatedImageGroup.querySelector('.cnet-modal iframe');

                await navigateIframe(modalIframe);
                modalIframe.contentWindow.postMessage({
                    modalId,
                    imageURL: inputImage.src,
                    poseURL: downloadLink.href,
                }, '*');
                // Focus the iframe so that the focus is no longer on the `Edit` button.
                // Pressing space when the focus is on `Edit` button will trigger
                // the click again to resend the frame message.
                modalIframe.contentWindow.focus();
            });

            window.addEventListener('message', (event) => {
                const message = event.data;
                const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
                const renderButton = generatedImageGroup.querySelector('.cnet-render-pose');
                const poseTextbox = generatedImageGroup.querySelector('.cnet-pose-json textarea');
                const modalId = editButton.id.replace('cnet-modal-open-', '');
                const closeModalButton = generatedImageGroup.querySelector('.cnet-modal .cnet-modal-close');

                if (message.modalId !== modalId) return;
                /* 
                * Writes the pose data URL to an link element on input image group.
                * Click a hidden button to trigger a backend rendering of the pose JSON.
                * 
                * The backend should:
                * - Set the rendered pose image as preprocessor generated image.
                */
                downloadLink.href = message.poseURL;
                poseTextbox.value = message.poseURL;
                updateInput(poseTextbox);
                renderButton.click();
                closeModalButton.click();
            });
        });
    }

    function loadPlaceHolder() {
        const imageRows = gradioApp().querySelectorAll('.cnet-image-row');
        imageRows.forEach(imageRow => {
            if (cnetOpenposeEditorRegisteredElements.has(imageRow)) return;
            cnetOpenposeEditorRegisteredElements.add(imageRow);

            const generatedImageGroup = imageRow.querySelector('.cnet-generated-image-group');
            const editButton = generatedImageGroup.querySelector('.cnet-edit-pose');
            const modalContent = generatedImageGroup.querySelector('.cnet-modal-content');

            modalContent.classList.add('alert');
            modalContent.innerHTML = `
        <div>
            <p>Openpose editor not found. Please make sure you have an openpose
            editor available on /openpose_editor_index. To hide the edit button,
            you can check "Disable openpose edit" in Settings.<br>
            
            Following extension(s) provide integration with ControlNet:</p>
            <ul style="list-style-type:none;">
                <li><a href="https://github.com/huchenlei/sd-webui-openpose-editor">
                    huchenlei/sd-webui-openpose-editor</a></li>
            </ul>
        </div>
        `;

            editButton.innerHTML = '<del>' + editButton.innerHTML + '</del>';
        });
    }

    checkEditorAvailable().then(editorAvailable => {
        onUiUpdate(() => {
            if (editorAvailable)
                loadOpenposeEditor();
            else
                loadPlaceHolder();
        });
    });
})();