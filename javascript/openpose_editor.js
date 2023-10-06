(function () {
    async function checkEditorAvailable() {
        const LOCAL_EDITOR_PATH = '/openpose_editor_index';
        const REMOTE_EDITOR_PATH = 'https://huchenlei.github.io/sd-webui-openpose-editor/';

        async function testEditorPath(path) {
            const res = await fetch(path);
            return res.status === 200 ? path : undefined;
        }

        // Use local editor if the user has the extension installed. Fallback 
        // onto remote editor if the local editor is not ready yet.
        // See https://github.com/huchenlei/sd-webui-openpose-editor/issues/53
        // for more details.
        return await testEditorPath(LOCAL_EDITOR_PATH) || await testEditorPath(REMOTE_EDITOR_PATH);
    }

    const cnetOpenposeEditorRegisteredElements = new Set();
    function loadOpenposeEditor(editorURL) {
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
                const darkThemeParam = document.body.classList.contains('dark') ?
                    new URLSearchParams({ theme: 'dark' }).toString() :
                    '';

                window.addEventListener('message', (event) => {
                    const message = event.data;
                    if (message['ready']) resolve();
                }, { once: true });

                if (getPathname(iframe.src) !== editorURL) {
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
            <p>
                OpenPose editor not found. Please make sure you have an OpenPose editor available on <code>/openpose_editor_index</code>. To hide the edit button, check "Disable openpose edit" in Settings.<br>
                <br>
                The following extension(s) provide integration with ControlNet:
            </p>
            <ul>
                <li>
                    <a href="https://github.com/huchenlei/sd-webui-openpose-editor">huchenlei/sd-webui-openpose-editor</a>
                </li>
            </ul>
        </div>
        `;

            editButton.innerHTML = '<del>' + editButton.innerHTML + '</del>';
        });
    }

    checkEditorAvailable().then(editorURL => {
        onUiUpdate(() => {
            if (editorURL)
                loadOpenposeEditor(editorURL);
            else
                loadPlaceHolder();
        });
    });
})();