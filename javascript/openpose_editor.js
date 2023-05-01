const cnetOpenposeEditorRegisteredElements = new Set();
onUiUpdate(() => {
    // Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
    // will only visible on web page and not sent to python.
    function updateInput(target){
        let e = new Event("input", { bubbles: true })
        Object.defineProperty(e, "target", {value: target})
        target.dispatchEvent(e);
    }

    const imageRows = gradioApp().querySelectorAll('.cnet-image-row');
    imageRows.forEach(imageRow => {
        if (cnetOpenposeEditorRegisteredElements.has(imageRow)) return;
        cnetOpenposeEditorRegisteredElements.add(imageRow);

        const generatedImageGroup = imageRow.querySelector('.cnet-generated-image-group');
        const editButton = generatedImageGroup.querySelector('.cnet-edit-pose');

        editButton.addEventListener('click', () => {
            const inputImageGroup = imageRow.querySelector('.cnet-input-image-group');
            const inputImage = inputImageGroup.querySelector('.cnet-image img');
            const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
            const modalId = editButton.id.replace('cnet-modal-open-', '');
            const modalIframe = generatedImageGroup.querySelector('.cnet-modal iframe');

            modalIframe.contentWindow.postMessage({
                modalId,
                imageURL: inputImage.src,
                poseURL: downloadLink.href,
            }, '*');
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
});