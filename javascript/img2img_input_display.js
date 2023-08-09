/**
 * Display the currently active img2img tab's input image as an uninteractable
 * background image behind ControlNet's image input. This change will hint user
 * that if no ControlNet input image is uploaded, by default ControlNet will
 * fallback onto img2img input image.
 */
(function () {
    function getActiveImg2ImgTabImgSrc(img2imgTabs) {
        const tabs = img2imgTabs.querySelectorAll('.tabitem');
        const activeTabs = [...tabs].filter(tab => tab.style.display !== 'none');
        if (!activeTabs) return;
        const image = activeTabs[0].querySelector('.image-container img')
        return image ? image.src : undefined;
    }

    function updateControlNetInputFallbackPreview(cnetInputContainers, imgDataURL) {
        for (const container of cnetInputContainers) {
            const badge = container.querySelector('.cnet-badge');
            if (badge) badge.remove();

            if (imgDataURL) {
                // Do not add fallback image if controlnet input already exists.
                if (container.querySelector('img')) {
                    continue;
                }
                // Set the background image
                container.style.backgroundImage = `url('${imgDataURL}')`;

                // Set other background properties
                container.style.backgroundPosition = 'center';
                container.style.backgroundRepeat = 'no-repeat';
                container.style.backgroundSize = 'contain';
                container.title = "Img2Img input will be used if no ControlNet input is specified.";

                const div = document.createElement('div');
                div.classList.add('cnet-badge', 'primary', 'cnet-a1111-badge');
                div.innerHTML = 'A1111';
                container.appendChild(div);
            } else {
                container.style.backgroundImage = 'none';
            }
        }
    }

    const ImgChangeType = {
        NO_CHANGE: 0,
        REMOVE: 1,
        ADD: 2,
        SRC_CHANGE: 3,
    };

    function imgChangeObserved(mutationsList) {
        // Iterate over all mutations that just occured
        for (let mutation of mutationsList) {
            // Check if the mutation is an addition or removal of a node
            if (mutation.type === 'childList') {
                // Check if nodes were added
                if (mutation.addedNodes.length > 0) {
                    for (const node of mutation.addedNodes) {
                        if (node.tagName === 'IMG') {
                            return ImgChangeType.ADD;
                        }
                    }
                }

                // Check if nodes were removed
                if (mutation.removedNodes.length > 0) {
                    for (const node of mutation.removedNodes) {
                        if (node.tagName === 'IMG') {
                            return ImgChangeType.REMOVE;
                        }
                    }
                }
            }
            // Check if the mutation is a change of an attribute
            else if (mutation.type === 'attributes') {
                if (mutation.target.tagName === 'IMG' && mutation.attributeName === 'src') {
                    return ImgChangeType.SRC_CHANGE;
                }
            }
        }
        return ImgChangeType.NO_CHANGE;
    }

    let callback_registered = false;
    onUiUpdate(() => {
        if (callback_registered) return;

        const cnetInputContainers = gradioApp().querySelectorAll(
            "#img2img_controlnet_tabs .cnet-input-image-group .cnet-image");
        if (!cnetInputContainers) return;

        const img2imgTabs = gradioApp().querySelector("#mode_img2img");
        if (!img2imgTabs) return;

        // Every time img2img input updates, update fallback preview.
        const img2imgContainers = img2imgTabs.querySelectorAll('.tabitem .image-container');
        for (const container of img2imgContainers) {
            new MutationObserver((mutationsList) => {
                if (imgChangeObserved(mutationsList) !== ImgChangeType.NO_CHANGE) {
                    updateControlNetInputFallbackPreview(
                        cnetInputContainers,
                        getActiveImg2ImgTabImgSrc(img2imgTabs)
                    );
                    return;
                }
            }).observe(container, {
                childList: true,
                attributes: true,
                attributeFilter: ['src'],
                subtree: true,
            });
        }

        // Every time controlnet input updates, update fallback preview.
        for (const container of cnetInputContainers) {
            new MutationObserver((mutationsList) => {
                const changeObserved = imgChangeObserved(mutationsList);
                if (changeObserved === ImgChangeType.REMOVE) {
                    updateControlNetInputFallbackPreview(
                        [container],
                        getActiveImg2ImgTabImgSrc(img2imgTabs)
                    );
                    return;
                }
                if (changeObserved === ImgChangeType.ADD ||
                    changeObserved === ImgChangeType.SRC_CHANGE) {
                    updateControlNetInputFallbackPreview(
                        [container],
                        undefined
                    );
                    return;
                }
            }).observe(container, {
                childList: true,
                attributes: true,
                attributeFilter: ['src'],
                subtree: true,
            });
        }

        // Every time the img2img tab is switched, update fallback preview.
        new MutationObserver((mutationsList) => {
            for (const mutation of mutationsList) {
                if (mutation.type === 'childList') {
                    updateControlNetInputFallbackPreview(
                        cnetInputContainers,
                        getActiveImg2ImgTabImgSrc(img2imgTabs)
                    );
                    return;
                }
            }
        }).observe(img2imgTabs.querySelector('.tab-nav'), { childList: true });

        callback_registered = true;
    });
})();