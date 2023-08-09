/**
 * Give a badge on ControlNet Accordion indicating total number of active 
 * units.
 * Make active unit's tab name green.
 * Append control type to tab name.
 * Disable resize mode selection when A1111 img2img input is used.
 */
(function () {
    const cnetAllUnits = new Map/* <Element, ControlNetUnitTab> */();
    const cnetAllAccordions = new Set();
    onUiUpdate(() => {
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

        function childIndex(element) {
            // Get all child nodes of the parent
            let children = Array.from(element.parentNode.childNodes);

            // Filter out non-element nodes (like text nodes and comments)
            children = children.filter(child => child.nodeType === Node.ELEMENT_NODE);

            return children.indexOf(element);
        }

        function imageInputDisabledAlert() {
            alert('Inpaint control type must use a1111 input in img2img mode.');
        }

        class ControlNetUnitTab {
            constructor(tab) {
                this.tab = tab;
                this.isImg2Img = tab.querySelector('.cnet-unit-enabled').id.includes('img2img');

                this.enabledCheckbox = tab.querySelector('.cnet-unit-enabled input');
                this.inputImage = tab.querySelector('.cnet-input-image-group .cnet-image input[type="file"]');
                this.inputImageContainer = tab.querySelector('.cnet-input-image-group .cnet-image');
                this.controlTypeRadios = tab.querySelectorAll('.controlnet_control_type_filter_group input[type="radio"]');
                this.resizeModeRadios = tab.querySelectorAll('.controlnet_resize_mode_radio input[type="radio"]');
                this.runPreprocessorButton = tab.querySelector('.cnet-run-preprocessor');

                const tabs = tab.parentNode;
                this.tabNav = tabs.querySelector('.tab-nav');
                this.tabIndex = childIndex(tab) - 1; // -1 because tab-nav is also at the same level.

                this.attachEnabledButtonListener();
                this.attachControlTypeRadioListener();
                this.attachTabNavChangeObserver();
                this.attachImageUploadListener();
                this.attachImageStateChangeObserver();

                // Initial updates:
                if (this.isImg2Img)
                    this.updateResizeModeState();
            }

            getTabNavButton() {
                return this.tabNav.querySelector(`:nth-child(${this.tabIndex + 1})`);
            }

            getActiveControlType() {
                for (let radio of this.controlTypeRadios) {
                    if (radio.checked) {
                        return radio.value;
                    }
                }
                return undefined;
            }

            updateActiveState() {
                const tabNavButton = this.getTabNavButton();
                if (!tabNavButton) return;

                if (this.enabledCheckbox.checked) {
                    tabNavButton.classList.add('cnet-unit-active');
                } else {
                    tabNavButton.classList.remove('cnet-unit-active');
                }
            }

            /**
             * Add the active control type to tab displayed text.
             */
            updateActiveControlType() {
                const tabNavButton = this.getTabNavButton();
                if (!tabNavButton) return;

                // Remove the control if exists
                const controlTypeSuffix = tabNavButton.querySelector('.control-type-suffix');
                if (controlTypeSuffix) controlTypeSuffix.remove();

                // Add new suffix.
                const controlType = this.getActiveControlType();
                if (controlType === 'All') return;

                const span = document.createElement('span');
                span.innerHTML = `[${controlType}]`;
                span.classList.add('control-type-suffix');
                tabNavButton.appendChild(span);
            }

            /**
             * When 'Inpaint' control type is selected in img2img:
             * - Make image input disabled
             * - Clear existing image input
             */
            updateImageInputState() {
                if (!this.isImg2Img) return;

                const tabNavButton = this.getTabNavButton();
                if (!tabNavButton) return;

                const controlType = this.getActiveControlType();
                if (controlType.toLowerCase() === 'inpaint') {
                    this.inputImage.disabled = true;
                    this.inputImage.parentNode.addEventListener('click', imageInputDisabledAlert);
                    const removeButton = this.tab.querySelector(
                        '.cnet-input-image-group .cnet-image button[aria-label="Remove Image"]');
                    if (removeButton) removeButton.click();
                } else {
                    this.inputImage.disabled = false;
                    this.inputImage.parentNode.removeEventListener('click', imageInputDisabledAlert);
                }
            }

            /**
             * For img2img, disable resize mode selection when using A1111
             * input, as the selected resize mode won't take any effect in
             * the backend when using A1111 input.
             */
            updateResizeModeState() {
                const img = this.inputImageContainer.querySelector('img');
                for (const radio of this.resizeModeRadios) {
                    if (img) {
                        radio.disabled = false;
                        radio.parentNode.classList.remove('cnet-disabled-radio');
                        radio.parentNode.removeAttribute('title');
                    } else {
                        radio.disabled = true;
                        radio.parentNode.classList.add('cnet-disabled-radio');
                        radio.parentNode.title = "Use A1111 resize mode when input is from A1111.";
                    }
                }
            }

            attachEnabledButtonListener() {
                this.enabledCheckbox.addEventListener('change', () => {
                    this.updateActiveState();
                });
            }

            attachControlTypeRadioListener() {
                for (const radio of this.controlTypeRadios) {
                    radio.addEventListener('change', () => {
                        this.updateActiveControlType();
                    });
                }
            }

            /**
             * Each time the active tab change, all tab nav buttons are cleared and
             * regenerated by gradio. So we need to reapply the active states on 
             * them.
             */
            attachTabNavChangeObserver() {
                new MutationObserver((mutationsList) => {
                    for (const mutation of mutationsList) {
                        if (mutation.type === 'childList') {
                            this.updateActiveState();
                            this.updateActiveControlType();
                        }
                    }
                }).observe(this.tabNav, { childList: true });
            }

            attachImageUploadListener() {
                // Automatically check `enable` checkbox when image is uploaded.
                this.inputImage.addEventListener('change', (event) => {
                    if (!event.target.files) return;
                    if (!this.enabledCheckbox.checked)
                        this.enabledCheckbox.click();
                });
            }

            attachImageStateChangeObserver() {
                new MutationObserver((mutationsList) => {
                    const changeObserved = imgChangeObserved(mutationsList);
                    if (changeObserved === ImgChangeType.ADD ||
                        changeObserved === ImgChangeType.REMOVE) {
                        if (this.isImg2Img)
                            this.updateResizeModeState();
                    }

                    if (changeObserved === ImgChangeType.ADD) {
                        // enabling the run preprocessor button
                        this.runPreprocessorButton.removeAttribute("disabled");
                        this.runPreprocessorButton.title = 'Run preprocessor';
                    }

                    if (changeObserved === ImgChangeType.REMOVE) {
                        // disabling the run preprocessor button
                        this.runPreprocessorButton.setAttribute("disabled", true);
                        this.runPreprocessorButton.title = "No ControlNet input image available";
                    }
                }).observe(this.inputImageContainer, {
                    childList: true,
                    subtree: true,
                });
            }
        }

        gradioApp().querySelectorAll('.cnet-unit-tab').forEach(tab => {
            if (cnetAllUnits.has(tab)) return;
            cnetAllUnits.set(tab, new ControlNetUnitTab(tab));
        });

        function getActiveUnitCount(checkboxes) {
            let activeUnitCount = 0;
            for (const checkbox of checkboxes) {
                if (checkbox.checked)
                    activeUnitCount++;
            }
            return activeUnitCount;
        }

        gradioApp().querySelectorAll('#controlnet').forEach(accordion => {
            if (cnetAllAccordions.has(accordion)) return;
            const checkboxes = accordion.querySelectorAll('.cnet-unit-enabled input');
            if (!checkboxes) return;

            const span = accordion.querySelector('.label-wrap span');
            checkboxes.forEach(checkbox => {
                checkbox.addEventListener('change', () => {
                    // Remove existing badge.
                    if (span.childNodes.length !== 1) {
                        span.removeChild(span.lastChild);
                    }
                    // Add new badge if necessary.
                    const activeUnitCount = getActiveUnitCount(checkboxes);
                    if (activeUnitCount > 0) {
                        const div = document.createElement('div');
                        div.classList.add('cnet-badge');
                        div.classList.add('primary');
                        div.innerHTML = `${activeUnitCount} unit${activeUnitCount > 1 ? 's' : ''}`;
                        span.appendChild(div);
                    }
                });
            });
            cnetAllAccordions.add(accordion);
        });
    });
})();