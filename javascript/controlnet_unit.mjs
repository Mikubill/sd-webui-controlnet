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

export class ControlNetUnit {
  constructor(index, tab, accordion) {
    this.index = index;
    this.tab = tab;
    this.accordion = accordion;
    this.isImg2Img = tab.querySelector('.cnet-unit-enabled').id.includes('img2img');

    this.enabledCheckbox = tab.querySelector('.cnet-unit-enabled input');
    this.inputImage = tab.querySelector('.cnet-input-image-group .cnet-image input[type="file"]');
    this.inputImageContainer = tab.querySelector('.cnet-input-image-group .cnet-image');
    this.inputImageGroup = tab.querySelector('.cnet-input-image-group');
    this.controlTypeSelector = tab.querySelectorAll('.controlnet_control_type_filter_group input');
    this.resizeModeRadios = tab.querySelectorAll('.controlnet_resize_mode_radio input[type="radio"]');
    this.runPreprocessorButton = tab.querySelector('.cnet-run-preprocessor');
    this.generatedImageGroup = tab.querySelector('.cnet-generated-image-group');
    this.poseEditButton = tab.querySelector('.cnet-edit-pose');
    this.allowPreviewCheckbox = tab.querySelector('.cnet-allow-preview input');
    this.effectiveRegionMaskImage = tab.querySelector(".cnet-effective-region-mask");

    const tabs = tab.parentNode;
    this.tabNav = tabs.querySelector('.tab-nav');
    this.tabIndex = childIndex(tab) - 1; // -1 because tab-nav is also at the same level.

    this.activeStateChangeCallbacks = [];
    this.attachEnabledButtonListener();
    this.attachControlTypeRadioListener();
    this.attachTabNavChangeObserver();
    this.attachImageUploadListener();
    this.attachImageStateChangeObserver();
    this.attachA1111SendInfoObserver();
  }

  getTabNavButton() {
    return this.tabNav.querySelector(`:nth-child(${this.tabIndex + 1})`);
  }

  // Control type selector can be
  // - Radio
  // - Dropdown
  controlTypeSelectorIsDropdown() {
    return this.controlTypeSelector.length == 1;
  }

  getActiveControlType() {
    if (this.controlTypeSelectorIsDropdown()) {
      return this.controlTypeSelector[0].value;
    }

    for (let radio of this.controlTypeSelector) {
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

  updateActiveUnitCount() {
    function getActiveUnitCount(checkboxes) {
      let activeUnitCount = 0;
      for (const checkbox of checkboxes) {
        if (checkbox.checked)
          activeUnitCount++;
      }
      return activeUnitCount;
    }

    const checkboxes = this.accordion.querySelectorAll('.cnet-unit-enabled input');
    const span = this.accordion.querySelector('.label-wrap span');

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

  isActive() {
    return this.getTabNavButton().classList.contains('selected');
  }

  onActiveStateChange(callback) {
    this.activeStateChangeCallbacks.push(callback);
  }

  isEnabled() {
    return this.enabledCheckbox.checked;
  }

  onEnabledStateChange(callback) {
    this.enabledCheckbox.addEventListener('change', callback);
  }

  attachEnabledButtonListener() {
    this.enabledCheckbox.addEventListener('change', () => {
      this.updateActiveState();
      this.updateActiveUnitCount();
    });
  }

  attachControlTypeRadioListener() {
    if (this.controlTypeSelectorIsDropdown()) {
      const input = this.controlTypeSelector[0];
      const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
      const tab = this;
      Object.defineProperty(input, "value", {
        get: desc.get,
        set: function (v) {
          desc.set.call(this, v);
          tab.updateActiveControlType();
        },
      });
      return;
    }
    for (const radio of this.controlTypeSelector) {
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
          this.activeStateChangeCallbacks.forEach(cb => cb());
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

    // Automatically check `enable` checkbox when JSON pose file is uploaded.
    this.tab.querySelector('.cnet-upload-pose input').addEventListener('change', (event) => {
      if (!event.target.files) return;
      if (!this.enabledCheckbox.checked)
        this.enabledCheckbox.click();
    });
  }

  attachImageStateChangeObserver() {
    new MutationObserver((mutationsList) => {
      const changeObserved = imgChangeObserved(mutationsList);

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

  /**
   * Observe send PNG info buttons in A1111, as they can also directly
   * set states of ControlNetUnit.
   */
  attachA1111SendInfoObserver() {
    const pasteButtons = gradioApp().querySelectorAll('#paste');
    const pngButtons = gradioApp().querySelectorAll(
      this.isImg2Img ?
        '#img2img_tab, #inpaint_tab' :
        '#txt2img_tab'
    );

    for (const button of [...pasteButtons, ...pngButtons]) {
      button.addEventListener('click', () => {
        // The paste/send img generation info feature goes
        // though gradio, which is pretty slow. Ideally we should
        // observe the event when gradio has done the job, but
        // that is not an easy task.
        // Here we just do a 2 second delay until the refresh.
        setTimeout(() => {
          this.updateActiveState();
          this.updateActiveUnitCount();
        }, 2000);
      });
    }
  }
}
