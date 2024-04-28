class GradioSlider {
  constructor(container) {
    this.container = container;
    this.number_input = container.querySelector('input[type="number"]');
    this.slider_input = container.querySelector('input[type="range"]');
  }

  getValue() {
    return this.number_input.value;
  }

  onChange(callback) {
    this.number_input.addEventListener("change", callback);
    this.slider_input.addEventListener("change", callback);
  }
}

export class A1111Context {
  constructor(appRoot, generationType) {
    this.width_slider = new GradioSlider(appRoot.querySelector(`#${generationType}_width`));
    this.height_slider = new GradioSlider(appRoot.querySelector(`#${generationType}_height`));
  }
};