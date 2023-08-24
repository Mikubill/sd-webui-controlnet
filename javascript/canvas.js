(function () {
    var hasApplied = false;
    onUiUpdate(function () {
        if (!hasApplied) {
            if (typeof window.applyZoomAndPanIntegration === "function") {  // webui 1.6.0 
                hasApplied = true;
                window.applyZoomAndPanIntegration("#txt2img_controlnet",Array.from({ length: 20 }, (_, i) => `#txt2img_controlnet_ControlNet-${i}_input_image`));
                window.applyZoomAndPanIntegration("#img2img_controlnet",Array.from({ length: 20 }, (_, i) => `#img2img_controlnet_ControlNet-${i}_input_image`));
                if (typeof window.applyZoomAndPan === "function") {  // check is nested because window.applyZoomAndPan was added in webui 1.4.0
                    window.applyZoomAndPan("#txt2img_controlnet_ControlNet_input_image");
                    window.applyZoomAndPan("#img2img_controlnet_ControlNet_input_image");
                }
                //console.log("window.applyZoomAndPanIntegration applied.");
            } else {
                //console.log("window.applyZoomAndPanIntegration is not available.");
            }
        }
    });
})();
