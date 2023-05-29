onUiUpdate(function () {
    // mouseover tooltips for various UI elements
    const titles = {
        '🔄': 'Refresh',
        '\u2934': 'Send dimensions to stable diffusion',
        '💥': 'Run preprocessor',
        '📝': 'Open new canvas',
        '📷': 'Enable webcam',
        '⇄': 'Mirror webcam',
        '⬅': 'Use preview image as input image',
    };
    gradioApp().querySelectorAll('.cnet-toolbutton').forEach(function (button) {
        const tooltip = titles[button.textContent];
        if (tooltip) {
            button.title = tooltip;
        }
    })
});