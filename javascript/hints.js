// mouseover tooltips for various UI elements
titles_controlnet = {
    '🔄': 'Refresh',
    '\u2934': 'Send dimensions to stable diffusion',
    '💥': 'Run preprocessor',
    '📝': 'Open new canvas',
    '📷': 'Enable webcam',
    '⇄': 'Mirror webcam',
};

onUiUpdate(function(){
	gradioApp().querySelectorAll('.cnet-toolbutton').forEach(function(button){
		tooltip = titles_controlnet[button.textContent];
		if(tooltip){
			button.title = tooltip;
		}
	})
});