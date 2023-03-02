cnet_titles = {
	"⏫": "Set this size as final processing output size",
	"🔄": "Refresh",
	"⇅": "Switch height/width",
	"📷": "Enable WebCam",
	"⇄": "Mirror WebCam"

}

onUiUpdate(function () {
	gradioApp().querySelectorAll('.gr-button-tool').forEach(function (span) {
		tooltip = cnet_titles[span.textContent];

		if (!tooltip) {
			tooltip = cnet_titles[span.value];
		}

		if (!tooltip) {
			for (const c of span.classList) {
				if (c in cnet_titles) {
					tooltip = cnet_titles[c];
					break;
				}
			}
		}

		if (tooltip) {
			span.title = tooltip;
		}
	})

	gradioApp().querySelectorAll('select').forEach(function (select) {
		if (select.onchange != null) return;

		select.onchange = function () {
			select.title = titles[select.value] || "";
		}
	})

})