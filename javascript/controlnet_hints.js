cnet_titles = {
	"⏫": "Set this size as final processing output size",
	"🔄": "Refresh",
	"⇅": "Switch height/width",
	"📷": "Enable WebCam",
	"⇄": "Mirror WebCam",
	"🆕":"Create empty canvas (using size from below canvas size)",
	"👁":"Toggle Annotator preview on/off",
	"↔\u00a0512": "Set width to 512 and height accordingly",
	"↔\u00a0768": "Set width to 768 and height accordingly",
	"↕\u00a0512": "Set height to 512 and width accordingly",
	"↕\u00a0768": "Set height to 768 and width accordingly"
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