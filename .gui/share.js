"use strict";

function show_main_window() {
	document.getElementById('spinner').style.display = 'none';
	document.getElementById('main_window').style.display = 'contents';
}

function initialize_tabs () {
	function setupTabs(container) {
		const tabs = container.querySelectorAll('[role="tab"]');
		const tabPanels = container.querySelectorAll('[role="tabpanel"]');

		if (tabs.length === 0 || tabPanels.length === 0) {
			return;
		}

		tabs.forEach(tab => tab.setAttribute("aria-selected", "false"));
		tabPanels.forEach(panel => panel.hidden = true);

		const firstTab = tabs[0];
		const firstPanel = tabPanels[0];

		if (firstTab && firstPanel) {
			firstTab.setAttribute("aria-selected", "true");
			firstPanel.hidden = false;
		}

		tabs.forEach(tab => {
			tab.addEventListener("click", function () {
				const parentContainer = tab.closest(".tabs");

				const parentTabs = parentContainer.querySelectorAll('[role="tab"]');
				const parentPanels = parentContainer.querySelectorAll('[role="tabpanel"]');

				parentTabs.forEach(t => t.setAttribute("aria-selected", "false"));
				parentPanels.forEach(panel => panel.hidden = true);

				this.setAttribute("aria-selected", "true");
				const targetPanel = document.getElementById(this.getAttribute("aria-controls"));
				if (targetPanel) {
					targetPanel.hidden = false;

					const nestedTabs = targetPanel.querySelector(".tabs");
					if (nestedTabs) {
						setupTabs(nestedTabs);
					}
				}
			});
		});
	}

	document.querySelectorAll(".tabs").forEach(setupTabs);
}

function showFullscreenSpinnerWithMessage(message) {
	try {
		var spinnerOverlayId = 'fullscreen-spinner-overlay';

		if (document.getElementById(spinnerOverlayId)) {
			return;
		}

		if (!document.getElementById('fullscreen-spinner-style')) {
			var styleEl = document.createElement('style');
			styleEl.type = 'text/css';
			styleEl.id = 'fullscreen-spinner-style';
			var keyFrames = '@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }';
			styleEl.appendChild(document.createTextNode(keyFrames));
			document.head.appendChild(styleEl);
		}

		var overlay = document.createElement('div');
		overlay.classList.add('invert_in_dark_mode');
		overlay.id = spinnerOverlayId;
		overlay.style.position = 'fixed';
		overlay.style.top = '0';
		overlay.style.left = '0';
		overlay.style.width = '100vw';
		overlay.style.height = '100vh';
		overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.85)';
		overlay.style.display = 'flex';
		overlay.style.flexDirection = 'column';
		overlay.style.justifyContent = 'center';
		overlay.style.alignItems = 'center';
		overlay.style.zIndex = '99999';

		var spinner = document.createElement('div');
		spinner.style.border = '16px solid #f3f3f3';
		spinner.style.borderTop = '16px solid #3498db';
		spinner.style.borderRadius = '50%';
		spinner.style.width = '120px';
		spinner.style.height = '120px';
		spinner.style.animation = 'spin 2s linear infinite';

		var text = document.createElement('div');
		text.style.color = 'white';
		text.style.fontSize = '1.5rem';
		text.style.marginTop = '20px';
		text.style.textAlign = 'center';
		text.textContent = message || '';

		overlay.appendChild(spinner);
		overlay.appendChild(text);

		document.body.appendChild(overlay);
	} catch (error) {
		console.error('Error at showing the spinner:', error);
	}
}

function hideFullscreenSpinner() {
	try {
		var spinnerOverlayId = 'fullscreen-spinner-overlay';
		var overlay = document.getElementById(spinnerOverlayId);
		if (overlay) {
			overlay.parentNode.removeChild(overlay);
		}
	} catch (error) {
		console.error('Error at removing the spinners:', error);
	}
}
