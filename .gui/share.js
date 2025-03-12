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
