function close_main_window() {
	// Hole die aktuelle URL
	const url = new URL(window.location.href);

	// Prüfe, ob run_nr gesetzt ist, dann entferne es
	if (url.searchParams.has('run_nr')) {
		url.searchParams.delete('run_nr');
	}

	// Wenn run_nr nicht gesetzt ist, entferne stattdessen experiment_name
	else if (url.searchParams.has('experiment_name')) {
		url.searchParams.delete('experiment_name');
	}

	// Entferne abschließend user_id, wenn vorhanden
	if (url.searchParams.has('user_id')) {
		url.searchParams.delete('user_id');
	}

	// Aktualisiere die URL und verhalte dich wie ein Link-Klick
	window.location.assign(url.toString()); // Wie ein Klick auf einen neuen Link
}

function show_main_window() {
	document.getElementById('spinner').style.display = 'none';
	document.getElementById('main_window').style.display = 'block';
}

fetchData().then(data => {
	createScatter2D(data);
	createScatter3D(data);
	createParallelPlot(data);
	createTable(data);

	show_main_window();
});

function initialize_tabs () {
	function setupTabs(container) {
		const tabs = container.querySelectorAll('[role="tab"]');
		const tabPanels = container.querySelectorAll('[role="tabpanel"]');

		if (tabs.length === 0 || tabPanels.length === 0) {
			return;
		}

		// Automatisch den ersten Tab aktiv setzen
		tabs.forEach(tab => tab.setAttribute("aria-selected", "false"));
		tabPanels.forEach(panel => panel.hidden = true);

		const firstTab = tabs[0];
		const firstPanel = tabPanels[0];

		if (firstTab && firstPanel) {
			firstTab.setAttribute("aria-selected", "true");
			firstPanel.hidden = false;
		}

		// Event-Listener für Tab-Wechsel
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

					// Falls sich darin ein weiterer `.tabs`-Container befindet, aktiviere auch dessen ersten Tab
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

document.addEventListener("DOMContentLoaded", initialize_tabs);

if($("#spinner").length) {
	document.getElementById('spinner').style.display = 'block';
}
