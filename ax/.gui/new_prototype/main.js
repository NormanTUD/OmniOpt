fetchData().then(data => {

	createScatter2D(data);
	createScatter3D(data);
	createParallelPlot(data);
	createTable(data);

	document.getElementById('scatter2d').on('plotly_relayout', (eventData) =>
		filterTableOnZoom(eventData, data, 'learning_rate', 'accuracy')
	);

	document.getElementById('spinner').style.display = 'none';
	document.getElementById('main_window').style.display = 'block';
});

document.addEventListener("DOMContentLoaded", function () {
	function setupTabs(container) {
		const tabs = container.querySelectorAll('[role="tab"]');
		const tabPanels = container.querySelectorAll('[role="tabpanel"]');

		tabs.forEach(tab => {
			tab.addEventListener("click", function () {
				// Finde das aktuelle Tab-Container-Element (damit es auch für verschachtelte Tabs funktioniert)
				const parentContainer = tab.closest(".tabs");

				// Deaktiviere alle Tabs und verstecke alle Panels im aktuellen Tab-Bereich
				const parentTabs = parentContainer.querySelectorAll('[role="tab"]');
				const parentPanels = parentContainer.querySelectorAll('[role="tabpanel"]');

				parentTabs.forEach(t => t.setAttribute("aria-selected", "false"));
				parentPanels.forEach(panel => panel.hidden = true);

				// Aktuelles Tab aktivieren
				this.setAttribute("aria-selected", "true");
				const targetPanel = document.getElementById(this.getAttribute("aria-controls"));
				if (targetPanel) {
					targetPanel.hidden = false;
				}
			});
		});
	}

	// Setzt Tabs für alle `.tabs`-Container
	document.querySelectorAll(".tabs").forEach(setupTabs);
});

document.getElementById('spinner').style.display = 'block';
