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
	const tabs = document.querySelectorAll('[role="tab"]');
	const tabPanels = document.querySelectorAll('[role="tabpanel"]');

	tabs.forEach(tab => {
		tab.addEventListener("click", function () {
			tabs.forEach(t => t.setAttribute("aria-selected", "false"));
			tabPanels.forEach(panel => panel.hidden = true);

			this.setAttribute("aria-selected", "true");
			const targetPanel = document.getElementById(this.getAttribute("aria-controls"));
			if (targetPanel) {
				targetPanel.hidden = false;
			}
		});
	});
});

document.getElementById('spinner').style.display = 'block';
