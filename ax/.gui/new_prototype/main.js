fetchData().then(data => {
	// Hide the spinner when data is loaded
	document.getElementById('spinner').style.display = 'none';

	createScatter2D(data);
	createScatter3D(data);
	createParallelPlot(data);
	createTable(data);

	document.getElementById('scatter2d').on('plotly_relayout', (eventData) =>
		filterTableOnZoom(eventData, data, 'learning_rate', 'accuracy')
	);

	// Set up tab navigation
	document.querySelectorAll('.tab-button').forEach(button => {
		button.addEventListener('click', () => {
			showTab(button.getAttribute('data-tab'));
		});
	});
});

// Show spinner while data is loading
document.getElementById('spinner').style.display = 'block';
