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

// Show spinner while data is loading
document.getElementById('spinner').style.display = 'block';
