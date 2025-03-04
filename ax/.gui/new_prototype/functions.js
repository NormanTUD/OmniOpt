async function fetchData() {
	const response = await fetch('data.php');
	return response.json();
}

function enable_dark_mode() {
	document.body.classList.add('dark-mode');
}

function disable_dark_mode() {
	document.body.classList.remove('dark-mode');
}

function createScatter2D(data) {
	if(!$("#scatter2d").length) {
		console.warn("#scatter2d not found");
		return;
	}
	const minVal = Math.min(...data.map(d => d.accuracy));
	const maxVal = Math.max(...data.map(d => d.accuracy));

	// Farbskala von Rot (max) bis Grün (min)
	const colorscale = 'RdYlGn'; // Farbskala (rot bis grün)

	Plotly.newPlot('scatter2d', [{
		x: data.map(d => d.learning_rate),
		y: data.map(d => d.accuracy),
		mode: 'markers',
		type: 'scatter',
		marker: {
			color: data.map(d => d.accuracy),  // Farbe basierend auf Accuracy
			colorscale: colorscale,  // Farbskala
			colorbar: {  // Farbbalken
				title: 'Accuracy',
				tickvals: [minVal, (minVal + maxVal) / 2, maxVal],
				ticktext: [minVal.toFixed(2), ((minVal + maxVal) / 2).toFixed(2), maxVal.toFixed(2)],
				tickmode: 'array'
			}
		}
	}], {
		title: '2D Scatter Plot',
		plot_bgcolor: 'rgba(0, 0, 0, 0)',  // Hintergrund transparent
		paper_bgcolor: 'rgba(0, 0, 0, 0)',  // Papierhintergrund transparent
		showlegend: true,  // Legende anzeigen
		legend: {
			x: 0.8,  // Position der Legende (x-Achse)
			y: 0.9,  // Position der Legende (y-Achse)
			title: 'Accuracy',  // Titel der Legende
			font: {
				size: 12
			}
		}
	});

	document.getElementById('scatter2d').on('plotly_relayout', (eventData) =>
		filterTableOnZoom(eventData, data, 'learning_rate', 'accuracy')
	);
}

function createScatter3D(data) {
	if(!$("#scatter3d").length) {
		console.warn("#scatter3d not found");
		return;
	}
	Plotly.newPlot('scatter3d', [{
		x: data.map(d => d.learning_rate),
		y: data.map(d => d.batch_size),
		z: data.map(d => d.accuracy),
		mode: 'markers',
		type: 'scatter3d'
	}], {
		title: '3D Scatter Plot',
		lot_bgcolor: 'rgba(0, 0, 0, 0)',
		paper_bgcolor: 'rgba(0, 0, 0, 0)'
	});
}

function createParallelPlot(data) {
	if(!$("#parallel").length) {
		console.warn("#parallel not found");
		return;
	}
	Plotly.newPlot('parallel', [{
		type: 'parcoords',
		dimensions: [
			{ label: 'Learning Rate', values: data.map(d => d.learning_rate) },
			{ label: 'Batch Size', values: data.map(d => d.batch_size) },
			{ label: 'Accuracy', values: data.map(d => d.accuracy) }
		]
	}], {
		title: 'Parallel Coordinates',
		lot_bgcolor: 'rgba(0, 0, 0, 0)',
		paper_bgcolor: 'rgba(0, 0, 0, 0)'
	});
}

function createTable(data, headers, table_name) {
	if (!$("#" + table_name).length) {
		console.warn("#" + table_name + " not found");
		return;
	}

	new gridjs.Grid({
		columns: headers,
		data: data,
		search: true,
		sort: true
	}).render(document.getElementById(table_name));
}

function filterTableOnZoom(eventData, data, keyX, keyY) {
	if(!$("#table").length) {
		console.warn("#table not found");
		return;
	}
	const xRange = eventData['xaxis.range'];
	const yRange = eventData['yaxis.range'];
	if (!xRange || !yRange) return;

	const filtered = data.filter(row =>
		row[keyX] >= xRange[0] && row[keyX] <= xRange[1] &&
		row[keyY] >= yRange[0] && row[keyY] <= yRange[1]
	);

	document.getElementById('table').innerHTML = '';
	createTable(filtered);
}

function copy_to_clipboard_base64(text) {
	copy_to_clipboard(atob(text));
}

function copy_to_clipboard(text) {
	if (!navigator.clipboard) {
		// Fallback für ältere Browser
		let textarea = document.createElement("textarea");
		textarea.value = text;
		document.body.appendChild(textarea);
		textarea.select();
		try {
			document.execCommand("copy");
		} catch (err) {
			console.error("Copy failed:", err);
		}
		document.body.removeChild(textarea);
		return;
	}

	navigator.clipboard.writeText(text).then(() => {
		console.log("Text copied to clipboard");
	}).catch(err => {
		console.error("Failed to copy text:", err);
	});
}
