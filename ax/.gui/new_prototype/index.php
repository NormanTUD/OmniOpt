<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hyperparameter Dashboard</title>
  <link href="https://unpkg.com/tabler@latest/dist/css/tabler.min.css" rel="stylesheet">
  <style>
    body.dark-mode { background-color: #1e1e1e; color: #fff; }
    .plot-container { margin-bottom: 2rem; }
    .spinner {
      border: 4px solid #f3f3f3;
      border-top: 4px solid #3498db;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 2s linear infinite;
      margin: auto;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    .tabs { margin-bottom: 20px; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .tab-button.active { background-color: #3498db; color: white; }
  </style>
<?php
	$theme = "xp";

	if($_GET["theme"] == 98) {
		$theme = "98";
	}
?>
  <link rel="stylesheet" href="https://unpkg.com/<?php print $theme; ?>.css">
</head>
<body>
  <div class="page window" style='font-family: sans-serif'>
	  <div class="title-bar">
        <div class="title-bar-text">
		Hyperparameter Optimization Dashboard
        </div>

        <div class="title-bar-controls">
          <!--<button aria-label="Minimize"></button>
	  <button aria-label="Maximize"></button>-->
          <button aria-label="Close"></button>
        </div>
      </div>
    <div class="container py-4">
      <h1 class="mb-4"></h1>
      
      <!-- Tab navigation -->
      <div class="tabs">
        <button class="tab-button active" data-tab="scatter2d">2D Scatter</button>
        <button class="tab-button" data-tab="scatter3d">3D Scatter</button>
        <button class="tab-button" data-tab="parallel">Parallel Coordinates</button>
        <button class="tab-button" data-tab="table">Data Table</button>
        <button class="tab-button" data-tab="logs">Logs</button>
      </div>

      <!-- Tab content -->
      <div class="tab-content active" id="scatter2d"></div>
      <div class="tab-content" id="scatter3d"></div>
      <div class="tab-content" id="parallel"></div>
      <div class="tab-content" id="table"></div>
	<div class="tab-content" id="logs">
		<div class="tabs">
			<button class="tab-button active" data-tab="output1">A</button>
			<button class="tab-button" data-tab="output2">B</button>
		</div>
		<div class="tab-content" id="output1">
			<pre>Hallo</pre>
		</div>
		<div class="tab-content" id="output2">
			<pre>Hallo</pre>
		</div>
	</div>

      <!-- Loading spinner -->
      <div id="spinner" class="spinner"></div>
    </div>
  </div>

  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script src="https://unpkg.com/gridjs/dist/gridjs.umd.js"></script>
  <link href="https://unpkg.com/gridjs/dist/theme/mermaid.min.css" rel="stylesheet" />

  <script>
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
	}


	function createScatter3D(data) {
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

	function createTable(data) {
		new gridjs.Grid({
		columns: Object.keys(data[0]),
			data: data.map(Object.values),
			search: true,
			pagination: true,
			sort: true
	}).render(document.getElementById('table'));
	}

	function filterTableOnZoom(eventData, data, keyX, keyY) {
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

	function showTab(tabId) {
		const tabs = document.querySelectorAll('.tab-content');
		tabs.forEach(tab => tab.classList.remove('active'));

		const tabButtons = document.querySelectorAll('.tab-button');
		tabButtons.forEach(button => button.classList.remove('active'));

		document.getElementById(tabId).classList.add('active');
		const activeButton = document.querySelector(`[data-tab="${tabId}"]`);
		activeButton.classList.add('active');
	}

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
  </script>
</body>
</html>
