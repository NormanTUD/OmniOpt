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

	pre {
		color: green;
		background-color: black !important;
	}
  </style>
	<?php include("css.php"); ?>
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
    <div class="container py-4 window-body has-space">
      <h1 class="mb-4"></h1>

<section class="tabs" style="max-width: 500px">
  <menu role="tablist" aria-label="Sample Tabs">
    <button role="tab" aria-selected="true" aria-controls="tab-A">Tab A</button>
    <button role="tab" aria-controls="tab-B">Tab B</button>
    <button role="tab" aria-controls="tab-C">Tab C</button>
  </menu>
  <!-- the tab content -->
  <article role="tabpanel" id="tab-A">
    <h3>Tab Content</h3>
    <p>
      You create the tabs, you would use a <code>menu role="tablist"</code> element then for the tab titles you use a <code>button</code> with the <code>aria-controls</code> parameter set to match the relative <code>role="tabpanel"</code>'s element.
    </p>
    <p>
      Read more at <a href="https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles/Tab_Role" target="_blank">MDN Web docs - ARIA: tab role</a>
    </p>

    <pre>Microsoft&#10094;R&#10095; Windows DOS
&#10094;C&#10095; Copyright Microsoft Corp 1990-2001.
      <br>C:&#92;WINDOWS&#92;SYSTEM32> You can build a command line easily with a window and pre tag
    </pre>
  </article>
  <article role="tabpanel" hidden id="tab-B">
    <h3>More...</h3>
    <p>This tab contains a GroupBox</p>
    <fieldset>
      <legend>Today's mood</legend>
      <div class="field-row">
        <input id="radio10" type="radio" name="fieldset-example2">
        <label for="radio10">Claire Saffitz</label>
      </div>
      <div class="field-row">
        <input id="radio11" type="radio" name="fieldset-example2">
        <label for="radio11">Brad Leone</label>
      </div>
      <div class="field-row">
        <input id="radio12" type="radio" name="fieldset-example2">
        <label for="radio12">Chris Morocco</label>
      </div>
      <div class="field-row">
        <input id="radio13" type="radio" name="fieldset-example2">
        <label for="radio13">Carla Lalli Music</label>
      </div>
    </fieldset>
  </article>
  <article role="tabpanel" hidden id="tab-C">
    <h3>Tab 3</h3>
    <p>Lorem Ipsum Dolor Sit</p>
  </article>
</section>
      
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

  <script src="functions.js"></script>
  <script>
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
