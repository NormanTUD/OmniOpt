"use strict";

function paretoCloudFrontFlags(points, xMinimize, yMinimize) {
	var normalized = points.map(function (point) {
		return {
			x: xMinimize ? point.x : -point.x,
			y: yMinimize ? point.y : -point.y
		};
	});

	var flags = new Array(points.length).fill(true);

	for (var i = 0; i < normalized.length; i++) {
		for (var j = 0; j < normalized.length; j++) {
			if (i === j) {
				continue;
			}

			var candidate = normalized[i];
			var other = normalized[j];

			var otherDominatesCandidate =
				other.x <= candidate.x &&
				other.y <= candidate.y &&
				(other.x < candidate.x || other.y < candidate.y);

			if (otherDominatesCandidate) {
				flags[i] = false;
				break;
			}
		}
	}

	return flags;
}

function paretoCloudHoverText(point, xName, yName, trialIndex, statusIndex) {
	var lines = [];

	if (trialIndex !== -1) {
		lines.push("<b>Trial Index:</b> " + point.row[trialIndex]);
	}

	if (statusIndex !== -1) {
		lines.push("<b>Status:</b> " + point.row[statusIndex]);
	}

	lines.push("<b>" + xName + ":</b> " + point.x);
	lines.push("<b>" + yName + ":</b> " + point.y);

	return lines.join("<br>");
}

function plotParetoCloud() {
	if ($("#plotParetoCloud").data("loaded") == "true") {
		return;
	}

	var plotDiv = document.getElementById("plotParetoCloud");

	if (!plotDiv) {
		console.error("DIV with id 'plotParetoCloud' not found.");
		return;
	}

	plotDiv.innerHTML = "";

	if (!Array.isArray(result_names) || result_names.length < 2) {
		plotDiv.innerHTML = "<div class='caveat warning'>At least two result values are needed to plot a Pareto cloud.</div>";
		return;
	}

	if (!Array.isArray(tab_results_headers_json) || !Array.isArray(tab_results_csv_json)) {
		plotDiv.innerHTML = "<div class='caveat alarm'>Result data is not available.</div>";
		return;
	}

	var trialIndex = tab_results_headers_json.indexOf("trial_index");
	var statusIndex = tab_results_headers_json.indexOf("trial_status");
	var rendered = 0;

	for (var i = 0; i < result_names.length; i++) {
		for (var j = i + 1; j < result_names.length; j++) {
			var xName = result_names[i];
			var yName = result_names[j];
			var xIndex = tab_results_headers_json.indexOf(xName);
			var yIndex = tab_results_headers_json.indexOf(yName);

			if (xIndex === -1 || yIndex === -1) {
				console.warn("Cannot plot Pareto cloud for", xName, yName, ": header not found.");
				continue;
			}

			var points = [];

			for (var r = 0; r < tab_results_csv_json.length; r++) {
				var row = tab_results_csv_json[r];
				var x = parseFloat(row[xIndex]);
				var y = parseFloat(row[yIndex]);

				if (!isFinite(x) || !isFinite(y)) {
					continue;
				}

				points.push({ x: x, y: y, row: row });
			}

			if (!points.length) {
				continue;
			}

			var xMinimize = String(result_min_max[i]).toLowerCase() === "min";
			var yMinimize = String(result_min_max[j]).toLowerCase() === "min";
			var frontFlags = paretoCloudFrontFlags(points, xMinimize, yMinimize);

			var regularPoints = [];
			var frontPoints = [];

			for (var p = 0; p < points.length; p++) {
				if (frontFlags[p]) {
					frontPoints.push(points[p]);
				} else {
					regularPoints.push(points[p]);
				}
			}

			var regularTrace = {
				x: regularPoints.map(function (point) { return point.x; }),
				y: regularPoints.map(function (point) { return point.y; }),
				text: regularPoints.map(function (point) { return paretoCloudHoverText(point, xName, yName, trialIndex, statusIndex); }),
				hoverinfo: "text",
				mode: "markers",
				type: "scatter",
				name: "All points",
				marker: {
					size: get_marker_size(),
					color: "rgba(31, 119, 180, 0.55)"
				}
			};

			var frontTrace = {
				x: frontPoints.map(function (point) { return point.x; }),
				y: frontPoints.map(function (point) { return point.y; }),
				text: frontPoints.map(function (point) { return paretoCloudHoverText(point, xName, yName, trialIndex, statusIndex); }),
				hoverinfo: "text",
				mode: "markers",
				type: "scatter",
				name: "Pareto front",
				marker: {
					size: get_marker_size() * 2.5,
					color: "rgb(255, 127, 14)",
					line: {
						width: 2,
						color: "black"
					}
				}
			};

			var wrapper = document.createElement("div");
			wrapper.style.marginBottom = "30px";

			var titleEl = document.createElement("h3");
			titleEl.textContent = "Pareto-Cloud: " + xName + " (" + result_min_max[i] + ") vs " + yName + " (" + result_min_max[j] + ")";
			wrapper.appendChild(titleEl);

			var infoEl = document.createElement("div");
			infoEl.textContent = frontPoints.length + " of " + points.length + " points are on the Pareto front.";
			wrapper.appendChild(infoEl);

			var plotElement = document.createElement("div");
			plotElement.style.width = "100%";
			plotElement.style.height = "500px";
			wrapper.appendChild(plotElement);
			plotDiv.appendChild(wrapper);

			var layout = {
				xaxis: { title: get_axis_title_data(xName) },
				yaxis: { title: get_axis_title_data(yName) },
				margin: { t: 10, l: 60, r: 20, b: 50 },
				hovermode: "closest",
				showlegend: true,
				height: 500
			};

			Plotly.newPlot(plotElement, [regularTrace, frontTrace], add_default_layout_data(layout, 1));
			rendered++;
		}
	}

	if (!rendered) {
		plotDiv.innerHTML = "<div class='caveat warning'>No valid result pairs could be plotted.</div>";
		return;
	}

	$("#plotParetoCloud").data("loaded", "true");

	if (typeof apply_theme_based_on_system_preferences === 'function') {
		apply_theme_based_on_system_preferences();
	}
}
