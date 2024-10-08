"use strict";
let logs = [];

function generateStackTraceTable(stacktrace) {
	if (!stacktrace || stacktrace.trim() === "") {
		return "";
	}

	const lines = stacktrace.split("\n").filter(line => line.trim() !== "");

	let html = `
		<table border="1" cellpadding="5" cellspacing="0">
			<thead>
			    <tr>
				<th>Function</th>
				<th>File</th>
				<th>Line</th>
				<th>Column</th>
			    </tr>
			</thead>
		<tbody>
	`;

	lines.forEach(line => {
		const match = line.match(/(.*?)@(.+?):(\d+):(\d+)/);
		if (match) {
			const functionName = match[1] || "(anonymous)";
			const fileName = match[2];
			const lineNumber = match[3];
			const columnNumber = match[4];

			html += `
				<tr class='stacktrace_table'>
					<td>${functionName}</td>
					<td>${fileName}</td>
					<td>${lineNumber}</td>
					<td>${columnNumber}</td>
				</tr>
			`;
		}
	});

	html += `
		</tbody>
	</table>
	`;

	return html;
}

function log(message) {
	console.log(message);
	appendLog('log', message, new Error().stack);
}

function error(message) {
	console.error(message);
	appendLog('error', message, new Error().stack);
}

function warn(message) {
	console.warn(message);
	appendLog('warn', message, new Error().stack);
}

function debug_function(message) {
	console.debug(message);
	appendLog('debug_function', message, new Error().stack);
}

function debug(message) {
	console.debug(message);
	appendLog('debug', message, new Error().stack);
}

function appendLog(type, message, stacktrace) {
	if($("#statusBar").length == 0) { add_status_bar() };

	const timestamp = new Date().toLocaleTimeString();
	const logMessage = `[${timestamp}] ${message}`;
	logs.push({ type, message: logMessage, stacktrace });

	$('#statusLogs').prepend(`
		<div class="log-entry ${type}" data-stacktrace="${stacktrace || ''}">
			${logMessage}
			<div class="stacktrace" style="display:none; font-size:12px; color:#ccc;">${stacktrace ? generateStackTraceTable(stacktrace) : ''}</div>
		</div>
	`);

	$('.log-entry').first().click(function() {
		$(this).find('.stacktrace').slideToggle();
	});

	let statusColor;
	switch(type) {
		case 'log':
			statusColor = '#00ff00';
			break;
		case 'error':
			statusColor = '#ff0000';
			break;
		case 'warn':
			statusColor = '#ffff00';
			break;
		case 'debug_function':
			statusColor = '#ff8800';
			break;
		case 'debug':
			statusColor = '#00ffff';
			break;
		default:
			statusColor = '#ffffff';
	}

	$('#currentStatus').html(`<span style="color:${statusColor};">${message}</span>`);
}

function inject_status_bar_css () {
	const css = `
		#currentStatus {
			cursor: pointer;
		}

		#statusBar {
			position: fixed;
			bottom: 0;
			left: 0;
			width: 100%;
			background-color: #333;
			color: white;
			padding: 10px;
			font-family: Arial, sans-serif;
			z-index: 1000;
			box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.5);
		}

		#statusBar .toggleMenu {
			cursor: pointer;
			float: right;
			margin-right: 10px;
			background-color: #444;
			padding: 5px;
			border-radius: 50%;
		}

		#statusLogs {
			display: none;
			background-color: #222;
			max-height: 200px;
			overflow-y: auto;
			padding: 10px;
			margin-top: 10px;
			border-top: 1px solid #444;
		}

		.log-entry {
			padding: 5px 0;
			font-size: 14px;
			cursor: pointer;
		}

		.log-entry.log {
			color: #00ff00;
		}

		.log-entry.error {
			color: #ff0000;
		}
		
		.log-entry.warn {
			color: #ffff00;
		}

		.log-entry.debug_function {
			color: #ff8800;
		}

		.log-entry.debug {
			color: #00ffff;
		}

		.stacktrace {
			display: none;
			color: #ccc;
		}
	`;

	$('<style></style>').text(css).appendTo('head');
}

function add_status_bar () {
	$('body').append(`
		<div id="statusBar" class="invert_in_dark_mode">
			<span id="currentStatus">Ready</span>
			<div class="toggleMenu">▼</div>
			<div id="statusLogs"></div>
		</div>
	`);

	$('#currentStatus').click(function() {
		$('#statusLogs').slideToggle();
	});

	$('.toggleMenu').click(function() {
		$('#statusLogs').slideToggle();
		$(this).text($(this).text() === '▼' ? '▲' : '▼');
	});

	inject_status_bar_css();

	$("body").css("margin-bottom", "100px")
}

function showSpinnerOverlay(text) {
	if (document.getElementById('spinner-overlay')) {
		return;
	}

	var overlay = document.createElement('div');
	overlay.id = 'spinner-overlay';

	var container = document.createElement('div');
	container.id = 'spinner-container';

	var spinner = document.createElement('div');
	spinner.classList.add('spinner');

	var spinnerText = document.createElement('div');
	spinnerText.id = 'spinner-text';
	spinnerText.innerText = text;

	container.appendChild(spinner);
	container.appendChild(spinnerText);

	overlay.appendChild(container);

	document.body.appendChild(overlay);
}

function removeSpinnerOverlay() {
	var overlay = document.getElementById('spinner-overlay');
	if (overlay) {
		overlay.remove();
	}
}

function copy_to_clipboard(text) {
	var dummy = document.createElement("textarea");
	document.body.appendChild(dummy);
	dummy.value = text;
	dummy.select();
	document.execCommand("copy");
	document.body.removeChild(dummy);
}

function find_closest_element_behind_and_copy_content_to_clipboard (clicked_element, element_to_search_for_class) {
	var prev_element = $(clicked_element).prev();

	while (!$(clicked_element).prev().hasClass(element_to_search_for_class)) {
		prev_element = $(clicked_element).prev();

		if(!prev_element) {
			error(`Could not find ${element_to_search_for_class} from clicked_element:`, clicked_element);
			return;
		}
	}

	var found_element_text = $(prev_element).text();

	copy_to_clipboard(found_element_text);

	var oldText = $(clicked_element).text();

	$(clicked_element).text("✅Raw data copied to clipboard");

	setTimeout(() => {
		$(clicked_element).text(oldText);
	}, 1000);
}

function parse_csv(csv) {
	try {
		var rows = csv.trim().split('\n').map(row => row.split(','));
		return rows;
	} catch (error) {
		error("Error parsing CSV:", error);
		return [];
	}
}

function normalizeArrayLength(array) {
	let maxColumns = array.reduce((max, row) => Math.max(max, row.length), 0);

	return array.map(row => {
		let filledRow = [...row];
		while (filledRow.length < maxColumns) {
			filledRow.push("");
		}
		return filledRow;
	});
}

function create_table_from_csv_data(csvData, table_container, new_table_id, optionalColumnTitles = null) {
	// Parse CSV data
	var data = parse_csv(csvData);
	data = normalizeArrayLength(data);
	var tableContainer = document.getElementById(table_container);

	// Create table element
	var table = document.createElement('table');
	$(table).addClass('display').attr('id', new_table_id);

	// Create header row
	var thead = document.createElement('thead');
	var headRow = document.createElement('tr');
	headRow.classList.add('invert_in_dark_mode');

	var headers = optionalColumnTitles ? optionalColumnTitles : data[0];

	headers.forEach(header => {
		var th = document.createElement('th');
		th.textContent = header.trim();
		headRow.appendChild(th);
	});

	thead.appendChild(headRow);
	table.appendChild(thead);

	// Create body rows
	var tbody = document.createElement('tbody');
	var startRow = optionalColumnTitles ? 0 : 1; // Start at row 0 if optional titles provided

	data.slice(startRow).forEach(row => {
		var tr = document.createElement('tr');
		row.forEach(cell => {
			var td = document.createElement('td');
			td.textContent = cell.trim();
			tr.appendChild(td);
		});
		tbody.appendChild(tr);
	});

	table.appendChild(tbody);
	tableContainer.appendChild(table);

	// Hide the raw CSV data and add a button for toggling visibility
	var rawDataElement = document.createElement('pre');
	rawDataElement.textContent = csvData;
	rawDataElement.style.display = 'none'; // Initially hide raw data

	var toggle_raw_data_button = document.createElement('button');
	toggle_raw_data_button.classList.add("invert_in_dark_mode");
	toggle_raw_data_button.textContent = 'Show Raw Data';

	toggle_raw_data_button.addEventListener('click', function() {
		if (rawDataElement.style.display === 'none') {
			rawDataElement.style.display = 'block';
			toggle_raw_data_button.textContent = 'Hide Raw Data';
		} else {
			rawDataElement.style.display = 'none';
			toggle_raw_data_button.textContent = 'Show Raw Data';
		}
	});


	toggle_raw_data_button.classList.add('toggle_raw_data');

	// Append the button and raw data element under the table
	tableContainer.appendChild(toggle_raw_data_button);
	tableContainer.appendChild(rawDataElement);

	$(document).ready(function() {
		var _new_table_id = `#${new_table_id}`;
		$(_new_table_id).DataTable({
			"ordering": true,  // Enable sorting on all columns
			"paging": false    // Show all entries on one page (no pagination)
		});
	});
}

function initialize_autotables() {
	$('.autotable').each(function(index) {
		var csvText = $(this).text().trim();
		var tableContainer = $(this).parent();

		$(this).hide();

		var _optionalColumnTitles = null;

		if ($(this).data("header_columns")) {
			_optionalColumnTitles = $(this).data("header_columns").split(",");
		}

		var table_container_id = tableContainer.attr('id');

		create_table_from_csv_data(
			csvText, 
			table_container_id, 
			`autotable_${index}`,
			_optionalColumnTitles
		);
	});

	apply_theme_based_on_system_preferences();
}

function generateTOC() {
	try {
		// Check if the TOC div exists
		var $tocDiv = $('#toc');
		if ($tocDiv.length === 0) {
			return;
		}

		// Create the TOC structure
		var $tocContainer = $('<div class="toc"></div>');
		var $tocHeader = $('<h2>Table of Contents</h2>');
		var $tocList = $('<ul></ul>');

		$tocContainer.append($tocHeader);
		$tocContainer.append($tocList);

		// Get all h2, h3, h4, h5, h6 elements
		var headers = $('h2, h3, h4, h5, h6');
		var tocItems = [];

		headers.each(function() {
			var $header = $(this);
			var headerTag = $header.prop('tagName').toLowerCase();
			var headerText = $header.text();
			var headerId = $header.attr('id');

			if (!headerId) {
				headerId = headerText.toLowerCase().replace(/\s+/g, '-');
				$header.attr('id', headerId);
			}

			tocItems.push({
				tag: headerTag,
				text: headerText,
				id: headerId
			});
		});

		// Generate the nested list for TOC
		var currentLevel = 2; // Since we start with h2
		var listStack = [$tocList];

		tocItems.forEach(function(item) {
			var level = parseInt(item.tag.replace('h', ''), 10);
			var $li = $('<li></li>');
			var $a = $('<a></a>').attr('href', '#' + item.id).text(item.text);
			$li.append($a);

			if (level > currentLevel) {
				var $newList = $('<ul></ul>');
				listStack[listStack.length - 1].append($newList);
				listStack.push($newList);
			} else if (level < currentLevel) {
				listStack.pop();
			}

			listStack[listStack.length - 1].append($li);
			currentLevel = level;
		});

		$tocDiv.html($tocContainer);
	} catch (error) {
		error('Error generating TOC:', error);
	}
}

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

function add_tab(tab_id, tab_name, tab_html_content) {
	var container_id = "#main_tabbed";

	if ($(container_id).length) {
		$(container_id).tabs();
	}

	if ($(container_id + " #" + tab_id + "-content").length > 0) {
		//warn(`#${tab_id} already exists.`);
		return;
	}

	if ($(container_id + " ul.ui-tabs-nav").length === 0) {
		$(container_id).prepend('<ul class="ui-tabs-nav ui-helper-hidden ui-helper-clearfix"></ul>');
	}

	var tabButton = $('<li id="' + tab_id + '-tab"><a href="#' + tab_id + '-content">' + tab_name + '</a></li>');
	$(container_id + " ul").append(tabButton);

	var tabContent = $('<div id="' + tab_id + '-content">' + tab_html_content + '</div>');
	$(container_id).append(tabContent);

	$(container_id).tabs("refresh");
	//debug(`Tab ${tab_name} (#${tab_id}) added`);

	open_first_tab_when_none_is_open();
}

function remove_tab(tab_id) {
	var tabSelector = "#" + tab_id + "-tab";
	var contentSelector = "#" + tab_id + "-content";

	if ($(tabSelector).length > 0) {
		$(tabSelector).remove();
	} else {
		warn(`#${tab_id} doesn't exist`);
	}

	if ($(contentSelector).length > 0) {
		$(contentSelector).remove();
	} else {
		warn(`#${tab_id} doesn't exist`);
	}

	$("#main_tabbed").tabs("refresh");
}

function open_first_tab_when_none_is_open() {
	if($("#main_tabbed").tabs("option", "active") === false) {
		$("#main_tabbed").tabs("option", "active", 0);
	}
}
