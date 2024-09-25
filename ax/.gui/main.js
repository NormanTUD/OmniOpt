function showSpinnerOverlay(text) {
	// Überprüfen, ob bereits ein Overlay existiert
	if (document.getElementById('spinner-overlay')) {
		return; // Wenn ja, wird nichts getan
	}

	// Erstelle das Overlay
	var overlay = document.createElement('div');
	overlay.id = 'spinner-overlay';

	// Erstelle den Container für den Spinner und den Text
	var container = document.createElement('div');
	container.id = 'spinner-container';

	// Erstelle den Spinner
	var spinner = document.createElement('div');
	spinner.classList.add('spinner');

	// Erstelle den Text
	var spinnerText = document.createElement('div');
	spinnerText.id = 'spinner-text';
	spinnerText.innerText = text;

	// Füge den Spinner und den Text zum Container hinzu
	container.appendChild(spinner);
	container.appendChild(spinnerText);

	// Füge den Container zum Overlay hinzu
	overlay.appendChild(container);

	// Füge das Overlay zum Body hinzu
	document.body.appendChild(overlay);
}

function removeSpinnerOverlay() {
	// Überprüfe, ob das Overlay existiert
	var overlay = document.getElementById('spinner-overlay');
	if (overlay) {
		overlay.remove(); // Entferne das Overlay
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
			console.error(`Could not find ${element_to_search_for_class} from clicked_element:`, clicked_element);
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
		console.error("Error parsing CSV:", error);
		return [];
	}
}

function normalizeArrayLength(array) {
	// Schritt 1: Bestimme die maximale Länge der Unterarrays
	let maxColumns = array.reduce((max, row) => Math.max(max, row.length), 0);

	// Schritt 2: Fülle die kürzeren Arrays auf die maximale Länge auf
	return array.map(row => {
		let filledRow = [...row]; // Kopiere das Array
		while (filledRow.length < maxColumns) {
			filledRow.push(""); // Füge leere Strings hinzu
		}
		return filledRow;
	});
}

function create_table_from_csv_data(csvData, table_container, new_table_id, optionalColumnTitles = null) {
	// Parse CSV data
	var data = parse_csv(csvData);
	data = normalizeArrayLength(data);
	var tableContainer = document.getElementById(table_container);

	if (!tableContainer) {
		console.error(`Table container "${table_container}" not found`);
		return;
	}

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

	var toggleButton = document.createElement('button');
	toggleButton.classList.add("invert_in_dark_mode");
	toggleButton.textContent = 'Show Raw Data';

	toggleButton.addEventListener('click', function() {
		if (rawDataElement.style.display === 'none') {
			rawDataElement.style.display = 'block';
			toggleButton.textContent = 'Hide Raw Data';
		} else {
			rawDataElement.style.display = 'none';
			toggleButton.textContent = 'Show Raw Data';
		}
	});

	// Append the button and raw data element under the table
	tableContainer.appendChild(toggleButton);
	tableContainer.appendChild(rawDataElement);

	$(document).ready(function() {
		var _new_table_id = `#${new_table_id}`;
		console.log(_new_table_id);
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

		//function create_table_from_csv_data(csvData, table_container, new_table_id, optionalColumnTitles = null) {
		create_table_from_csv_data(
			csvText, 
			tableContainer.attr('id'), 
			`autotable_${index}`,
			_optionalColumnTitles
		);
	});

	apply_theme_based_on_system_preferences();
}
