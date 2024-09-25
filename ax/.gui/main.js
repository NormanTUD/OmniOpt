function showSpinnerOverlay(text) {
	// Überprüfen, ob bereits ein Overlay existiert
	if (document.getElementById('spinner-overlay')) {
		return; // Wenn ja, wird nichts getan
	}

	// Erstelle das Overlay
	const overlay = document.createElement('div');
	overlay.id = 'spinner-overlay';

	// Erstelle den Container für den Spinner und den Text
	const container = document.createElement('div');
	container.id = 'spinner-container';

	// Erstelle den Spinner
	const spinner = document.createElement('div');
	spinner.classList.add('spinner');

	// Erstelle den Text
	const spinnerText = document.createElement('div');
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
	const overlay = document.getElementById('spinner-overlay');
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

	$(clicked_element).text("✅Copied to clipboard");

	setTimeout(() => {
		$(clicked_element).text(oldText);
	}, 1000);
}
