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
