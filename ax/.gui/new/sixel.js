
function getElementByXpath(path) {
	return document.evaluate(path, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
}

function replaceSixelWithImage(element) {
	const text = element.textContent;

	// Sehr simples Regex für Sixel (du kannst das feiner abstimmen)
	const sixelRegex = /(P[0-9;]+q.*?)(?=(P|$|\s))/gs;

	let newHTML = text;
	let match;
	while ((match = sixelRegex.exec(text)) !== null) {
		const sixelCode = match[1];
		const imgTag = `<img src="${renderSixelToDataUrl(sixelCode)}" alt="Sixel Image" style="display:inline-block;">`;
		newHTML = newHTML.replace(sixelCode, imgTag);
	}

	element.innerHTML = newHTML;
}

function renderSixelToDataUrl(sixelCode) {
	const colorPalette = {};
	let maxX = 0;
	let maxY = 0;
	let x = 0;
	let y = 0;
	let currentColor = 0;

	const cellHeight = 6;
	const dataStart = sixelCode.indexOf('"');
	if (dataStart === -1) return '';

	const data = sixelCode.slice(dataStart + 1);
	let i = 0;

	// Farbpalette auslesen
	const colorRegex = /#(\d+);(\d+);(\d+);(\d+)/g;
	let colorMatch;
	while ((colorMatch = colorRegex.exec(sixelCode)) !== null) {
		const index = parseInt(colorMatch[1], 10);
		const r = Math.round((parseInt(colorMatch[2], 10) / 100) * 255);
		const g = Math.round((parseInt(colorMatch[3], 10) / 100) * 255);
		const b = Math.round((parseInt(colorMatch[4], 10) / 100) * 255);
		colorPalette[index] = `rgb(${r},${g},${b})`;
	}

	if (!colorPalette[0]) colorPalette[0] = 'black'; // Fallback
	let pixelData = [];

	while (i < data.length) {
		const char = data[i];

		if (char === '$') {
			x = 0; // Zeilenumbruch horizontal
		} else if (char === '-') {
			x = 0;
			y += cellHeight; // Zeilenvorschub vertikal
		} else if (char === '#') {
			const match = data.slice(i).match(/^#(\d+)/);
			if (match) {
				currentColor = parseInt(match[1], 10);
				i += match[0].length - 1;
			}
		} else if (char === '!') {
			const count = parseInt(data[++i], 10);
			const repeatChar = data[++i];
			for (let r = 0; r < count; r++) {
				processSixelChar(repeatChar, x, y, currentColor, pixelData);
				x += 1;
			}
		} else if (char >= '?' && char <= '~') {
			processSixelChar(char, x, y, currentColor, pixelData);
			x += 1;
		}

		maxX = Math.max(maxX, x);
		maxY = Math.max(maxY, y + cellHeight);
		i++;
	}

	// Canvas basierend auf echter Größe erstellen
	const canvas = document.createElement('canvas');
	canvas.width = maxX;
	canvas.height = maxY;
	const ctx = canvas.getContext('2d');

	// Antialiasing ausschalten
	ctx.imageSmoothingEnabled = false;

	// Hintergrund auf weiß setzen
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	// Pixel ausmalen
	for (const { x, y, color } of pixelData) {
		ctx.fillStyle = colorPalette[color] || colorPalette[0];
		ctx.fillRect(x, y, 1, 1);
	}

	return canvas.toDataURL('image/png');
}

// Hilfsfunktion zum Verarbeiten eines Sixel-Zeichens
function processSixelChar(char, x, y, colorIndex, pixelData) {
	const bits = char.charCodeAt(0) - 63;
	for (let bit = 0; bit < 6; bit++) {
		if (bits & (1 << bit)) {
			pixelData.push({ x, y: y + bit, color: colorIndex });
		}
	}
}

function test_sixel_replace() {
	var xpath = "/html/body/div/div[3]/section/article[7]/pre/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span";
	replaceSixelWithImage(getElementByXpath(xpath));
}
