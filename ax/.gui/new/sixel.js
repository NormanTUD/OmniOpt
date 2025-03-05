
function getElementByXpath(path) {
	return document.evaluate(path, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
}

function replaceSixelWithImage(element) {
	const text = element.textContent;

	const sixelRegex = /(P[0-9;]+q.*?)(?=(P|$|\s))/gs;

	let newHTML = text;
	let match;
	while ((match = sixelRegex.exec(text)) !== null) {
		const sixelCode = match[1];
		const imgTag = `<img src="${renderSixelToDataUrl(sixelCode)}" alt="Sixel Image" style="display:inline-block;"><br>`;
		newHTML = newHTML.replace(sixelCode, imgTag);
	}

	element.innerHTML = newHTML;
}

function test_sixel_replace() {
	var xpath = "/html/body/div/div[3]/section/article[7]/pre/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span/span";
	replaceSixelWithImage(getElementByXpath(xpath));
}

// =====================================

// Hauptfunktion, die den gesamten Sixel-Code verarbeitet und das Bild rendert
function renderSixelToDataUrl(sixelCode) {
  const colorPalette = parseColorPalette(sixelCode);  // Farbpalette extrahieren
  const pixelData = parseSixelData(sixelCode, colorPalette);  // Sixel-Daten extrahieren und in Pixel umwandeln
	const width = get_width(sixelCode)
	const height = get_height(sixelCode)

  return generateImage(pixelData, colorPalette, width, height);  // Bild auf Canvas erstellen und als Base64 PNG zurückgeben
}

// 1. Farbpalette aus dem Sixel-Header extrahieren
function parseColorPalette(sixelCode) {
  const colorPalette = {};
  const colorRegex = /#(\d+);(\d+);(\d+);(\d+)/g;
  let colorMatch;

  while ((colorMatch = colorRegex.exec(sixelCode)) !== null) {
    const index = parseInt(colorMatch[1], 10);
    const r = Math.round((parseInt(colorMatch[2], 10) / 100) * 255);
    const g = Math.round((parseInt(colorMatch[3], 10) / 100) * 255);
    const b = Math.round((parseInt(colorMatch[4], 10) / 100) * 255);
    colorPalette[index] = `rgb(${r},${g},${b})`;
  }

  if (!colorPalette[0]) colorPalette[0] = 'black'; // Fallback-Farbe
  return colorPalette;
}

// 2. Sixel-Daten parsen und Pixel in einer Liste sammeln
function parseSixelData(sixelCode, colorPalette) {
  let pixelData = [];
  let x = 0, y = 0, currentColor = 0;
  const dataStart = sixelCode.indexOf('"');
  const data = sixelCode.slice(dataStart + 1);

  let i = 0;
  while (i < data.length) {
    const char = data[i];

    if (char === '$') {
      x = 0; // Zeilenumbruch horizontal
    } else if (char === '-') {
      x = 0;
      y += 6; // Zeilenvorschub vertikal (6 Pixel pro Zeile)
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
    i++;
  }

  return pixelData;
}

// 3. Berechnung der Bilddimensionen (Breite und Höhe)
function calculateDimensions(pixelData) {
  let maxX = 0, maxY = 0;

  for (const { x, y } of pixelData) {
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y + 6);  // 6 Pixel pro Zeile (Sixel)
  }

  return { width: maxX + 1, height: maxY };  // maxX + 1, weil x bei 0 beginnt
}

// 4. Canvas erstellen und Pixel darauf zeichnen
function generateImage(pixelData, colorPalette, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.imageSmoothingEnabled = false;  // Antialiasing ausschalten
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Pixel ausmalen
  for (const { x, y, color } of pixelData) {
    ctx.fillStyle = colorPalette[color] || colorPalette[0];
    ctx.fillRect(x, y, 1, 1);
  }

  return canvas.toDataURL('image/png');
}

// 5. Sixel-Zeichen in Pixel umwandeln (jede 6 Pixel hoch)
function processSixelChar(char, x, y, colorIndex, pixelData) {
  const bits = char.charCodeAt(0) - 63; // Umwandlung von ASCII-Wert
  for (let bit = 0; bit < 6; bit++) {
    if (bits & (1 << bit)) {
      pixelData.push({ x, y: y + bit, color: colorIndex });
    }
  }
}

function get_width(sixelCode) {
    // Extrahiert die Breite des Bildes aus dem Sixel-Code
    const widthMatch = /q"1;1;(\d+);(\d+)/.exec(sixelCode);  // Sucht nach der Breite
    if (widthMatch) {
        console.log('Width found:', widthMatch[1]);  // Der zweite Wert ist die Breite
        return parseInt(widthMatch[1], 10);
    }

    console.error('No width found.');
    return 0;  // Rückgabe von 0, wenn keine Breite gefunden wurde
}

function get_height(sixelCode) {
    // Extrahiert die Höhe des Bildes aus dem Sixel-Code
    const heightMatch = /q"1;1;(\d+);(\d+)/.exec(sixelCode);  // Sucht nach der Höhe
    if (heightMatch) {
        console.log('Height found:', heightMatch[2]);  // Der erste Wert ist die Höhe
        return parseInt(heightMatch[2], 10);
    }

    console.error('No height found.');
    return 0;  // Rückgabe von 0, wenn keine Höhe gefunden wurde
}
