"use strict";
$(document).ready(function() {
	apply_theme_based_on_system_preferences();

	document.addEventListener("keydown", function(event) {
		if (event.key === "Escape") {
			var deleteButton = document.getElementById("del_search_button");
			if (deleteButton && getComputedStyle(deleteButton).display !== "none") {
				delete_search();
			}
		}
	});

	var themeSelect = document.getElementById("themeSelect");

	var currentTheme = get_cookie("theme");
	if (currentTheme) {
		themeSelect.value = currentTheme;
		if (currentTheme === "dark") {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	}

	themeSelect.addEventListener("change", function() {
		if (themeSelect.value === "dark") {
			enable_dark_mode();
		} else {
			enable_light_mode();
		}
	});

	// Listen for changes to the system color scheme preferences
	window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function(e) {
		if(!get_cookie("theme")) {
			if (e.matches) {
				enable_dark_mode();
			} else {
				enable_light_mode();
			}
		}
	});

	Prism.highlightAll();
	generateTOC();
});


function wrapEmojisInSpans() {
	const emojiRegex = /(\p{Emoji_Presentation}|\p{Extended_Pictographic})/gu;

	try {
		if (!(emojiRegex instanceof RegExp)) {
			throw new TypeError('emojiRegex muss ein RegExp-Objekt sein');
		}

		var startTime = performance && performance.now ? performance.now() : Date.now();
		console.time && console.time('wrapEmojisInSpans');

		// Tags, in denen wir NICHT suchen/ersetzen wollen:
		var forbiddenTags = {
			'SCRIPT': true,
			'STYLE': true,
			'TEXTAREA': true,
			'CODE': true,
			'PRE': true
		};

		// Hilfsfunktion: prüft, ob ein Knoten innerhalb eines verbotenen Tags oder
		// innerhalb einer .tutorial_icon-Elementkette liegt.
		function isInsideForbidden(node) {
			var cur = node.parentNode;
			while (cur && cur.nodeType === Node.ELEMENT_NODE) {
				if (forbiddenTags[cur.tagName]) {
					return true;
				}
				if (cur.classList && cur.classList.contains('tutorial_icon')) {
					return true;
				}
				cur = cur.parentNode;
			}
			return false;
		}

		// TreeWalker: alle Textknoten besuchen, Filter macht akzeptiert/nicht akzeptiert
		var walker = document.createTreeWalker(
			document.body,
			NodeFilter.SHOW_TEXT,
			{
				acceptNode: function(node) {
					// quick reject: leerer Text
					if (!node || !node.nodeValue || node.nodeValue.trim() === '') {
						return NodeFilter.FILTER_REJECT;
					}
					// enthält der Text überhaupt ein Emoji (schneller Test)?
					if (!emojiRegex.test(node.nodeValue)) {
						return NodeFilter.FILTER_REJECT;
					}
					// reset lastIndex falls regex global war (safety)
					try { emojiRegex.lastIndex = 0; } catch (e) {}

					// prüfe, ob der Knoten in einem verbotenen Container steht
					if (isInsideForbidden(node)) {
						return NodeFilter.FILTER_REJECT;
					}
					// sonst akzeptieren
					return NodeFilter.FILTER_ACCEPT;
				}
			},
			false
		);

		// Sammle die relevanten Textknoten
		var textNodes = [];
		var current;
		while ((current = walker.nextNode())) {
			textNodes.push(current);
		}

		if (textNodes.length === 0) {
			console.timeEnd && console.timeEnd('wrapEmojisInSpans');
			var emptyResultTime = performance && performance.now ? performance.now() - startTime : Date.now() - startTime;
			return {
				replacedTextNodes: 0,
				affectedParents: 0,
				durationMs: emptyResultTime
			};
		}

		// Gruppiere Textknoten nach parentNode
		var parentsMap = new Map();
		for (var i = 0; i < textNodes.length; i++) {
			var tn = textNodes[i];
			var p = tn.parentNode;
			if (!p) continue;
			var arr = parentsMap.get(p);
			if (!arr) {
				arr = [];
				parentsMap.set(p, arr);
			}
			arr.push(tn);
		}

		var totalReplacedTextNodes = 0;
		var affectedParents = 0;

		// Hilfsfunktion: verarbeitet einen Text-String, gibt DocumentFragment zurück
		function fragmentFromTextWithEmojis(text) {
			var frag = document.createDocumentFragment();
			// reset lastIndex for global regex safety
			try { emojiRegex.lastIndex = 0; } catch (e) {}
			var matches = Array.from(text.matchAll(emojiRegex));
			if (matches.length === 0) {
				frag.appendChild(document.createTextNode(text));
				return frag;
			}

			var lastIndex = 0;
			for (var m = 0; m < matches.length; m++) {
				var match = matches[m];
				var emoji = match[0];
				var index = match.index;
				if (index > lastIndex) {
					frag.appendChild(document.createTextNode(text.slice(lastIndex, index)));
				}
				var span = document.createElement('span');
				span.className = 'tutorial_icon invert_in_dark_mode no_cursive';
				span.textContent = emoji;
				frag.appendChild(span);
				lastIndex = index + emoji.length;
			}
			if (lastIndex < text.length) {
				frag.appendChild(document.createTextNode(text.slice(lastIndex)));
			}
			return frag;
		}

		// Verarbeite jeden Parent in einem einzigen Schreibschritt
		parentsMap.forEach(function(textNodeArray, parent) {
			try {
				// Erstelle ein Fragment mit allen neuen Kindknoten (Reihenfolge beibehalten)
				var frag = document.createDocumentFragment();
				var childSnapshot = Array.from(parent.childNodes);

				// Erstelle ein Set für schnelle Lookup, welche Textknoten ersetzt werden müssen
				var toReplaceSet = new Set(textNodeArray);

				for (var c = 0; c < childSnapshot.length; c++) {
					var child = childSnapshot[c];
					if (child.nodeType === Node.TEXT_NODE && toReplaceSet.has(child)) {
						// Ersetze diesen Textknoten durch Fragment mit Text + Emoji-Spans
						var newFrag = fragmentFromTextWithEmojis(child.nodeValue);
						// appendChild bewegt Kindelemente aus newFrag in frag
						frag.appendChild(newFrag);
						totalReplacedTextNodes += 1;
					} else {
						// Bewahre unveränderte Knoten: wir clonen sie, damit Event-Handler etc. nicht doppelt
						// (cloneNode(true) kopiert DOM-Struktur — Vorsicht: Event-Listener gehen verloren)
						frag.appendChild(child.cloneNode(true));
					}
				}

				// Ein einziger DOM-Akt: children ersetzen
				// Wir löschen zuerst die Kinder und fügen dann das fragment hinzu.
				// Das minimiert Layout/Reflow-Phasen.
				parent.innerHTML = '';
				parent.appendChild(frag);

				affectedParents += 1;
			} catch (innerErr) {
				console.error('Fehler beim Verarbeiten eines Parents:', innerErr);
			}
		});

		console.timeEnd && console.timeEnd('wrapEmojisInSpans');
		var duration = performance && performance.now ? performance.now() - startTime : Date.now() - startTime;

		return {
			replacedTextNodes: totalReplacedTextNodes,
			affectedParents: affectedParents,
			durationMs: duration
		};
	} catch (err) {
		console.error('wrapEmojisInSpans failed:', err);
		throw err;
	}
}

window.addEventListener('DOMContentLoaded', wrapEmojisInSpans);
