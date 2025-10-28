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
	try {
		const emojiRegex = /(\p{Emoji_Presentation}|\p{Extended_Pictographic})/gu;

		function run() {
			if (!document.body) {
				document.addEventListener('DOMContentLoaded', run);
				return;
			}

			const forbiddenTags = {
				'SCRIPT': true,
				'STYLE': true,
				'TEXTAREA': true,
				'CODE': true,
				'PRE': true
			};

			function isInsideForbidden(node) {
				let cur = node.parentNode;
				while (cur && cur.nodeType === Node.ELEMENT_NODE) {
					if (forbiddenTags[cur.tagName]) return true;
					if (cur.classList && cur.classList.contains('tutorial_icon')) return true;
					cur = cur.parentNode;
				}
				return false;
			}

			const walker = document.createTreeWalker(
				document.body,
				NodeFilter.SHOW_TEXT,
				{
					acceptNode: function(node) {
						if (!node || !node.nodeValue || node.nodeValue.trim() === '') return NodeFilter.FILTER_REJECT;
						if (!emojiRegex.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
						try { emojiRegex.lastIndex = 0; } catch (e) {}
						if (isInsideForbidden(node)) return NodeFilter.FILTER_REJECT;
						return NodeFilter.FILTER_ACCEPT;
					}
				},
				false
			);

			const textNodes = [];
			let current;
			while ((current = walker.nextNode())) {
				textNodes.push(current);
			}

			if (textNodes.length === 0) return;

			const parentsMap = new Map();
			for (const tn of textNodes) {
				const p = tn.parentNode;
				if (!p) continue;
				let arr = parentsMap.get(p);
				if (!arr) {
					arr = [];
					parentsMap.set(p, arr);
				}
				arr.push(tn);
			}

			function fragmentFromTextWithEmojis(text) {
				const frag = document.createDocumentFragment();
				try { emojiRegex.lastIndex = 0; } catch (e) {}
				const matches = Array.from(text.matchAll(emojiRegex));
				if (matches.length === 0) {
					frag.appendChild(document.createTextNode(text));
					return frag;
				}

				let lastIndex = 0;
				for (const match of matches) {
					const emoji = match[0];
					const index = match.index;
					if (index > lastIndex) {
						frag.appendChild(document.createTextNode(text.slice(lastIndex, index)));
					}
					const span = document.createElement('span');
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

			parentsMap.forEach((nodes, parent) => {
				try {
					const frag = document.createDocumentFragment();
					const childSnapshot = Array.from(parent.childNodes);
					const toReplace = new Set(nodes);

					for (const child of childSnapshot) {
						if (child.nodeType === Node.TEXT_NODE && toReplace.has(child)) {
							frag.appendChild(fragmentFromTextWithEmojis(child.nodeValue));
						} else {
							frag.appendChild(child.cloneNode(true));
						}
					}

					parent.innerHTML = '';
					parent.appendChild(frag);
				} catch (err) {
					console.error('Error while working on a parent:', err);
				}
			});
		}

		run();
	} catch (err) {
		console.error('wrapEmojisInSpans failed:', err);
	}
}

window.addEventListener('DOMContentLoaded', wrapEmojisInSpans);
