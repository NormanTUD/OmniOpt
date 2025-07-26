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

const emojiRegex = /(\p{Emoji_Presentation}|\p{Extended_Pictographic})/gu;

function wrapEmojisInSpans() {
  function processNode(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      if (emojiRegex.test(node.textContent)) {
        const frag = document.createDocumentFragment();
        let lastIndex = 0;
        node.textContent.replace(emojiRegex, (match, offset) => {
          if (offset > lastIndex) {
            frag.appendChild(document.createTextNode(node.textContent.slice(lastIndex, offset)));
          }
          const span = document.createElement('span');
          span.className = 'tutorial_icon invert_in_dark_mode no_cursive';
          span.textContent = match;
          frag.appendChild(span);
          lastIndex = offset + match.length;
        });
        if (lastIndex < node.textContent.length) {
          frag.appendChild(document.createTextNode(node.textContent.slice(lastIndex)));
        }
        node.parentNode.replaceChild(frag, node);
      }
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      const forbiddenTags = ['SCRIPT', 'STYLE', 'TEXTAREA', 'CODE', 'PRE'];
      if (!forbiddenTags.includes(node.tagName)) {
        for (let child of Array.from(node.childNodes)) {
          processNode(child);
        }
      }
    }
  }

  processNode(document.body);
}

window.addEventListener('DOMContentLoaded', wrapEmojisInSpans);
