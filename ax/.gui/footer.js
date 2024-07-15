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

		$tocDiv.append($tocContainer);
	} catch (error) {
		console.error('Error generating TOC:', error);
	}
}


$(document).ready(function() {
	Prism.highlightAll();
	generateTOC();
});
