function generateTOC() {
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

	// Get all h1, h2, h3, etc. elements
	var headers = $('h1, h2, h3, h4, h5, h6');
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
	var currentLevel = 1;
	var $currentList = $tocList;

	tocItems.forEach(function(item) {
		var level = parseInt(item.tag.replace('h', ''), 10);
		var $li = $('<li></li>');
		var $a = $('<a></a>').attr('href', '#' + item.id).text(item.text);
		$li.append($a);

		if (level > currentLevel) {
			var $newList = $('<ul></ul>');
			$currentList.append($newList);
			$currentList = $newList;
		} else if (level < currentLevel) {
			$currentList = $currentList.parent();
		}

		$currentList.append($li);
		currentLevel = level;
	});

	$tocDiv.append($tocContainer);
}

$(document).ready(function() {
    generateTOC();
});
