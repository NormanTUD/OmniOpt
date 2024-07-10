function generateTOC() {
    // Check if the TOC div exists
    var tocDiv = document.getElementById('toc');
    if (!tocDiv) {
        return;
    }

    // Create the TOC structure
    var tocContainer = document.createElement('div');
    tocContainer.className = 'toc';
    var tocHeader = document.createElement('h2');
    tocHeader.innerText = 'Table of Contents';
    var tocList = document.createElement('ul');

    tocContainer.appendChild(tocHeader);
    tocContainer.appendChild(tocList);

    // Get all h2, h3, h4, h5, h6 elements
    var headers = document.querySelectorAll('h2, h3, h4, h5, h6');
    var tocItems = [];

    headers.forEach(function(header) {
        var headerTag = header.tagName.toLowerCase();
        var headerText = header.innerText;
        var headerId = header.id;

        if (!headerId) {
            headerId = headerText.toLowerCase().replace(/\s+/g, '-');
            header.id = headerId;
        }

        tocItems.push({
            tag: headerTag,
            text: headerText,
            id: headerId
        });
    });

    // Generate the nested list for TOC
    var currentLevel = 2; // starting from h2
    var currentList = tocList;
    var parents = [{ level: currentLevel, list: currentList }]; // stack to keep track of parent lists

    tocItems.forEach(function(item) {
        var level = parseInt(item.tag.replace('h', ''), 10);
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.href = '#' + item.id;
        a.innerText = item.text;
        li.appendChild(a);

        if (level > currentLevel) {
            var newList = document.createElement('ul');
            li.appendChild(newList);
            currentList.appendChild(li);
            parents.push({ level: currentLevel, list: currentList });
            currentList = newList;
        } else {
            while (level < currentLevel) {
                var parent = parents.pop();
                currentLevel = parent.level;
                currentList = parent.list;
            }
            currentList.appendChild(li);
        }

        currentLevel = level;
    });

    tocDiv.appendChild(tocContainer);
}


$(document).ready(function() {
	Prism.highlightAll();
	generateTOC();
});
