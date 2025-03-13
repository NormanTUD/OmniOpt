"use strict";

$(document).ready(function() {
	generateTOC();
});

async function start_gremlins () {
	(function() {
		function callback() {
			var horde = gremlins.createHorde({
				species: [
					gremlins.species.clicker(),
					gremlins.species.toucher(),
					gremlins.species.formFiller(),
					gremlins.species.scroller(),
					gremlins.species.typer()
				],
				mogwais: [
					gremlins.mogwais.alert(),
					gremlins.mogwais.gizmo()
				],
				strategies: [
					gremlins.strategies.distribution()
				]
			});

			horde.unleash();
		}
		var s = document.createElement("script");
		s.src = "https://unpkg.com/gremlins.js";
		if (s.addEventListener) {
			s.addEventListener("load", callback, false);
		} else if (s.readyState) {
			s.onreadystatechange = callback;
		}
		document.body.appendChild(s);
	})();
}
