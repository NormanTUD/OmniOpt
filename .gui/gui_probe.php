<?php include("gui.php"); ?>
<script>
(function () {
	var $ = window.jQuery;
	function el(id) { return document.getElementById(id); }
	function waitUntil(fn, timeout, step) {
		return new Promise(function (resolve) {
			var start = Date.now();
			(function poll() {
				var v; try { v = fn(); } catch (e) { v = false; }
				if (v) return resolve(v);
				if (Date.now() - start > (timeout || 40000)) return resolve(false);
				setTimeout(poll, step || 60);
			})();
		});
	}
	async function main() {
		window.__oattach = [];
		var res = [];
		function decAll(root) {
			var o = ""; var els = root.querySelectorAll("mjx-c, .mjx-c");
			for (var i = 0; i < els.length; i++) {
				var m = /(?:^|\s)mjx-c([0-9A-Fa-f]{2,6})(?:\s|$)/.exec(els[i].className);
				if (m) o += String.fromCodePoint(parseInt(m[1], 16));
			}
			return o;
		}
		try {
			el("rp_tab_formula").click();
			var pane = el("formula_pane_text");
			pane.value = "f(a,b) = \\int_{x = a}^b (\\underbrace{a}_{\\substack{[10,\\, 1000] \\in \\mathbb{R} \\\\ \\text{continuous}}} + \\underbrace{b}_{\\substack{[-10,\\, 10] \\in \\mathbb{R} \\\\ \\text{continuous}}})";
			pane.dispatchEvent(new Event("input", { bubbles: true }));
			el("formula_apply_btn").click();
			await waitUntil(function () { return $(".parameterRow .parameterName").length >= 2; }, 30000);
			var pi = get_current_parameter_info();
			res.push("pi=" + JSON.stringify(pi));
			await waitUntil(function () { return el("formula_preview").querySelectorAll("mjx-munder, .mjx-munder").length >= 2; }, 30000);
			res.push("munder=" + el("formula_preview").querySelectorAll("mjx-munder, .mjx-munder").length);
			res.push("unders=" + el("formula_preview").querySelectorAll("mjx-under, .mjx-under").length);
			await waitUntil(function () { return el("formula_preview").querySelectorAll(".omniopt_bound_overlay").length > 0 || (window.__oattach || []).length > 1; }, 20000);
			res.push("ovs=" + el("formula_preview").querySelectorAll(".omniopt_bound_overlay").length);
			var ovs = el("formula_preview").querySelectorAll(".omniopt_bound_overlay");
			for (var i = 0; i < ovs.length; i++) res.push("ov " + ovs[i].getAttribute("data-name") + ":" + ovs[i].getAttribute("data-side"));
			res.push("btns=" + el("formula_preview").querySelectorAll(".omniopt_edit_btn").length);
			var mis = el("formula_preview").querySelectorAll("mjx-mi, .mjx-mi");
			for (var mi = 0; mi < mis.length; mi++) res.push("mi '" + decAll(mis[mi]) + "' parent=" + mis[mi].parentElement.tagName);
			res.push("LOG=" + window.__oattach.join(" | "));
			var us = el("formula_preview").querySelectorAll("mjx-under, .mjx-under");
			for (var ui = 0; ui < us.length; ui++) {
				var txt = decAll(us[ui]);
				if (!txt) continue;
				var nm = "";
				try { nm = _resolve_under_label_param(us[ui]); } catch (e2) { nm = "THREW"; }
				res.push("label '" + txt + "' -> name '" + nm + "' piHas=" + (pi[nm] !== undefined));
			}
		} catch (e) {
			res.push("ERROR: " + e + (e.stack ? " :: " + String(e.stack).slice(0, 400) : ""));
		}
		res.push("LOG=" + window.__oattach.join(" | "));
		document.title = res.join("\n");
	}
	(function waitInit() {
		if (el("rp_tab_run") && el("site") && el("site").style.display !== "none") setTimeout(main, 0);
		else setTimeout(waitInit, 50);
	})();
})();
</script>