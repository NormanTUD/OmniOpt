<?php include("gui.php"); ?>
<script>
var _dbgDiv = document.createElement("pre");
_dbgDiv.id = "omniopt_harness_debug";
_dbgDiv.textContent = "STARTED";
document.body.appendChild(_dbgDiv);
function _dbgSet(v) { var d = document.getElementById("omniopt_harness_debug"); if (d) d.textContent = v; }
window.onerror = function (msg, src, line, col, err) {
	_dbgSet("ONERROR: " + msg + " @ " + src + ":" + line + ":" + col + " :: " + (err && err.stack ? err.stack : ""));
	return false;
};
(function () {
	var $ = window.jQuery;
	function el(id) { return document.getElementById(id); }
	function waitUntil(fn, timeout, step) {
		return new Promise(function (resolve) {
			var start = Date.now();
			(function poll() {
				var v;
				try { v = fn(); } catch (e) { v = false; }
				if (v) return resolve(v);
				if (Date.now() - start > (timeout || 25000)) return resolve(false);
				setTimeout(poll, step || 60);
			})();
		});
	}
	var results = [];
	function check(label, cond) { results.push((cond ? "PASS: " : "FAIL: ") + label + (cond ? "" : "   <== CHECK")); }
	function post(label, val) { results.push("INFO: " + label + " = " + val); }

	function setFormula(text) {
		el("rp_tab_formula").click();
		var pane = el("formula_pane_text");
		pane.value = text;
		pane.dispatchEvent(new Event("input", { bubbles: true }));
		el("formula_apply_btn").click();
	}
	function formulaTextVal() {
		return (el("formula_pane_text") || {}).value || "";
	}
	function setOptionTo(name, kind) {
		var row = null;
		$(".parameterRow").each(function () {
			if ($(this).find(".parameterName").val().trim() === name) row = $(this);
		});
		if (!row) return null;
		row.find(".optionSelect").val(kind).trigger("change");
		return row;
	}
	async function main() {
		try {
			// ============ Scenario 1: choice + fixed labels are editable ============
			setFormula("f(a,b) = a + b");
			await waitUntil(function () { return $(".parameterRow .parameterName").filter(function () { return this.value === "a"; }).length; });
			await waitUntil(function () { return el("formula_preview").querySelectorAll("mjx-under, .mjx-under").length >= 2; }, 25000);
			post("scenario1: base formula active", formulaTextVal());
			{ // convert a -> choice, b -> fixed, then re-render
				var ra = setOptionTo("a", "choice");
				ra.find(".choiceValues").val("yes, no");
				var rb = setOptionTo("b", "fixed");
				rb.find(".fixedValue").val("2.5");
				if (typeof _formula_preview_callback === "function") {
					try { _formula_preview_callback(); } catch (e) { post("callback threw", String(e && e.stack || e)); }
				}
				await waitUntil(function () {
					var os = el("formula_preview").querySelectorAll(".omniopt_bound_overlay");
					return os.length >= 2;
				}, 25000);
				post("underbrace count", el("formula_preview").querySelectorAll("mjx-under, .mjx-under").length);
				post("formula_error text", (el("formula_error") || {}).textContent || "(none)");
				var decoded = [];
				var us = el("formula_preview").querySelectorAll("mjx-under, .mjx-under");
				for (var ui = 0; ui < us.length; ui++) {
					var glyphs = el("formula_preview").querySelectorAll ? undefined : undefined;
					var t = "";
					var ue = us[ui];
					var par = ue.parentElement;
					var baseTxt = "";
					if (par) {
						var kids = par.children;
						for (var k = 0; k < kids.length; k++) {
							if (kids[k].tagName.toLowerCase() === "mjx-base") {
								var g2 = _decode_mjx_glyphs(kids[k]);
								for (var gi = 0; gi < g2.length; gi++) baseTxt += g2[gi].ch;
							}
						}
					}
					var g3 = _decode_mjx_glyphs(ue);
					for (var gi2 = 0; gi2 < g3.length; gi2++) t += g3[gi2].ch;
					decoded.push(baseTxt + " << " + t);
				}
				post("decoded underbrace labels", decoded.join(" | "));
			}
			var ovs = el("formula_preview").querySelectorAll(".omniopt_bound_overlay");
			var kinds = Array.prototype.map.call(ovs, function (o) { return o.getAttribute("data-name") + ":" + o.getAttribute("data-kind") + ":" + o.getAttribute("data-side"); });
			post("scenario1: overlay kinds", kinds.join(", "));
			check("choice overlay present", kinds.indexOf("a:choice:values") !== -1);
			check("fixed overlay present", kinds.indexOf("b:fixed:value") !== -1);

			// click the choice overlay -> popover names the variable
			var choiceOv = Array.prototype.filter.call(ovs, function (o) { return o.getAttribute("data-name") === "a" && o.getAttribute("data-kind") === "choice"; })[0];
			choiceOv.click();
			await waitUntil(function () { return document.querySelector(".omniopt_edit_pop input"); }, 5000);
			var pop = document.querySelector(".omniopt_edit_pop");
			check("popover shows variable name", (pop.querySelector(".omniopt_edit_name") || {}).textContent === "a");
			check("popover shows choice chip", (pop.querySelector(".omniopt_edit_chip") || {}).textContent === "choice");
			var inp = pop.querySelector("input");
			post("popover choice input val", inp.value);
			check("popover meta present", (pop.querySelector(".omniopt_edit_meta") || {}).textContent.indexOf("comma") !== -1);
			inp.value = "yes, no, maybe";
			inp.dispatchEvent(new Event("keydown", { bubbles: true, key: "Enter" }));
			await waitUntil(function () {
				var row = null;
				$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "a") row = $(this); });
				return row && (row.find(".choiceValues").val() || "").indexOf("maybe") !== -1;
			}, 8000);
			var ra2 = null;
			$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "a") ra2 = $(this); });
			check("choice values written to table", ra2.find(".choiceValues").val(), "yes,no,maybe");

			// fixed overlay edit
			await waitUntil(function () { return el("formula_preview").querySelectorAll(".omniopt_bound_overlay[data-name='b']").length === 1; }, 8000);
			var fixedOv = el("formula_preview").querySelector(".omniopt_bound_overlay[data-name='b']");
			fixedOv.click();
			await waitUntil(function () { return document.querySelector(".omniopt_edit_pop input"); }, 5000);
			var pop2 = document.querySelector(".omniopt_edit_pop");
			check("popover shows fixed chip", (pop2.querySelector(".omniopt_edit_chip") || {}).textContent === "fixed");
			inp = pop2.querySelector("input");
			inp.value = "3";
			inp.dispatchEvent(new Event("keydown", { bubbles: true, key: "Enter" }));
			await waitUntil(function () {
				var row = null;
				$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "b") row = $(this); });
				return row && parseFloat(row.find(".fixedValue").val()) === 3;
			}, 8000);
			var rb2 = null;
			$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "b") rb2 = $(this); });
			check("fixed value written to table", rb2.find(".fixedValue").val(), "3");

			// ============ Scenario 2: exponent keeps underbrace, body badge fills gap ============
			setFormula("f(a,b) = a^b");
			await waitUntil(function () {
				var rowA = null, rowB = null;
				$(".parameterRow").each(function () { var n = $(this).find(".parameterName").val().trim(); if (n === "a") rowA = $(this); if (n === "b") rowB = $(this); });
				return rowA && rowB && rowA.find(".optionSelect").val() === "range" && rowB.find(".optionSelect").val() === "range";
			}, 25000);

			// a keeps its underbrace (as before), b (inside the exponent) gets a body badge.
			await waitUntil(function () {
				var underCount = el("formula_preview").querySelectorAll("mjx-under, .mjx-under").length;
				var bBadge = el("formula_preview").querySelector(".omniopt_bound_overlay[data-name='b'][data-side='body']");
				return underCount >= 1 && !!bBadge;
			}, 25000);
			check("underbrace on a kept for a^b", el("formula_preview").querySelectorAll("mjx-under, .mjx-under").length >= 1);
			var bBadge = el("formula_preview").querySelector(".omniopt_bound_overlay[data-name='b'][data-side='body']");
			check("body badge on b (exponent member)", !!bBadge);

			// edit b via its body badge ("min, max" combined)
			bBadge.click();
			await waitUntil(function () { return document.querySelector(".omniopt_edit_pop input"); }, 5000);
			var pop3 = document.querySelector(".omniopt_edit_pop");
			check("badge popover names variable", (pop3.querySelector(".omniopt_edit_name") || {}).textContent === "b");
			check("badge popover chip = range", (pop3.querySelector(".omniopt_edit_chip") || {}).textContent === "range");
			check("badge popover label = min, max", (pop3.querySelector(".omniopt_edit_inputrow label") || {}).textContent === "min, max");
			inp = pop3.querySelector("input");
			post("badge popover input val", inp.value);
			inp.value = "0, 5";
			inp.dispatchEvent(new Event("keydown", { bubbles: true, key: "Enter" }));
			await waitUntil(function () {
				var row = null;
				$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "b") row = $(this); });
				return row && row.find(".minValue").val() === "0" && row.find(".maxValue").val() === "5";
			}, 8000);
			var rb3 = null;
			$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "b") rb3 = $(this); });
			check("badge edit set b min", rb3.find(".minValue").val(), "0");
			check("badge edit set b max", rb3.find(".maxValue").val(), "5");

			// a is still editable through its underbrace label numbers.
			await waitUntil(function () { return el("formula_preview").querySelectorAll(".omniopt_bound_overlay[data-name='a'][data-side='max']").length === 1; }, 8000);
			var aMax = el("formula_preview").querySelector(".omniopt_bound_overlay[data-name='a'][data-side='max']");
			check("a still editable via its label", !!aMax);
			aMax.click();
			await waitUntil(function () { return document.querySelector(".omniopt_edit_pop input"); }, 5000);
			var pop4 = document.querySelector(".omniopt_edit_pop");
			check("label popover chip = range max", (pop4.querySelector(".omniopt_edit_chip") || {}).textContent === "range max");
			check("label popover input label = max", (pop4.querySelector(".omniopt_edit_inputrow label") || {}).textContent === "max");
			inp = pop4.querySelector("input");
			inp.value = "7";
			inp.dispatchEvent(new Event("keydown", { bubbles: true, key: "Enter" }));
			await waitUntil(function () {
				var row = null;
				$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "a") row = $(this); });
				return row && row.find(".maxValue").val() === "7";
			}, 8000);
			var ra2b = null;
			$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "a") ra2b = $(this); });
			check("label edit set a max", ra2b.find(".maxValue").val(), "7");

			// Escape must cancel without writing
			var aBadge = el("formula_preview").querySelector(".omniopt_bound_overlay[data-name='b'][data-side='body']");
			if (aBadge) aBadge.click();
			await waitUntil(function () { return document.querySelector(".omniopt_edit_pop input"); }, 5000);
			var pop5 = document.querySelector(".omniopt_edit_pop");
			inp = pop5.querySelector("input");
			inp.value = "99, 99";
			inp.dispatchEvent(new Event("keydown", { bubbles: true, key: "Escape" }));
			await waitUntil(function () { return !document.querySelector(".omniopt_edit_pop"); }, 5000);
			var rb4 = null;
			$(".parameterRow").each(function () { if ($(this).find(".parameterName").val().trim() === "b") rb4 = $(this); });
			check("escape cancels edit", rb4.find(".minValue").val(), "0");

		} catch (e) {
			results.push("ERROR: " + e && e.stack ? e.stack : String(e));
		}
		results.push("ATTACHLOG: " + (window.__oattach || []).join(" | "));
		results.push("FINAL_TITLE_MARKER");
		document.title = results.join("\n");
	}
	(function waitInit() {
		if (el("rp_tab_run") && el("site") && el("site").style.display !== "none") setTimeout(main, 0);
		else setTimeout(waitInit, 50);
	})();
})();
</script>