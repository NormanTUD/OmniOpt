<?php include("gui.php"); ?>
<script>
(function () {
	function el(id) { return document.getElementById(id); }
	function waitUntil(fn, timeout, step) {
		return new Promise(function (resolve) {
			var start = Date.now();
			(function poll() {
				var v;
				try { v = fn(); } catch (e) { v = false; }
				if (v) return resolve(v);
				if (Date.now() - start > (timeout || 20000)) return resolve(false);
				setTimeout(poll, step || 60);
			})();
		});
	}
	var results = [];
	function check(label, cond) { results.push((cond ? "PASS: " : "FAIL: ") + label + (cond ? "" : "   <== CHECK")); }
	function post(label, val) { results.push("INFO: " + label + " = " + val); }
	async function main() {
		try {
			el("rp_tab_formula").click();
			await waitUntil(function () { return el("formula_pane_text"); });
			var pane = el("formula_pane_text");
			pane.value = "f(a) = a + sin(b*2)";
			pane.dispatchEvent(new Event("input", { bubbles: true }));
			el("formula_apply_btn").click();
			await waitUntil(function () {
				var o = document.querySelectorAll("#formula_preview .omniopt_bound_overlay");
				return o.length >= 2 && o[0].getBoundingClientRect().width > 0;
			}, 20000);
			var ovs = document.querySelectorAll("#formula_preview .omniopt_bound_overlay");
			post("overlay count", ovs.length);
			var noText = Array.prototype.every.call(ovs, function (o) { return (o.textContent || "").trim() === ""; });
			check("overlays contain no visible text", noText);
			var cs = window.getComputedStyle(ovs[0]);
			post("default background", cs.backgroundColor);
			post("default border", cs.border);
			check("no background by default", cs.backgroundColor === "rgba(0, 0, 0, 0)" || cs.backgroundColor === "transparent");

			// Each rendered number is still covered by an exact hit box.
			var any = false;
			Array.prototype.forEach.call(ovs, function (o) {
				var r = o.getBoundingClientRect();
				post("ov w/h", Math.round(r.width) + "x" + Math.round(r.height));
				if (r.width > 0 && r.height > 0) any = true;
			});
			check("hit boxes have size", any);

			// Clicking still edits + syncs.
			ovs[0].click();
			await waitUntil(function () { return el("formula_preview").querySelector("input"); }, 5000);
			check("edit input opens on click", !!el("formula_preview").querySelector("input"));
			var inp = el("formula_preview").querySelector("input");
			post("edit input value", inp && inp.value);
			var name0 = ovs[0].getAttribute("data-name"), side0 = ovs[0].getAttribute("data-side");
			inp.value = "3.5";
			inp.dispatchEvent(new Event("blur", { bubbles: true }));
			await waitUntil(function () {
				var row = null;
				Array.prototype.forEach.call(document.querySelectorAll(".parameterRow"), function (r) {
					if (r.querySelector(".parameterName").value.trim() === name0) row = r;
				});
				if (!row) return false;
				return parseFloat(row.querySelector(side0 === "min" ? ".minValue" : ".maxValue").value) === 3.5;
			}, 8000);
			var row0 = null;
			Array.prototype.forEach.call(document.querySelectorAll(".parameterRow"), function (r) {
				if (r.querySelector(".parameterName").value.trim() === name0) row0 = r;
			});
			var synced = row0 && row0.querySelector(side0 === "min" ? ".minValue" : ".maxValue").value;
			check(name0 + "/" + side0 + " synced to 3.5 (got " + synced + ")", parseFloat(synced) === 3.5);
		} catch (e) {
			results.push("ERROR: " + (e && e.stack ? e.stack : String(e)));
		}
		document.title = results.join("\n");
	}
	(function waitInit() {
		if (el("rp_tab_run") && el("site") && el("site").style.display !== "none") setTimeout(main, 0);
		else setTimeout(waitInit, 50);
	})();
})();
</script>