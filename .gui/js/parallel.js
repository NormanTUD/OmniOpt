function createParallelPlot(dataArray, headers, resultNames, ignoreColumns = [], reload = false) {
	try {
		if ($("#parallel-plot").data("loaded") === "true" && !reload) {
			return;
		}

		dataArray = filterNonEmptyRows(dataArray);

		const ignoreSet = new Set(ignoreColumns);
		const enable_slurm_id_if_exists = $("#enable_slurm_id_if_exists").is(":checked");
		const numericalCols = [];
		const categoricalCols = [];

		parallel_classify_columns(headers, dataArray, ignoreSet, enable_slurm_id_if_exists, numericalCols, categoricalCols, {});
		const precomputedMappings = parallel_precompute_category_mappings(dataArray, categoricalCols);

		const { controlContainer, columnVisibility, minMaxLimits, headerIndex } = parallel_create_controls(
			headers, dataArray, ignoreSet, enable_slurm_id_if_exists, numericalCols
		);

		const { resultSelect, colorValuesRef } = parallel_create_result_selector(resultNames, numericalCols, dataArray);

		// Setup dual-canvas architecture
		const container = document.getElementById("parallel-plot");
		container.innerHTML = "";
		container.style.position = "relative";
		container.style.borderRadius = "16px";
		container.style.overflow = "hidden";
		container.style.boxShadow = "0 8px 32px rgba(0,0,0,0.12)";

		const dpr = window.devicePixelRatio || 1;
		const displayW = container.clientWidth || 1200;
		const displayH = 560;

		// Line canvas (offscreen-style, handles all the heavy drawing)
		const lineCanvas = document.createElement("canvas");
		lineCanvas.width = displayW * dpr;
		lineCanvas.height = displayH * dpr;
		lineCanvas.style.width = displayW + "px";
		lineCanvas.style.height = displayH + "px";
		lineCanvas.style.position = "absolute";
		lineCanvas.style.top = "0";
		lineCanvas.style.left = "0";
		container.appendChild(lineCanvas);

		// UI canvas (axes, labels, brushes — always crisp)
		const uiCanvas = document.createElement("canvas");
		uiCanvas.width = displayW * dpr;
		uiCanvas.height = displayH * dpr;
		uiCanvas.style.width = displayW + "px";
		uiCanvas.style.height = displayH + "px";
		uiCanvas.style.position = "absolute";
		uiCanvas.style.top = "0";
		uiCanvas.style.left = "0";
		uiCanvas.style.pointerEvents = "none";
		container.appendChild(uiCanvas);

		// Interaction canvas (transparent, captures mouse)
		const interactCanvas = document.createElement("canvas");
		interactCanvas.width = displayW * dpr;
		interactCanvas.height = displayH * dpr;
		interactCanvas.style.width = displayW + "px";
		interactCanvas.style.height = displayH + "px";
		interactCanvas.style.position = "absolute";
		interactCanvas.style.top = "0";
		interactCanvas.style.left = "0";
		interactCanvas.style.cursor = "crosshair";
		container.appendChild(interactCanvas);

		container.style.width = displayW + "px";
		container.style.height = displayH + "px";

		const brushes = {};

		let _rafId = null;
		function updatePlot() {
			if (_rafId) cancelAnimationFrame(_rafId);
			_rafId = requestAnimationFrame(() => {
				canvasParallelRender({
					lineCanvas, uiCanvas, dpr, displayW, displayH,
					dataArray, numericalCols, categoricalCols,
					columnVisibility, minMaxLimits, precomputedMappings,
					colorValuesRef, resultSelect, brushes
				});
			});
		}

		setupCanvasBrushing(interactCanvas, dpr, brushes, updatePlot, numericalCols, categoricalCols, columnVisibility, displayW, displayH);

		parallel_bind_min_max_and_checkbox_events(headers, dataArray, numericalCols, columnVisibility, minMaxLimits, updatePlot);

		resultSelect.off("change").on("change", function () {
			parallel_handle_color_change(this, numericalCols, dataArray, resultNames, colorValuesRef, updatePlot);
		});

		if (resultNames.length === 1) {
			resultSelect.val(resultNames[0]).trigger("change");
		}

		updatePlot();
		$("#parallel-plot").data("loaded", "true");

		const ro = new ResizeObserver(() => {
			const newW = container.clientWidth || 1200;
			lineCanvas.width = newW * dpr;
			lineCanvas.style.width = newW + "px";
			uiCanvas.width = newW * dpr;
			uiCanvas.style.width = newW + "px";
			interactCanvas.width = newW * dpr;
			interactCanvas.style.width = newW + "px";
			updatePlot();
		});
		ro.observe(container);

	} catch (err) {
		console.error("Error in createParallelPlot:", err);
	}
}

// ============================================================
// GORGEOUS CANVAS RENDERER
// ============================================================
function canvasParallelRender(ctx) {
	const t0 = performance.now();
	const {
		lineCanvas, uiCanvas, dpr, displayW, displayH,
		dataArray, numericalCols, categoricalCols,
		columnVisibility, minMaxLimits, precomputedMappings,
		colorValuesRef, resultSelect, brushes
	} = ctx;

	const lc = lineCanvas.getContext("2d");
	const uc = uiCanvas.getContext("2d");
	const W = lineCanvas.width;
	const H = lineCanvas.height;

	lc.setTransform(dpr, 0, 0, dpr, 0, 0);
	uc.setTransform(dpr, 0, 0, dpr, 0, 0);

	const isDark = (typeof theme !== "undefined" && theme === "dark");
	const bgColor = isDark ? "#1a1a2e" : "#fafbff";
	const textColor = isDark ? "#e0e0e0" : "#2d3436";
	const axisColor = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.15)";
	const labelColor = isDark ? "#ffffff" : "#1a1a2e";

	const PAD_TOP = 80;
	const PAD_BOTTOM = 50;
	const PAD_LEFT = 70;
	const PAD_RIGHT = 70;
	const plotH = displayH - PAD_TOP - PAD_BOTTOM;
	const plotW = displayW - PAD_LEFT - PAD_RIGHT;

	// Clear both canvases
	lc.clearRect(0, 0, displayW, displayH);
	uc.clearRect(0, 0, displayW, displayH);

	// Background gradient
	const bgGrad = lc.createLinearGradient(0, 0, 0, displayH);
	if (isDark) {
		bgGrad.addColorStop(0, "#1a1a2e");
		bgGrad.addColorStop(1, "#16213e");
	} else {
		bgGrad.addColorStop(0, "#f8f9ff");
		bgGrad.addColorStop(1, "#eef1f8");
	}
	lc.fillStyle = bgGrad;
	lc.fillRect(0, 0, displayW, displayH);

	// Determine visible axes
	const axes = [];
	numericalCols.forEach(col => {
		if (!columnVisibility[col.name]) return;
		axes.push({ name: col.name, index: col.index, type: "num" });
	});
	categoricalCols.forEach(col => {
		if (!columnVisibility[col.name]) return;
		axes.push({ name: col.name, index: col.index, type: "cat" });
	});

	if (axes.length < 2) {
		uc.fillStyle = textColor;
		uc.font = "bold 16px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
		uc.textAlign = "center";
		uc.fillText("Enable at least 2 axes to display the plot", displayW / 2, displayH / 2);
		return;
	}

	const axisSpacing = plotW / (axes.length - 1);

	// Compute ranges
	axes.forEach((axis, i) => {
		axis.x = PAD_LEFT + i * axisSpacing;
		if (axis.type === "num") {
			let min = Infinity, max = -Infinity;
			for (let r = 0; r < dataArray.length; r++) {
				const v = parseFloat(dataArray[r][axis.index]);
				if (!isNaN(v)) {
					if (v < min) min = v;
					if (v > max) max = v;
				}
			}
			if (!isFinite(min)) { min = 0; max = 100; }
			// Add 5% padding to range
			const pad = (max - min) * 0.02;
			axis.min = min - pad;
			axis.max = max + pad;
			axis.realMin = min;
			axis.realMax = max;
			axis.range = (axis.max - axis.min) || 1;
		} else {
			const mapping = precomputedMappings[axis.name];
			const keys = Object.keys(mapping);
			axis.mapping = mapping;
			axis.keys = keys;
			axis.min = 0;
			axis.max = keys.length - 1;
			axis.range = axis.max || 1;
		}
	});

	function normalize(axis, row) {
		if (axis.type === "num") {
			const v = parseFloat(row[axis.index]);
			if (isNaN(v)) return null;
			return (v - axis.min) / axis.range;
		} else {
			const v = axis.mapping[row[axis.index]];
			return v !== undefined ? v / axis.range : null;
		}
	}

	function yPos(norm) {
		return PAD_TOP + plotH * (1 - norm);
	}

	// Filter data
	const filteredIndices = [];
	for (let i = 0; i < dataArray.length; i++) {
		const row = dataArray[i];
		let pass = true;
		for (let j = 0; j < axes.length; j++) {
			const axis = axes[j];
			if (axis.type === "num") {
				const v = parseFloat(row[axis.index]);
				if (isNaN(v)) { pass = false; break; }
				const limits = minMaxLimits[axis.name];
				if (limits.min !== null && v < limits.min) { pass = false; break; }
				if (limits.max !== null && v > limits.max) { pass = false; break; }
			}
		}
		if (pass) filteredIndices.push(i);
	}

	// Color setup
	let colorCol = null;
	let colorMin = Infinity, colorMax = -Infinity;
	if (colorValuesRef.values && resultSelect.val() !== "none") {
		const selectedName = resultSelect.val();
		colorCol = numericalCols.find(col => col.name.toLowerCase() === selectedName.toLowerCase());
		if (colorCol) {
			for (let i = 0; i < filteredIndices.length; i++) {
				const v = parseFloat(dataArray[filteredIndices[i]][colorCol.index]);
				if (!isNaN(v)) {
					if (v < colorMin) colorMin = v;
					if (v > colorMax) colorMax = v;
				}
			}
		}
	}
	const colorRange = colorMax - colorMin || 1;
	const colorScale = colorValuesRef.scale;

	function getLineColor(row, alpha) {
		if (!colorCol) return `rgba(99, 140, 255, ${alpha})`;
		const v = parseFloat(row[colorCol.index]);
		if (isNaN(v)) return `rgba(99, 140, 255, ${alpha})`;
		const t = Math.max(0, Math.min(1, (v - colorMin) / colorRange));
		return interpolateColorHSL(colorScale, t, alpha);
	}

	// Brush check
	function passesBrush(row) {
		for (const axisIdx in brushes) {
			const [bMin, bMax] = brushes[axisIdx];
			const axis = axes[axisIdx];
			const norm = normalize(axis, row);
			if (norm === null || norm < bMin || norm > bMax) return false;
		}
		return true;
	}

	// === DRAW LINES ===
	const totalLines = filteredIndices.length;
	const hasBrush = Object.keys(brushes).length > 0;

	// Adaptive line width and alpha
	let lineWidth, baseAlpha;
	if (totalLines <= 50) {
		lineWidth = 2.5;
		baseAlpha = 0.7;
	} else if (totalLines <= 200) {
		lineWidth = 2.0;
		baseAlpha = 0.5;
	} else if (totalLines <= 500) {
		lineWidth = 1.8;
		baseAlpha = 0.35;
	} else if (totalLines <= 2000) {
		lineWidth = 1.5;
		baseAlpha = 0.2;
	} else {
		lineWidth = 1.2;
		baseAlpha = Math.max(0.05, 15 / totalLines);
	}

	lc.lineCap = "round";
	lc.lineJoin = "round";

	// Draw unhighlighted lines first (if brush active)
	if (hasBrush) {
		lc.lineWidth = lineWidth * 0.7;
		for (let i = 0; i < totalLines; i++) {
			const row = dataArray[filteredIndices[i]];
			if (passesBrush(row)) continue;
			lc.strokeStyle = isDark ? "rgba(80, 80, 120, 0.06)" : "rgba(180, 180, 200, 0.08)";
			lc.beginPath();
			let started = false;
			for (let j = 0; j < axes.length; j++) {
				const norm = normalize(axes[j], row);
				if (norm === null) { started = false; continue; }
				const x = axes[j].x;
				const y = yPos(norm);
				if (!started) { lc.moveTo(x, y); started = true; }
				else { lc.lineTo(x, y); }
			}
			lc.stroke();
		}
	}

	// Draw highlighted / all lines
	lc.lineWidth = lineWidth;
	for (let i = 0; i < totalLines; i++) {
		const row = dataArray[filteredIndices[i]];
		if (hasBrush && !passesBrush(row)) continue;
		const alpha = hasBrush ? Math.min(baseAlpha * 2.5, 0.85) : baseAlpha;
		lc.strokeStyle = getLineColor(row, alpha);
		lc.beginPath();
		let started = false;
		for (let j = 0; j < axes.length; j++) {
			const norm = normalize(axes[j], row);
			if (norm === null) { started = false; continue; }
			const x = axes[j].x;
			const y = yPos(norm);
			if (!started) { lc.moveTo(x, y); started = true; }
			else { lc.lineTo(x, y); }
		}
		lc.stroke();
	}

	// === DRAW UI LAYER (axes, labels, ticks) ===
	const fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

	axes.forEach((axis, i) => {
		const x = axis.x;

		// Axis line with glow
		uc.strokeStyle = axisColor;
		uc.lineWidth = 1.5;
		uc.beginPath();
		uc.moveTo(x, PAD_TOP);
		uc.lineTo(x, PAD_TOP + plotH);
		uc.stroke();

		// Axis endpoint dots
		uc.fillStyle = isDark ? "rgba(99, 140, 255, 0.6)" : "rgba(45, 52, 54, 0.4)";
		uc.beginPath();
		uc.arc(x, PAD_TOP, 3, 0, Math.PI * 2);
		uc.fill();
		uc.beginPath();
		uc.arc(x, PAD_TOP + plotH, 3, 0, Math.PI * 2);
		uc.fill();

		// Header label — positioned ABOVE the axis with a background pill
		uc.save();
		uc.font = `bold 11px ${fontFamily}`;
		uc.textAlign = "center";
		uc.textBaseline = "bottom";

		const labelText = axis.name.length > 18 ? axis.name.substring(0, 16) + "…" : axis.name;
		const labelW = uc.measureText(labelText).width + 14;
		const labelH = 20;
		const labelX = x;
		const labelY = PAD_TOP - 28;

		// Pill background
		uc.fillStyle = isDark ? "rgba(30, 40, 70, 0.9)" : "rgba(255, 255, 255, 0.95)";
		uc.shadowColor = isDark ? "rgba(0,0,0,0.4)" : "rgba(0,0,0,0.08)";
		uc.shadowBlur = 6;
		uc.shadowOffsetY = 2;
		roundRect(uc, labelX - labelW / 2, labelY - labelH + 4, labelW, labelH, 6);
		uc.fill();
		uc.shadowBlur = 0;

		// Label text
		uc.fillStyle = labelColor;
		uc.fillText(labelText, labelX, labelY);
		uc.restore();

		// Tick values
		uc.font = `10px ${fontFamily}`;
		uc.textAlign = "center";
		uc.fillStyle = isDark ? "rgba(200,200,220,0.7)" : "rgba(80,80,100,0.8)";

		if (axis.type === "num") {
			// Top value
			uc.textBaseline = "bottom";
			uc.fillText(formatNumber(axis.realMax), x, PAD_TOP - 4);
			// Bottom value
			uc.textBaseline = "top";
			uc.fillText(formatNumber(axis.realMin), x, PAD_TOP + plotH + 6);
		} else {
			const keys = axis.keys;
			uc.textAlign = "left";
			uc.textBaseline = "middle";
			uc.font = `9px ${fontFamily}`;
			const maxLabels = Math.min(keys.length, 15);
			const step = keys.length > 1 ? plotH / (keys.length - 1) : 0;
			for (let k = 0; k < maxLabels; k++) {
				const y = PAD_TOP + plotH - k * step;
				const txt = keys[k].length > 12 ? keys[k].substring(0, 10) + "…" : keys[k];
				uc.fillText(txt, x + 10, y);
			}
			if (keys.length > maxLabels) {
				uc.fillText(`+${keys.length - maxLabels} more`, x + 10, PAD_TOP + plotH + 16);
			}
		}

		// Brush indicator
		if (brushes[i]) {
			const [bMin, bMax] = brushes[i];
			const y1 = yPos(bMax);
			const y2 = yPos(bMin);

			// Brush highlight rectangle
			const brushGrad = uc.createLinearGradient(x - 12, y1, x + 12, y1);
			brushGrad.addColorStop(0, "rgba(99, 140, 255, 0.0)");
			brushGrad.addColorStop(0.3, "rgba(99, 140, 255, 0.25)");
			brushGrad.addColorStop(0.7, "rgba(99, 140, 255, 0.25)");
			brushGrad.addColorStop(1, "rgba(99, 140, 255, 0.0)");
			uc.fillStyle = brushGrad;
			uc.fillRect(x - 12, y1, 24, y2 - y1);

			// Brush handles
			uc.fillStyle = "rgba(99, 140, 255, 0.9)";
			roundRect(uc, x - 10, y1 - 3, 20, 6, 3);
			uc.fill();
			roundRect(uc, x - 10, y2 - 3, 20, 6, 3);
			uc.fill();
		}
	});

	// Color legend (if color active)
	if (colorCol) {
		drawColorLegend(uc, colorScale, colorMin, colorMax, displayW, PAD_TOP, plotH, isDark, fontFamily);
	}

	// Line count badge
	uc.font = `11px ${fontFamily}`;
	uc.textAlign = "left";
	uc.textBaseline = "top";
	uc.fillStyle = isDark ? "rgba(200,200,220,0.5)" : "rgba(100,100,120,0.5)";
	const brushedCount = hasBrush ? filteredIndices.filter(idx => passesBrush(dataArray[idx])).length : totalLines;
	uc.fillText(`${brushedCount.toLocaleString()} / ${totalLines.toLocaleString()} lines`, 12, displayH - 22);

	const t1 = performance.now();
	console.debug(`⚡ Parallel render: ${(t1 - t0).toFixed(1)}ms | ${totalLines} lines × ${axes.length} axes`);
}

// ============================================================
// COLOR LEGEND
// ============================================================
function drawColorLegend(ctx, colorScale, min, max, displayW, padTop, plotH, isDark, fontFamily) {
	const legendW = 14;
	const legendH = Math.min(plotH * 0.5, 200);
	const legendX = displayW - 45;
	const legendY = padTop + (plotH - legendH) / 2;

	// Gradient bar
	const grad = ctx.createLinearGradient(0, legendY + legendH, 0, legendY);
	const c1 = colorScale[0][1];
	const c2 = colorScale[1][1];
	grad.addColorStop(0, namedToRGB(c1));
	grad.addColorStop(1, namedToRGB(c2));

	ctx.fillStyle = grad;
	roundRect(ctx, legendX, legendY, legendW, legendH, 4);
	ctx.fill();

	// Border
	ctx.strokeStyle = isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.1)";
	ctx.lineWidth = 1;
	roundRect(ctx, legendX, legendY, legendW, legendH, 4);
	ctx.stroke();

	// Labels
	ctx.font = `9px ${fontFamily}`;
	ctx.textAlign = "left";
	ctx.fillStyle = isDark ? "rgba(200,200,220,0.7)" : "rgba(80,80,100,0.8)";
	ctx.textBaseline = "bottom";
	ctx.fillText(formatNumber(max), legendX - 2, legendY - 4);
	ctx.textBaseline = "top";
	ctx.fillText(formatNumber(min), legendX - 2, legendY + legendH + 4);
}

// ============================================================
// BRUSH INTERACTION
// ============================================================
function setupCanvasBrushing(canvas, dpr, brushes, updatePlot, numericalCols, categoricalCols, columnVisibility, displayW, displayH) {
	let dragging = false;
	let dragAxisIdx = null;
	let dragStartY = null;

	const PAD_TOP = 80;
	const PAD_BOTTOM = 50;
	const PAD_LEFT = 70;
	const PAD_RIGHT = 70;
	const plotH = displayH - PAD_TOP - PAD_BOTTOM;
	const plotW = displayW - PAD_LEFT - PAD_RIGHT;

	function getAxesCount() {
		let count = 0;
		numericalCols.forEach(col => { if (columnVisibility[col.name]) count++; });
		categoricalCols.forEach(col => { if (columnVisibility[col.name]) count++; });
		return count;
	}

	function getAxisAtX(mouseX) {
		const axesCount = getAxesCount();
		if (axesCount < 2) return -1;
		const spacing = plotW / (axesCount - 1);
		for (let i = 0; i < axesCount; i++) {
			const axisX = PAD_LEFT + i * spacing;
			if (Math.abs(mouseX - axisX) < 25) return i;
		}
		return -1;
	}

	function yToNorm(y) {
		return Math.max(0, Math.min(1, 1 - (y - PAD_TOP) / plotH));
	}

	function getMousePos(e) {
		const rect = canvas.getBoundingClientRect();
		return {
			x: (e.clientX - rect.left),
			y: (e.clientY - rect.top)
		};
	}

	canvas.addEventListener("mousedown", (e) => {
		const { x, y } = getMousePos(e);
		const axisIdx = getAxisAtX(x);
		if (axisIdx >= 0) {
			dragging = true;
			dragAxisIdx = axisIdx;
			dragStartY = y;
			canvas.style.cursor = "ns-resize";
		}
	});

	canvas.addEventListener("mousemove", (e) => {
		const { x, y } = getMousePos(e);
		if (!dragging) {
			const axisIdx = getAxisAtX(x);
			canvas.style.cursor = axisIdx >= 0 ? "crosshair" : "default";
			return;
		}
		const n1 = yToNorm(dragStartY);
		const n2 = yToNorm(y);
		brushes[dragAxisIdx] = [Math.min(n1, n2), Math.max(n1, n2)];
		updatePlot();
	});

	canvas.addEventListener("mouseup", () => {
		dragging = false;
		canvas.style.cursor = "crosshair";
	});

	canvas.addEventListener("mouseleave", () => {
		dragging = false;
	});

	canvas.addEventListener("dblclick", (e) => {
		const { x } = getMousePos(e);
		const axisIdx = getAxisAtX(x);
		if (axisIdx >= 0 && brushes[axisIdx]) {
			delete brushes[axisIdx];
			updatePlot();
		} else if (axisIdx < 0) {
			// Double-click on empty space clears all brushes
			Object.keys(brushes).forEach(k => delete brushes[k]);
			updatePlot();
		}
	});
}

// ============================================================
// UTILITIES
// ============================================================
function roundRect(ctx, x, y, w, h, r) {
	ctx.beginPath();
	ctx.moveTo(x + r, y);
	ctx.lineTo(x + w - r, y);
	ctx.quadraticCurveTo(x + w, y, x + w, y + r);
	ctx.lineTo(x + w, y + h - r);
	ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
	ctx.lineTo(x + r, y + h);
	ctx.quadraticCurveTo(x, y + h, x, y + h - r);
	ctx.lineTo(x, y + r);
	ctx.quadraticCurveTo(x, y, x + r, y);
	ctx.closePath();
}

function formatNumber(n) {
	if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + "M";
	if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
	if (Number.isInteger(n)) return n.toString();
	if (Math.abs(n) < 0.01) return n.toExponential(2);
	return n.toPrecision(4);
}

function interpolateColorHSL(colorScale, t, alpha) {
	// HSL interpolation for beautiful gradients
	const c1 = namedToHSL(colorScale[0][1]);
	const c2 = namedToHSL(colorScale[1][1]);

	const h = c1.h + (c2.h - c1.h) * t;
	const s = c1.s + (c2.s - c1.s) * t;
	const l = c1.l + (c2.l - c1.l) * t;

	return `hsla(${h.toFixed(0)}, ${s.toFixed(0)}%, ${l.toFixed(0)}%, ${alpha})`;
}

function namedToHSL(name) {
	const map = {
		red: { h: 0, s: 80, l: 50 },
		green: { h: 140, s: 70, l: 40 },
		blue: { h: 220, s: 80, l: 55 },
	};
	return map[name] || { h: 220, s: 60, l: 55 };
}

function namedToRGB(name) {
	const map = {
		red: "rgb(220, 60, 60)",
		green: "rgb(40, 167, 70)",
		blue: "rgb(60, 120, 220)",
	};
	return map[name] || "rgb(100, 150, 200)";
}

function getTextColor() {
	return (typeof theme !== "undefined" && theme === "dark") ? "#e0e0e0" : "#2d3436";
}

// ============================================================
// EXISTING HELPERS (unchanged logic)
// ============================================================
function parallel_classify_columns(headers, dataArray, ignoreSet, enable_slurm_id_if_exists, numericalCols, categoricalCols, categoryMappings) {
	headers.forEach((header, colIndex) => {
		if (ignoreSet.has(header)) return;
		if (!enable_slurm_id_if_exists && header === "OO_Info_SLURM_JOB_ID") return;

		let isNumerical = true;
		for (let i = 0; i < dataArray.length; i++) {
			if (isNaN(parseFloat(dataArray[i][colIndex]))) {
				isNumerical = false;
				break;
			}
		}

		if (isNumerical) {
			numericalCols.push({ name: header, index: colIndex });
		} else {
			categoricalCols.push({ name: header, index: colIndex });
			const uniqueValues = [...new Set(dataArray.map(row => row[colIndex]))];
			categoryMappings[header] = Object.fromEntries(uniqueValues.map((val, i) => [val, i]));
		}
	});
}

function parallel_precompute_category_mappings(dataArray, categoricalCols) {
	const precomputedMappings = {};
	categoricalCols.forEach(col => {
		const uniqueValues = [...new Set(dataArray.map(row => row[col.index]))];
		precomputedMappings[col.name] = Object.fromEntries(uniqueValues.map((val, i) => [val, i]));
	});
	return precomputedMappings;
}

function parallel_create_controls(headers, dataArray, ignoreSet, enable_slurm_id_if_exists, numericalCols) {
	const controlContainerId = "parallel-plot-controls";
	let controlContainer = $("#" + controlContainerId);
	if (controlContainer.length === 0) {
		controlContainer = $('<div id="' + controlContainerId + '" style="margin-bottom:10px; display: flex; flex-wrap: wrap; gap: 8px;"></div>');
		$("#parallel-plot").before(controlContainer);
	} else {
		controlContainer.empty();
	}

	const columnVisibility = {};
	const minMaxLimits = {};
	const headerIndex = {};
	headers.forEach((h, i) => headerIndex[h] = i);

	const fragment = document.createDocumentFragment();

	headers.forEach(header => {
		try {
			if (ignoreSet.has(header)) return;
			if (!enable_slurm_id_if_exists && header === "OO_Info_SLURM_JOB_ID") return;
			const isNumerical = numericalCols.some(col => col.name === header);
			const box = parallel_create_control_box(header, dataArray, headerIndex, isNumerical, columnVisibility, minMaxLimits);
			fragment.appendChild(box[0]);
		} catch (error) {
			console.error(`Error at header '${header}':`, error);
		}
	});

	controlContainer.append(fragment);
	return { controlContainer, columnVisibility, minMaxLimits, headerIndex };
}

function parallel_create_control_box(header, dataArray, headerIndex, isNumerical, columnVisibility, minMaxLimits) {
	const checkboxId = `chk_${header}`;
	columnVisibility[header] = true;
	minMaxLimits[header] = { min: null, max: null };

	const boxWrapper = $('<div></div>').css({
		border: "1px solid #ddd",
		borderRadius: "8px",
		padding: "12px 16px",
		marginBottom: "12px",
		boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
		backgroundColor: "#fff",
		display: "flex",
		flexWrap: "wrap",
		alignItems: "center",
		gap: "15px",
		maxWidth: "350px",
		width: "100%",
		boxSizing: "border-box"
	});

	const container = $('<div></div>').css({
		display: "flex",
		alignItems: "center",
		gap: "10px",
		flexWrap: "wrap",
		flexGrow: 1,
		minWidth: "0"
	});

	const checkbox = $(`<input type="checkbox" id="${checkboxId}" checked />`);
	const label = $(`<label for="${checkboxId}" style="font-weight: 600; min-width: 140px; cursor: pointer; white-space: nowrap;">${header}</label>`);
	container.append(checkbox).append(label);

	if (isNumerical) {
		const { minWrapper, maxWrapper } = parallel_create_min_max_inputs(header, dataArray, headerIndex);
		container.append(minWrapper).append(maxWrapper);
	}

	boxWrapper.append(container);
	return boxWrapper;
}

function parallel_create_min_max_inputs(header, dataArray, headerIndex) {
	const colIdx = headerIndex[header];
	let minVal = Infinity, maxVal = -Infinity;
	for (let i = 0; i < dataArray.length; i++) {
		const v = parseFloat(dataArray[i][colIdx]);
		if (!isNaN(v)) {
			if (v < minVal) minVal = v;
			if (v > maxVal) maxVal = v;
		}
	}
	if (!isFinite(minVal)) { minVal = 0; maxVal = 100; }

	function makeInput(labelText, id, placeholder, valMin, valMax) {
		const wrapper = $('<div></div>').css({
			display: "flex",
			flexDirection: "column",
			alignItems: "flex-start",
			minWidth: "90px"
		});
		const lbl = $('<label></label>').attr("for", id).text(labelText).css({
			fontSize: "0.75rem",
			color: "#555",
			marginBottom: "2px"
		});
		const input = $(`<input type="number" id="${id}" placeholder="${placeholder}" />`).css({
			width: "80px",
			padding: "5px 8px",
			borderRadius: "5px",
			border: "1px solid #ccc",
			boxShadow: "inset 0 1px 3px rgba(0,0,0,0.1)",
			transition: "border-color 0.3s ease"
		});
		input.attr("min", valMin).attr("max", valMax);
		input.on("focus", function () { $(this).css("border-color", "#007BFF"); });
		input.on("blur", function () { $(this).css("border-color", "#ccc"); });
		wrapper.append(lbl).append(input);
		return wrapper;
	}

	return {
		minWrapper: makeInput("Min", `min_${header}`, "min", minVal, maxVal),
		maxWrapper: makeInput("Max", `max_${header}`, "max", minVal, maxVal)
	};
}

function parallel_bind_min_max_and_checkbox_events(headers, dataArray, numericalCols, columnVisibility, minMaxLimits, updatePlot) {
	let _inputDebounce = null;
	headers.forEach(header => {
		const minInput = $(`#min_${header}`);
		const maxInput = $(`#max_${header}`);
		const checkbox = $(`#chk_${header}`);

		if (minInput.length > 0) {
			minInput.on("input", function () {
				const val = parseFloat($(this).val());
				minMaxLimits[header].min = isNaN(val) ? null : val;
				clearTimeout(_inputDebounce);
				_inputDebounce = setTimeout(updatePlot, 150);
			});
		}
		if (maxInput.length > 0) {
			maxInput.on("input", function () {
				const val = parseFloat($(this).val());
				minMaxLimits[header].max = isNaN(val) ? null : val;
				clearTimeout(_inputDebounce);
				_inputDebounce = setTimeout(updatePlot, 150);
			});
		}
		if (checkbox.length > 0) {
			checkbox.on("change", function () {
				columnVisibility[header] = $(this).is(":checked");
				updatePlot();
			});
		}
	});
}

function parallel_create_result_selector(resultNames, numericalCols, dataArray) {
	const resultSelectId = "result-select";
	let resultSelect = $(`#${resultSelectId}`);
	if (resultSelect.length === 0) {
		resultSelect = $(`<select id="${resultSelectId}"></select>`);
		$("#parallel-plot-controls").before(resultSelect);
	} else {
		resultSelect.empty();
	}
	resultSelect.append('<option value="none">No color</option>');

	for (let i = 0; i < resultNames.length; i++) {
		let minMaxInfo = "min [auto]";
		if (typeof result_min_max !== "undefined" && result_min_max[i] !== undefined) {
			minMaxInfo = result_min_max[i];
		}
		resultSelect.append(`<option value="${resultNames[i]}">${resultNames[i]} (${minMaxInfo})</option>`);
	}
	const colorValuesRef = { values: null, scale: null };
	return { resultSelect, colorValuesRef };
}

function parallel_handle_color_change(selectElem, numericalCols, dataArray, resultNames, colorValuesRef, updatePlot) {
	const selectedResult = $(selectElem).val();
	if (selectedResult === "none") {
		colorValuesRef.values = null;
		colorValuesRef.scale = null;
		updatePlot();
		return;
	}
	const col = numericalCols.find(c => c.name.toLowerCase() === selectedResult.toLowerCase());
	if (!col) {
		colorValuesRef.values = null;
		colorValuesRef.scale = null;
		updatePlot();
		return;
	}
	colorValuesRef.values = dataArray.map(row => parseFloat(row[col.index]));

	let invertColor = false;
	if (typeof result_min_max !== "undefined") {
		const idx = resultNames.indexOf(selectedResult);
		if (idx !== -1) {
			invertColor = result_min_max[idx] === "max";
		}
	}
	colorValuesRef.scale = invertColor ? [[0, 'red'], [1, 'green']] : [[0, 'green'], [1, 'red']];
	updatePlot();
}
