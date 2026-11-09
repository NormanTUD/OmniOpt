#!/usr/bin/env node

"use strict";

var landscape = require("./js/landscape.js");

var failedTests = 0;
var totalTests = 0;

var expect = function (label, actual, expected) {
	totalTests++;
	var isEqual = JSON.stringify(actual) === JSON.stringify(expected);

	if (isEqual) {
		if (process.env.SHOW_SUCCESS) {
			console.log("PASS: " + label);
		}
	} else {
		console.log("FAIL: " + label);
		console.log("  Expected: " + JSON.stringify(expected));
		console.log("  Actual:   " + JSON.stringify(actual));
		failedTests++;
	}
};

var expectTrue = function (label, actual) {
	expect(label, !!actual, true);
};

var expectClose = function (label, actual, expected, tolerance) {
	totalTests++;
	var ok = typeof actual === "number" && Math.abs(actual - expected) <= tolerance;

	if (ok) {
		if (process.env.SHOW_SUCCESS) {
			console.log("PASS: " + label);
		}
	} else {
		console.log("FAIL: " + label);
		console.log("  Expected: " + expected + " +/- " + tolerance);
		console.log("  Actual:   " + actual);
		failedTests++;
	}
};

console.log("--- Testing: numeric parameter detection ---");
{
	var headers = ["trial_index", "x", "y", "name", "OO_Info_x", "trial_status", "RESULT1"];
	var rows = [
		[0, 1, 2, "a", 4, "COMPLETED", 10],
		[1, 2, 3, "b", 5, "COMPLETED", 9],
		[2, 3, 4, "c", 6, "FAILED", ""]
	];

	var columns = landscape.getNumericParameterColumns(headers, rows, ["RESULT1"], ["trial_index", "trial_status"]);
	expect("detect: only real numeric params", columns, ["x", "y"]);

	var constantHeaders = ["trial_index", "constant", "x", "RESULT1"];
	var constantRows = [
		[0, 7, 1, 10],
		[1, 7, 2, 9],
		[2, 7, 3, 8]
	];

	var constantColumns = landscape.getNumericParameterColumns(constantHeaders, constantRows, ["RESULT1"], ["trial_index"]);
	expect("detect: constant numeric params are ignored", constantColumns, ["x"]);
}

console.log("\n--- Testing: trial collection and failure separation ---");
{
	var headers = ["trial_index", "x", "y", "trial_status", "RESULT1"];
	var rows = [
		[5, 5, 50, "COMPLETED", 2],
		[1, 1, 10, "COMPLETED", 9],
		[3, 3, 30, "FAILED", ""],
		[2, 2, 20, "COMPLETED", 4]
	];

	var data = landscape.collectLandscapeData(headers, rows, ["x", "y"], "RESULT1");
	expect("collect: valid count", data.valid.length, 3);
	expect("collect: failed count", data.failed.length, 1);
	expect("collect: sorted by trial index", data.valid.map(function (point) { return point.trial; }), [1, 2, 5]);
	expect("collect: values keep parameter order", data.valid[0].values, [1, 10]);
	expect("collect: failed keeps params", data.failed[0].values, [3, 30]);

	var bestMin = landscape.findBestPoint(data.valid, "min");
	var bestMax = landscape.findBestPoint(data.valid, "max");
	expect("best: min", bestMin.trial, 5);
	expect("best: max", bestMax.trial, 1);

	var multiHeaders = ["trial_index", "x", "y", "trial_status", "RESULT1", "RESULT2"];
	var multiRows = [
		[5, 5, 50, "COMPLETED", 2, 8],
		[1, 1, 10, "COMPLETED", 9, 1],
		[3, 3, 30, "FAILED", "", ""],
		[2, 2, 20, "COMPLETED", 4, 6]
	];
	var multi = landscape.collectLandscapeDataForResults(multiHeaders, multiRows, ["x", "y"], ["RESULT1", "RESULT2"]);
	expect("collect-multi: first result valid", multi.buckets[0].valid.length, 3);
	expect("collect-multi: second result valid", multi.buckets[1].valid.length, 3);
	expect("collect-multi: first sorted", multi.buckets[0].valid.map(function (point) { return point.trial; }), [1, 2, 5]);
	expect("collect-multi: failed shared", multi.failed.length, 1);
	expect("collect-multi: failed attached", multi.buckets[0].failed, multi.failed);
}

console.log("\n--- Testing: 1D interpolation ---");
{
	var points = [];

	for (var i = 0; i < 20; i++) {
		points.push({
			values: [i / 10],
			loss: (i / 10 - 1) * (i / 10 - 1),
			trial: i
		});
	}

	var curve = landscape.interpolate1D(points);
	expectTrue("1D: returns a curve", !!curve);
	expectTrue("1D: x range is padded", curve.x[0] < 0 && curve.x[curve.x.length - 1] > 1.9);
	expectClose("1D: minimum is close to the true minimum", Math.min.apply(null, curve.z), 0, 0.02);
}

console.log("\n--- Testing: 2D interpolation ---");
{
	var points = [];

	for (var x = 0; x <= 10; x++) {
		for (var y = 0; y <= 10; y++) {
			points.push({
				values: [x, y],
				loss: x + y,
				trial: x * 11 + y
			});
		}
	}

	var surface = landscape.interpolate2D(points, 40);
	expectTrue("2D: returns a surface", !!surface);
	expect("2D: has x grid", surface.x.length, 40);
	expect("2D: has y grid", surface.y.length, 40);
	expect("2D: has z rows", surface.z.length, 40);

	var middle = surface.z[20][20];
	var expected = surface.x[20] + surface.y[20];
	expectClose("2D: interpolates a plane", middle, expected, 0.25);
	expectTrue("2D: grid is finite", surface.z.every(function (row) {
		return row.every(function (value) { return isFinite(value); });
	}));
}

console.log("\n--- Testing: 3D interpolation ---");
{
	var points = [];

	for (var x = 0; x <= 6; x++) {
		for (var y = 0; y <= 6; y++) {
			for (var z = 0; z <= 6; z++) {
				points.push({
					values: [x, y, z],
					loss: x + 2 * y + 3 * z,
					trial: points.length
				});
			}
		}
	}

	var surface = landscape.interpolate3D(points, 24);
	var vertexCount = 24 * 24 * 24;
	expectTrue("3D: returns a volume", !!surface);
	expect("3D: flat x grid", surface.x.length, vertexCount);
	expect("3D: flat y grid", surface.y.length, vertexCount);
	expect("3D: flat z grid", surface.z.length, vertexCount);
	expect("3D: flat values", surface.value.length, vertexCount);
	expect("3D: axis lengths", [surface.xValues.length, surface.yValues.length, surface.zValues.length], [24, 24, 24]);

	var midIndex = (12 * 24 + 12) * 24 + 12;
	var mid = surface.value[midIndex];
	var expected = surface.xValues[12] + 2 * surface.yValues[12] + 3 * surface.zValues[12];
	expectClose("3D: interpolates a linear field", mid, expected, 0.6);
}

console.log("\n--- Testing: surface holes and slice filtering ---");
{
	var points = [];

	for (var x = 0; x <= 10; x += 0.5) {
		for (var y = 0; y <= 10; y += 0.5) {
			points.push({
				values: [x, y],
				loss: x + y,
				trial: points.length
			});
		}
	}

	var failed = [{ values: [5, 5], loss: null, trial: 999 }];
	var surface = landscape.interpolate2D(points, 40, {
		maxGapCells: 4,
		failedPoints: failed,
		failedRadiusCells: 2
	});

	expectTrue("holes: surface exists", !!surface);
	expectTrue("holes: failed position has no height", landscape.surfaceHeightAt(surface, 5, 5) === null);
	expectTrue("holes: valid position has a height", isFinite(landscape.surfaceHeightAt(surface, 2, 2)));

	var sliced = landscape.filterDataBySlice({
		valid: [
			{ values: [0, 0, 0], loss: 1, trial: 0 },
			{ values: [0, 0, 5], loss: 2, trial: 1 },
			{ values: [0, 0, 10], loss: 3, trial: 2 }
		],
		failed: []
	}, "low");
	expectTrue("slice: low keeps the low third", sliced.valid.length >= 1 && sliced.valid.every(function (point) { return point.values[2] <= 5; }));
}

console.log("\n--- Testing: colorscale direction and formatting ---");
{
	var minScale = landscape.getLossColorscale("min");
	var maxScale = landscape.getLossColorscale("max");
	expect("colorscale: min starts green", minScale[0][1], "#0f766e");
	expect("colorscale: max starts red", maxScale[0][1], "#dc2626");
	expect("format: small numbers", landscape.formatNumber(0.00001234), "1.234e-5");
	expect("format: ordinary numbers", landscape.formatNumber(12.3456789), "12.3457");
	expect("label: min", landscape.getResultLabel("RESULT1", ["RESULT1", "RESULT2"], ["min", "max"]), "RESULT1 (min)");
	expect("label: max", landscape.getResultLabel("RESULT2", ["RESULT1", "RESULT2"], ["min", "max"]), "RESULT2 (max)");
}

console.log("\n--- Testing: additional landscape edge cases ---");
{
	var headers = ["TRIAL_INDEX", "X", "Y", "RESULT", "trial_status", "OO_Info_X", "name"];
	var rows = [
		[0, "1", "2", "3", "COMPLETED", "4", "a"],
		[1, "2", "3", "4", "COMPLETED", "5", "b"],
		[2, "3", "4", "5", "COMPLETED", "6", "c"]
	];

	expect("edge: case-insensitive exclusion",
		landscape.getNumericParameterColumns(headers, rows, ["result"], ["trial_index", "trial_status"]),
		["X", "Y"]);

	var mixedRows = [
		[0, "1", "2", "3", "COMPLETED", "4", "a"],
		[1, "nope", "3", "4", "COMPLETED", "5", "b"]
	];
	expect("edge: mixed numeric/string column excluded",
		landscape.getNumericParameterColumns(["trial_index", "mixed", "ok"], mixedRows, ["RESULT"], ["trial_index"]),
		["ok"]);

	expect("edge: single observation excluded",
		landscape.getNumericParameterColumns(["trial_index", "x"], [[0, 1]], ["RESULT"], ["trial_index"]),
		[]);

	expect("goal: object max", landscape.getResultGoal("R", ["R"], { "R": "max" }), "max");
	expect("goal: case-insensitive max", landscape.getResultGoal("r", ["R"], ["MAX"]), "max");
	expect("goal: missing defaults to min", landscape.getResultGoal("missing", ["R"], ["max"]), "min");
	expect("goal: malformed defaults to min", landscape.getResultGoal("R", ["R"], null), "min");

	var collectHeaders = ["Trial_Index", "x", "trial_status", "RESULT"];
	var collectRows = [
		[2, 1, "COMPLETED", 5],
		[1, 2, "COMPLETED", 4],
		[3, 3, "FAILED", 9],
		[4, 4, "COMPLETED", ""]
	];
	var collected = landscape.collectLandscapeData(collectHeaders, collectRows, ["x"], "result");
	expect("collect: case-insensitive result", collected.valid.length, 2);
	expect("collect: sorted trials", collected.valid.map(function (point) { return point.trial; }), [1, 2]);
	expect("collect: failed with numeric result stays failed", collected.failed.length, 1);
	expect("collect: missing result column", landscape.collectLandscapeData(collectHeaders, collectRows, ["x"], "NOPE").valid.length, 0);
}

console.log("\n--- Testing: interpolation null / boundary cases ---");
{
	expect("1D: empty returns null", landscape.interpolate1D([]), null);
	expect("1D: one point returns null", landscape.interpolate1D([{ values: [1], loss: 2 }]), null);
	expect("1D: equal x returns null", landscape.interpolate1D([
		{ values: [1], loss: 2 },
		{ values: [1], loss: 3 }
	]), null);

	var linearPoints = [];
	for (var i = 0; i <= 10; i++) {
		linearPoints.push({ values: [i], loss: 2 * i + 1, trial: i });
	}
	var linearCurve = landscape.interpolate1D(linearPoints);
	expectClose("1D: linear midpoint", landscape.interpolate1DHeightAt(linearCurve, 5.5), 12, 0.25);
	expect("1D: below range returns null", landscape.interpolate1DHeightAt(linearCurve, -1), null);
	expect("1D: above range returns null", landscape.interpolate1DHeightAt(linearCurve, 11), null);

	expect("2D: fewer than 3 points returns null", landscape.interpolate2D([
		{ values: [0, 0], loss: 1 },
		{ values: [1, 1], loss: 2 }
	], 20), null);
	expect("2D: equal x returns null", landscape.interpolate2D([
		{ values: [1, 0], loss: 1 },
		{ values: [1, 1], loss: 2 },
		{ values: [1, 2], loss: 3 }
	], 20), null);

	var sparse = [
		{ values: [0, 0], loss: 1 },
		{ values: [5, 5], loss: 2 },
		{ values: [10, 0], loss: 3 }
	];
	var sparseSurface = landscape.interpolate2D(sparse, 20, { maxGapCells: 0 });
	var sparseHasNaN = false;
	for (var sy = 0; sy < sparseSurface.z.length; sy++) {
		for (var sx = 0; sx < sparseSurface.z[sy].length; sx++) {
			if (!isFinite(sparseSurface.z[sy][sx])) {
				sparseHasNaN = true;
			}
		}
	}
	expectTrue("2D: coverage mask leaves holes", sparseHasNaN);

	var dense = [];
	for (var dx = 0; dx <= 10; dx += 0.5) {
		for (var dy = 0; dy <= 10; dy += 0.5) {
			dense.push({ values: [dx, dy], loss: dx + 2 * dy, trial: dense.length });
		}
	}
	var plane = landscape.interpolate2D(dense, 60);
	expectClose("2D: bilinear height on a plane", landscape.surfaceHeightAt(plane, 3.3, 4.4), 12.1, 0.5);
	expect("2D: outside range returns null", landscape.surfaceHeightAt(plane, -1, 4), null);

	var optionsDense = landscape.surfaceInterpolationOptions(dense, [], 56);
	var optionsSparse = landscape.surfaceInterpolationOptions(sparse, [], 56);
	expect("options: grid size is honoured", optionsDense.gridSize, 56);
	expectTrue("options: failed radius is finite", optionsDense.failedRadiusCells >= 2);
	expectTrue("options: sparse data allows larger gaps", optionsSparse.maxGapCells >= optionsDense.maxGapCells);

	expect("3D: fewer than 4 points returns null", landscape.interpolate3D([
		{ values: [0, 0, 0], loss: 1 },
		{ values: [1, 1, 1], loss: 2 },
		{ values: [2, 2, 2], loss: 3 }
	], 20), null);
}

console.log("\n--- Testing: slicing and helpers ---");
{
	var data = {
		valid: [],
		failed: []
	};

	for (var i = 0; i < 10; i++) {
		data.valid.push({ values: [i, i, i], loss: i, trial: i });
	}
	data.failed.push({ values: [5, 5, 5], loss: null, trial: 100 });

	var all = landscape.filterDataBySlice(data, "all");
	var low = landscape.filterDataBySlice(data, "low");
	var middle = landscape.filterDataBySlice(data, "middle");
	var high = landscape.filterDataBySlice(data, "high");

	expect("slice: all keeps all", all.valid.length, 10);
	expectTrue("slice: low keeps at least one", low.valid.length >= 1);
	expectTrue("slice: middle keeps at least one", middle.valid.length >= 1);
	expectTrue("slice: high keeps at least one", high.valid.length >= 1);
	expect("slice: partitions all valid points", low.valid.length + middle.valid.length + high.valid.length, 10);
	expectTrue("slice: low values", low.valid.every(function (point) { return point.values[2] <= 4; }));
	expectTrue("slice: high values", high.valid.every(function (point) { return point.values[2] >= 6; }));

	expect("sample: returns all when short", landscape.sampleEvenly([1, 2, 3], 5).length, 3);
	var sampled = landscape.sampleEvenly([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4);
	expect("sample: respects max count", sampled.length, 4);
	expect("sample: keeps first", sampled[0], 0);
	expect("sample: keeps last", sampled[sampled.length - 1], 9);

	expect("format: zero", landscape.formatNumber(0), "0");
	expect("format: negative", landscape.formatNumber(-1.23456789), "-1.23457");
	expect("format: not available", landscape.formatNumber(NaN), "–");
	expect("format: very large", landscape.formatNumber(1e7), "1.000e+7");
}

console.log("\n--- Testing: live-update signature ---");
{
	var entries = [{ resultName: "RESULT1" }, { resultName: "RESULT2" }];
	var base = landscape.landscapeSignature(3, ["x", "y"], entries, false, false);
	expect("signature: stable", landscape.landscapeSignature(3, ["x", "y"], entries, false, false), base);
	expectTrue("signature: mode changes", landscape.landscapeSignature(2, ["x", "y"], entries, false, false) !== base);
	expectTrue("signature: params change", landscape.landscapeSignature(3, ["x", "z"], entries, false, false) !== base);
	expectTrue("signature: result changes", landscape.landscapeSignature(3, ["x", "y"], [{ resultName: "RESULT1" }], false, false) !== base);
	expectTrue("signature: failed toggle changes", landscape.landscapeSignature(3, ["x", "y"], entries, true, false) !== base);
	expectTrue("signature: points toggle changes", landscape.landscapeSignature(3, ["x", "y"], entries, false, true) !== base);
}

console.log("\n--- SUMMARY ---");

if (failedTests === 0) {
	console.log("All " + totalTests + " landscape tests passed successfully.");
} else {
	console.log(failedTests + " of " + totalTests + " landscape test(s) failed.");
}

process.exit(failedTests === 0 ? 0 : 1);
