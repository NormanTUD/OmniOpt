#!/usr/bin/env node

"use strict";

var analysis = require("./js/results_analysis.js");

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

var expectFalse = function (label, actual) {
	expect(label, !!actual, false);
};

var buildFailureData = function () {
	var headers = ["trial_index", "x", "y", "trial_status", "RESULT1"];
	var data = [];

	for (var x = 0; x < 10; x++) {
		for (var y = 0; y < 10; y++) {
			var failed = x >= 7 && y <= 2;
			var result = failed ? 5 : 10 - (x >= 5 ? 2 : 0) - (y <= 5 ? 2 : 0);
			data.push([data.length, x, y, failed ? "FAILED" : "COMPLETED", result]);
		}
	}

	return { headers: headers, data: data };
};

var treeDepth = function (node) {
	if (!node || node.leaf) {
		return 1;
	}
	return 1 + Math.max(treeDepth(node.left), treeDepth(node.right));
};

console.log("--- Testing: result query parser ---");
{
	var headers = ["x", "name", "trial_status", "RESULT1"];
	var rows = [
		[1, "foo", "COMPLETED", 0.5],
		[10, "bar", "FAILED", 1.5],
		[2, "FooBar", "FAILED", 2.5]
	];

	var query = analysis.compileResultQuery("x > 5 AND trial_status = FAILED", headers);
	expectTrue("query: valid numeric/string query compiles", query.ok);
	expect("query: normalized", query.normalized, "x > 5 AND trial_status = \"FAILED\"");
	expect("query: matches", rows.filter(query.predicate).length, 1);

	query = analysis.compileResultQuery("status = FAILED", headers);
	expectTrue("query: status alias compiles", query.ok);
	expect("query: status alias normalized", query.normalized, "trial_status = \"FAILED\"");
	expect("query: status alias matches", rows.filter(query.predicate).length, 2);

	query = analysis.compileResultQuery("name contains \"foo\"", headers);
	expect("query: contains is case-insensitive", rows.filter(query.predicate).length, 2);

	query = analysis.compileResultQuery("(x > 5 OR name = foo) AND trial_status = COMPLETED", headers);
	expect("query: parentheses and precedence", rows.filter(query.predicate).length, 1);

	query = analysis.compileResultQuery("x != 1 AND RESULT1 >= 1", headers);
	expect("query: != and result filter", rows.filter(query.predicate).length, 2);

	query = analysis.compileResultQuery("x > abc", headers);
	expectTrue("query: non-numeric comparison compiles", query.ok);
	expect("query: non-numeric comparison matches nothing", rows.filter(query.predicate).length, 0);

	query = analysis.compileResultQuery("nope = 1", headers);
	expectFalse("query: unknown column rejected", query.ok);
	expectTrue("query: unknown column has an error", /Unknown column/.test(query.error));

	query = analysis.compileResultQuery("x > 5 AND", headers);
	expectFalse("query: dangling AND rejected", query.ok);
	expectTrue("query: dangling AND has an error", /Expected a column name/.test(query.error));

	query = analysis.compileResultQuery("x >", headers);
	expectFalse("query: missing value rejected", query.ok);
	expectTrue("query: missing value has an error", /Expected a value/.test(query.error));

	query = analysis.compileResultQuery("", headers);
	expectFalse("query: empty query rejected", query.ok);

	var aggregate = analysis.computeAggregate(rows, headers, ["RESULT1"], function (row) {
		return row[2] === "FAILED";
	});
	expect("aggregate: count", aggregate.count, 2);
	expect("aggregate: total", aggregate.total, 3);
	expect("aggregate: result count", aggregate.results[0].count, 2);
	expect("aggregate: min", aggregate.results[0].min, 1.5);
	expect("aggregate: median", aggregate.results[0].median, 2);
	expect("aggregate: max", aggregate.results[0].max, 2.5);
}

console.log("\n--- Testing: feature selection ---");
{
	var headers = ["x", "OO_Info_x", "OO_Info_y", "trial_status", "RESULT1", "generation_node", "run_time"];
	var columns = analysis.selectFeatureColumns(headers, {
		resultNames: ["RESULT1"],
		specialColumns: analysis.DEFAULT_SPECIAL_COLUMNS
	});
	expect("features: ignores every OO_Info column", columns.map(function (c) { return c.name; }), ["x"]);

	columns = analysis.selectFeatureColumns(headers, {
		resultNames: ["RESULT1"],
		specialColumns: analysis.DEFAULT_SPECIAL_COLUMNS,
		extraColumns: ["generation_node"]
	});
	expect("features: extra categorical column is kept", columns.map(function (c) { return c.name; }), ["x", "generation_node"]);
}

console.log("\n--- Testing: failure decision tree ---");
{
	var fixture = buildFailureData();
	var result = analysis.analyzeFailures(fixture.data, fixture.headers, {
		resultNames: ["RESULT1"],
		maxDepth: 4,
		minLeaf: 3
	});

	expectTrue("failure tree: analysis succeeds", result.ok);
	expect("failure tree: total", result.total, 100);
	expect("failure tree: positives", result.positives, 9);
	expect("failure tree: negatives", result.negatives, 91);
	expect("failure tree: baseline", result.baselineRate, 0.09);
	expectTrue("failure tree: top rule exists", result.topRules.length > 0);
	expect("failure tree: top rule is pure", result.topRules[0].rate, 1);
	expect("failure tree: top rule support", result.topRules[0].samples, 9);

	var ruleText = analysis.conditionsToQuery(result.topRules[0].conditions);
	expectTrue("failure tree: rule contains x split", /x >/.test(ruleText));
	expectTrue("failure tree: rule contains y split", /y <=/.test(ruleText));

	var compiled = analysis.compileResultQuery(ruleText, fixture.headers);
	expectTrue("failure tree: rule query compiles", compiled.ok);
	expect("failure tree: rule query matches the same rows",
		fixture.data.filter(compiled.predicate).length, result.topRules[0].samples);

	var allRules = analysis.treeToRules(result.tree);
	var covered = allRules.reduce(function (sum, rule) { return sum + rule.samples; }, 0);
	expect("failure tree: leaves partition the data", covered, 100);
	expectTrue("failure tree: max depth respected", treeDepth(result.tree) <= 5);
	expect("failure tree: confusion total", result.confusion.total, 100);
	expect("failure tree: confusion counts", result.confusion.tp + result.confusion.fp + result.confusion.tn + result.confusion.fn, 100);
	expect("failure tree: training accuracy", result.confusion.accuracy, 1);
	expectTrue("failure tree: importance contains x", result.featureImportance.some(function (entry) { return entry.name === "x"; }));
	expectTrue("failure tree: importance contains y", result.featureImportance.some(function (entry) { return entry.name === "y"; }));

	var depthLimited = analysis.analyzeFailures(fixture.data, fixture.headers, {
		resultNames: ["RESULT1"],
		maxDepth: 1,
		minLeaf: 3
	});
	expectTrue("failure tree: depth limit still succeeds", depthLimited.ok);
	expectTrue("failure tree: depth limit respected", treeDepth(depthLimited.tree) <= 2);
}

console.log("\n--- Testing: failure tree ignores OO_Info columns ---");
{
	var headers = ["trial_index", "OO_Info_ZEIT", "n_conv", "n_dense", "trial_status", "RESULT1"];
	var data = [];
	for (var i = 0; i < 200; i++) {
		var nConv = i % 10;
		var nDense = Math.floor(i / 10) % 10;
		var failed = nConv > 5 && nDense < 2;
		data.push([i, failed ? 10000 + i : i, nConv, nDense, failed ? "FAILED" : "COMPLETED", failed ? 1 : 10]);
	}
	var result = analysis.analyzeFailures(data, headers, { resultNames: ["RESULT1"] });
	expectTrue("failure tree: OO_Info analysis succeeds", result.ok);
	expectTrue("failure tree: OO_Info rule exists", result.topRules.length > 0);
	var ruleText = analysis.conditionsToQuery(result.topRules[0].conditions);
	expectFalse("failure tree: OO_Info is not used", /OO_Info_/.test(ruleText));
	expectTrue("failure tree: real parameter n_conv is used", /n_conv/.test(ruleText));
	expectTrue("failure tree: real parameter n_dense is used", /n_dense/.test(ruleText));
	expectTrue("failure tree: rule is based on all rows", result.total === 200);
}

console.log("\n--- Testing: failure tree exit-code fallback ---");
{
	var headers = ["trial_index", "x", "exit_code", "RESULT1"];
	var data = [];
	for (var i = 0; i < 100; i++) {
		var x = i % 10;
		data.push([i, x, x >= 7 ? 1 : 0, 10]);
	}
	var result = analysis.analyzeFailures(data, headers, { resultNames: ["RESULT1"] });
	expectTrue("failure tree: exit_code fallback succeeds", result.ok);
	expectTrue("failure tree: exit_code fallback finds a rule", result.topRules.length > 0);
	expectTrue("failure tree: exit_code fallback uses x", /x/.test(analysis.conditionsToQuery(result.topRules[0].conditions)));
}

console.log("\n--- Testing: failure tree without failures ---");
{
	var fixture = buildFailureData();
	fixture.data.forEach(function (row) {
		row[3] = "COMPLETED";
	});
	var result = analysis.analyzeFailures(fixture.data, fixture.headers, { resultNames: ["RESULT1"] });
	expectFalse("failure tree: no failures is rejected", result.ok);
	expectTrue("failure tree: no failures gives a reason", /only one outcome class/.test(result.reason));
}

console.log("\n--- Testing: subgroup / result patterns ---");
{
	var fixture = buildFailureData();
	var result = analysis.findResultPatterns(fixture.data, fixture.headers, ["RESULT1"], ["min"], {
		minSupport: 5,
		minEffect: 0.05
	});

	expectTrue("patterns: analysis succeeds", result.ok);
	expect("patterns: one result group", result.results.length, 1);
	expect("patterns: only completed trials are considered", result.results[0].considered, 91);
	expect("patterns: goal", result.results[0].goal, "min");
	expectTrue("patterns: better patterns exist", result.results[0].good.length > 0);
	expectTrue("patterns: worse patterns exist", result.results[0].bad.length > 0);

	var best = result.results[0].good[0];
	var text = analysis.conditionsToQuery(best.conditions);
	expectTrue("patterns: best rule contains a parameter", /x|y/.test(text));
	expectTrue("patterns: best rule is restricted to completed trials", /trial_status = \"COMPLETED\"/.test(text));
	expectTrue("patterns: best rule improves the result", best.stats.mean < best.stats.complementMean);
	expectTrue("patterns: best rule has support", best.stats.n >= 5);
	expectTrue("patterns: all better rules have support", result.results[0].good.every(function (rule) { return rule.stats.n >= 5; }));
	expectTrue("patterns: all worse rules have support", result.results[0].bad.every(function (rule) { return rule.stats.n >= 5; }));

	var compiled = analysis.compileResultQuery(text, fixture.headers);
	expectTrue("patterns: rule query compiles", compiled.ok);
	expect("patterns: rule query matches its support", fixture.data.filter(compiled.predicate).length, best.stats.n);

	var maxFixture = buildFailureData();
	maxFixture.data.forEach(function (row) {
		row[4] = row[3] === "FAILED" ? 100 : (row[1] >= 5 && row[2] <= 5 ? 20 : 10);
	});
	var maxResult = analysis.findResultPatterns(maxFixture.data, maxFixture.headers, ["RESULT1"], ["max"], {
		minSupport: 5,
		minEffect: 0.05
	});
	expectTrue("patterns: max goal succeeds", maxResult.ok);
	expect("patterns: max goal", maxResult.results[0].goal, "max");
	expectTrue("patterns: max goal finds a better group", maxResult.results[0].good.length > 0);
	expectTrue("patterns: max goal group is larger than complement", maxResult.results[0].good[0].stats.mean > maxResult.results[0].good[0].stats.complementMean);

	var partial = analysis.findResultPatterns(fixture.data, fixture.headers, ["MISSING", "RESULT1"], ["min", "max"], {
		minSupport: 5,
		minEffect: 0.05
	});
	expectTrue("patterns: partially missing result list still works", partial.ok);
	expect("patterns: goal mapping uses the original result list", partial.results[0].goal, "max");

	var invalid = analysis.findResultPatterns(fixture.data, fixture.headers, ["MISSING"], ["min"], { minSupport: 5 });
	expectFalse("patterns: missing result is rejected", invalid.ok);
}

console.log("\n--- Testing: formatting and condition helpers ---");
{
	var condition = { column: "name", operator: "==", value: "a\"b" };
	expect("format: quotes and escapes strings", analysis.formatCondition(condition), "name = \"a\\\"b\"");
	expect("format: number", analysis.formatNumber(1.234567890123), "1.2345679");
	expect("format: integer", analysis.formatNumber(5), "5");

	var row = [1, "foo", "FAILED"];
	var conditions = [{ column: "x", index: 0, operator: ">", value: 0 }];
	expect("conditions: match row", analysis.conditionsMatchRow(conditions, row, { x: 0 }), true);
	expect("conditions: do not match row", analysis.conditionsMatchRow([{ column: "x", index: 0, operator: ">", value: 2 }], row, { x: 0 }), false);
}

console.log("\n--- Testing: results table external filter ---");
{
	var fs = require("fs");
	var tableCode = fs.readFileSync(__dirname + "/js/results_table.js", "utf8");
	var tableHelpers = new Function(
		tableCode + "\nreturn { _resultsFilterRows: _resultsFilterRows };"
	)();
	var data = [
		[1, "foo", "COMPLETED"],
		[2, "bar", "FAILED"],
		[3, "foo", "FAILED"]
	];

	expect("table filter: no filters keeps all rows",
		tableHelpers._resultsFilterRows(data, "", null), [0, 1, 2]);
	expect("table filter: search is case-insensitive",
		tableHelpers._resultsFilterRows(data, "FOO", null), [0, 2]);
	expect("table filter: external filter selects rows",
		tableHelpers._resultsFilterRows(data, "", function (row) { return row[2] === "FAILED"; }), [1, 2]);
	expect("table filter: search and external filter compose",
		tableHelpers._resultsFilterRows(data, "foo", function (row) { return row[2] === "FAILED"; }), [2]);
	expect("table filter: row index is passed to the filter",
		tableHelpers._resultsFilterRows(data, "", function (row, index) { return index === 1; }), [1]);
}

console.log("\n---------------------------------");
if (failedTests === 0) {
	console.log("SUMMARY: All " + totalTests + " results-analysis tests passed successfully.");
	process.exit(0);
} else {
	console.log("SUMMARY: " + failedTests + " of " + totalTests + " results-analysis test(s) failed.");
	process.exit(1);
}
