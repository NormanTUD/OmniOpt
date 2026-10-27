(function (root, factory) {
	if (typeof module === "object" && module.exports) {
		module.exports = factory(root);
	} else {
		root.OmniOptResultsAnalysis = factory(root);
	}
})(typeof globalThis !== "undefined" ? globalThis : this, function (root) {
	"use strict";

	var DEFAULT_SPECIAL_COLUMNS = [
		"trial_index",
		"worker_generator_uuid",
		"arm_name",
		"trial_status",
		"generation_method",
		"generation_node",
		"hostname",
		"run_time",
		"start_time",
		"exit_code",
		"signal",
		"end_time",
		"program_string",
		"submit_time",
		"queue_time"
	];

	function isBlank(value) {
		return value === null || value === undefined || String(value).trim() === "";
	}

	function toNumber(value) {
		if (value === null || value === undefined) {
			return null;
		}
		if (typeof value === "number") {
			return isFinite(value) ? value : null;
		}
		var text = String(value).trim();
		if (text === "") {
			return null;
		}
		var number = Number(text);
		return isFinite(number) ? number : null;
	}

	function valuesEqual(left, right) {
		var leftNumber = toNumber(left);
		var rightNumber = toNumber(right);

		if (leftNumber !== null && rightNumber !== null) {
			return leftNumber === rightNumber;
		}

		return String(left === null || left === undefined ? "" : left) ===
			String(right === null || right === undefined ? "" : right);
	}

	function compareValues(left, right, operator) {
		var leftNumber = toNumber(left);
		var rightNumber = toNumber(right);

		if (leftNumber === null || rightNumber === null) {
			return false;
		}

		switch (operator) {
			case "<":
				return leftNumber < rightNumber;
			case "<=":
				return leftNumber <= rightNumber;
			case ">":
				return leftNumber > rightNumber;
			case ">=":
				return leftNumber >= rightNumber;
			default:
				return false;
		}
	}

	function conditionMatchesValue(condition, value) {
		switch (condition.operator) {
			case "contains":
				return String(value === null || value === undefined ? "" : value)
					.toLowerCase()
					.indexOf(String(condition.value === null || condition.value === undefined ? "" : condition.value).toLowerCase()) !== -1;
			case "=":
			case "==":
				return valuesEqual(value, condition.value);
			case "!=":
				return !valuesEqual(value, condition.value);
			case "<":
			case "<=":
			case ">":
			case ">=":
				return compareValues(value, condition.value, condition.operator);
			default:
				return false;
		}
	}

	function invertOperator(operator) {
		switch (operator) {
			case "<":
				return ">=";
			case "<=":
				return ">";
			case ">":
				return "<=";
			case ">=":
				return "<";
			case "==":
				return "!=";
			case "!=":
				return "==";
			default:
				return operator;
		}
	}

	function inverseCondition(condition) {
		return {
			column: condition.column,
			index: condition.index,
			operator: invertOperator(condition.operator),
			value: condition.value
		};
	}

	function median(values) {
		if (!values.length) {
			return null;
		}
		var sorted = values.slice().sort(function (a, b) { return a - b; });
		var middle = Math.floor(sorted.length / 2);
		if (sorted.length % 2 === 1) {
			return sorted[middle];
		}
		return (sorted[middle - 1] + sorted[middle]) / 2;
	}

	function mean(values) {
		if (!values.length) {
			return null;
		}
		var sum = 0;
		for (var i = 0; i < values.length; i++) {
			sum += values[i];
		}
		return sum / values.length;
	}

	function standardDeviation(values, average) {
		if (values.length < 2) {
			return 0;
		}
		var sum = 0;
		for (var i = 0; i < values.length; i++) {
			var difference = values[i] - average;
			sum += difference * difference;
		}
		return Math.sqrt(sum / values.length);
	}

	function quantileSorted(sorted, q) {
		if (!sorted.length) {
			return null;
		}
		var position = (sorted.length - 1) * q;
		var base = Math.floor(position);
		var rest = position - base;
		var next = sorted[Math.min(base + 1, sorted.length - 1)];
		return sorted[base] + (next - sorted[base]) * rest;
	}

	function quantile(values, q) {
		if (!values.length) {
			return null;
		}
		var sorted = values.slice().sort(function (a, b) { return a - b; });
		return quantileSorted(sorted, q);
	}

	function uniqueNumbers(values) {
		var unique = [];
		for (var i = 0; i < values.length; i++) {
			if (!isFinite(values[i])) {
				continue;
			}
			if (!unique.length || unique[unique.length - 1] !== values[i]) {
				unique.push(values[i]);
			}
		}
		return unique;
	}

	function formatNumber(value) {
		if (value === null || value === undefined || value === "") {
			return "";
		}
		var number = toNumber(value);
		if (number === null) {
			return String(value);
		}
		if (number === 0) {
			return "0";
		}
		var absolute = Math.abs(number);
		if (absolute >= 1000000 || absolute < 0.0001) {
			return Number(number.toPrecision(6)).toExponential(4).replace(/\.?0+e/, "e");
		}
		return Number(number.toPrecision(8)).toString();
	}

	function formatValue(value) {
		if (typeof value === "number") {
			return formatNumber(value);
		}
		return String(value === null || value === undefined ? "" : value);
	}

	function quoteValue(value) {
		if (typeof value === "number") {
			return String(Number(value.toPrecision(12)));
		}
		return "\"" + String(value === null || value === undefined ? "" : value)
			.replace(/\\/g, "\\\\")
			.replace(/"/g, "\\\"") + "\"";
	}

	function formatCondition(condition) {
		var operator = condition.operator === "==" ? "=" : condition.operator;
		return condition.column + " " + operator + " " + quoteValue(condition.value);
	}

	function conditionsToQuery(conditions) {
		return conditions.map(formatCondition).join(" AND ");
	}

	function conditionsKey(conditions) {
		return conditions.map(function (condition) {
			return condition.column + "\u0000" + condition.operator + "\u0000" + String(condition.value);
		}).sort().join("\u0001");
	}

	function headerIndexMap(headers) {
		var map = {};
		for (var i = 0; i < headers.length; i++) {
			map[headers[i]] = i;
		}
		return map;
	}

	function goalForResult(resultName, resultNames, resultMinMax) {
		var value = null;

		if (Array.isArray(resultMinMax)) {
			var index = resultNames.indexOf(resultName);
			if (index >= 0) {
				value = resultMinMax[index];
			}
		} else if (resultMinMax && Object.prototype.hasOwnProperty.call(resultMinMax, resultName)) {
			value = resultMinMax[resultName];
		}

		value = String(value === null || value === undefined ? "min" : value).toLowerCase();
		return value.indexOf("max") !== -1 ? "max" : "min";
	}

	function tokenizeQuery(input) {
		var tokens = [];
		var i = 0;

		function push(type, value, position) {
			tokens.push({ type: type, value: value, position: position });
		}

		while (i < input.length) {
			var character = input[i];

			if (/\s/.test(character)) {
				i++;
				continue;
			}

			if (character === "(" || character === ")") {
				push(character, character, i);
				i++;
				continue;
			}

			if (character === "\"" || character === "'") {
				var quote = character;
				var start = i;
				i++;
				var text = "";
				var closed = false;
				while (i < input.length) {
					if (input[i] === "\\" && i + 1 < input.length) {
						text += input[i + 1];
						i += 2;
						continue;
					}
					if (input[i] === quote) {
						closed = true;
						i++;
						break;
					}
					text += input[i];
					i++;
				}
				if (!closed) {
					throw new Error("Unterminated string at position " + start);
				}
				push("string", text, start);
				continue;
			}

			var operator = null;
			var twoCharacters = input.substr(i, 2);
			if (["<=", ">=", "!=", "=="].indexOf(twoCharacters) !== -1) {
				operator = twoCharacters;
			} else if (["<", ">", "="].indexOf(character) !== -1) {
				operator = character;
			}
			if (operator !== null) {
				push("operator", operator, i);
				i += operator.length;
				continue;
			}

			var wordStart = i;
			while (i < input.length && !/[\s()<>=!]/.test(input[i])) {
				i++;
			}
			var word = input.slice(wordStart, i);
			if (word === "") {
				throw new Error("Unexpected character '" + input[wordStart] + "' at position " + wordStart);
			}
			var lower = word.toLowerCase();
			if (lower === "and" || lower === "or") {
				push("keyword", lower, wordStart);
			} else if (lower === "contains") {
				push("operator", "contains", wordStart);
			} else if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(word)) {
				push("number", Number(word), wordStart);
			} else {
				push("word", word, wordStart);
			}
		}

		return tokens;
	}

	function compileResultQuery(input, headers, options) {
		options = options || {};
		var aliases = options.aliases || { status: "trial_status" };
		var headerMap = headerIndexMap(headers);
		var lowerHeaderMap = {};

		for (var i = 0; i < headers.length; i++) {
			var lower = headers[i].toLowerCase();
			if (!Object.prototype.hasOwnProperty.call(lowerHeaderMap, lower)) {
				lowerHeaderMap[lower] = headers[i];
			}
		}

		function resolveColumn(name) {
			var alias = aliases[name] || aliases[String(name).toLowerCase()];
			if (alias) {
				name = alias;
			}
			if (Object.prototype.hasOwnProperty.call(headerMap, name)) {
				return { name: name, index: headerMap[name] };
			}
			var canonical = lowerHeaderMap[String(name).toLowerCase()];
			if (canonical) {
				return { name: canonical, index: headerMap[canonical] };
			}
			return null;
		}

		var tokens;
		try {
			tokens = tokenizeQuery(String(input));
		} catch (error) {
			return { ok: false, error: error.message };
		}

		if (!tokens.length) {
			return { ok: false, error: "Empty query" };
		}

		var position = 0;

		function peek() {
			return tokens[position] || null;
		}

		function consume(type, value) {
			var token = peek();
			if (!token || token.type !== type || (value !== undefined && token.value !== value)) {
				return null;
			}
			position++;
			return token;
		}

		function parseOr() {
			var node = parseAnd();
			while (true) {
				var token = peek();
				if (token && token.type === "keyword" && token.value === "or") {
					position++;
					node = { type: "or", left: node, right: parseAnd() };
				} else {
					return node;
				}
			}
		}

		function parseAnd() {
			var node = parsePrimary();
			while (true) {
				var token = peek();
				if (token && token.type === "keyword" && token.value === "and") {
					position++;
					node = { type: "and", left: node, right: parsePrimary() };
				} else {
					return node;
				}
			}
		}

		function parsePrimary() {
			if (consume("(")) {
				var node = parseOr();
				if (!consume(")")) {
					throw new Error("Expected ')'");
				}
				return node;
			}
			return parseCondition();
		}

		function parseCondition() {
			var columnToken = peek();
			if (!columnToken || (columnToken.type !== "word" && columnToken.type !== "string")) {
				throw new Error("Expected a column name");
			}
			position++;

			var operatorToken = peek();
			if (!operatorToken || operatorToken.type !== "operator") {
				throw new Error("Expected an operator after '" + columnToken.value + "'");
			}
			position++;

			var valueToken = peek();
			if (!valueToken || ["string", "number", "word"].indexOf(valueToken.type) === -1) {
				throw new Error("Expected a value after '" + operatorToken.value + "'");
			}
			position++;

			var resolved = resolveColumn(columnToken.value);
			if (!resolved) {
				throw new Error("Unknown column '" + columnToken.value + "'");
			}

			return {
				type: "condition",
				column: resolved.name,
				index: resolved.index,
				operator: operatorToken.value,
				value: valueToken.value
			};
		}

		var ast;
		try {
			ast = parseOr();
			if (position !== tokens.length) {
				throw new Error("Unexpected token '" + tokens[position].value + "'");
			}
		} catch (error) {
			return { ok: false, error: error.message };
		}

		function astToString(node, parentPrecedence) {
			if (node.type === "condition") {
				return formatCondition(node);
			}
			var precedence = node.type === "or" ? 1 : 2;
			var text = astToString(node.left, precedence) + " " + node.type.toUpperCase() + " " +
				astToString(node.right, precedence);
			if (precedence < parentPrecedence) {
				return "(" + text + ")";
			}
			return text;
		}

		function astMatches(node, row) {
			if (node.type === "condition") {
				return conditionMatchesValue(node, row[node.index]);
			}
			if (node.type === "and") {
				return astMatches(node.left, row) && astMatches(node.right, row);
			}
			return astMatches(node.left, row) || astMatches(node.right, row);
		}

		return {
			ok: true,
			ast: ast,
			normalized: astToString(ast, 0),
			predicate: function (row) {
				return astMatches(ast, row);
			}
		};
	}

	function computeAggregate(data, headers, resultNames, predicate) {
		var headerMap = headerIndexMap(headers);
		var names = (resultNames || []).filter(function (name) {
			return Object.prototype.hasOwnProperty.call(headerMap, name);
		});
		var values = {};
		for (var n = 0; n < names.length; n++) {
			values[names[n]] = [];
		}

		var count = 0;
		for (var r = 0; r < data.length; r++) {
			var row = data[r];
			if (predicate && !predicate(row)) {
				continue;
			}
			count++;
			for (var k = 0; k < names.length; k++) {
				var number = toNumber(row[headerMap[names[k]]]);
				if (number !== null) {
					values[names[k]].push(number);
				}
			}
		}

		return {
			count: count,
			total: data.length,
			results: names.map(function (name) {
				var resultValues = values[name];
				var average = mean(resultValues);
				return {
					name: name,
					count: resultValues.length,
					min: resultValues.length ? Math.min.apply(null, resultValues) : null,
					max: resultValues.length ? Math.max.apply(null, resultValues) : null,
					mean: average,
					median: median(resultValues)
				};
			})
		};
	}

	function selectFeatureColumns(headers, options) {
		options = options || {};
		var resultNames = options.resultNames || [];
		var specialColumns = options.specialColumns || DEFAULT_SPECIAL_COLUMNS;
		var extraColumns = options.extraColumns || [];
		var headerSet = {};
		var excluded = {};

		for (var h = 0; h < headers.length; h++) {
			headerSet[headers[h]] = true;
		}

		resultNames.concat(specialColumns).forEach(function (name) {
			excluded[name] = true;
		});

		var columns = [];
		for (var i = 0; i < headers.length; i++) {
			var name = headers[i];
			if (!name) {
				continue;
			}
			if (name.indexOf("OO_Info_") === 0 && headerSet[name.slice("OO_Info_".length)]) {
				continue;
			}
			if (excluded[name] && extraColumns.indexOf(name) === -1) {
				continue;
			}
			columns.push({ name: name, index: i });
		}
		return columns;
	}

	function prepareFeatures(data, headers, featureColumns) {
		return featureColumns.map(function (column) {
			var rawValues = new Array(data.length);
			var nonEmpty = 0;
			var numericCount = 0;

			for (var i = 0; i < data.length; i++) {
				var raw = data[i][column.index];
				rawValues[i] = raw;
				if (isBlank(raw)) {
					continue;
				}
				nonEmpty++;
				if (toNumber(raw) !== null) {
					numericCount++;
				}
			}

			var numeric = nonEmpty > 0 && numericCount / nonEmpty >= 0.8;
			var values;

			if (numeric) {
				values = rawValues.map(function (raw) {
					var number = toNumber(raw);
					return number === null ? NaN : number;
				});
			} else {
				values = rawValues.map(function (raw) {
					return raw === null || raw === undefined ? "" : String(raw);
				});
			}

			return {
				name: column.name,
				index: column.index,
				numeric: numeric,
				rawValues: rawValues,
				values: values
			};
		});
	}

	function numericThresholds(feature, maxThresholds) {
		var values = [];
		for (var i = 0; i < feature.values.length; i++) {
			if (isFinite(feature.values[i])) {
				values.push(feature.values[i]);
			}
		}
		if (!values.length) {
			return [];
		}
		values.sort(function (a, b) { return a - b; });

		var thresholds = [];
		var count = Math.max(1, Math.min(maxThresholds, values.length - 1));
		for (var t = 1; t <= count; t++) {
			thresholds.push(quantileSorted(values, t / (count + 1)));
		}
		return uniqueNumbers(thresholds);
	}

	function categoricalCandidates(feature, minLeaf, maxValues) {
		var counts = {};
		for (var i = 0; i < feature.values.length; i++) {
			var value = feature.values[i];
			if (value === "") {
				continue;
			}
			counts[value] = (counts[value] || 0) + 1;
		}

		return Object.keys(counts)
			.filter(function (value) { return counts[value] >= minLeaf; })
			.sort(function (a, b) {
				return counts[b] - counts[a] || a.localeCompare(b);
			})
			.slice(0, maxValues);
	}

	function impurity(positives, negatives, positiveWeight, negativeWeight) {
		var positive = positives * positiveWeight;
		var negative = negatives * negativeWeight;
		var total = positive + negative;
		if (total === 0) {
			return 0;
		}
		var p = positive / total;
		var n = negative / total;
		return 1 - p * p - n * n;
	}

	function buildDecisionTree(data, headers, options) {
		options = options || {};
		var labelFn = options.labelFn;
		var statusIndex = headers.indexOf("trial_status");

		if (!labelFn) {
			if (statusIndex === -1) {
				return { empty: true, reason: "Column 'trial_status' is missing" };
			}
			labelFn = function (row) {
				return String(row[statusIndex]).toUpperCase() === "FAILED";
			};
		}

		var featureColumns = options.featureColumns || selectFeatureColumns(headers, options);
		var features = prepareFeatures(data, headers, featureColumns);
		var maxDepth = options.maxDepth === undefined ? 4 : options.maxDepth;
		var minLeaf = options.minLeaf === undefined ? 5 : options.minLeaf;
		var minGain = options.minGain === undefined ? 1e-6 : options.minGain;
		var maxThresholds = options.maxThresholds === undefined ? 8 : options.maxThresholds;
		var maxCategoricalValues = options.maxCategoricalValues === undefined ? 20 : options.maxCategoricalValues;
		var balanceClasses = options.balanceClasses !== false;

		var indices = [];
		var labels = new Uint8Array(data.length);
		var positives = 0;
		var negatives = 0;

		for (var i = 0; i < data.length; i++) {
			var label = labelFn(data[i]);
			if (label === null || label === undefined) {
				continue;
			}
			indices.push(i);
			labels[i] = label ? 1 : 0;
			if (label) {
				positives++;
			} else {
				negatives++;
			}
		}

		if (!indices.length) {
			return { empty: true, reason: "No rows with a usable status" };
		}
		if (positives === 0 || negatives === 0) {
			return {
				empty: true,
				reason: "The data contains only one outcome class (" + positives + " positive, " + negatives + " negative)"
			};
		}

		var positiveWeight = balanceClasses ? indices.length / (2 * positives) : 1;
		var negativeWeight = balanceClasses ? indices.length / (2 * negatives) : 1;
		var importance = {};

		for (var f = 0; f < features.length; f++) {
			var feature = features[f];
			if (feature.numeric) {
				feature.thresholds = numericThresholds(feature, maxThresholds);
			} else {
				feature.candidates = categoricalCandidates(feature, minLeaf, maxCategoricalValues);
			}
		}

		function splitCandidates(feature) {
			if (feature.numeric) {
				return feature.thresholds.map(function (threshold) {
					return { operator: "<=", value: threshold };
				});
			}
			return feature.candidates.map(function (value) {
				return { operator: "==", value: value };
			});
		}

		function valueMatchesSplit(feature, value, candidate) {
			if (candidate.operator === "<=") {
				return value <= candidate.value;
			}
			return value === candidate.value;
		}

		function buildNode(subset, depth) {
			var nodePositives = 0;
			for (var s = 0; s < subset.length; s++) {
				nodePositives += labels[subset[s]];
			}
			var nodeNegatives = subset.length - nodePositives;

			var node = {
				leaf: true,
				samples: subset.length,
				positives: nodePositives,
				negatives: nodeNegatives,
				rate: subset.length ? nodePositives / subset.length : 0
			};

			if (depth >= maxDepth || subset.length < 2 * minLeaf || nodePositives === 0 || nodeNegatives === 0) {
				return node;
			}

			var parentImpurity = impurity(nodePositives, nodeNegatives, positiveWeight, negativeWeight);
			var best = null;

			for (var featureIndex = 0; featureIndex < features.length; featureIndex++) {
				var feature = features[featureIndex];
				var candidates = splitCandidates(feature);

				for (var candidateIndex = 0; candidateIndex < candidates.length; candidateIndex++) {
					var candidate = candidates[candidateIndex];
					var left = [];
					var right = [];
					var leftPositives = 0;
					var rightPositives = 0;

					for (var rowIndex = 0; rowIndex < subset.length; rowIndex++) {
						var sourceIndex = subset[rowIndex];
						var value = feature.values[sourceIndex];
						if (valueMatchesSplit(feature, value, candidate)) {
							left.push(sourceIndex);
							leftPositives += labels[sourceIndex];
						} else {
							right.push(sourceIndex);
							rightPositives += labels[sourceIndex];
						}
					}

					if (left.length < minLeaf || right.length < minLeaf) {
						continue;
					}

					var leftNegatives = left.length - leftPositives;
					var rightNegatives = right.length - rightPositives;
					var leftImpurity = impurity(leftPositives, leftNegatives, positiveWeight, negativeWeight);
					var rightImpurity = impurity(rightPositives, rightNegatives, positiveWeight, negativeWeight);
					var leftWeight = leftPositives * positiveWeight + leftNegatives * negativeWeight;
					var rightWeight = rightPositives * positiveWeight + rightNegatives * negativeWeight;
					var totalWeight = leftWeight + rightWeight;
					var gain = parentImpurity -
						(leftWeight / totalWeight) * leftImpurity -
						(rightWeight / totalWeight) * rightImpurity;

					if (gain <= minGain) {
						continue;
					}

					if (best === null || gain > best.gain) {
						best = {
							feature: feature,
							candidate: candidate,
							left: left,
							right: right,
							gain: gain,
							impurity: parentImpurity
						};
					}
				}
			}

			if (best === null) {
				return node;
			}

			var condition = {
				column: best.feature.name,
				index: best.feature.index,
				operator: best.candidate.operator,
				value: best.candidate.value
			};

			importance[best.feature.name] = (importance[best.feature.name] || 0) + best.gain * subset.length;

			node.leaf = false;
			node.condition = condition;
			node.gain = best.gain;
			node.impurity = best.impurity;
			node.left = buildNode(best.left, depth + 1);
			node.right = buildNode(best.right, depth + 1);
			return node;
		}

		var tree = buildNode(indices, 0);
		var importanceTotal = 0;
		var importanceList = Object.keys(importance).map(function (name) {
			importanceTotal += importance[name];
			return { name: name, value: importance[name] };
		});
		if (importanceTotal > 0) {
			importanceList.forEach(function (entry) {
				entry.value /= importanceTotal;
			});
		}
		importanceList.sort(function (a, b) { return b.value - a.value; });

		tree.featureImportance = importanceList;
		tree.samples = indices.length;
		tree.positives = positives;
		tree.negatives = negatives;
		tree.baselineRate = positives / indices.length;
		return tree;
	}

	function treeToRules(tree) {
		var rules = [];

		function walk(node, conditions) {
			if (node.leaf) {
				if (conditions.length) {
					rules.push({
						conditions: conditions,
						samples: node.samples,
						positives: node.positives,
						negatives: node.negatives,
						rate: node.rate
					});
				}
				return;
			}
			walk(node.left, conditions.concat([node.condition]));
			walk(node.right, conditions.concat([inverseCondition(node.condition)]));
		}

		walk(tree, []);
		return rules;
	}

	function evaluateTree(tree, data, headers, labelFn) {
		var statusIndex = headers.indexOf("trial_status");
		if (!labelFn) {
			labelFn = function (row) {
				return String(row[statusIndex]).toUpperCase() === "FAILED";
			};
		}

		var tp = 0;
		var fp = 0;
		var tn = 0;
		var fn = 0;
		var total = 0;
		var positives = 0;

		for (var i = 0; i < data.length; i++) {
			var label = labelFn(data[i]);
			if (label === null || label === undefined) {
				continue;
			}
			total++;
			if (label) {
				positives++;
			}

			var node = tree;
			while (!node.leaf) {
				var condition = node.condition;
				var value = data[i][condition.index];
				node = conditionMatchesValue(condition, value) ? node.left : node.right;
			}
			var predicted = node.rate >= 0.5;
			if (predicted && label) {
				tp++;
			} else if (predicted && !label) {
				fp++;
			} else if (!predicted && label) {
				fn++;
			} else {
				tn++;
			}
		}

		var baseline = Math.max(positives, total - positives) / (total || 1);
		return {
			tp: tp,
			fp: fp,
			tn: tn,
			fn: fn,
			total: total,
			accuracy: total ? (tp + tn) / total : 0,
			baselineAccuracy: total ? baseline : 0
		};
	}

	function sampleData(data, maxRows) {
		if (!maxRows || data.length <= maxRows) {
			return { rows: data, sampled: false, originalSize: data.length };
		}
		var stride = Math.ceil(data.length / maxRows);
		var rows = [];
		for (var i = 0; i < data.length; i += stride) {
			rows.push(data[i]);
		}
		return { rows: rows, sampled: true, originalSize: data.length };
	}

	function analyzeFailures(data, headers, options) {
		options = options || {};
		var sampled = sampleData(data, options.maxRows === undefined ? 50000 : options.maxRows);
		var rows = sampled.rows;
		var statusIndex = headers.indexOf("trial_status");

		if (statusIndex === -1) {
			return { ok: false, reason: "Column 'trial_status' is missing" };
		}

		var positiveStatuses = (options.positiveStatuses || ["FAILED"]).map(function (value) {
			return String(value).toUpperCase();
		});
		var negativeStatuses = (options.negativeStatuses || ["COMPLETED"]).map(function (value) {
			return String(value).toUpperCase();
		});

		var labelFn = function (row) {
			var status = String(row[statusIndex] === null || row[statusIndex] === undefined ? "" : row[statusIndex]).toUpperCase();
			if (positiveStatuses.indexOf(status) !== -1) {
				return true;
			}
			if (negativeStatuses.indexOf(status) !== -1) {
				return false;
			}
			return null;
		};

		var featureColumns = options.featureColumns || selectFeatureColumns(headers, {
			resultNames: options.resultNames || [],
			specialColumns: options.specialColumns || DEFAULT_SPECIAL_COLUMNS,
			extraColumns: options.extraColumns || ["generation_node", "hostname"]
		});

		var tree = buildDecisionTree(rows, headers, {
			labelFn: labelFn,
			featureColumns: featureColumns,
			maxDepth: options.maxDepth === undefined ? 4 : options.maxDepth,
			minLeaf: options.minLeaf === undefined ? 5 : options.minLeaf,
			minGain: options.minGain,
			maxThresholds: options.maxThresholds,
			maxCategoricalValues: options.maxCategoricalValues,
			balanceClasses: options.balanceClasses
		});

		if (tree.empty) {
			return {
				ok: false,
				reason: tree.reason || "Not enough data",
				total: rows.length,
				sampled: sampled.sampled,
				originalSize: sampled.originalSize
			};
		}

		var rules = treeToRules(tree);
		var baseline = tree.baselineRate;
		rules.forEach(function (rule) {
			rule.baselineRate = baseline;
			rule.lift = baseline > 0 ? rule.rate / baseline : 0;
			rule.support = tree.samples > 0 ? rule.samples / tree.samples : 0;
			rule.score = (rule.rate - baseline) * Math.sqrt(rule.samples);
		});

		var topRules = rules
			.filter(function (rule) {
				return rule.positives > 0 && rule.rate > baseline && rule.samples >= (options.minRuleSamples || 1);
			})
			.sort(function (a, b) {
				return b.score - a.score || b.samples - a.samples;
			})
			.slice(0, options.maxRules === undefined ? 8 : options.maxRules);

		return {
			ok: true,
			tree: tree,
			rules: rules,
			topRules: topRules,
			baselineRate: baseline,
			total: tree.samples,
			positives: tree.positives,
			negatives: tree.negatives,
			featureImportance: tree.featureImportance || [],
			confusion: evaluateTree(tree, rows, headers, labelFn),
			sampled: sampled.sampled,
			originalSize: sampled.originalSize
		};
	}

	function buildConditionPool(features, minSupport) {
		var conditions = [];

		features.forEach(function (feature) {
			if (feature.numeric) {
				var values = [];
				for (var i = 0; i < feature.values.length; i++) {
					if (isFinite(feature.values[i])) {
						values.push(feature.values[i]);
					}
				}
				if (values.length < minSupport) {
					return;
				}
				values.sort(function (a, b) { return a - b; });
				var thresholds = uniqueNumbers([0.2, 0.35, 0.5, 0.65, 0.8].map(function (q) {
					return quantileSorted(values, q);
				}));

				thresholds.forEach(function (threshold) {
					conditions.push({
						column: feature.name,
						index: feature.index,
						operator: "<=",
						value: threshold
					});
					conditions.push({
						column: feature.name,
						index: feature.index,
						operator: ">",
						value: threshold
					});
				});
			} else {
				var counts = {};
				for (var j = 0; j < feature.values.length; j++) {
					var value = feature.values[j];
					if (value === "") {
						continue;
					}
					counts[value] = (counts[value] || 0) + 1;
				}
				Object.keys(counts)
					.filter(function (value) { return counts[value] >= minSupport; })
					.sort(function (a, b) { return counts[b] - counts[a] || a.localeCompare(b); })
					.slice(0, 10)
					.forEach(function (value) {
						conditions.push({
							column: feature.name,
							index: feature.index,
							operator: "==",
							value: value
						});
					});
			}
		});

		return conditions;
	}

	function filterIndices(indices, condition, featureByIndex) {
		var feature = featureByIndex[condition.index];
		if (!feature) {
			return [];
		}
		var matched = [];
		for (var i = 0; i < indices.length; i++) {
			if (conditionMatchesValue(condition, feature.values[indices[i]])) {
				matched.push(indices[i]);
			}
		}
		return matched;
	}

	function subgroupStats(matched, values, validIndices, goal, totals) {
		var matchedCount = matched.length;
		if (matchedCount < 2 || matchedCount >= validIndices.length) {
			return null;
		}

		var insideSum = 0;
		var insideSquares = 0;
		for (var i = 0; i < matchedCount; i++) {
			var value = values[matched[i]];
			insideSum += value;
			insideSquares += value * value;
		}

		var outsideCount = validIndices.length - matchedCount;
		var outsideSum = totals.sum - insideSum;
		var outsideSquares = totals.squares - insideSquares;
		var insideMean = insideSum / matchedCount;
		var outsideMean = outsideSum / outsideCount;
		var delta = insideMean - outsideMean;
		var deltaPct = outsideMean === 0 ? delta : delta / Math.abs(outsideMean);
		var insideVariance = Math.max(0, insideSquares / matchedCount - insideMean * insideMean);
		var outsideVariance = Math.max(0, outsideSquares / outsideCount - outsideMean * outsideMean);
		var pooled = Math.sqrt((insideVariance + outsideVariance) / 2);
		var effect = pooled === 0 ? 0 : delta / pooled;
		var score = goal === "max" ? deltaPct : -deltaPct;

		return {
			n: matchedCount,
			mean: insideMean,
			median: null,
			complementMean: outsideMean,
			complementMedian: null,
			deltaPct: deltaPct,
			effect: effect,
			score: score,
			support: matchedCount / validIndices.length
		};
	}

	function dedupeCandidates(candidates) {
		var byKey = {};
		candidates.forEach(function (candidate) {
			var key = conditionsKey(candidate.conditions);
			if (!byKey[key] || Math.abs(candidate.score) > Math.abs(byKey[key].score)) {
				byKey[key] = candidate;
			}
		});
		return Object.keys(byKey).map(function (key) { return byKey[key]; });
	}

	function attachMedians(candidate, values, validIndices, featureByIndex) {
		var matched = validIndices;
		for (var i = 0; i < candidate.conditions.length; i++) {
			matched = filterIndices(matched, candidate.conditions[i], featureByIndex);
			if (!matched.length) {
				return;
			}
		}
		var matchedSet = {};
		for (var j = 0; j < matched.length; j++) {
			matchedSet[matched[j]] = true;
		}
		var inside = [];
		var outside = [];
		for (var k = 0; k < validIndices.length; k++) {
			var sourceIndex = validIndices[k];
			if (matchedSet[sourceIndex]) {
				inside.push(values[sourceIndex]);
			} else {
				outside.push(values[sourceIndex]);
			}
		}
		candidate.stats.median = median(inside);
		candidate.stats.complementMedian = median(outside);
	}

	function findResultPatterns(data, headers, resultNames, resultMinMax, options) {
		options = options || {};
		var sampled = sampleData(data, options.maxRows === undefined ? 50000 : options.maxRows);
		var rows = sampled.rows;
		var names = (resultNames || []).filter(function (name) {
			return headers.indexOf(name) !== -1;
		});
		var goalNames = (resultNames || []).slice();

		if (!names.length) {
			return { ok: false, reason: "No result columns were found", results: [] };
		}

		var features = prepareFeatures(rows, headers, selectFeatureColumns(headers, {
			resultNames: names,
			specialColumns: options.specialColumns || DEFAULT_SPECIAL_COLUMNS,
			extraColumns: options.extraColumns || []
		})).slice(0, options.maxFeatures === undefined ? 14 : options.maxFeatures);

		var featureByIndex = {};
		features.forEach(function (feature) {
			featureByIndex[feature.index] = feature;
		});

		var minSupport = options.minSupport === undefined
			? Math.max(5, Math.ceil(rows.length * 0.02))
			: options.minSupport;
		var minEffect = options.minEffect === undefined ? 0.05 : options.minEffect;
		var maxRules = options.maxRules === undefined ? 5 : options.maxRules;
		var pool = buildConditionPool(features, minSupport);
		var results = [];
		var statusIndex = headers.indexOf("trial_status");
		var includeStatuses = options.includeStatuses;
		if (includeStatuses === undefined && statusIndex !== -1) {
			includeStatuses = ["COMPLETED"];
		}
		if (includeStatuses) {
			includeStatuses = includeStatuses.map(function (status) {
				return String(status).toUpperCase();
			});
		}

		names.forEach(function (resultName) {
			var resultIndex = headers.indexOf(resultName);
			var values = new Array(rows.length);
			var validIndices = [];

			for (var i = 0; i < rows.length; i++) {
				if (includeStatuses && statusIndex !== -1) {
					var status = String(rows[i][statusIndex] === null || rows[i][statusIndex] === undefined ? "" : rows[i][statusIndex]).toUpperCase();
					if (includeStatuses.indexOf(status) === -1) {
						continue;
					}
				}
				var number = toNumber(rows[i][resultIndex]);
				values[i] = number === null ? NaN : number;
				if (number !== null) {
					validIndices.push(i);
				}
			}

			if (validIndices.length < minSupport * 2) {
				return;
			}

			var totalSum = 0;
			var totalSquares = 0;
			for (var v = 0; v < validIndices.length; v++) {
				var totalValue = values[validIndices[v]];
				totalSum += totalValue;
				totalSquares += totalValue * totalValue;
			}
			var totals = { sum: totalSum, squares: totalSquares };

			var goal = goalForResult(resultName, goalNames.length ? goalNames : names, resultMinMax);
			var singles = [];

			for (var p = 0; p < pool.length; p++) {
				var matched = filterIndices(validIndices, pool[p], featureByIndex);
				if (matched.length < minSupport) {
					continue;
				}
				var stats = subgroupStats(matched, values, validIndices, goal, totals);
				if (stats && Math.abs(stats.deltaPct) >= minEffect) {
					singles.push({
						conditions: [pool[p]],
						matchIndices: matched,
						stats: stats
					});
				}
			}

			var beam = singles
				.slice()
				.sort(function (a, b) { return Math.abs(b.stats.score) - Math.abs(a.stats.score); })
				.slice(0, options.beamWidth === undefined ? 30 : options.beamWidth);
			var candidates = singles.map(function (single) {
				return {
					conditions: single.conditions,
					stats: single.stats
				};
			});

			beam.forEach(function (base) {
				var baseColumnIndexes = {};
				base.conditions.forEach(function (condition) {
					baseColumnIndexes[condition.index] = true;
				});

				pool.forEach(function (condition) {
					if (baseColumnIndexes[condition.index]) {
						return;
					}
					var matched = filterIndices(base.matchIndices, condition, featureByIndex);
					if (matched.length < minSupport) {
						return;
					}
					var stats = subgroupStats(matched, values, validIndices, goal, totals);
					if (!stats || Math.abs(stats.deltaPct) < minEffect) {
						return;
					}
					candidates.push({
						conditions: base.conditions.concat([condition]),
						stats: stats
					});
				});
			});

			var deduped = dedupeCandidates(candidates);
			var good = deduped
				.filter(function (candidate) { return candidate.stats.score > 0; })
				.sort(function (a, b) {
					return b.stats.score * Math.sqrt(b.stats.n) - a.stats.score * Math.sqrt(a.stats.n);
				})
				.slice(0, maxRules);
			var bad = deduped
				.filter(function (candidate) { return candidate.stats.score < 0; })
				.sort(function (a, b) {
					return a.stats.score * Math.sqrt(a.stats.n) - b.stats.score * Math.sqrt(b.stats.n);
				})
				.slice(0, maxRules);

			var singleStatusCondition = null;
			if (includeStatuses && includeStatuses.length === 1 && statusIndex !== -1) {
				singleStatusCondition = {
					column: "trial_status",
					index: statusIndex,
					operator: "==",
					value: includeStatuses[0]
				};
			}

			good.concat(bad).forEach(function (candidate) {
				attachMedians(candidate, values, validIndices, featureByIndex);
				if (singleStatusCondition) {
					candidate.conditions = candidate.conditions.concat([singleStatusCondition]);
				}
			});

			results.push({
				result: resultName,
				goal: goal,
				considered: validIndices.length,
				good: good,
				bad: bad
			});
		});

		return {
			ok: results.length > 0,
			results: results,
			reason: results.length ? null : "Not enough valid result values",
			minSupport: minSupport,
			total: rows.length,
			sampled: sampled.sampled,
			originalSize: sampled.originalSize
		};
	}

	function createElement(tagName, className, text) {
		var element = document.createElement(tagName);
		if (className) {
			element.className = className;
		}
		if (text !== undefined && text !== null) {
			element.textContent = text;
		}
		return element;
	}

	function formatPercent(value) {
		return (value * 100).toFixed(value >= 0.1 ? 0 : 1) + "%";
	}

	function formatSignedPercent(value) {
		var sign = value > 0 ? "+" : "";
		return sign + (value * 100).toFixed(1) + "%";
	}

	function renderRuleCard(rule, kind, onApply) {
		var card = createElement("div", "analysis-rule analysis-rule-" + kind);
		var query = conditionsToQuery(rule.conditions);
		var title = createElement("div", "analysis-rule-query", query);
		card.appendChild(title);

		var meta = createElement("div", "analysis-rule-meta");
		if (kind === "failure") {
			meta.textContent =
				"n=" + rule.samples +
				" · failed " + formatPercent(rule.rate) +
				" · baseline " + formatPercent(rule.baselineRate) +
				" · lift " + formatNumber(rule.lift) + "×";
		} else {
			meta.textContent =
				"n=" + rule.stats.n +
				" · mean " + formatNumber(rule.stats.mean) +
				" vs " + formatNumber(rule.stats.complementMean) +
				" (" + formatSignedPercent(rule.stats.deltaPct) + ")" +
				" · median " + formatNumber(rule.stats.median) +
				" vs " + formatNumber(rule.stats.complementMedian);
		}
		card.appendChild(meta);

		var applyButton = createElement("button", "analysis-apply", "Apply as filter");
		applyButton.type = "button";
		applyButton.addEventListener("click", function () {
			onApply(query);
		});
		card.appendChild(applyButton);

		return card;
	}

	function renderTree(node) {
		var wrapper = createElement("div", "analysis-tree-node");
		if (node.leaf) {
			wrapper.textContent = "→ " + formatPercent(node.rate) + " failed (n=" + node.samples + ")";
			wrapper.classList.add("analysis-tree-leaf");
			return wrapper;
		}
		var split = createElement("div", "analysis-tree-split", "if " + formatCondition(node.condition));
		wrapper.appendChild(split);
		var children = createElement("div", "analysis-tree-children");
		children.appendChild(renderTree(node.left));
		children.appendChild(renderTree(node.right));
		wrapper.appendChild(children);
		return wrapper;
	}

	function initializeResultsAnalysis(containerId, data, headers, controller, options) {
		if (typeof document === "undefined") {
			return null;
		}

		var container = document.getElementById(containerId);
		if (!container || container.dataset.analysisInitialized === "true") {
			return null;
		}
		container.dataset.analysisInitialized = "true";

		options = options || {};
		var resultNames = options.resultNames || root.result_names || [];
		var resultMinMax = options.resultMinMax || root.result_min_max || [];
		var specialColumns = options.specialColumns || root.special_col_names || DEFAULT_SPECIAL_COLUMNS;

		var panel = createElement("section", "results-analysis");
		var title = createElement("h3", "results-analysis-title", "Find results and patterns");
		panel.appendChild(title);

		var queryRow = createElement("div", "results-analysis-query");
		var queryInput = createElement("input", "results-analysis-input");
		queryInput.type = "search";
		queryInput.placeholder = "e.g. RESULT1 < 1.5 AND int_param > 5";
		queryInput.setAttribute("aria-label", "Result filter query");
		queryRow.appendChild(queryInput);

		var applyButton = createElement("button", "results-button", "Filter");
		applyButton.type = "button";
		queryRow.appendChild(applyButton);

		var clearButton = createElement("button", "results-button", "Clear");
		clearButton.type = "button";
		queryRow.appendChild(clearButton);

		var linkButton = createElement("button", "results-button", "Copy link");
		linkButton.type = "button";
		queryRow.appendChild(linkButton);
		panel.appendChild(queryRow);

		var queryError = createElement("div", "results-analysis-error");
		panel.appendChild(queryError);

		var aggregate = createElement("div", "results-analysis-aggregate");
		panel.appendChild(aggregate);

		var builder = createElement("details", "results-analysis-builder");
		var builderSummary = createElement("summary", null, "Build a filter");
		builder.appendChild(builderSummary);
		var builderRow = createElement("div", "results-analysis-builder-row");

		var columnSelect = createElement("select", "results-analysis-select");
		headers.forEach(function (header) {
			var option = createElement("option", null, header);
			option.value = header;
			columnSelect.appendChild(option);
		});
		builderRow.appendChild(columnSelect);

		var operatorSelect = createElement("select", "results-analysis-select");
		["=", "!=", "<", "<=", ">", ">=", "contains"].forEach(function (operator) {
			var option = createElement("option", null, operator);
			option.value = operator;
			operatorSelect.appendChild(option);
		});
		builderRow.appendChild(operatorSelect);

		var valueInput = createElement("input", "results-analysis-value");
		valueInput.type = "text";
		valueInput.placeholder = "value";
		builderRow.appendChild(valueInput);

		var addButton = createElement("button", "results-button", "Add");
		addButton.type = "button";
		builderRow.appendChild(addButton);
		builder.appendChild(builderRow);
		panel.appendChild(builder);

		var actions = createElement("div", "results-analysis-actions");
		var failureButton = createElement("button", "results-button", "Why do trials fail?");
		failureButton.type = "button";
		var patternButton = createElement("button", "results-button", "Find patterns");
		patternButton.type = "button";
		actions.appendChild(failureButton);
		actions.appendChild(patternButton);
		panel.appendChild(actions);

		var output = createElement("div", "results-analysis-output");
		panel.appendChild(output);

		container.parentNode.insertBefore(panel, container);

		var statusIndex = headers.indexOf("trial_status");
		if (statusIndex === -1) {
			failureButton.disabled = true;
			failureButton.title = "No trial_status column";
		}
		if (!resultNames.length) {
			patternButton.disabled = true;
			patternButton.title = "No result columns";
		}

		function setOutputMessage(text) {
			output.innerHTML = "";
			output.appendChild(createElement("p", "results-analysis-message", text));
		}

		function updateUrl(query) {
			if (!root.history || !root.URLSearchParams || !root.location) {
				return;
			}
			try {
				var url = new URL(root.location.href);
				if (query) {
					url.searchParams.set("result_query", query);
				} else {
					url.searchParams.delete("result_query");
				}
				url.hash = "tab_results";
				root.history.replaceState(null, "", url.toString());
			} catch (error) {
				console.error("Could not update URL", error);
			}
		}

		function applyQuery() {
			var query = queryInput.value.trim();
			queryError.textContent = "";

			if (!query) {
				controller.clearExternalFilter();
				aggregate.textContent = "Showing all " + data.length + " trials.";
				updateUrl("");
				return true;
			}

			var compiled = compileResultQuery(query, headers, { aliases: { status: "trial_status" } });
			if (!compiled.ok) {
				queryError.textContent = compiled.error;
				controller.clearExternalFilter();
				aggregate.textContent = "";
				return false;
			}

			controller.setExternalFilter(compiled.predicate);
			var stats = computeAggregate(data, headers, resultNames, compiled.predicate);
			var parts = [stats.count + " of " + stats.total + " trials"];

			stats.results.forEach(function (result) {
				if (!result.count) {
					parts.push(result.name + ": no numeric values");
					return;
				}
				parts.push(
					result.name +
					": min " + formatNumber(result.min) +
					", median " + formatNumber(result.median) +
					", max " + formatNumber(result.max)
				);
			});

			aggregate.textContent = parts.join(" · ");
			updateUrl(compiled.normalized);
			return true;
		}

		function applyRuleQuery(query) {
			queryInput.value = query;
			applyQuery();
			output.innerHTML = "";
			output.appendChild(createElement("p", "results-analysis-message", "Filter applied. The results table below now shows only the matching trials."));
		}

		var queryTimer = null;
		queryInput.addEventListener("input", function () {
			if (queryTimer) {
				clearTimeout(queryTimer);
			}
			queryTimer = setTimeout(function () {
				queryTimer = null;
				applyQuery();
			}, 180);
		});
		queryInput.addEventListener("keydown", function (event) {
			if (event.key === "Enter") {
				event.preventDefault();
				applyQuery();
			}
		});
		applyButton.addEventListener("click", applyQuery);
		clearButton.addEventListener("click", function () {
			queryInput.value = "";
			applyQuery();
		});

		function buildQueryValue() {
			var value = valueInput.value;
			var operator = operatorSelect.value;
			if (operator === "contains") {
				return quoteValue(value);
			}
			if (toNumber(value) !== null) {
				return String(toNumber(value));
			}
			return quoteValue(value);
		}

		addButton.addEventListener("click", function () {
			var condition = columnSelect.value + " " + operatorSelect.value + " " + buildQueryValue();
			var current = queryInput.value.trim();
			queryInput.value = current ? current + " AND " + condition : condition;
			valueInput.value = "";
			applyQuery();
		});

		linkButton.addEventListener("click", function () {
			var query = queryInput.value.trim();
			var url = new URL(root.location.href);
			if (query) {
				url.searchParams.set("result_query", compileResultQuery(query, headers).ok
					? compileResultQuery(query, headers).normalized
					: query);
			} else {
				url.searchParams.delete("result_query");
			}
			url.hash = "tab_results";
			var text = url.toString();
			if (typeof root.copy_to_clipboard === "function") {
				root.copy_to_clipboard(text);
			} else if (root.navigator && root.navigator.clipboard) {
				root.navigator.clipboard.writeText(text);
			}
			queryError.textContent = "Link copied.";
		});

		function runAnalysis(kind) {
			failureButton.disabled = true;
			patternButton.disabled = true;
			setOutputMessage("Analyzing...");

			setTimeout(function () {
				try {
					if (kind === "failure") {
						var failureResult = analyzeFailures(data, headers, {
							resultNames: resultNames,
							specialColumns: specialColumns,
							extraColumns: ["generation_node", "hostname"]
						});
						renderFailures(failureResult);
					} else {
						var patternResult = findResultPatterns(data, headers, resultNames, resultMinMax, {
							specialColumns: specialColumns
						});
						renderPatterns(patternResult);
					}
				} catch (error) {
					console.error(error);
					setOutputMessage("Analysis failed: " + error.message);
				} finally {
					if (statusIndex !== -1) {
						failureButton.disabled = false;
					}
					if (resultNames.length) {
						patternButton.disabled = false;
					}
				}
			}, 20);
		}

		function renderFailures(result) {
			output.innerHTML = "";

			if (!result.ok) {
				setOutputMessage(result.reason);
				return;
			}

			var heading = createElement("h4", "results-analysis-heading", "Failure patterns");
			output.appendChild(heading);

			var summary = createElement(
				"p",
				"results-analysis-summary",
				"Analyzed " + result.total + " trials: " +
				result.positives + " failed, " + result.negatives + " completed. Baseline failure rate: " +
				formatPercent(result.baselineRate) + "."
			);
			output.appendChild(summary);

			if (result.sampled) {
				output.appendChild(createElement("p", "results-analysis-warning",
					"Large run: analyzed a deterministic sample of " + result.total + " of " + result.originalSize + " trials."));
			}

			if (result.featureImportance.length) {
				var importanceTitle = createElement("h5", "results-analysis-subheading", "Most important signals");
				output.appendChild(importanceTitle);
				var importanceList = createElement("div", "analysis-importance-list");
				result.featureImportance.slice(0, 6).forEach(function (entry) {
					var row = createElement("div", "analysis-importance-row");
					var label = createElement("span", "analysis-importance-label", entry.name);
					var barWrapper = createElement("span", "analysis-importance-bar-wrapper");
					var bar = createElement("span", "analysis-importance-bar");
					bar.style.width = Math.max(2, entry.value * 100) + "%";
					barWrapper.appendChild(bar);
					var value = createElement("span", "analysis-importance-value", formatPercent(entry.value));
					row.appendChild(label);
					row.appendChild(barWrapper);
					row.appendChild(value);
					importanceList.appendChild(row);
				});
				output.appendChild(importanceList);
			}

			if (!result.topRules.length) {
				output.appendChild(createElement("p", "results-analysis-message",
					"No failure pattern with enough support was found. Try more trials or lower the minimum leaf size."));
			} else {
				var rulesTitle = createElement("h5", "results-analysis-subheading", "Rules with the strongest failure signal");
				output.appendChild(rulesTitle);
				result.topRules.forEach(function (rule) {
					output.appendChild(renderRuleCard(rule, "failure", applyRuleQuery));
				});
			}

			var treeDetails = createElement("details", "analysis-tree-details");
			treeDetails.appendChild(createElement("summary", null, "Show decision tree"));
			treeDetails.appendChild(renderTree(result.tree));
			treeDetails.appendChild(createElement(
				"p",
				"results-analysis-note",
				"Training accuracy: " + formatPercent(result.confusion.accuracy) +
				" (majority baseline: " + formatPercent(result.confusion.baselineAccuracy) + "). " +
				"Rules describe correlations in this run, not proven causes."
			));
			output.appendChild(treeDetails);
		}

		function renderPatterns(result) {
			output.innerHTML = "";

			if (!result.ok) {
				setOutputMessage(result.reason);
				return;
			}

			var heading = createElement("h4", "results-analysis-heading", "Result patterns");
			output.appendChild(heading);

			if (result.sampled) {
				output.appendChild(createElement("p", "results-analysis-warning",
					"Large run: analyzed a deterministic sample of " + result.total + " of " + result.originalSize + " trials."));
			}

			result.results.forEach(function (resultGroup) {
				var groupTitle = createElement(
					"h5",
					"results-analysis-subheading",
					resultGroup.result + " (" + (resultGroup.goal === "max" ? "maximize" : "minimize") + ")" +
					(resultGroup.considered ? " · " + resultGroup.considered + " completed trials" : "")
				);
				output.appendChild(groupTitle);

				var columns = createElement("div", "analysis-pattern-columns");

				[["Better than the rest", resultGroup.good, "good"], ["Worse than the rest", resultGroup.bad, "bad"]].forEach(function (part) {
					var column = createElement("div", "analysis-pattern-column");
					column.appendChild(createElement("div", "analysis-pattern-column-title", part[0]));
					if (!part[1].length) {
						column.appendChild(createElement("p", "results-analysis-message", "No pattern found."));
					} else {
						part[1].forEach(function (candidate) {
							column.appendChild(renderRuleCard(candidate, part[2], applyRuleQuery));
						});
					}
					columns.appendChild(column);
				});

				output.appendChild(columns);
			});

			output.appendChild(createElement(
				"p",
				"results-analysis-note",
				"Patterns compare a subgroup with the remaining trials. Use the support (n) and the effect size to judge how trustworthy a pattern is."
			));
		}

		failureButton.addEventListener("click", function () { runAnalysis("failure"); });
		patternButton.addEventListener("click", function () { runAnalysis("patterns"); });

		var initialQuery = null;
		try {
			initialQuery = new URLSearchParams(root.location.search).get("result_query");
		} catch (error) {
			initialQuery = null;
		}

		if (initialQuery) {
			queryInput.value = initialQuery;
			applyQuery();
		} else {
			aggregate.textContent = "Showing all " + data.length + " trials.";
		}

		return {
			panel: panel,
			applyQuery: applyQuery,
			clear: function () {
				queryInput.value = "";
				applyQuery();
			}
		};
	}

	return {
		compileResultQuery: compileResultQuery,
		conditionMatchesValue: conditionMatchesValue,
		conditionsMatchRow: function (conditions, row, headerMap) {
			for (var i = 0; i < conditions.length; i++) {
				var condition = conditions[i];
				var index = condition.index;
				if (index === undefined) {
					index = headerMap[condition.column];
				}
				if (index === undefined || !conditionMatchesValue(condition, row[index])) {
					return false;
				}
			}
			return true;
		},
		conditionsToQuery: conditionsToQuery,
		computeAggregate: computeAggregate,
		selectFeatureColumns: selectFeatureColumns,
		buildDecisionTree: buildDecisionTree,
		treeToRules: treeToRules,
		evaluateTree: evaluateTree,
		analyzeFailures: analyzeFailures,
		findResultPatterns: findResultPatterns,
		formatCondition: formatCondition,
		formatNumber: formatNumber,
		initializeResultsAnalysis: initializeResultsAnalysis,
		DEFAULT_SPECIAL_COLUMNS: DEFAULT_SPECIAL_COLUMNS
	};
});
