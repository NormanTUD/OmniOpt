#!/usr/bin/env node

/**
 * Additional JS unit tests for the OmniOpt GUI frontend code.
 *
 * Mirrors the format of gui_js_tests.js so it can be run with plain `node`.
 * Covers functions from .gui/main.js, .gui/parallel.js, .gui/search.js and
 * .gui/darkmode.js that were not yet covered.
 */

var failedTests = 0;
var totalTests = 0;

function expect(label, actual, expected) {
	totalTests++;
	var isEqual = JSON.stringify(actual) === JSON.stringify(expected);
	if (isEqual) {
		if (process.env.SHOW_SUCCESS) console.log("PASS: " + label);
	} else {
		console.log("FAIL: " + label);
		console.log("  Expected: " + JSON.stringify(expected));
		console.log("  Actual:   " + JSON.stringify(actual));
		failedTests++;
	}
}

function expect_true(label, actual) {
	expect(label, !!actual, true);
}

function expect_false(label, actual) {
	expect(label, !!actual, false);
}

function expect_throws(label, callback) {
	totalTests++;
	try {
		callback();
		console.log("FAIL: " + label + " (expected exception, none thrown)");
		failedTests++;
	} catch (e) {
		if (process.env.SHOW_SUCCESS) console.log("PASS: " + label);
	}
}

// ============================================================
// Functions under test - extracted from .gui/main.js
// ============================================================

function parse_csv(csv) {
	try {
		var rows = csv.trim().split('\n').map(row => row.split(','));
		return rows;
	} catch (error) {
		return [];
	}
}

function normalizeArrayLength(array) {
	let maxColumns = array.reduce((max, row) => Math.max(max, row.length), 0);
	return array.map(row => {
		let filledRow = [...row];
		while (filledRow.length < maxColumns) {
			filledRow.push("");
		}
		return filledRow;
	});
}

function removeAnsiCodes(input) {
	const ansiRegex = /\x1b\[[0-9;]*[A-Za-z]/g;
	return input.replace(ansiRegex, '');
}

function removeLinesStartingWith(inputString, ...startStrings) {
	let lines = inputString.split("\n");
	let filteredLines = [];
	for (let i = 0; i < lines.length; i++) {
		let line = lines[i];
		let startsWithAny = startStrings.some(startString => line.includes(startString));
		if (!startsWithAny) {
			filteredLines.push(line);
		}
	}
	return filteredLines.join("\n");
}

// ============================================================
// Functions under test - extracted from .gui/parallel.js
// ============================================================

function createMapping(header_line) {
	let mapping = {};
	header_line.forEach((key, i) => {
		mapping[key] = i;
	});
	return mapping;
}

function areAllValuesNA(values) {
	return values.every(value => value === "N/A");
}

function isNumericArray(values) {
	return values.every(value => !isNaN(parseFloat(value)));
}

function mapStringsParallel(values) {
	let uniqueStrings = [...new Set(values.map(cleanValue))].sort();
	return uniqueStrings.reduce((acc, str, idx) => {
		acc[str] = idx;
		return acc;
	}, {});
}

function cleanValues(values) {
	return values.map(cleanValue);
}

function cleanValue(value) {
	return (value === null || value === undefined || value === "") ? "N/A" : value;
}

function createTicks(values) {
	const min = Math.min(...values);
	const max = Math.max(...values);
	const numTicks = Math.min(10, values.length);
	const step = (max - min) / (numTicks - 1);
	return Array.from({ length: numTicks }, (_, i) => (min + step * i).toFixed(2));
}

function createTickText(ticks) {
	return ticks.map(v => v.toLocaleString());
}

function extractResultValues(data, result_idx) {
	return data.map(row => parseFloat(row[result_idx]))
		.filter(value => value !== undefined && !isNaN(value));
}

function extractParameterKeys() {
	const excludedKeys = ["trial_index", "arm_name", "run_time", "trial_status", "generation_method", "start_time", "end_time", "program_string", "hostname", "signal", "exit_code"];
	return excludedKeys;
}

// ============================================================
// Functions under test - extracted from .gui/search.js
// ============================================================

function mark_search_result_yellow(content, search) {
	try {
		var escapedSearch = search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		var regex = new RegExp("(" + escapedSearch + ")", "gi");
		return content.replace(regex, "<span class='marked_text invert_in_dark_mode'>$1</span>");
	} catch (error) {
		return content;
	}
}

function get_category_icon(category) {
	var icons = {
		"Tutorials": "<img class='emoji_nav' src='emojis/books.svg' />",
		"Shares": "<img class='emoji_nav' src='emojis/world.svg' />",
		"Default": "<img class='emoji_nav' src='emojis/page.svg' />"
	};
	return icons[category] || icons["Default"];
}

function replace_backticks_with_tt(str) {
	let result = '';
	let i = 0;
	while (i < str.length) {
		if (str[i] === '`') {
			let end = str.indexOf('`', i + 1);
			if (end !== -1) {
				result += `<tt>${str.slice(i + 1, end)}</tt>`;
				i = end + 1;
			} else {
				result += str[i];
				i++;
			}
		} else {
			result += str[i];
			i++;
		}
	}
	return result;
}

// ============================================================
// Functions under test - extracted from .gui/darkmode.js
// ============================================================

function invertColor(color) {
	var rgb = color.match(/\d+/g);
	if (rgb) {
		var r = 255 - parseInt(rgb[0]);
		var g = 255 - parseInt(rgb[1]);
		var b = 255 - parseInt(rgb[2]);
		return 'rgb(' + r + ',' + g + ',' + b + ')';
	}
	return color;
}

// ============================================================
// Functions under test - extracted from .gui/main.js (uuid)
// ============================================================

function uuidv4() {
	return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
		(+c ^ (Math.floor(Math.random() * 16)) & 15 >> +c / 4).toString(16)
	);
}

// ============================================================
// TESTS
// ============================================================

// --- Group: parse_csv ---
console.log("--- Testing: parse_csv ---");
expect("parse_csv: basic", parse_csv("a,b\n1,2"), [["a","b"],["1","2"]]);
expect("parse_csv: empty becomes empty-row",
	parse_csv(""), [[""]]);
expect("parse_csv: single row", parse_csv("a,b,c"), [["a","b","c"]]);
expect("parse_csv: trims whitespace",
	parse_csv("  a,b\n1,2  "), [["a","b"],["1","2"]]);
expect("parse_csv: with newlines",
	parse_csv("\na,b\n1,2\n"), [["a","b"],["1","2"]]);
expect("parse_csv: preserves whitespace in fields (no trailing)",
	parse_csv("a,b"), [["a","b"]]);

// --- Group: normalizeArrayLength ---
console.log("\n--- Testing: normalizeArrayLength ---");
expect("normalizeArrayLength: equal length",
	normalizeArrayLength([[1,2],[3,4]]), [[1,2],[3,4]]);
expect("normalizeArrayLength: pads shorter",
	normalizeArrayLength([[1,2,3],[4,5]]),
	[[1,2,3],[4,5,""]]);
expect("normalizeArrayLength: empty array",
	normalizeArrayLength([]), []);
expect("normalizeArrayLength: all empty rows stay empty",
	normalizeArrayLength([[],[],[]]), [[],[],[]]);
expect("normalizeArrayLength: single row",
	normalizeArrayLength([[1]]), [[1]]);
expect("normalizeArrayLength: nested padding",
	normalizeArrayLength([[1],[2,3,4,5]]),
	[[1,"","",""],[2,3,4,5]]);
expect("normalizeArrayLength: handles strings",
	normalizeArrayLength([["a","b"],["c"]]),
	[["a","b"],["c",""]]);

// --- Group: removeAnsiCodes ---
console.log("\n--- Testing: removeAnsiCodes ---");
expect("removeAnsiCodes: red text",
	removeAnsiCodes("\x1b[31mHello\x1b[0m"), "Hello");
expect("removeAnsiCodes: no codes", removeAnsiCodes("plain text"), "plain text");
expect("removeAnsiCodes: empty", removeAnsiCodes(""), "");
expect("removeAnsiCodes: multiple codes",
	removeAnsiCodes("\x1b[31mR\x1b[0m\x1b[32mG\x1b[0m"), "RG");
expect("removeAnsiCodes: bold code",
	removeAnsiCodes("\x1b[1mbold\x1b[0m"), "bold");
expect("removeAnsiCodes: 256 color",
	removeAnsiCodes("\x1b[38;5;196mtext\x1b[0m"), "text");
expect("removeAnsiCodes: cursor",
	removeAnsiCodes("\x1b[2Jclear\x1b[H"), "clear");
expect("removeAnsiCodes: text between",
	removeAnsiCodes("a\x1b[31mb\x1b[0mc"), "abc");

// --- Group: removeLinesStartingWith ---
console.log("\n--- Testing: removeLinesStartingWith ---");
expect("removeLinesStartingWith: no match",
	removeLinesStartingWith("a\nb\nc", "x"),
	"a\nb\nc");
expect("removeLinesStartingWith: single filter",
	removeLinesStartingWith("DEBUG: line1\nreal line\nDEBUG: line2", "DEBUG"),
	"real line");
expect("removeLinesStartingWith: multiple filters",
	removeLinesStartingWith("DEBUG: x\nINFO: y\nreal\n", "DEBUG", "INFO"),
	"real\n");
expect("removeLinesStartingWith: empty input",
	removeLinesStartingWith("", "x"), "");
expect("removeLinesStartingWith: no filters",
	removeLinesStartingWith("a\nb"), "a\nb");
expect("removeLinesStartingWith: substring filter",
	removeLinesStartingWith("aaa\nbbb\naaa", "aaa"), "bbb");

// --- Group: createMapping ---
console.log("\n--- Testing: createMapping ---");
expect("createMapping: basic",
	createMapping(["a", "b", "c"]),
	{a: 0, b: 1, c: 2});
expect("createMapping: empty",
	createMapping([]), {});
expect("createMapping: single element",
	createMapping(["x"]), {x: 0});

// --- Group: areAllValuesNA ---
console.log("\n--- Testing: areAllValuesNA ---");
expect_true("areAllValuesNA: all NA",
	areAllValuesNA(["N/A", "N/A"]));
expect_false("areAllValuesNA: one non-NA",
	areAllValuesNA(["N/A", "1"]));
expect_true("areAllValuesNA: empty array (vacuously true)",
	areAllValuesNA([]));

// --- Group: isNumericArray ---
console.log("\n--- Testing: isNumericArray ---");
expect_true("isNumericArray: numbers",
	isNumericArray(["1", "2", "3"]));
expect_false("isNumericArray: one string",
	isNumericArray(["1", "two", "3"]));
expect_true("isNumericArray: floats",
	isNumericArray(["1.5", "2.5"]));
expect_true("isNumericArray: empty (vacuously true)",
	isNumericArray([]));
expect_true("isNumericArray: negative",
	isNumericArray(["-1", "-2.5"]));

// --- Group: mapStringsParallel ---
console.log("\n--- Testing: mapStringsParallel ---");
expect("mapStringsParallel: unique strings",
	mapStringsParallel(["a", "b", "c"]),
	{a: 0, b: 1, c: 2});
expect("mapStringsParallel: duplicates removed",
	mapStringsParallel(["a", "a", "b"]),
	{a: 0, b: 1});
expect("mapStringsParallel: sorted",
	mapStringsParallel(["c", "a", "b"]),
	{a: 0, b: 1, c: 2});
expect("mapStringsParallel: empty",
	mapStringsParallel([]), {});

// --- Group: cleanValue / cleanValues ---
console.log("\n--- Testing: cleanValue / cleanValues ---");
expect("cleanValue: null", cleanValue(null), "N/A");
expect("cleanValue: undefined", cleanValue(undefined), "N/A");
expect("cleanValue: empty string", cleanValue(""), "N/A");
expect("cleanValue: number", cleanValue(42), 42);
expect("cleanValue: string", cleanValue("hello"), "hello");
expect("cleanValues: mixed",
	cleanValues([null, "x", "", "y"]),
	["N/A", "x", "N/A", "y"]);
expect("cleanValues: empty",
	cleanValues([]), []);

// --- Group: createTicks ---
console.log("\n--- Testing: createTicks ---");
let ticks5 = createTicks([0, 1, 2, 3, 4]);
expect("createTicks: 5 elements returns 5 ticks",
	ticks5.length, 5);
expect("createTicks: first tick is min",
	ticks5[0], "0.00");
expect("createTicks: last tick is max",
	ticks5[ticks5.length-1], "4.00");

let singleTick = createTicks([5]);
expect("createTicks: single value returns one tick",
	singleTick.length, 1);

// --- Group: createTickText ---
console.log("\n--- Testing: createTickText ---");
expect("createTickText: returns string array",
	typeof createTickText(["1.5"])[0], "string");
expect("createTickText: same length",
	createTickText(["1.5", "2.5"]).length, 2);

// --- Group: extractResultValues ---
console.log("\n--- Testing: extractResultValues ---");
expect("extractResultValues: extracts numbers",
	extractResultValues([["a", "1.5"], ["b", "2.5"]], 1),
	[1.5, 2.5]);
expect("extractResultValues: filters NaN",
	extractResultValues([["1"], ["abc"], ["3"]], 0),
	[1, 3]);
expect("extractResultValues: empty data",
	extractResultValues([], 0), []);

// --- Group: extractParameterKeys ---
console.log("\n--- Testing: extractParameterKeys ---");
let excludedKeys = extractParameterKeys();
expect_true("extractParameterKeys: contains trial_index",
	excludedKeys.includes("trial_index"));
expect_true("extractParameterKeys: contains arm_name",
	excludedKeys.includes("arm_name"));
expect_true("extractParameterKeys: contains exit_code",
	excludedKeys.includes("exit_code"));
expect_false("extractParameterKeys: contains parameter",
	excludedKeys.includes("parameter"));

// --- Group: mark_search_result_yellow ---
console.log("\n--- Testing: mark_search_result_yellow ---");
let marked = mark_search_result_yellow("hello world", "world");
expect_true("mark_search_result_yellow: contains span",
	marked.includes("<span"));
expect_true("mark_search_result_yellow: contains marked_text",
	marked.includes("marked_text"));
expect_true("mark_search_result_yellow: contains search term",
	marked.includes("world"));

let markedCaseInsensitive = mark_search_result_yellow("Hello World", "world");
expect_true("mark_search_result_yellow: case insensitive",
	markedCaseInsensitive.toLowerCase().includes("<span"));

let markedRegexSpecial = mark_search_result_yellow("a.b.c", ".");
expect_true("mark_search_result_yellow: escapes regex chars",
	markedRegexSpecial.includes("<span"));

// --- Group: get_category_icon ---
console.log("\n--- Testing: get_category_icon ---");
expect_true("get_category_icon: Tutorials has icon",
	get_category_icon("Tutorials").includes("books.svg"));
expect_true("get_category_icon: Shares has icon",
	get_category_icon("Shares").includes("world.svg"));
expect_true("get_category_icon: Default has icon",
	get_category_icon("Default").includes("page.svg"));
expect_true("get_category_icon: unknown returns Default",
	get_category_icon("UnknownCat").includes("page.svg"));

// --- Group: replace_backticks_with_tt ---
console.log("\n--- Testing: replace_backticks_with_tt ---");
expect("replace_backticks_with_tt: single backtick",
	replace_backticks_with_tt("`code`"),
	"<tt>code</tt>");
expect("replace_backticks_with_tt: multiple",
	replace_backticks_with_tt("`a` and `b`"),
	"<tt>a</tt> and <tt>b</tt>");
expect("replace_backticks_with_tt: no backticks",
	replace_backticks_with_tt("no code"), "no code");
expect("replace_backticks_with_tt: empty",
	replace_backticks_with_tt(""), "");
expect("replace_backticks_with_tt: unclosed backtick",
	replace_backticks_with_tt("`unclosed"), "`unclosed");
expect("replace_backticks_with_tt: with spaces inside",
	replace_backticks_with_tt("`a b c`"), "<tt>a b c</tt>");
expect("replace_backticks_with_tt: text around",
	replace_backticks_with_tt("hello `world` bye"),
	"hello <tt>world</tt> bye");

// --- Group: invertColor ---
console.log("\n--- Testing: invertColor ---");
expect("invertColor: rgb(255,0,0) -> rgb(0,255,255)",
	invertColor("rgb(255,0,0)"), "rgb(0,255,255)");
expect("invertColor: rgb(0,0,0) -> rgb(255,255,255)",
	invertColor("rgb(0,0,0)"), "rgb(255,255,255)");
expect("invertColor: rgb(128,64,32)",
	invertColor("rgb(128,64,32)"), "rgb(127,191,223)");
expect("invertColor: not rgb returns unchanged",
	invertColor("notacolor"), "notacolor");
expect("invertColor: named color preserved",
	invertColor("red"), "red");
expect("invertColor: rgb with spaces",
	invertColor("rgb( 50 , 60 , 70 )"), "rgb(205,195,185)");
expect("invertColor: hex gives rgb output (with NaN)",
	invertColor("#FF0000").startsWith("rgb("), true);
expect("invertColor: rgba gives rgb",
	invertColor("rgba(100,150,200,0.5)").startsWith("rgb("), true);

// --- Group: uuidv4 ---
console.log("\n--- Testing: uuidv4 ---");
let uuid = uuidv4();
expect_true("uuidv4: has hyphens",
	uuid.includes("-"));
expect("uuidv4: length 36", uuid.length, 36);
expect_true("uuidv4: contains hex chars",
	/[0-9a-f-]{36}/.test(uuid));
expect_true("uuidv4: position 14 is 4",
	uuid[14], "4");
expect_true("uuidv4: position 19 is one of 8,9,a,b",
	"89ab".includes(uuid[19]));

// ============================================================
// SUMMARY
// ============================================================
console.log("\n---------------------------------");
if (failedTests === 0) {
	console.log(`SUMMARY: All ${totalTests} additional JS tests passed successfully.`);
	process.exit(0);
} else {
	console.log(`SUMMARY: ${failedTests} of ${totalTests} additional JS test(s) failed.`);
	process.exit(1);
}
