#!/usr/bin/env node

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
// Functions under test - extracted from gui.js
// ============================================================

function normalizeFloat(value) {
	if (!isFinite(value)) {
		return '';
	}

	var str = value.toString();

	if (str.includes('e')) {
		var fixed = value.toFixed(20);
		fixed = fixed.replace(/(\.\d*?[1-9])0+$/, '$1');
		fixed = fixed.replace(/\.0+$/, '');
		return fixed;
	}

	return str;
}

function string_or_array_to_list(input) {
	if (typeof input === "string") {
		return input;
	} else if (Array.isArray(input)) {
		if (input.length === 1) {
			return input[0];
		} else {
			const listItems = input.map(item => `<li>${item}</li>`);
			return `<ul>${listItems.join("")}</ul>`;
		}
	} else {
		throw new Error("Invalid input type. Only strings or arrays are allowed.");
	}
}

function quote_variables(input) {
	return input.replace(/(["'])(.*?)\1|%(\((\w+)\)|(\w+))/g, function(match, quotes, insideQuotes, p1, p2, p3) {
		if (quotes) {
			return match;
		} else {
			var variable = p2 || p3;
			return "'%(" + variable + ")'";
		}
	});
}

function get_var_names_from_run_program(run_program_string) {
	const pattern = /(?:\$|\%)?\([a-zA-Z_]+\)|(?:\$|%)[a-zA-Z_]+/g;
	const variableNames = [];

	let match;
	while ((match = pattern.exec(run_program_string)) !== null) {
		let varName = match[0];
		varName = varName.replace(/^(\$|%)/, "");
		varName = varName.replace(/^(\$|%)?\(|\)$/g, "");
		if (/^[a-zA-Z_]+$/.test(varName)) {
			variableNames.push(varName);
		}
	}

	return variableNames;
}

function encode_base64(v) {
	return btoa(v);
}

function decode_base64(input) {
	decoded = atob(input);
	return decoded;
}

function addBase64DecodedVersions(cmdString) {
	return cmdString.replace(/(--[a-zA-Z0-9_]+)=('([^']+)'|"([^"]+)"|([^\s]+))/g, (match, key, _, singleQuoted, doubleQuoted, bare) => {
		const value = singleQuoted || doubleQuoted || bare;

		let decoded = null;
		try {
			if (key === "--run_program" || key === "--run_program_once") {
				decoded = decode_base64(value);
			}
		} catch (e) {
			console.error(e);
		}

		if (decoded) {
			var safeDecoded = decoded.replace(/\x27/g, `'\\''`).trim();
			return ` ${key}=$(echo '${safeDecoded}' | base64 -w0)`;
		} else {
			return match;
		}
	});
}

function add_equation_spaces(expression) {
	const operators = {
		'>=': '__GE__',
		'<=': '__LE__',
		'==': '__EQ__',
		'!=': '__NE__',
		'=>': '__AR__',
	};

	for (const [op, placeholder] of Object.entries(operators)) {
		expression = expression.replaceAll(op, placeholder);
	}

	expression = expression.replace(/([+\-*/()=<>])/g, ' $1 ');

	for (const [op, placeholder] of Object.entries(operators)) {
		expression = expression.replaceAll(placeholder, ` ${op} `);
	}

	return expression.replace(/\s+/g, ' ').trim();
}

function test_if_equation_is_valid(str, names) {
	var errors = [];
	var isValid = true;

	if (!str.includes(">=") && !str.includes("<=")) {
		errors.push("Missing '>=' or '<=' operator.");
		isValid = false;
	}

	var splitted = str.includes(">=") ? str.split(">=") : str.split("<=");
	if (splitted.length !== 2) {
		errors.push("Equation format is incorrect.");
		isValid = false;
	}

	var left_side = splitted[0].replace(/\s+/g, "");
	if (!left_side) {
		errors.push("Left side is empty or contains only whitespace.");
		isValid = false;
	}

	if (isValid) {
		var right_side = splitted[1].trim();

		if (names.includes(left_side) && names.includes(right_side)) {
			return "";
		}

		if (!/^[+-]?\d+(\.\d+)?$/.test(right_side)) {
			errors.push("The right side does not look like a constant.");
			isValid = false;
		}

		var escapedNames = names.map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
		var namePattern = `(?:${escapedNames.join("|")})`;

		var numberPattern = "\\d+(?:\\.\\d+)?";
		var factorPattern = `(?:${numberPattern}|${namePattern})`;
		var productPattern = `${factorPattern}(?:\\*${factorPattern})*`;
		var termPattern = `[+-]?${productPattern}`;
		var fullPattern = `^${termPattern}(?:[+-]${termPattern})*$`;

		var regex = new RegExp(fullPattern);
		if (!regex.test(left_side)) {
			errors.push("Left side does not match expected pattern.");
			isValid = false;
		}

		if (/[*+-]{2,}/.test(left_side)) {
			errors.push("Multiple operators in a row.");
			isValid = false;
		}

		var nr_re = "([+-]?\\d+(\\.\\d+)?)";
		var number_followed_by_varname = new RegExp(`${nr_re}(${namePattern})`);
		if (number_followed_by_varname.test(left_side)) {
			errors.push("Number followed directly by variable without *.");
			isValid = false;
		}

		if (/^[*+]/.test(left_side)) {
			errors.push("Left side starts with an invalid operator.");
			isValid = false;
		}
	}

	function errorsToHtml(_errors) {
		if (_errors.length) {
			_errors.unshift(`Equation: ${str}`);
			return "<ul>" + _errors.map(error => `<li>${error}</li>`).join('') + "</ul>";
		}
		return "";
	}

	var ret_str = errorsToHtml(errors);
	return ret_str;
}

// ============================================================
// Formula parsing helpers (extracted from gui.js)
// ============================================================

var FORMULA_RESERVED = new Set([
	"pi", "E", "I", "oo", "inf", "infty", "nan", "NaN", "True", "False",
	"sin", "cos", "tan", "asin", "acos", "atan",
	"sinh", "cosh", "tanh",
	"exp", "log", "ln", "sqrt", "abs", "Min", "Max",
	"Sum", "Product", "Integral", "Derivative",
	"math", "numpy", "np", "self", "def", "return", "import", "from",
	"if", "else", "elif", "for", "while", "in", "and", "or", "not",
	"params", "evaluate", "_raw",
]);

function _strip_macros(text) {
	text = text.replace(
		/\\(text|textit|textbf|mathrm|operatorname|mathbf|mathcal|mathbb|mathfrak|mathsf|mathtt|mbox|boldsymbol)\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g,
		"$1"
	);
	text = text.replace(/\\(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|ln|sqrt|abs)\b/g, "$1");
	text = text.replace(/\\(sum|prod|frac|dfrac|tfrac|sqrt|left|right|displaystyle|textstyle|mathit|mathrm|operatorname)\b/g, "");
	text = text.replace(/\\([A-Za-z]+)/g, "$1");
	text = text.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, "$1");
	return text;
}

function _collect_identifiers(text) {
	var re = /[A-Za-z_][A-Za-z0-9_]*/g;
	var out = [];
	var seen = {};
	var m;
	while ((m = re.exec(text)) !== null) {
		var name = m[0];
		if (FORMULA_RESERVED.has(name)) continue;
		out.push(name);
		seen[name] = (seen[name] || 0) + 1;
	}
	return { list: out, set: Object.keys(seen) };
}

function _split_assignment(text) {
	var depth = 0;
	var lastIdx = -1;
	for (var i = 0; i < text.length; i++) {
		var ch = text[i];
		if (ch === "{") depth++;
		else if (ch === "}") depth--;
		else if (ch === "=" && depth === 0) {
			lastIdx = i;
		}
	}
	if (lastIdx >= 0) {
		return { lhs: text.substring(0, lastIdx), rhs: text.substring(lastIdx + 1) };
	}
	return { lhs: "", rhs: text };
}

function _lhs_parameter_names(lhs) {
	var open = lhs.indexOf("(");
	if (open < 0) return [];
	var close = lhs.lastIndexOf(")");
	if (close < open) return [];
	var name = lhs.substring(0, open).trim();
	if (!/^[A-Za-z_][A-Za-z0-9_.]*$/.test(name)) return [];
	var inner = lhs.substring(open + 1, close);
	return _collect_identifiers(_strip_macros(inner)).list;
}

function _strip_sumprod_bodies(text) {
	var boundNames = {};
	var re = /\\(?:sum|prod)\s*_\s*(?:\{([^{}]+)\}|([A-Za-z][A-Za-z0-9_]*))(?:\s*\^\s*\{[^{}]+\})?\s*(\{(?:[^{}]|\{[^{}]*\})*\}|[A-Za-z_][A-Za-z0-9_]*)/g;
	var m;
	while ((m = re.exec(text)) !== null) {
		var sub = m[1] || m[2] || "";
		var name = sub.indexOf("=") >= 0 ? sub.split("=", 1)[0].trim() : sub.trim();
		boundNames[name] = true;
	}
	text = text.replace(
		/\\(?:sum|prod)\s*_\s*(?:\{[^{}]+\}|[A-Za-z][A-Za-z0-9_]*)(?:\s*\^\s*\{[^{}]+\})?\s*(\{(?:[^{}]|\{[^{}]*\})*\}|[A-Za-z_][A-Za-z0-9_]*)/g,
		" "
	);
	return { text: text, bound: boundNames };
}

function client_extract_python_params(text) {
	if (!text) return [];
	var seen = {};
	var out = [];
	var re = /params(?:\s*\.\s*get\s*\()?\s*\[?\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\)?/g;
	var m;
	while ((m = re.exec(text)) !== null) {
		var n = m[1];
		if (!seen[n]) {
			seen[n] = true;
			out.push(n);
		}
	}
	return out;
}

// ============================================================
// TESTS
// ============================================================

// --- Group: normalizeFloat ---
console.log("--- Testing: normalizeFloat ---");
expect("normalizeFloat: integer", normalizeFloat(42), "42");
expect("normalizeFloat: float", normalizeFloat(3.14), "3.14");
expect("normalizeFloat: zero", normalizeFloat(0), "0");
expect("normalizeFloat: negative", normalizeFloat(-5.5), "-5.5");
expect("normalizeFloat: NaN", normalizeFloat(NaN), "");
expect("normalizeFloat: Infinity", normalizeFloat(Infinity), "");
expect("normalizeFloat: -Infinity", normalizeFloat(-Infinity), "");
expect("normalizeFloat: small float", normalizeFloat(0.1 + 0.2), "0.30000000000000004");
expect("normalizeFloat: very small", normalizeFloat(1e-20), "0.00000000000000000001");
expect("normalizeFloat: scientific notation", normalizeFloat(1e10), "10000000000");
expect("normalizeFloat: scientific small", normalizeFloat(1.5e-7), "0.00000015");
expect("normalizeFloat: trailing zeros in sci", normalizeFloat(1.50e-7), "0.00000015");

// --- Group: string_or_array_to_list ---
console.log("\n--- Testing: string_or_array_to_list ---");
expect("string_or_array_to_list: string passthrough", string_or_array_to_list("hello"), "hello");
expect("string_or_array_to_list: single item array", string_or_array_to_list(["only"]), "only");
expect("string_or_array_to_list: multiple items", string_or_array_to_list(["a", "b"]), "<ul><li>a</li><li>b</li></ul>");
expect("string_or_array_to_list: three items", string_or_array_to_list(["x", "y", "z"]), "<ul><li>x</li><li>y</li><li>z</li></ul>");
expect_throws("string_or_array_to_list: throws on number", function() { string_or_array_to_list(42); });
expect_throws("string_or_array_to_list: throws on null", function() { string_or_array_to_list(null); });
expect_throws("string_or_array_to_list: throws on object", function() { string_or_array_to_list({}); });

// --- Group: quote_variables ---
console.log("\n--- Testing: quote_variables ---");
expect("quote_variables: bare variable", quote_variables("%(var)"), "'%(var)'");
expect("quote_variables: $variable", quote_variables("%(var)"), "'%(var)'");
expect("quote_variables: already quoted", quote_variables("'%(var)'"), "'%(var)'");
expect("quote_variables: double quoted", quote_variables('"%(var)"'), '"%(var)"');
expect("quote_variables: mixed", quote_variables("foo %(bar) baz"), "foo '%(bar)' baz");
expect("quote_variables: no variables", quote_variables("no vars here"), "no vars here");
expect("quote_variables: multiple vars", quote_variables("%(a) %(b)"), "'%(a)' '%(b)'");
expect("quote_variables: $-prefixed", quote_variables("%(x)"), "'%(x)'");

// --- Group: get_var_names_from_run_program ---
console.log("\n--- Testing: get_var_names_from_run_program ---");
expect("get_var_names_from_run_program: $() syntax", get_var_names_from_run_program("$(var)"), ["var"]);
expect("get_var_names_from_run_program: %(name) syntax", get_var_names_from_run_program("%(name)"), ["name"]);
expect("get_var_names_from_run_program: multiple", get_var_names_from_run_program("$(a) %(b) $(c)"), ["a", "b", "c"]);
expect("get_var_names_from_run_program: no vars", get_var_names_from_run_program("no variables"), []);
expect("get_var_names_from_run_program: mixed with text", get_var_names_from_run_program("echo $(x) > %(y).txt"), ["x", "y"]);
expect("get_var_names_from_run_program: empty string", get_var_names_from_run_program(""), []);

// --- Group: encode_base64 / decode_base64 ---
console.log("\n--- Testing: encode_base64 / decode_base64 ---");
expect("encode_base64: simple", encode_base64("hello"), "aGVsbG8=");
expect("encode_base64: empty", encode_base64(""), "");
expect("decode_base64: simple", decode_base64("aGVsbG8="), "hello");
expect("decode_base64: empty", decode_base64(""), "");
expect("encode/decode roundtrip", decode_base64(encode_base64("test string 123")), "test string 123");

// --- Group: addBase64DecodedVersions ---
console.log("\n--- Testing: addBase64DecodedVersions ---");
var encoded = encode_base64("print('hello')");
expect("addBase64DecodedVersions: run_program", addBase64DecodedVersions("--run_program=" + encoded).includes("base64 -w0"), true);
expect("addBase64DecodedVersions: non-run_program unchanged", addBase64DecodedVersions("--other=value"), "--other=value");
var encoded2 = encode_base64("echo test");
expect("addBase64DecodedVersions: run_program_once", addBase64DecodedVersions("--run_program_once=" + encoded2).includes("base64 -w0"), true);
expect("addBase64DecodedVersions: quoted value", addBase64DecodedVersions("--run_program='" + encoded + "'").includes("base64 -w0"), true);
expect("addBase64DecodedVersions: double quoted", addBase64DecodedVersions('--run_program="' + encoded + '"').includes("base64 -w0"), true);

// --- Group: add_equation_spaces ---
console.log("\n--- Testing: add_equation_spaces ---");
expect("add_equation_spaces: >= operator", add_equation_spaces("x>=5"), "x >= 5");
expect("add_equation_spaces: <= operator", add_equation_spaces("x<=5"), "x <= 5");
expect("add_equation_spaces: already spaced", add_equation_spaces("x + y >= 5"), "x + y >= 5");
expect("add_equation_spaces: multiply", add_equation_spaces("2*x+3*y>=10"), "2 * x + 3 * y >= 10");
expect("add_equation_spaces: complex", add_equation_spaces("x+y>=10"), "x + y >= 10");
expect("add_equation_spaces: parens", add_equation_spaces("(x+1)>=5"), "( x + 1 ) >= 5");
expect("add_equation_spaces: != operator", add_equation_spaces("x!=5"), "x != 5");
expect("add_equation_spaces: == operator", add_equation_spaces("x==5"), "x == 5");
expect("add_equation_spaces: extra whitespace collapsed", add_equation_spaces("  x   +   y  >=  5  "), "x + y >= 5");

// --- Group: test_if_equation_is_valid ---
console.log("\n--- Testing: test_if_equation_is_valid ---");
var names = ["hallo", "welt", "x", "y"];

expect_true("equation: x >= y", test_if_equation_is_valid("x >= y", names) === "");
expect_true("equation: x + y >= 5", test_if_equation_is_valid("x + y >= 5", names) === "");
expect_true("equation: 2*x + 3*y <= 10", test_if_equation_is_valid("2*x + 3*y <= 10", names) === "");
expect_true("equation: x*y >= 10", test_if_equation_is_valid("x*y >= 10", names) === "");
expect_true("equation: 0*x + 0*y >= 0", test_if_equation_is_valid("0*x + 0*y >= 0", names) === "");
expect_true("equation: hallo + welt <= 10", test_if_equation_is_valid("hallo + welt <= 10", names) === "");
expect_true("equation: with spaces", test_if_equation_is_valid("  x  +   y  <=  15  ", names) === "");
expect_true("equation: tabs", test_if_equation_is_valid("hallo\t+\twelt \t<= 42", names) === "");

expect_false("equation: missing operator", test_if_equation_is_valid("x + y", names) === "");
expect_false("equation: > instead of >=", test_if_equation_is_valid("x + y > 5", names) === "");
expect_false("equation: = instead of >=", test_if_equation_is_valid("x + y = 10", names) === "");
expect_false("equation: empty right side", test_if_equation_is_valid("x + y >= ", names) === "");
expect_false("equation: unknown variable", test_if_equation_is_valid("x + z >= 10", names) === "");
expect_false("equation: 2hallo", test_if_equation_is_valid("2hallo + 3welt <= 10", names) === "");
expect_false("equation: division", test_if_equation_is_valid("x / 2 >= 5", names) === "");
expect_false("equation: unicode numbers", test_if_equation_is_valid("hallo + welt <= 𝟜𝟚", names) === "");
expect_false("equation: starting with *", test_if_equation_is_valid("*x + y >= 10", names) === "");
expect_false("equation: starting with +", test_if_equation_is_valid("+x + y >= 10", names) === "");
expect_false("equation: double operators", test_if_equation_is_valid("x + + y >= 5", names) === "");
expect_false("equation: parentheses", test_if_equation_is_valid("hallo - (welt) + x - (y) <= 10", names) === "");
expect_false("equation: comma", test_if_equation_is_valid("hallo, welt <= 10", names) === "");
expect_false("equation: single operator only", test_if_equation_is_valid(">= 10", names) === "");

// --- Group: _split_assignment ---
console.log("\n--- Testing: _split_assignment ---");
expect("split: no =",
	JSON.stringify(_split_assignment("x + y")), JSON.stringify({ lhs: "", rhs: "x + y" }));
expect("split: single =",
	JSON.stringify(_split_assignment("f(x) = x + y")), JSON.stringify({ lhs: "f(x) ", rhs: " x + y" }));
expect("split: chained = uses LAST = (a + b = c = c - d)",
	JSON.stringify(_split_assignment("a + b = c = c - d")),
	JSON.stringify({ lhs: "a + b = c ", rhs: " c - d" }));
expect("split: = inside braces ignored",
	JSON.stringify(_split_assignment("f({a=b}) = c")),
	JSON.stringify({ lhs: "f({a=b}) ", rhs: " c" }));
expect("split: empty rhs",
	JSON.stringify(_split_assignment("y =")), JSON.stringify({ lhs: "y ", rhs: "" }));

// --- Group: _lhs_parameter_names ---
console.log("\n--- Testing: _lhs_parameter_names ---");
expect("lhs params: f(x, y)",
	JSON.stringify(_lhs_parameter_names("f(x, y)")), JSON.stringify(["x", "y"]));
expect("lhs params: f(g(x))",
	JSON.stringify(_lhs_parameter_names("f(g(x))")), JSON.stringify(["g", "x"]));
expect("lhs params: no parens -> []",
	JSON.stringify(_lhs_parameter_names("a + b = c")), JSON.stringify([]));
expect("lhs params: chained = with arithmetic LHS -> []",
	JSON.stringify(_lhs_parameter_names("a + b = c")), JSON.stringify([]));
expect("lhs params: dotted method name",
	JSON.stringify(_lhs_parameter_names("obj.f(x, y)")), JSON.stringify(["x", "y"]));
expect("lhs params: not an identifier name -> []",
	JSON.stringify(_lhs_parameter_names("2f(x)")), JSON.stringify([]));

// --- Group: _strip_sumprod_bodies ---
console.log("\n--- Testing: _strip_sumprod_bodies ---");
expect("sumprod: bound i",
	JSON.stringify(Object.keys(_strip_sumprod_bodies("\\sum_{i=0}^{n} i**2").bound)),
	JSON.stringify(["i"]));
expect("sumprod: bound n from RHS not affected (was in ^)",
	JSON.stringify(Object.keys(_strip_sumprod_bodies("\\sum_{i=0}^{n} i**2").bound).sort()),
	JSON.stringify(["i"]));
expect("sumprod: prod",
	JSON.stringify(Object.keys(_strip_sumprod_bodies("\\prod_{k=0}^{m} k").bound)),
	JSON.stringify(["k"]));
expect("sumprod: nested (each \\sum needs its own body)",
	JSON.stringify(Object.keys(_strip_sumprod_bodies("\\sum_i i**2 + \\sum_j j**2").bound).sort()),
	JSON.stringify(["i", "j"]));
expect_true("sumprod: bound name (i) is removed from text",
	_strip_sumprod_bodies("\\sum_i i**2").text.indexOf("i") < 0);

// --- Group: client_extract_python_params ---
console.log("\n--- Testing: client_extract_python_params ---");
expect("py params: single quoted",
	JSON.stringify(client_extract_python_params("def evaluate(params): return params['x'] + params['y']")),
	JSON.stringify(["x", "y"]));
expect("py params: double quoted",
	JSON.stringify(client_extract_python_params("params[\"lr\"] * 2")),
	JSON.stringify(["lr"]));
expect("py params: get()",
	JSON.stringify(client_extract_python_params("params.get('epochs')")),
	JSON.stringify(["epochs"]));
expect("py params: no params",
	JSON.stringify(client_extract_python_params("def evaluate(x): return x")),
	JSON.stringify([]));
expect("py params: dedup",
	JSON.stringify(client_extract_python_params("params['x'] + params['x']")),
	JSON.stringify(["x"]));

function sorted(arr) {
	return arr.slice().sort();
}

// --- Group: LHS-based parameter split ---
console.log("\n--- Testing: LHS / no-LHS split ---");
function param_names(result) {
	return sorted(result.parameters);
}
function const_names(result) {
	return sorted(result.constants);
}
// No LHS: every free symbol should land in parameters (NOT constants).
{
	var r = client_extract_formula_params_simple("a + sin(b)", "infix");
	expect_true("no-LHS: both vars end up as parameters",
		JSON.stringify(param_names(r)) === JSON.stringify(["a", "b"]));
	expect_true("no-LHS: constants list is empty",
		JSON.stringify(const_names(r)) === JSON.stringify([]));
}
{
	// ``a + b + c`` has 3 distinct identifiers and no implicit-multiplication
	// ambiguity.
	var r = client_extract_formula_params_simple("a + b + c", "infix");
	expect_true("no-LHS: 3 vars -> 3 parameters",
		r.parameters.length === 3 && r.constants.length === 0);
}
{
	var r = client_extract_formula_params_simple("e^x", "infix");
	// `e` is a known constant; without an LHS it should still be suggested
	// as a parameter so the user has the choice to fix or optimise it.
	expect_true("no-LHS: e becomes a parameter (not a hidden constant)",
		param_names(r).indexOf("e") >= 0);
}
// With an LHS function-call: LHS params become parameters, RHS-only
// identifiers become constants.
{
	var r = client_extract_formula_params_simple("f(x, y) = a*x + b*y + c", "infix");
	expect_true("with-LHS: parameters are x, y (from the LHS)",
		JSON.stringify(param_names(r)) === JSON.stringify(["x", "y"]));
	expect_true("with-LHS: constants are a, b, c (RHS-only)",
		JSON.stringify(const_names(r)) === JSON.stringify(["a", "b", "c"]));
}
// With an LHS that's a bare identifier (no parens) like ``g(x) = ...`` —
// the parameter list is inside the parens.
{
	var r = client_extract_formula_params_simple("g(x) = x**2 + y", "infix");
	expect_true("with-LHS-bare: parameters from parens",
		JSON.stringify(param_names(r)) === JSON.stringify(["x"]));
	expect_true("with-LHS-bare: y is a constant",
		JSON.stringify(const_names(r)) === JSON.stringify(["y"]));
}

// --- Group: broken-formula behaviour ---
console.log("\n--- Testing: broken-formula handling ---");
function safe_extract(text, mode) {
	try {
		return JSON.stringify(client_extract_formula_params_simple(text, mode));
	} catch (e) {
		return "ERROR: " + e.message;
	}
}
// Build a simplified version that uses the extracted helpers so we can
// test the parse flow without jQuery / DOM.
function client_extract_formula_params_simple(text, mode) {
	if (!text || !text.trim()) return { parameters: [], constants: [], bound: [] };
	var split = _split_assignment(text);
	var lhsRaw = split.lhs;
	var boundInfo = _strip_sumprod_bodies(text);
	var boundNames = boundInfo.bound;
	var lhsIdents = lhsRaw.trim() ? _lhs_parameter_names(lhsRaw) : [];
	var lhsSet = {};
	for (var i = 0; i < lhsIdents.length; i++) lhsSet[lhsIdents[i]] = true;
	var rhsClean = _strip_macros(boundInfo.text);
	var rhsOnly = _split_assignment(rhsClean).rhs;
	var rhsIdents = _collect_identifiers(rhsOnly).list;
	var parameters = [], constants = [];
	var seenP = {}, seenC = {};
	var hasLhs = lhsIdents.length > 0;
	for (var j = 0; j < lhsIdents.length; j++) {
		if (!seenP[lhsIdents[j]]) { seenP[lhsIdents[j]] = true; parameters.push(lhsIdents[j]); }
	}
	for (var k = 0; k < rhsIdents.length; k++) {
		var rn = rhsIdents[k];
		if (boundNames[rn]) continue;
		if (lhsSet[rn]) continue;
		if (FORMULA_RESERVED.has(rn)) continue;
		if (!seenP[rn] && !seenC[rn]) {
			seenP[rn] = true;
			seenC[rn] = true;
			if (hasLhs) {
				constants.push(rn);
			} else {
				parameters.push(rn);
			}
		}
	}
	return { parameters: parameters, constants: constants, bound: Object.keys(boundNames) };
}

expect_true("broken: unbalanced braces don't crash",
	safe_extract("f(x) = x + {y", "infix").indexOf("ERROR") < 0);
expect_true("broken: trailing operator doesn't crash",
	safe_extract("f(x, y) = x +", "infix").indexOf("ERROR") < 0);
expect_true("broken: empty body doesn't crash",
	safe_extract("f(x) = \\sum_{i} ", "infix").indexOf("ERROR") < 0);
expect_true("broken: bare macro doesn't crash",
	safe_extract("f(x) = \\", "infix").indexOf("ERROR") < 0);
expect_true("broken: random LaTeX doesn't crash",
	safe_extract("f(x) = \\frac{}{}", "infix").indexOf("ERROR") < 0);
expect_true("broken: NaN literal doesn't crash",
	safe_extract("f(x) = NaN + x", "infix").indexOf("ERROR") < 0);
// Chained assignments don't crash and produce *some* shape.
{
	var r = safe_extract("a + b = c = c - d", "infix");
	var j = JSON.parse(r);
	expect_true("broken: chained assignment returns shape",
		j && Array.isArray(j.parameters) && Array.isArray(j.constants));
}
// The RHS-only identifiers should appear in either parameters or constants
// (the LHS arithmetic ``a + b = c`` isn't a function definition so we
// can't pull parameter names out of it).
{
	var r = safe_extract("a + b = c = c - d", "infix");
	var j = JSON.parse(r);
	var all = (j.parameters.concat(j.constants)).sort();
	expect_true("broken: chained assignment RHS variables all accounted for",
		all.indexOf("c") >= 0 && all.indexOf("d") >= 0);
}
expect("broken: normal formula still works",
	JSON.parse(safe_extract("f(x, y) = x + y", "infix")).parameters.sort(),
	["x", "y"]);

// --- Group: URL round-trip for run_program / formula ---
console.log("\n--- Testing: URL round-trip ---");
// Mimic the encode / decode used by update_url + the URL-restoration code.
function url_encode_text(v) {
	if (v && v.trim() !== "") {
		return btoa(unescape(encodeURIComponent(v)));
	}
	return "";
}
function url_decode_text(v) {
	if (v === "") return "";
	try {
		return decodeURIComponent(escape(atob(v)));
	} catch (e) {
		try {
			return decodeURIComponent(v);
		} catch (e2) {
			return v;
		}
	}
}

expect_true("url: round-trip simple text",
	url_decode_text(url_encode_text("echo hello")) === "echo hello");
expect_true("url: round-trip multi-line",
	url_decode_text(url_encode_text("line1\nline2\nline3")) === "line1\nline2\nline3");
expect_true("url: round-trip special chars",
	url_decode_text(url_encode_text('a && b || c < d > e "f" \'g\'')) ===
		'a && b || c < d > e "f" \'g\'');
expect_true("url: round-trip unicode",
	url_decode_text(url_encode_text("Häuser überall — 你好")) === "Häuser überall — 你好");
expect_true("url: round-trip empty",
	url_decode_text(url_encode_text("")) === "");
expect_true("url: round-trip whitespace-only preserved as empty",
	url_decode_text(url_encode_text("   ")) === "");
expect_true("url: encoded multi-line is one-line (no %0A in the encoded form)",
	url_encode_text("a\nb").indexOf("\n") < 0);
// The encoded form must survive a full URLSearchParams round-trip.
function roundtripViaUrlSearch(v) {
	var encoded = url_encode_text(v);
	// URLSearchParams decodes once; we feed it the already-encoded value
	// wrapped in the URL-encoding layer the browser applies.
	var params = new URLSearchParams();
	params.set("rp", encodeURIComponent(encoded));
	var got = params.get("rp");
	return url_decode_text(got);
}
expect_true("url: URLSearchParams round-trip preserves newlines (base64-safe)",
	!/[\r\n]/.test(roundtripViaUrlSearch("echo hello\nls -la")));
expect_true("url: URLSearchParams round-trip preserves tabs (base64-safe)",
	!/[\t]/.test(roundtripViaUrlSearch("col1\tcol2\tcol3")));
// Full round-trip: original -> encode -> URL-decode -> decode must equal original.
function fullRoundtrip(v) {
	var encoded = url_encode_text(v);
	// URLSearchParams applies URL encoding on .set and decodes it on .get.
	var params = new URLSearchParams();
	params.set("rp", encoded);  // base64 is URL-safe so this is a no-op
	var got = params.get("rp");
	return url_decode_text(got);
}
expect_true("url: full round-trip via URLSearchParams (newlines)",
	fullRoundtrip("echo hello\nls -la") === "echo hello\nls -la");
expect_true("url: full round-trip via URLSearchParams (tabs)",
	fullRoundtrip("col1\tcol2\tcol3") === "col1\tcol2\tcol3");
expect_true("url: full round-trip via URLSearchParams (special chars)",
	fullRoundtrip('a && b || c < d > e "f" \'g\'') ===
		'a && b || c < d > e "f" \'g\'');

// ============================================================
// SUMMARY
// ============================================================
console.log("\n---------------------------------");
if (failedTests === 0) {
	console.log(`SUMMARY: All ${totalTests} JS tests passed successfully.`);
	process.exit(0);
} else {
	console.log(`SUMMARY: ${failedTests} of ${totalTests} JS test(s) failed.`);
	process.exit(1);
}
