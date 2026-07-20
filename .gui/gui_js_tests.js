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
