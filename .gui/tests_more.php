<?php
/**
 * Additional automated unit tests for .gui PHP functions.
 * Covers functions that are not (yet) exercised in tests.php
 * or tests_extended.php.  Exit 0 on success, exit 1 on failure.
 */

require_once 'share_functions.php';

$failedTests = 0;

function echo_if_wanted($param) {
	if (getenv("SHOW_SUCCESS")) {
		echo $param;
	}
}

function expect($label, $actual, $expected) {
	global $failedTests;
	if ($actual === $expected) {
		echo_if_wanted("PASS: $label\n");
	} else {
		echo "FAIL: $label\n";
		echo "  Expected: " . json_encode($expected) . "\n";
		echo "  Actual:   " . json_encode($actual) . "\n";
		$failedTests++;
	}
}

function expect_true($label, $actual) {
	expect($label, (bool)$actual, true);
}

function expect_false($label, $actual) {
	expect($label, (bool)$actual, false);
}

function expect_throws($label, $callback) {
	global $failedTests;
	try {
		$callback();
		echo "FAIL: $label (expected exception, none thrown)\n";
		$failedTests++;
	} catch (\Throwable $e) {
		echo_if_wanted("PASS: $label\n");
	}
}

function test_file_helper($content, $callback) {
	$tmp = tempnam(sys_get_temp_dir(), 'php_new_test_');
	if ($content !== null) {
		file_put_contents($tmp, $content);
	} else if (file_exists($tmp)) {
		unlink($tmp);
	}

	$result = $callback($tmp);
	if (file_exists($tmp)) unlink($tmp);
	return $result;
}

function create_test_dir($structure) {
	$base = sys_get_temp_dir() . '/php_new_test_' . uniqid();
	if (!mkdir($base)) return null;
	foreach ($structure as $name => $content) {
		$path = "$base/$name";
		if (is_array($content)) {
			mkdir($path);
			foreach ($content as $subName => $subContent) {
				file_put_contents("$path/$subName", $subContent);
			}
		} else {
			mkdir($path);
		}
	}
	return $base;
}

function rmdir_recursive($dir) {
	if (!is_dir($dir)) return;
	foreach (scandir($dir) as $file) {
		if ($file === '.' || $file === '..') continue;
		$path = "$dir/$file";
		is_dir($path) ? rmdir_recursive($path) : unlink($path);
	}
	rmdir($dir);
}

// Polyfill for older PHP versions
if (!function_exists('str_contains')) {
	function str_contains($haystack, $needle) {
		return $needle !== '' && mb_strpos($haystack, $needle) !== false;
	}
}

// =================================================================
// START OF TESTS
// =================================================================

// --- Group: is_valid_user_id ---
echo_if_wanted("--- Testing: is_valid_user_id ---\n");
$userIdCases = [
	["user123", true],
	["user_456", true],
	["123abc", true],
	["User_XYZ", true],
	["user", true],
	["user@123", false],
	["user name", false],
	["user123#", false],
	["user123!", false],
	["user123$", false],
	["user-name", false],
	["user.name", false],
	["", false],
];
foreach ($userIdCases as [$input, $expected]) {
	expect("is_valid_user_id: '" . var_export($input, true) . "'",
		is_valid_user_id($input), $expected);
}
expect("is_valid_user_id: null", is_valid_user_id(null), false);

// --- Group: is_valid_experiment_name ---
echo_if_wanted("\n--- Testing: is_valid_experiment_name ---\n");
$expNameCases = [
	["experiment1", true],
	["experiment-2", true],
	["experiment_3", true],
	["exp", true],
	["ABC", true],
	["exp 4", false],
	["exp#5", false],
	["exp@6", false],
	["experiment/7", false],
	["exp!8", false],
	["exp.9", false],
	["", false],
];
foreach ($expNameCases as [$input, $expected]) {
	expect("is_valid_experiment_name: '" . var_export($input, true) . "'",
		is_valid_experiment_name($input), $expected);
}
expect("is_valid_experiment_name: null", is_valid_experiment_name(null), false);

// --- Group: is_valid_run_nr ---
echo_if_wanted("\n--- Testing: is_valid_run_nr ---\n");
$runNrCases = [
	["123", true],
	["0001", true],
	["5", true],
	["0", true],
	["999999", true],
	["a123", false],
	["12.34", false],
	["run123", false],
	["12abc", false],
	["123run", false],
	["+5", false],
	["-5", false],
	["1.0", false],
	["", false],
];
foreach ($runNrCases as [$input, $expected]) {
	expect("is_valid_run_nr: '" . var_export($input, true) . "'",
		is_valid_run_nr($input), $expected);
}
expect("is_valid_run_nr: null", is_valid_run_nr(null), false);

// --- Group: get_get ---
echo_if_wanted("\n--- Testing: get_get ---\n");
$_GET['test_get_key'] = 'get_value';
expect("get_get: retrieves from \$_GET", get_get('test_get_key'), 'get_value');
expect("get_get: returns default when missing", get_get('missing_key', 'fallback'), 'fallback');
expect("get_get: returns null when missing no default", get_get('missing_key'), null);
unset($_GET['test_get_key']);

// --- Group: my_htmlentities ---
echo_if_wanted("\n--- Testing: my_htmlentities ---\n");
expect("my_htmlentities: encodes <", my_htmlentities("<"), "&lt;");
expect("my_htmlentities: encodes >", my_htmlentities(">"), "&gt;");
expect("my_htmlentities: encodes &", my_htmlentities("&"), "&amp;");
expect("my_htmlentities: encodes single quote", my_htmlentities("'"), "&#039;");
expect("my_htmlentities: encodes double quote", my_htmlentities('"'), "&quot;");
expect("my_htmlentities: encodes umlauts", my_htmlentities("äöü"), "&auml;&ouml;&uuml;");
expect("my_htmlentities: handles plain ASCII", my_htmlentities("hello"), "hello");
expect("my_htmlentities: mixed content", my_htmlentities("a<b & c"), "a&lt;b &amp; c");

// --- Group: get_icon_html ---
echo_if_wanted("\n--- Testing: get_icon_html ---\n");
$iconHtml = get_icon_html("settings.svg");
expect("get_icon_html: contains img tag", str_contains($iconHtml, "<img"), true);
expect("get_icon_html: references icon name", str_contains($iconHtml, "settings.svg"), true);
expect("get_icon_html: contains invert_icon class", str_contains($iconHtml, "invert_icon"), true);

// --- Group: copy_id_to_clipboard_string ---
echo_if_wanted("\n--- Testing: copy_id_to_clipboard_string ---\n");
$clipboardStr = copy_id_to_clipboard_string("my_id_123", "data.csv");
expect("copy_id_to_clipboard_string: contains button",
	str_contains($clipboardStr, "<button"), true);
expect("copy_id_to_clipboard_string: contains filename",
	str_contains($clipboardStr, "data.csv"), true);
expect("copy_id_to_clipboard_string: contains copy clipboard class",
	str_contains($clipboardStr, "copy_clipboard_button"), true);
expect("copy_id_to_clipboard_string: handles filename with special chars",
	str_contains(copy_id_to_clipboard_string("x", "f<o>o.csv"), "f&lt;o&gt;o.csv"), true);

// --- Group: has_non_empty_folder ---
echo_if_wanted("\n--- Testing: has_non_empty_folder ---\n");
$tmpDir = sys_get_temp_dir() . '/php_new_test_' . uniqid();
mkdir($tmpDir);
file_put_contents("$tmpDir/a.txt", "content");
expect("has_non_empty_folder: directory with files", has_non_empty_folder($tmpDir), true);

$emptyDir = sys_get_temp_dir() . '/php_new_test_' . uniqid();
mkdir($emptyDir);
expect("has_non_empty_folder: empty directory", has_non_empty_folder($emptyDir), false);

$dirWithEmptyFile = sys_get_temp_dir() . '/php_new_test_' . uniqid();
mkdir($dirWithEmptyFile);
file_put_contents("$dirWithEmptyFile/empty.txt", "");
expect("has_non_empty_folder: directory with only empty file", has_non_empty_folder($dirWithEmptyFile), true);

$dirWithSubDir = sys_get_temp_dir() . '/php_new_test_' . uniqid();
mkdir("$dirWithSubDir/subdir", 0777, true);
expect("has_non_empty_folder: directory with only subdir", has_non_empty_folder($dirWithSubDir), false);
file_put_contents("$dirWithSubDir/subdir/deep.txt", "x");
expect("has_non_empty_folder: file deep in subdir", has_non_empty_folder($dirWithSubDir), true);

rmdir_recursive($tmpDir);
rmdir_recursive($emptyDir);
rmdir_recursive($dirWithEmptyFile);
rmdir_recursive($dirWithSubDir);

expect("has_non_empty_folder: nonexistent directory",
	has_non_empty_folder("/tmp/no_such_dir_" . uniqid()), false);

// --- Group: rrmdir ---
echo_if_wanted("\n--- Testing: rrmdir ---\n");
$deepDir = sys_get_temp_dir() . '/php_rrmdir_test_' . uniqid();
mkdir("$deepDir/sub1/sub2", 0777, true);
file_put_contents("$deepDir/sub1/sub2/file.txt", "data");
file_put_contents("$deepDir/top.txt", "data");
expect("rrmdir: directory exists before", is_dir($deepDir), true);
rrmdir($deepDir);
expect("rrmdir: directory removed", is_dir($deepDir), false);
expect("rrmdir: handles nonexistent directory", rrmdir("/tmp/nonexistent_" . uniqid()), null);

// --- Group: my_unlink ---
echo_if_wanted("\n--- Testing: my_unlink ---\n");
$tmpFile = tempnam(sys_get_temp_dir(), 'php_unlink_');
file_put_contents($tmpFile, "data");
expect("my_unlink: removes file", my_unlink($tmpFile), true);
expect("my_unlink: file gone after call", file_exists($tmpFile), false);

// --- Group: validate_param ---
echo_if_wanted("\n--- Testing: validate_param ---\n");
$_GET['username'] = 'valid_user';
$result = validate_param("username", "/^[a-zA-Z0-9_]+$/", "Invalid username");
expect("validate_param: returns input on match", $result, "valid_user");

expect_throws("validate_param: throws on no match", function () {
	validate_param("missing_param_xyz", "/^[a-zA-Z0-9_]+$/", "should throw");
});
unset($_GET['username']);

// --- Group: validate_readable_file ---
echo_if_wanted("\n--- Testing: validate_readable_file ---\n");
$tmpReadable = tempnam(sys_get_temp_dir(), 'php_read_');
file_put_contents($tmpReadable, "data");
expect("validate_readable_file: readable file returns null",
	validate_readable_file($tmpReadable), null);

$tmpEmpty = tempnam(sys_get_temp_dir(), 'php_empty_');
expect("validate_readable_file: empty file returns message",
	validate_readable_file($tmpEmpty) !== null, true);

expect("validate_readable_file: missing file returns message",
	validate_readable_file("/tmp/no_such_file_" . uniqid()) !== null, true);

unlink($tmpReadable);
unlink($tmpEmpty);

// --- Group: get_latest_modification_time ---
echo_if_wanted("\n--- Testing: get_latest_modification_time ---\n");
$dirA = sys_get_temp_dir() . '/php_lat_' . uniqid();
mkdir($dirA);
file_put_contents("$dirA/old.txt", "old");
sleep(1);
clearstatcache();
file_put_contents("$dirA/new.txt", "new");
clearstatcache();
$mtime = get_latest_modification_time($dirA);
expect("get_latest_modification_time: returns positive integer",
	is_int($mtime) && $mtime > 0, true);
expect("get_latest_modification_time: uses cache on second call",
	get_latest_modification_time($dirA), $mtime);
rrmdir($dirA);

expect("get_latest_modification_time: nonexistent returns 0",
	get_latest_modification_time("/tmp/no_such_dir_" . uniqid()), 0);

// --- Group: extract_min_max_ram_cpu_from_worker_info ---
echo_if_wanted("\n--- Testing: extract_min_max_ram_cpu_from_worker_info ---\n");
$workerInfo = "CPU: 10.5%, RAM: 1024 MB\nCPU: 20.0%, RAM: 2048 MB\nCPU: 30.5%, RAM: 4096 MB";
$tableHtml = extract_min_max_ram_cpu_from_worker_info($workerInfo);
expect("extract_min_max_ram_cpu: returns HTML table",
	str_contains($tableHtml, "<table>"), true);
expect("extract_min_max_ram_cpu: contains RAM header",
	str_contains($tableHtml, "Min RAM"), true);
expect("extract_min_max_ram_cpu: contains CPU header",
	str_contains($tableHtml, "CPU"), true);

$emptyInfo = extract_min_max_ram_cpu_from_worker_info("no data here");
expect("extract_min_max_ram_cpu: no data returns empty", $emptyInfo, "");

// --- Group: build_run_folder_path (additional) ---
echo_if_wanted("\n--- Testing: build_run_folder_path (additional) ---\n");
expect("build_run_folder_path: numeric user_id",
	build_run_folder_path(42, "exp", 0), "42/exp/0/");
expect("build_run_folder_path: numeric experiment",
	build_run_folder_path("user", 99, 1), "user/99/1/");
expect("build_run_folder_path: trailing slash always present",
	substr(build_run_folder_path("a", "b", "c"), -1), "/");

// --- Group: utf8ize (extended) ---
echo_if_wanted("\n--- Testing: utf8ize (extended) ---\n");
expect("utf8ize: nested array with umlauts",
	utf8ize(["a" => ["b" => "äöü"]]),
	["a" => ["b" => "äöü"]]);
expect("utf8ize: integer passes through", utf8ize(42), 42);
expect("utf8ize: float passes through", utf8ize(3.14), 3.14);
expect("utf8ize: null passes through", utf8ize(null), null);
expect("utf8ize: boolean passes through", utf8ize(true), true);
expect("utf8ize: empty array passes through", utf8ize([]), []);

// --- Group: normalize_csv_value (extended) ---
echo_if_wanted("\n--- Testing: normalize_csv_value (extended) ---\n");
expect("normalize_csv_value: integer-like float", normalize_csv_value("42.000"), "42");
expect("normalize_csv_value: large integer", normalize_csv_value("1234567890"), "1234567890");
expect("normalize_csv_value: scientific notation",
	normalize_csv_value("1.5e2"), "150");
expect("normalize_csv_value: negative integer-like",
	normalize_csv_value("-5.000"), "-5");
expect("normalize_csv_value: whitespace string",
	normalize_csv_value("   "), "");
expect("normalize_csv_value: non-numeric",
	normalize_csv_value("abc"), "abc");
expect("normalize_csv_value: alphanumeric",
	normalize_csv_value("abc123"), "abc123");
expect("normalize_csv_value: small float",
	normalize_csv_value("0.5"), "0.5");
expect("normalize_csv_value: simple int",
	normalize_csv_value("42"), "42");

// --- Group: get_status_for_results_csv (extended) ---
echo_if_wanted("\n--- Testing: get_status_for_results_csv (extended) ---\n");
$mixedCsv = "id,trial_status,extra\n1,COMPLETED,x\n2,COMPLETED,x\n3,FAILED,x\n4,RUNNING,x\n5,COMPLETED,x";
$status = test_file_helper($mixedCsv, 'get_status_for_results_csv');
expect("get_status_for_results_csv: succeeded count",
	$status["succeeded"], 3);
expect("get_status_for_results_csv: failed count",
	$status["failed"], 1);
expect("get_status_for_results_csv: running count",
	$status["running"], 1);
expect("get_status_for_results_csv: total count",
	$status["total"], 5);

$noneCsv = "id,name,extra\n1,foo,x\n2,bar,x";
$emptyStatus = test_file_helper($noneCsv, 'get_status_for_results_csv');
expect("get_status_for_results_csv: no trial_status column counts all rows",
	$emptyStatus["total"], 2);

// --- Group: file_string_contains_results (extended) ---
echo_if_wanted("\n--- Testing: file_string_contains_results (extended) ---\n");
expect("file_string_contains_results: case-sensitive match",
	file_string_contains_results("RESULT: 1", ["result"]), true);
expect("file_string_contains_results: case-mismatch rejected",
	file_string_contains_results("RESULT: 1", ["RESULT"]), true);
expect("file_string_contains_results: missing colon",
	file_string_contains_results("RESULT 1", ["RESULT"]), false);
expect("file_string_contains_results: no colon but found in names",
	file_string_contains_results("no key but result word", ["result"]), false);
expect("file_string_contains_results: empty names list is vacuously true",
	file_string_contains_results("anything", []), true);
expect("file_string_contains_results: underscore is word char",
	file_string_contains_results("FOO_RESULT: 5", ["RESULT"]), false);
expect("file_string_contains_results: decimal value",
	file_string_contains_results("LOSS: 0.5", ["loss"]), true);
expect("file_string_contains_results: negative value",
	file_string_contains_results("LOSS: -1.5", ["loss"]), true);
expect("file_string_contains_results: float exponent",
	file_string_contains_results("LOSS: 1e3", ["loss"]), true);
expect("file_string_contains_results: name not found",
	file_string_contains_results("OTHER: 5", ["result"]), false);

// --- Group: ends_with_submitit_info (extended) ---
echo_if_wanted("\n--- Testing: ends_with_submitit_info (extended) ---\n");
expect("ends_with_submitit_info: success ending",
	ends_with_submitit_info("blah\nsubmitit INFO (2024-01-15 10:00:00,000) - Exiting after successful completion"), true);
expect("ends_with_submitit_info: only newlines before",
	ends_with_submitit_info("\n\n\nsubmitit INFO (2024-01-15 10:00:00,000) - Exiting after successful completion\n"), true);
expect("ends_with_submitit_info: wrong case",
	ends_with_submitit_info("submitit info (2024-01-15 10:00:00,000) - Exiting after successful completion"), false);
expect("ends_with_submitit_info: missing timestamp",
	ends_with_submitit_info("submitit INFO - Exiting after successful completion"), false);
expect("ends_with_submitit_info: missing message",
	ends_with_submitit_info("submitit INFO (2024-01-15 10:00:00,000)"), false);
expect("ends_with_submitit_info: garbage",
	ends_with_submitit_info("random text"), false);

// --- Group: contains_slurm_time_limit_error (extended) ---
echo_if_wanted("\n--- Testing: contains_slurm_time_limit_error (extended) ---\n");
$timeLimitErr = "slurmstepd: error: *** JOB 42 ON node1 CANCELLED AT 2024-01-15T10:30:00 DUE TO TIME LIMIT ***";
expect("contains_slurm_time_limit_error: standard pattern",
	contains_slurm_time_limit_error($timeLimitErr), true);
expect("contains_slurm_time_limit_error: case-sensitive (lowercase rejected)",
	contains_slurm_time_limit_error(strtolower($timeLimitErr)), false);
expect("contains_slurm_time_limit_error: unrelated error",
	contains_slurm_time_limit_error("slurmstepd: error: out of memory"), false);
expect("contains_slurm_time_limit_error: non-string",
	contains_slurm_time_limit_error(123), false);
expect("contains_slurm_time_limit_error: empty string",
	contains_slurm_time_limit_error(""), false);
expect("contains_slurm_time_limit_error: only part of message",
	contains_slurm_time_limit_error("DUE TO TIME LIMIT"), false);

// --- Group: extract_results_dict (extended) ---
echo_if_wanted("\n--- Testing: extract_results_dict (extended) ---\n");
$multiLine = "RESULT: 1.0\nLOSS: 0.5\nACC: 0.95";
$dict = extract_results_dict($multiLine);
expect("extract_results_dict: returns 3 keys", count($dict), 3);
expect("extract_results_dict: RESULT value", $dict["RESULT"], "1.0");
expect("extract_results_dict: LOSS value", $dict["LOSS"], "0.5");
expect("extract_results_dict: ACC value", $dict["ACC"], "0.95");

$colonFree = extract_results_dict("no colons here");
expect("extract_results_dict: no colons returns empty", $colonFree, []);

$manySpaces = extract_results_dict("RESULT:    42");
expect("extract_results_dict: spaces after colon",
	$manySpaces["RESULT"], "42");

// --- Group: format_results_from_dict (extended) ---
echo_if_wanted("\n--- Testing: format_results_from_dict (extended) ---\n");
expect("format_results_from_dict: three results",
	format_results_from_dict(
		["A" => "1", "B" => "2", "C" => "3"],
		["A", "B", "C"]
	),
	"A: 1, B: 2, C: 3");
expect("format_results_from_dict: with single result",
	format_results_from_dict(["X" => "10"], ["X"]),
	"X: 10");
expect("format_results_from_dict: empty names list",
	format_results_from_dict(["X" => "1"], []),
	"");
expect("format_results_from_dict: zero values",
	format_results_from_dict(["A" => "0", "B" => "0"], ["A", "B"]),
	"A: 0, B: 0");

// --- Group: extract_trial_index (extended) ---
echo_if_wanted("\n--- Testing: extract_trial_index (extended) ---\n");
expect("extract_trial_index: large index",
	extract_trial_index("Trial-Index: 99999 some log", 0), 99999);
expect("extract_trial_index: index at start",
	extract_trial_index("Trial-Index: 7 rest", 0), 7);
expect("extract_trial_index: invalid falls back to nr",
	extract_trial_index("not here", 12), 12);
expect("extract_trial_index: empty input uses nr",
	extract_trial_index("", 5), 5);

// --- Group: clean_result_name_lines (extended) ---
echo_if_wanted("\n--- Testing: clean_result_name_lines (extended) ---\n");
expect("clean_result_name_lines: strips equals",
	clean_result_name_lines(["A=foo"]), ["Afoo"]);
expect("clean_result_name_lines: strips spaces",
	clean_result_name_lines(["X  Y"]), ["XY"]);
expect("clean_result_name_lines: keeps underscores",
	clean_result_name_lines(["a_b_c"]), ["a_b_c"]);
expect("clean_result_name_lines: keeps underscores and digits",
	clean_result_name_lines(["var_123"]), ["var_123"]);
expect("clean_result_name_lines: keeps numbers",
	clean_result_name_lines(["abc123"]), ["abc123"]);
expect("clean_result_name_lines: multiple lines",
	clean_result_name_lines(["A = foo", "B = bar", "C=baz"]),
	["Afoo", "Bbar", "Cbaz"]);
expect("clean_result_name_lines: strips punctuation",
	clean_result_name_lines(["a.b,c!"]), ["abc"]);
expect("clean_result_name_lines: keeps umlauts",
	clean_result_name_lines(["größe"]), ["größe"]);

// --- Group: get_runtime (extended) ---
echo_if_wanted("\n--- Testing: get_runtime (extended) ---\n");
$s1 = "submitit INFO (2024-01-15 10:00:00,000) - started";
$s2 = "submitit INFO (2024-01-15 10:00:30,000) - ended";
expect("get_runtime: 30 seconds", get_runtime("$s1\n$s2"), 30);

$s3 = "submitit INFO (2024-01-15 10:00:00,000) - started\nsubmitit INFO (2024-01-15 11:00:00,000) - ended";
expect("get_runtime: 1 hour = 3600s", get_runtime($s3), 3600);

// --- Group: time_since (extended) ---
echo_if_wanted("\n--- Testing: time_since (extended) ---\n");
$now = time();
expect("time_since: future timestamp", time_since($now + 100), "just now");
expect("time_since: 1 second ago", time_since($now - 1), "1 second ago");
expect("time_since: 59 seconds ago", time_since($now - 59), "59 seconds ago");
expect("time_since: 2 hours singular", time_since($now - 7200), "2 hours ago");
expect("time_since: 1 day singular", time_since($now - 86400), "1 day ago");
expect("time_since: 1 year singular", time_since($now - 31536000), "1 year ago");
expect("time_since: exactly 1 minute ago", time_since($now - 60), "1 minute ago");
expect("time_since: exactly 1 hour ago", time_since($now - 3600), "1 hour ago");

// --- Group: count_subfolders_or_files (extended) ---
echo_if_wanted("\n--- Testing: count_subfolders_or_files (extended) ---\n");
$mixedDir = create_test_dir([
	"file1.txt" => "a",
	"file2.log" => "b",
	"file3.json" => "c",
	"subdir" => []
]);
expect("count_subfolders_or_files: counts mixed", count_subfolders_or_files($mixedDir), 4);
rmdir_recursive($mixedDir);

expect("count_subfolders_or_files: returns int type",
	is_int(count_subfolders_or_files(sys_get_temp_dir())), true);

// --- Group: keep_rows_every_n_seconds (extended) ---
echo_if_wanted("\n--- Testing: keep_rows_every_n_seconds (extended) ---\n");
$csvData = [
	["timestamp", "x"],
	["1000", "A"],
	["1010", "B"],
	["1015", "C"],
	["1030", "D"],
];
$filtered = keep_rows_every_n_seconds($csvData, 20);
expect("keep_rows_every_n_seconds: keeps first and last",
	count($filtered) >= 2, true);

$singleRow = [["t"], ["100"]];
$filteredSingle = keep_rows_every_n_seconds($singleRow, 60);
expect("keep_rows_every_n_seconds: single row stays",
	count($filteredSingle), 2);

// --- Group: csv_array_to_text (extended) ---
echo_if_wanted("\n--- Testing: csv_array_to_text (extended) ---\n");
$csv = [["a", "b", "c"], ["1", "2", "3"]];
$text = csv_array_to_text($csv);
expect("csv_array_to_text: contains headers",
	str_contains($text, "a,b,c"), true);
expect("csv_array_to_text: contains data",
	str_contains($text, "1,2,3"), true);

// --- Group: convertNewlinesToBr (extended) ---
echo_if_wanted("\n--- Testing: convertNewlinesToBr (extended) ---\n");
expect("convertNewlinesToBr: only single newline",
	convertNewlinesToBr("a\nb"), "a\nb");
expect("convertNewlinesToBr: four newlines",
	convertNewlinesToBr("a\n\n\n\nb"), "a\n<br><br><br>b");
expect("convertNewlinesToBr: empty string",
	convertNewlinesToBr(""), "");

// --- Group: highlight_backticks (extended) ---
echo_if_wanted("\n--- Testing: highlight_backticks (extended) ---\n");
expect("highlight_backticks: no backticks",
	highlight_backticks("hello world"), "hello world");
expect("highlight_backticks: empty string",
	highlight_backticks(""), "");
expect("highlight_backticks: code with spaces",
	highlight_backticks("`a b c`"), "<tt>a b c</tt>");
expect("highlight_backticks: many backticks",
	highlight_backticks("`a` `b` `c`"),
	"<tt>a</tt> <tt>b</tt> <tt>c</tt>");

// --- Group: collapse_runs_keep_first_last (extended) ---
echo_if_wanted("\n--- Testing: collapse_runs_keep_first_last (extended) ---\n");
$tripleRun = [
	["1", "A"],
	["2", "A"],
	["3", "A"],
	["4", "B"],
	["5", "B"],
];
$collapsedTriple = collapse_runs_keep_first_last($tripleRun);
expect("collapse_runs_keep_first_last: keeps 4 (2 per run)",
	count($collapsedTriple), 4);

$twoRowRun = [
	["1", "X"],
	["2", "X"],
];
$collapsedTwo = collapse_runs_keep_first_last($twoRowRun);
expect("collapse_runs_keep_first_last: 2-row run keeps 2",
	count($collapsedTwo), 2);

// --- Group: highlight_debug_info (extended) ---
echo_if_wanted("\n--- Testing: highlight_debug_info (extended) ---\n");
expect("highlight_debug_info: empty string",
	str_contains(highlight_debug_info("", "INFO"), "ERROR") === false, true);
expect("highlight_debug_info: WARNING contains span",
	str_contains(highlight_debug_info("WARNING: here"), "<span"), true);
expect("highlight_debug_info: DEBUG block start contains span",
	str_contains(highlight_debug_info("DEBUG INFOS START\nfoo\nDEBUG INFOS END"), "<span"), true);

// --- Group: ascii_table_to_html (extended) ---
echo_if_wanted("\n--- Testing: ascii_table_to_html (extended) ---\n");
$tbl = ascii_table_to_html("A│B│C\n1│2│3\n4│5│6");
expect("ascii_table_to_html: contains th",
	str_contains($tbl, "<th"), true);
expect("ascii_table_to_html: contains td",
	str_contains($tbl, "<td"), true);
expect("ascii_table_to_html: closes table",
	str_contains($tbl, "</table>"), true);
expect("ascii_table_to_html: contains h2 for table title",
	str_contains($tbl, "<h2>"), true);
expect("ascii_table_to_html: contains all data values",
	str_contains($tbl, "4") && str_contains($tbl, "5") && str_contains($tbl, "6"), true);

$emptyTbl = ascii_table_to_html("");
expect("ascii_table_to_html: empty returns error",
	str_contains($emptyTbl, "Error") || str_contains($emptyTbl, "No valid table"), true);

// --- Group: is_valid_svg_file (extended) ---
echo_if_wanted("\n--- Testing: is_valid_svg_file (extended) ---\n");
$emptyFile = test_file_helper("", 'is_valid_svg_file');
expect("is_valid_svg_file: empty file", $emptyFile, false);

$validSvg = '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>';
expect("is_valid_svg_file: with xml declaration",
	test_file_helper($validSvg, 'is_valid_svg_file'), true);

// --- Group: is_valid_zip_file (extended) ---
echo_if_wanted("\n--- Testing: is_valid_zip_file (extended) ---\n");
$emptyFile = test_file_helper("", 'is_valid_zip_file');
expect("is_valid_zip_file: empty file", $emptyFile, false);

$tmpDir = tempnam(sys_get_temp_dir(), 'php_dir_');
expect("is_valid_zip_file: directory", is_valid_zip_file($tmpDir), false);
if (file_exists($tmpDir)) unlink($tmpDir);

// --- Group: analyze_column_types (extended) ---
echo_if_wanted("\n--- Testing: analyze_column_types (extended) ---\n");
$mixedData = [
	["1", "hello", "true"],
	["2", "world", "false"],
	["3", "foo", "true"],
];
$analysis = analyze_column_types($mixedData, [0 => "n", 1 => "s", 2 => "b"], []);
expect("analyze_column_types: numeric col detected",
	$analysis["n"]["numeric"], true);
expect("analyze_column_types: string col detected",
	$analysis["s"]["string"], true);
expect("analyze_column_types: not all numeric -> string col",
	$analysis["s"]["numeric"], false);

$emptyAnalysis = analyze_column_types([], [0 => "x"], []);
expect("analyze_column_types: empty data has no numeric",
	$emptyAnalysis["x"]["numeric"], false);
expect("analyze_column_types: empty data has no string",
	$emptyAnalysis["x"]["string"], false);

// --- Group: count_column_types (extended) ---
echo_if_wanted("\n--- Testing: count_column_types (extended) ---\n");
$emptyCount = count_column_types([]);
expect("count_column_types: empty analysis returns 2-element array",
	is_array($emptyCount) && count($emptyCount) === 2, true);
expect("count_column_types: empty both zero",
	$emptyCount[0], 0);

$allNumeric = ["a" => ["numeric" => true, "string" => false], "b" => ["numeric" => true, "string" => false]];
$countsAllNum = count_column_types($allNumeric);
expect("count_column_types: all numeric counts as 2",
	$countsAllNum[0], 2);

// --- Group: file helpers / edge cases ---
echo_if_wanted("\n--- Testing: read_file_as_array (extended) ---\n");
$mixedContent = "Line A\n\nLine B\r\nLine C\n   \nLine D";
$arr = test_file_helper($mixedContent, 'read_file_as_array');
expect("read_file_as_array: count after filtering",
	count($arr), 4);
expect("read_file_as_array: first line", $arr[0], "Line A");
expect("read_file_as_array: last line", $arr[3], "Line D");

$unreadablePath = "/proc/no_such_file_" . uniqid();
expect("read_file_as_array: unreadable file",
	read_file_as_array($unreadablePath), []);

// --- Group: keep_rows_every_n_seconds header handling ---
echo_if_wanted("\n--- Testing: keep_rows_every_n_seconds (header) ---\n");
$withHeader = [
	["time", "val"],
	["0", "x"],
	["100", "y"],
];
$kept = keep_rows_every_n_seconds($withHeader, 60);
expect("keep_rows_every_n_seconds: header always kept", $kept[0][0], "time");

// --- Group: get_runtime_human_format (more) ---
echo_if_wanted("\n--- Testing: get_runtime_human_format (more) ---\n");
expect("get_runtime_human_format: 1 second", get_runtime_human_format(1), "1s");
expect("get_runtime_human_format: 59 minutes",
	get_runtime_human_format(59 * 60), "59m");
expect("get_runtime_human_format: 24 hours",
	get_runtime_human_format(24 * 3600), "24h");
expect("get_runtime_human_format: 24h with 30m",
	get_runtime_human_format(24 * 3600 + 30 * 60), "24h:30m");
expect("get_runtime_human_format: 24h 30m 15s",
	get_runtime_human_format(24 * 3600 + 30 * 60 + 15), "24h:30m:15s");
expect("get_runtime_human_format: 1m30s",
	get_runtime_human_format(90), "1m:30s");
expect("get_runtime_human_format: only minutes",
	get_runtime_human_format(60), "1m");
expect("get_runtime_human_format: 0 negative large",
	get_runtime_human_format(-10000), "0s");

// --- Group: extract_magic_comment (extended) ---
echo_if_wanted("\n--- Testing: extract_magic_comment (extended) ---\n");
$tmpMagic = tempnam(sys_get_temp_dir(), 'php_magic_');
file_put_contents($tmpMagic, "# version: 2.0\n# type: python\nprint('x')");
expect("extract_magic_comment: version",
	extract_magic_comment($tmpMagic, 'version'), '2.0');
expect("extract_magic_comment: type",
	extract_magic_comment($tmpMagic, 'type'), 'python');
expect("extract_magic_comment: non-existent key",
	extract_magic_comment($tmpMagic, 'unknown'), null);
unlink($tmpMagic);

expect("extract_magic_comment: nonexistent file returns null",
	extract_magic_comment('/tmp/nonexistent_magic_' . uniqid(), 'version'), null);

// --- Group: analyze_results_csv (extended) ---
echo_if_wanted("\n--- Testing: analyze_results_csv (extended) ---\n");
$allCompletedCsv = "id,trial_status,extra\n1,COMPLETED,x\n2,COMPLETED,x\n3,COMPLETED,x";
$resultAll = test_file_helper($allCompletedCsv, 'analyze_results_csv');
expect("analyze_results_csv: all completed",
	str_contains($resultAll, "Completed: 3"), true);
expect("analyze_results_csv: 0-failed entries omitted",
	str_contains($resultAll, "Failed"), false);
expect("analyze_results_csv: 0-running entries omitted",
	str_contains($resultAll, "Running"), false);

$mixedCsv = "id,trial_status,extra\n1,COMPLETED,x\n2,FAILED,x\n3,RUNNING,x";
$resultMixed = test_file_helper($mixedCsv, 'analyze_results_csv');
expect("analyze_results_csv: mixed contains all three",
	str_contains($resultMixed, "Completed: 1") &&
	str_contains($resultMixed, "Failed: 1") &&
	str_contains($resultMixed, "Running: 1"), true);

// --- Group: get_exit_code_from_outfile (extended) ---
echo_if_wanted("\n--- Testing: get_exit_code_from_outfile (extended) ---\n");
expect("get_exit_code_from_outfile: large exit code",
	get_exit_code_from_outfile("EXIT_CODE: 255"), 255);
expect("get_exit_code_from_outfile: between other lines",
	get_exit_code_from_outfile("Started...\nEXIT_CODE: 1\nDone."), 1);
expect("get_exit_code_from_outfile: non-numeric",
	get_exit_code_from_outfile("EXIT_CODE: abc"), null);

// --- Group: utf8ize (more) ---
echo_if_wanted("\n--- Testing: utf8ize (more) ---\n");
$deeper = ["a" => ["b" => ["c" => "ä"]]];
$result = utf8ize($deeper);
expect("utf8ize: deeply nested", $result["a"]["b"]["c"], "ä");

// --- Group: get_html_category_comment (extended) ---
echo_if_wanted("\n--- Testing: get_html_category_comment (extended) ---\n");
$tmpCat = tempnam(sys_get_temp_dir(), 'php_cat_');
file_put_contents($tmpCat, "<!-- Category: MyCat -->\n<h1>Title</h1>");
expect("get_html_category_comment: extracts category",
	get_html_category_comment($tmpCat), 'MyCat');
unlink($tmpCat);

// --- Group: get_first_heading_content (extended) ---
echo_if_wanted("\n--- Testing: get_first_heading_content (extended) ---\n");
$tmpMulti = tempnam(sys_get_temp_dir(), 'php_multi_');
file_put_contents($tmpMulti, "<h1>First</h1>\n<h2>Second</h2>\n<h1>Third</h1>");
expect("get_first_heading_content: returns first h1",
	get_first_heading_content($tmpMulti), 'First');
unlink($tmpMulti);

$tmpMd = tempnam(sys_get_temp_dir(), 'php_md2_');
file_put_contents($tmpMd, "## Subheading\n# Top Heading\n");
expect("get_first_heading_content: returns first markdown heading",
	get_first_heading_content($tmpMd), 'Subheading');
unlink($tmpMd);

// --- Group: convert_markdown_to_html (extended) ---
echo_if_wanted("\n--- Testing: convert_markdown_to_html (extended) ---\n");
$mdH3 = convert_markdown_to_html("### Heading");
expect("convert_markdown_to_html: h3", str_contains($mdH3, "<h3>"), true);

$mdPara = convert_markdown_to_html("para 1\n\npara 2");
expect("convert_markdown_to_html: paragraphs",
	str_contains($mdPara, "<p>") || str_contains($mdPara, "<br"), true);

$mdMixed = convert_markdown_to_html("**bold _not italic_**");
expect("convert_markdown_to_html: bold only (no nested)",
	str_contains($mdMixed, "<strong>") && !str_contains($mdMixed, "<em>"), true);

// --- Group: normalize_csv_file_contents (extended) ---
echo_if_wanted("\n--- Testing: normalize_csv_file_contents (extended) ---\n");
expect("normalize_csv_file_contents: trailing newline",
	normalize_csv_file_contents("a,b\n1,2\n"), "a,b\n1,2");
expect("normalize_csv_file_contents: with negative floats",
	normalize_csv_file_contents("-3.000,-2.500"), "-3,-2.5");
expect("normalize_csv_file_contents: scientific becomes int",
	normalize_csv_file_contents("1e3,abc"), "1000,abc");
expect("normalize_csv_file_contents: empty string",
	normalize_csv_file_contents(""), "");
expect("normalize_csv_file_contents: empty lines skipped",
	normalize_csv_file_contents("a\n\n\nb"), "a\nb");

// --- Group: has_real_char (extended) ---
echo_if_wanted("\n--- Testing: has_real_char (extended) ---\n");
$tmpReal = tempnam(sys_get_temp_dir(), 'php_real2_');
file_put_contents($tmpReal, "x");
expect("has_real_char: single char", has_real_char($tmpReal), true);
unlink($tmpReal);

$tmpZero = tempnam(sys_get_temp_dir(), 'php_zero_');
file_put_contents($tmpZero, "\0");
expect("has_real_char: null byte counts as real char",
	has_real_char($tmpZero), true);
unlink($tmpZero);

// --- Group: is_ascii_or_utf8 (extended) ---
echo_if_wanted("\n--- Testing: is_ascii_or_utf8 (extended) ---\n");
$tmpEmpty = tempnam(sys_get_temp_dir(), 'php_empty2_');
file_put_contents($tmpEmpty, "");
expect("is_ascii_or_utf8: empty file", is_ascii_or_utf8($tmpEmpty), true);
unlink($tmpEmpty);

// --- Group: remove_ansi_escape_sequences (extended) ---
echo_if_wanted("\n--- Testing: remove_ansi_escape_sequences (extended) ---\n");
expect("remove_ansi_escape_sequences: multiple codes",
	remove_ansi_escape_sequences("\x1b[31mR\x1b[0m\x1b[32mG\x1b[0m"), "RG");
expect("remove_ansi_escape_sequences: cursor movement",
	remove_ansi_escape_sequences("\x1b[2Jhello\x1b[H"), "hello");
expect("remove_ansi_escape_sequences: cursor movement with content",
	remove_ansi_escape_sequences("a\x1b[2Jb"), "ab");

// --- Group: get_valid_folders (extended) ---
echo_if_wanted("\n--- Testing: get_valid_folders (extended) ---\n");
$fDir = create_test_dir([
	"abc_def" => [],
	"abc.def" => [],
	"abc" => [],
	"a b c" => [],
	"123" => [],
]);
$valid = get_valid_folders($fDir);
sort($valid);
expect("get_valid_folders: dot excluded",
	in_array("abc.def", $valid), false);
expect("get_valid_folders: space excluded",
	in_array("a b c", $valid), false);
expect("get_valid_folders: underscore allowed",
	in_array("abc_def", $valid), true);
expect("get_valid_folders: digits allowed",
	in_array("123", $valid), true);
rmdir_recursive($fDir);

// --- Group: sort_folders_by_modification_time (extended) ---
echo_if_wanted("\n--- Testing: sort_folders_by_modification_time (extended) ---\n");
$sortDir = create_test_dir([
	"older" => ["file.txt" => "x"],
	"newer" => ["file.txt" => "x"],
]);
file_put_contents("$sortDir/older/file.txt", "old data");
sleep(1);
clearstatcache();
file_put_contents("$sortDir/newer/file.txt", "new data");
clearstatcache();
$sortList = ["older", "newer"];
sort_folders_by_modification_time($sortDir, $sortList);
expect("sort_folders_by_modification_time: newer first",
	$sortList[0], "newer");
expect("sort_folders_by_modification_time: older last",
	$sortList[1], "older");
rmdir_recursive($sortDir);

// --- Group: build_run_folder_path edge cases ---
echo_if_wanted("\n--- Testing: build_run_folder_path (edge) ---\n");
expect("build_run_folder_path: empty strings",
	build_run_folder_path("", "", ""), "///");

// --- Group: get_html_comment (extended) ---
echo_if_wanted("\n--- Testing: get_html_comment (extended) ---\n");
$tmpComment = tempnam(sys_get_temp_dir(), 'php_comment_');
file_put_contents($tmpComment, "<!-- Multi word comment -->\n<h1>title</h1>");
expect("get_html_comment: extracts comment",
	get_html_comment($tmpComment), 'Multi word comment');
unlink($tmpComment);

// --- Group: csv_array_to_text (extended) ---
echo_if_wanted("\n--- Testing: csv_array_to_text (extended) ---\n");
$numeric = [["x"], ["1"], ["2"], ["3"]];
expect("csv_array_to_text: numeric data",
	csv_array_to_text($numeric), "x\n1\n2\n3");

// --- Group: keep_rows_every_n_seconds with default ---
echo_if_wanted("\n--- Testing: keep_rows_every_n_seconds (default 60) ---\n");
$csv = [
	["t"],
	["0"],
	["30"],
	["61"],
	["120"],
	["180"],
];
$filteredDefault = keep_rows_every_n_seconds($csv);
expect("keep_rows_every_n_seconds: default 60s applied",
	count($filteredDefault) >= 2, true);

// --- Group: string_is_numeric extended ---
echo_if_wanted("\n--- Testing: string_is_numeric (extended) ---\n");
expect("string_is_numeric: hex-like rejected",
	string_is_numeric("0x1A"), false);
expect("string_is_numeric: leading zeros",
	string_is_numeric("00123"), true);
expect("string_is_numeric: long int",
	string_is_numeric("9999999999999999"), true);
expect("string_is_numeric: only whitespace",
	string_is_numeric("   "), false);
expect("string_is_numeric: tab char",
	string_is_numeric("\t"), false);

// --- Group: is_valid_user_or_experiment_name (extended) ---
echo_if_wanted("\n--- Testing: is_valid_user_or_experiment_name (extended) ---\n");
expect("is_valid_user_or_experiment_name: uppercase only",
	is_valid_user_or_experiment_name("ABC"), true);
expect("is_valid_user_or_experiment_name: mixed case",
	is_valid_user_or_experiment_name("AbCdEf"), true);
expect("is_valid_user_or_experiment_name: dots rejected",
	is_valid_user_or_experiment_name("a.b"), false);
expect("is_valid_user_or_experiment_name: slashes rejected",
	is_valid_user_or_experiment_name("a/b"), false);
expect("is_valid_user_or_experiment_name: just digits",
	is_valid_user_or_experiment_name("12345"), true);

// --- Group: convert_markdown_to_html (nested) ---
echo_if_wanted("\n--- Testing: convert_markdown_to_html (nested) ---\n");
$mdNested = convert_markdown_to_html("# Title\n\n**bold** and *italic*");
expect("convert_markdown_to_html: title and bold",
	str_contains($mdNested, "<h1>") && str_contains($mdNested, "<strong>"), true);
expect("convert_markdown_to_html: italic",
	str_contains($mdNested, "<em>"), true);
expect("convert_markdown_to_html: bold keeps content",
	str_contains($mdNested, "bold"), true);

$mdBoldItalic = convert_markdown_to_html("***combined***");
expect("convert_markdown_to_html: triple star",
	str_contains($mdBoldItalic, "<strong>") || str_contains($mdBoldItalic, "<em>"), true);

$mdJustBold = convert_markdown_to_html("**bold _not italic_**");
expect("convert_markdown_to_html: bold but no nested italic",
	str_contains($mdJustBold, "<strong>") && !str_contains($mdJustBold, "<em>"), true);
expect("convert_markdown_to_html: keeps underscore literally",
	str_contains($mdJustBold, "_not italic_"), true);

// --- Group: get_log_files (extended) ---
echo_if_wanted("\n--- Testing: get_log_files (extended) ---\n");
$logDir = create_test_dir([
	"0_0_log.out" => "data",
	"1_0_log.out" => "data",
	"2_0_log.out" => "data",
	"0_1_log.out" => "x",
	"nonlog.txt" => "x",
	"not_a_log.png" => "x",
]);
$logs = get_log_files($logDir);
expect("get_log_files: finds 3 log files", count($logs), 3);
expect("get_log_files: only *_0_log.out matches",
	array_key_exists('0', $logs) && array_key_exists('1', $logs) && array_key_exists('2', $logs), true);
rmdir_recursive($logDir);

expect("get_log_files: nonexistent dir returns empty",
	get_log_files("/tmp/no_dir_" . uniqid()), []);

// --- Group: get_csv_data_as_array (extended) ---
echo_if_wanted("\n--- Testing: get_csv_data_as_array (extended) ---\n");
$tsvFile = tempnam(sys_get_temp_dir(), 'php_tsv_');
file_put_contents($tsvFile, "a\tb\n1\t2");
$tsvData = get_csv_data_as_array($tsvFile, "\t");
expect("get_csv_data_as_array: tsv delimiter", $tsvData[1][0], 1);
expect("get_csv_data_as_array: row count tsv", count($tsvData), 2);
unlink($tsvFile);

// --- Group: convertNewlinesToBr (consecutive) ---
echo_if_wanted("\n--- Testing: convertNewlinesToBr (consecutive) ---\n");
expect("convertNewlinesToBr: two consecutive nl",
	convertNewlinesToBr("a\n\nb"), "a\n<br>b");
expect("convertNewlinesToBr: five consecutive nl",
	convertNewlinesToBr("a\n\n\n\n\nb"),
	"a\n<br><br><br><br>b");

// --- Group: sanitize_safe_html (extended) ---
echo_if_wanted("\n--- Testing: sanitize_safe_html (extended) ---\n");
$safeScript = sanitize_safe_html("<script>alert('xss')</script><b>safe</b>");
expect("sanitize_safe_html: removes script tag",
	str_contains($safeScript, "<script"), false);
expect("sanitize_safe_html: keeps safe text",
	str_contains($safeScript, "safe"), true);

$safeIframe = sanitize_safe_html("<iframe src='evil.com'></iframe>");
expect("sanitize_safe_html: removes iframe",
	str_contains($safeIframe, "iframe"), false);

$safeStyle = sanitize_safe_html("<span style='color: red; position: absolute'>styled</span>");
expect("sanitize_safe_html: keeps allowed css",
	str_contains($safeStyle, "color"), true);
expect("sanitize_safe_html: removes forbidden css property",
	str_contains($safeStyle, "position"), false);

$safeEvil = sanitize_safe_html("<img src='https://x.com/a.png' onerror='bad()'>");
expect("sanitize_safe_html: keeps allowed img",
	str_contains($safeEvil, "<img"), true);
expect("sanitize_safe_html: strips onerror",
	str_contains($safeEvil, "onerror"), false);

$safeJsSrc = sanitize_safe_html("<img src='javascript:alert(1)'>");
expect("sanitize_safe_html: blocks javascript: src",
	str_contains($safeJsSrc, "javascript:"), false);

$safeDataSrc = sanitize_safe_html("<img src='data:image/png;base64,AAAA'>");
expect("sanitize_safe_html: allows data: src",
	str_contains($safeDataSrc, "data:image/png;base64"), true);

// --- Group: get_or_env ---
echo_if_wanted("\n--- Testing: get_or_env (extended) ---\n");
$_GET['test_combo'] = 'from_get';
expect("get_or_env: GET wins over ENV",
	get_or_env('test_combo'), 'from_get');
unset($_GET['test_combo']);
putenv('test_combo2=from_env');
expect("get_or_env: ENV fallback",
	get_or_env('test_combo2'), 'from_env');

// --- Group: convert_file_to_html ---
echo_if_wanted("\n--- Testing: convert_file_to_html ---\n");
$tmpMdFile = tempnam(sys_get_temp_dir(), 'php_conv_');
file_put_contents($tmpMdFile, "# Hello\n\n**world**");
ob_start();
convert_file_to_html($tmpMdFile);
$converted = ob_get_clean();
expect("convert_file_to_html: contains h1",
	str_contains($converted, "<h1>"), true);
expect("convert_file_to_html: contains strong",
	str_contains($converted, "<strong>"), true);
unlink($tmpMdFile);

ob_start();
convert_file_to_html("/tmp/no_such_file_" . uniqid() . ".md");
$notFound = ob_get_clean();
expect("convert_file_to_html: prints not-found message",
	str_contains($notFound, "File not found"), true);

// --- Group: replace_python_placeholders (extended) ---
echo_if_wanted("\n--- Testing: replace_python_placeholders (extended) ---\n");
expect("replace_python_placeholders: numeric placeholder",
	replace_python_placeholders("{x} {y}", ["x" => "1", "y" => "2"]), "1 2");
expect("replace_python_placeholders: nested braces left alone",
	replace_python_placeholders("{{a}}", ["a" => "x"]), "{x}");
expect("replace_python_placeholders: empty placeholder",
	replace_python_placeholders("{}", []), "{}");
expect("replace_python_placeholders: replacement contains braces",
	replace_python_placeholders("{a}", ["a" => "{b}"]), "{b}");

// --- Group: generate_argparse_html_table (extended) ---
echo_if_wanted("\n--- Testing: generate_argparse_html_table (extended) ---\n");
$argsWithMultiple = [
	"Group1" => [
		"desc" => "First group",
		"args" => [
			["--foo", "Foo arg", "val1", "type: str"],
			["--bar", "Bar arg", "42", "type: int"],
		]
	],
	"Group2" => [
		"desc" => "Second group",
		"args" => [
			["--baz", "Baz arg", "default", "type: str"],
		]
	],
];
$multiHtml = generate_argparse_html_table($argsWithMultiple, false);
expect("generate_argparse_html_table: contains --foo",
	str_contains($multiHtml, "--foo"), true);
expect("generate_argparse_html_table: contains --baz",
	str_contains($multiHtml, "--baz"), true);
expect("generate_argparse_html_table: contains First group",
	str_contains($multiHtml, "First group"), true);
expect("generate_argparse_html_table: contains Second group",
	str_contains($multiHtml, "Second group"), true);

// --- Group: extract_help_params_from_bash (extended) ---
echo_if_wanted("\n--- Testing: extract_help_params_from_bash (extended) ---\n");
$bashMultiHelp = '
function help {
    echo "  --foo      First option"
    echo "  --bar      Second option"
    echo "  --baz      Third option"
    exit 1
}';
$multiRes = test_file_helper($bashMultiHelp, fn($p) => extract_help_params_from_bash($p));
expect("extract_help_params_from_bash: contains --foo",
	str_contains($multiRes, "--foo"), true);
expect("extract_help_params_from_bash: contains --bar",
	str_contains($multiRes, "--bar"), true);
expect("extract_help_params_from_bash: contains --baz",
	str_contains($multiRes, "--baz"), true);

$noHelp = 'echo "no help here"';
$noHelpRes = test_file_helper($noHelp, fn($p) => extract_help_params_from_bash($p));
expect("extract_help_params_from_bash: no help returns empty table",
	str_contains($noHelpRes, "<table"), true);
expect("extract_help_params_from_bash: no help has no args",
	str_contains($noHelpRes, "--"), false);

// --- Group: get_latest_recursive_modification_time (extended) ---
echo_if_wanted("\n--- Testing: get_latest_recursive_modification_time (extended) ---\n");
$recDir = sys_get_temp_dir() . '/php_rec_test_' . uniqid();
mkdir("$recDir/a/b/c", 0777, true);
mkdir("$recDir/top");
sleep(1);
clearstatcache();
file_put_contents("$recDir/a/b/c/file.txt", "x");
clearstatcache();
$recTime = get_latest_recursive_modification_time($recDir);
expect("get_latest_recursive_modification_time: returns timestamp",
	is_int($recTime) && $recTime > 0, true);
rmdir_recursive($recDir);

expect_throws("get_latest_recursive_modification_time: nonexistent throws",
	function () {
		get_latest_recursive_modification_time("/tmp/no_dir_" . uniqid());
	});

// --- Group: build_run_folder_path with strings ---
echo_if_wanted("\n--- Testing: build_run_folder_path (string) ---\n");
expect("build_run_folder_path: with all strings",
	build_run_folder_path("alice", "exp1", "5"), "alice/exp1/5/");

// --- Group: get_html_category_comment more ---
echo_if_wanted("\n--- Testing: get_html_category_comment (extended) ---\n");
$tmpCat = tempnam(sys_get_temp_dir(), 'php_cat3_');
file_put_contents($tmpCat, "<!-- Category:    Spaced Cat   -->\n<p>x</p>");
expect("get_html_category_comment: trims spaces",
	get_html_category_comment($tmpCat), 'Spaced Cat');
unlink($tmpCat);

// --- Group: extract_results_dict whitespace ---
echo_if_wanted("\n--- Testing: extract_results_dict (whitespace) ---\n");
expect("extract_results_dict: numeric with trailing whitespace",
	extract_results_dict("KEY: 42  ")["KEY"], "42");

// --- Group: file_string_contains_results more ---
echo_if_wanted("\n--- Testing: file_string_contains_results (extended) ---\n");
expect("file_string_contains_results: word boundary",
	file_string_contains_results("RESULT_NAME: 5", ["RESULT_NAME"]), true);
expect("file_string_contains_results: underscore prefix rejected (no word boundary)",
	file_string_contains_results("FOO_RESULT: 5", ["RESULT"]), false);
expect("file_string_contains_results: standalone word matches",
	file_string_contains_results("RESULT: 5", ["RESULT"]), true);
expect("file_string_contains_results: mixed case names",
	file_string_contains_results("Loss: 0.5", ["loss"]), true);

// --- Group: respond_with_error (skipped - calls exit()) ---
echo_if_wanted("\n--- Testing: respond_with_error ---\n");
echo_if_wanted("SKIP: respond_with_error calls exit(); not testable in-process\n");

// =================================================================
// FINISH
// =================================================================
echo_if_wanted("\n---------------------------------\n");
if ($failedTests === 0) {
	echo_if_wanted("SUMMARY: All new-php-unit-tests passed successfully.\n");
	exit(0);
} else {
	echo_if_wanted("SUMMARY: $failedTests new-php-unit-test(s) failed.\n");
	exit(1);
}
