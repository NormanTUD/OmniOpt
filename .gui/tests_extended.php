<?php
/**
 * Extended automated unit tests for .gui PHP functions.
 * Covers functions not tested in tests.php.
 * Exit 0 on success, exit 1 on failure.
 */

require_once 'share_functions.php';

$failedTests = 0;

function echo_if_wanted($param) {
	if(getenv("SHOW_SUCCESS")) {
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
	$tmp = tempnam(sys_get_temp_dir(), 'php_ext_test_');
	if ($content !== null) file_put_contents($tmp, $content);
	else if (file_exists($tmp)) unlink($tmp);

	$result = $callback($tmp);
	if (file_exists($tmp)) unlink($tmp);
	return $result;
}

function create_test_dir($structure) {
	$base = sys_get_temp_dir() . '/php_ext_test_' . uniqid();
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

// =================================================================
// START OF TESTS
// =================================================================

// --- Group: get_post / get_server ---
echo_if_wanted("--- Testing: get_post / get_server ---\n");
$_POST['test_key'] = 'post_value';
expect("get_post: retrieves from \$_POST", get_post('test_key'), 'post_value');
expect("get_post: returns default when missing", get_post('missing_key', 'default'), 'default');
expect("get_post: returns null when missing no default", get_post('missing_key'), null);
unset($_POST['test_key']);

$_SERVER['test_server_key'] = 'server_value';
expect("get_server: retrieves from \$_SERVER", get_server('test_server_key'), 'server_value');
expect("get_server: returns default when missing", get_server('missing_key', 'fallback'), 'fallback');
unset($_SERVER['test_server_key']);

// --- Group: get_html_category_comment ---
echo_if_wanted("\n--- Testing: get_html_category_comment ---\n");
$tmpHtml = tempnam(sys_get_temp_dir(), 'php_html_');
file_put_contents($tmpHtml, '<!-- Category: My Custom Category -->');
expect("get_html_category_comment: extracts category", get_html_category_comment($tmpHtml), 'My Custom Category');
unlink($tmpHtml);

$tmpNoCat = tempnam(sys_get_temp_dir(), 'php_html_');
file_put_contents($tmpNoCat, '<p>No category here</p>');
expect("get_html_category_comment: returns null when absent", get_html_category_comment($tmpNoCat), null);
unlink($tmpNoCat);

// get_html_category_comment/get_html_comment use file_get_contents without exists check
// so they throw with the project's error handler on nonexistent files

// --- Group: get_html_comment ---
echo_if_wanted("\n--- Testing: get_html_comment ---\n");
$tmpHtml2 = tempnam(sys_get_temp_dir(), 'php_html_');
file_put_contents($tmpHtml2, '<!-- This is a comment -->');
expect("get_html_comment: extracts comment", get_html_comment($tmpHtml2), 'This is a comment');
unlink($tmpHtml2);

// --- Group: get_first_heading_content ---
echo_if_wanted("\n--- Testing: get_first_heading_content ---\n");
$tmpH1 = tempnam(sys_get_temp_dir(), 'php_html_');
file_put_contents($tmpH1, "<p>text</p>\n<h1>First Heading</h1>\n<h2>Second</h2>");
expect("get_first_heading_content: extracts h1", get_first_heading_content($tmpH1), 'First Heading');
unlink($tmpH1);

$tmpMd = tempnam(sys_get_temp_dir(), 'php_md_');
file_put_contents($tmpMd, "# Markdown Heading\nSome text");
expect("get_first_heading_content: extracts markdown heading", get_first_heading_content($tmpMd), 'Markdown Heading');
unlink($tmpMd);

$tmpNoH = tempnam(sys_get_temp_dir(), 'php_html_');
file_put_contents($tmpNoH, "<p>No heading here</p>");
expect("get_first_heading_content: returns null when no heading", get_first_heading_content($tmpNoH), null);
unlink($tmpNoH);

// --- Group: replace_python_placeholders ---
echo_if_wanted("\n--- Testing: replace_python_placeholders ---\n");
expect("replace_python_placeholders: simple replacement", replace_python_placeholders("Hello {name}!", ['name' => 'World']), "Hello World!");
expect("replace_python_placeholders: multiple replacements", replace_python_placeholders("{a} and {b}", ['a' => 'X', 'b' => 'Y']), "X and Y");
expect("replace_python_placeholders: keeps unmatched", replace_python_placeholders("{a} {b}", ['a' => 'X']), "X {b}");
expect("replace_python_placeholders: empty input", replace_python_placeholders("", []), "");
expect("replace_python_placeholders: no placeholders", replace_python_placeholders("no placeholders", []), "no placeholders");
expect_throws("replace_python_placeholders: throws on non-string", function() { replace_python_placeholders(123, []); });
expect_throws("replace_python_placeholders: throws on non-array replacements", function() { replace_python_placeholders("test", "not_array"); });

// --- Group: extract_magic_comment ---
echo_if_wanted("\n--- Testing: extract_magic_comment ---\n");
$tmpPy = tempnam(sys_get_temp_dir(), 'php_py_');
file_put_contents($tmpPy, "# version: 1.2.3\n# other: value\nprint('hello')");
expect("extract_magic_comment: finds magic comment", extract_magic_comment($tmpPy, 'version'), '1.2.3');
unlink($tmpPy);

$tmpNoMagic = tempnam(sys_get_temp_dir(), 'php_py_');
file_put_contents($tmpNoMagic, "# just a comment\nprint('hello')");
expect("extract_magic_comment: returns null when absent", extract_magic_comment($tmpNoMagic, 'version'), null);
unlink($tmpNoMagic);

expect("extract_magic_comment: returns null for nonexistent", extract_magic_comment('/tmp/nonexistent_xyz_' . uniqid(), 'key'), null);

// --- Group: convert_markdown_to_html ---
echo_if_wanted("\n--- Testing: convert_markdown_to_html ---\n");
$md1 = convert_markdown_to_html("# Heading 1");
expect("convert_markdown_to_html: h1", str_contains($md1, "<h1>Heading 1</h1>"), true);

$md2 = convert_markdown_to_html("## Heading 2");
expect("convert_markdown_to_html: h2", str_contains($md2, "<h2>Heading 2</h2>"), true);

$md3 = convert_markdown_to_html("**bold text**");
expect("convert_markdown_to_html: bold", str_contains($md3, "<strong>bold text</strong>"), true);

$md4 = convert_markdown_to_html("*italic text*");
expect("convert_markdown_to_html: italic", str_contains($md4, "<em>italic"), true);

$md5 = convert_markdown_to_html("`inline code`");
expect("convert_markdown_to_html: inline code", str_contains($md5, "<code "), true);

$md6 = convert_markdown_to_html("[link](https://example.com)");
expect("convert_markdown_to_html: link", str_contains($md6, "href=\"https://example.com\""), true);

$md7 = convert_markdown_to_html("- item 1\n- item 2");
expect("convert_markdown_to_html: list items", str_contains($md7, "<li>"), true);

$md8 = convert_markdown_to_html("```bash\necho hello\n```");
expect("convert_markdown_to_html: bash codeblock", str_contains($md8, "language-bash"), true);

$md9 = convert_markdown_to_html("```python\nprint('hi')\n```");
expect("convert_markdown_to_html: python codeblock", str_contains($md9, "language-python"), true);

$md10 = convert_markdown_to_html("```json\n{\"key\": \"val\"}\n```");
expect("convert_markdown_to_html: json codeblock", str_contains($md10, "language-json"), true);

$md11 = convert_markdown_to_html("simple text without formatting");
expect("convert_markdown_to_html: plain text", str_contains($md11, "simple text"), true);

$md12 = convert_markdown_to_html("![alt text](image.png)");
expect("convert_markdown_to_html: image becomes link (link regex runs first)", str_contains($md12, "href=\"image.png\""), true);

// --- Group: csv_array_to_text ---
echo_if_wanted("\n--- Testing: csv_array_to_text ---\n");
$csv1 = [["h1","h2"],["a","b"],["c","d"]];
expect("csv_array_to_text: basic", csv_array_to_text($csv1), "h1,h2\na,b\nc,d");

$csv2 = [["single"]];
expect("csv_array_to_text: single cell", csv_array_to_text($csv2), "single");

$csv3 = [];
expect("csv_array_to_text: empty", csv_array_to_text($csv3), "");

// --- Group: normalize_csv_file_contents ---
echo_if_wanted("\n--- Testing: normalize_csv_file_contents ---\n");
expect("normalize_csv_file_contents: normalizes floats", normalize_csv_file_contents("10.000,hello\n20.500,world"), "10,hello\n20.5,world");
expect("normalize_csv_file_contents: preserves strings", normalize_csv_file_contents("abc,def"), "abc,def");
expect("normalize_csv_file_contents: skips empty lines", normalize_csv_file_contents("a,b\n\nc,d"), "a,b\nc,d");

// --- Group: remove_ansi_colors ---
echo_if_wanted("\n--- Testing: remove_ansi_colors ---\n");
expect("remove_ansi_colors: strips ANSI codes", remove_ansi_colors("\033[31mHello\033[0m"), "Hello");
expect("remove_ansi_colors: plain text", remove_ansi_colors("no codes here"), "no codes here");
expect("remove_ansi_colors: empty string", remove_ansi_colors(""), "");

// --- Group: has_real_char ---
echo_if_wanted("\n--- Testing: has_real_char ---\n");
$tmpReal = tempnam(sys_get_temp_dir(), 'php_real_');
file_put_contents($tmpReal, "has content");
expect("has_real_char: file with content", has_real_char($tmpReal), true);
unlink($tmpReal);

$tmpEmpty = tempnam(sys_get_temp_dir(), 'php_empty_');
file_put_contents($tmpEmpty, "");
expect("has_real_char: empty file", has_real_char($tmpEmpty), false);
unlink($tmpEmpty);

$tmpWhitespace = tempnam(sys_get_temp_dir(), 'php_ws_');
file_put_contents($tmpWhitespace, "  \n\t  ");
expect("has_real_char: whitespace only", has_real_char($tmpWhitespace), false);
unlink($tmpWhitespace);

expect("has_real_char: nonexistent file", has_real_char('/tmp/nonexistent_xyz_' . uniqid()), false);

// --- Group: is_ascii_or_utf8 ---
echo_if_wanted("\n--- Testing: is_ascii_or_utf8 ---\n");
$tmpAscii = tempnam(sys_get_temp_dir(), 'php_asc_');
file_put_contents($tmpAscii, "Hello ASCII");
expect("is_ascii_or_utf8: ASCII file", is_ascii_or_utf8($tmpAscii), true);
unlink($tmpAscii);

$tmpUtf = tempnam(sys_get_temp_dir(), 'php_utf_');
file_put_contents($tmpUtf, "Héllo Wörld Ñ");
expect("is_ascii_or_utf8: UTF-8 file", is_ascii_or_utf8($tmpUtf), true);
unlink($tmpUtf);

$tmpNonUtf = tempnam(sys_get_temp_dir(), 'php_nu8_');
file_put_contents($tmpNonUtf, "\x80\x81\x82");
expect("is_ascii_or_utf8: invalid bytes", is_ascii_or_utf8($tmpNonUtf), false);
unlink($tmpNonUtf);

expect("is_ascii_or_utf8: nonexistent file", is_ascii_or_utf8('/tmp/nonexistent_xyz_' . uniqid()), false);

// --- Group: analyze_results_csv ---
echo_if_wanted("\n--- Testing: analyze_results_csv ---\n");
$tmpCsv = tempnam(sys_get_temp_dir(), 'php_csv_');
file_put_contents($tmpCsv, "id,trial_status\n1,COMPLETED\n2,FAILED\n3,RUNNING\n4,COMPLETED");
$analysis = analyze_results_csv($tmpCsv);
expect("analyze_results_csv: counts completed", str_contains($analysis, "Completed: 2"), true);
expect("analyze_results_csv: counts failed", str_contains($analysis, "Failed: 1"), true);
expect("analyze_results_csv: counts running", str_contains($analysis, "Running: 1"), true);
unlink($tmpCsv);

$tmpNoCsv = tempnam(sys_get_temp_dir(), 'php_csv_');
file_put_contents($tmpNoCsv, "id,name\n1,test");
expect("analyze_results_csv: no trial_status column", analyze_results_csv($tmpNoCsv), "");
unlink($tmpNoCsv);

expect("analyze_results_csv: nonexistent file", analyze_results_csv('/tmp/nonexistent_xyz_' . uniqid()), "");

// --- Group: get_exit_code_from_outfile ---
echo_if_wanted("\n--- Testing: get_exit_code_from_outfile ---\n");
expect("get_exit_code_from_outfile: finds exit code", get_exit_code_from_outfile("some output\nEXIT_CODE: 42\nmore"), 42);
expect("get_exit_code_from_outfile: exit code 0", get_exit_code_from_outfile("EXIT_CODE: 0"), 0);
expect("get_exit_code_from_outfile: no exit code", get_exit_code_from_outfile("just some text"), null);
expect("get_exit_code_from_outfile: empty string", get_exit_code_from_outfile(""), null);

// --- Group: get_runtime ---
echo_if_wanted("\n--- Testing: get_runtime ---\n");
expect("get_runtime: empty string", get_runtime(""), null);
expect("get_runtime: null", get_runtime(null), null);
expect("get_runtime: no timestamps", get_runtime("just some text"), 0);

$s1 = "submitit INFO (2024-01-15 10:00:00,000) - started";
$s2 = "submitit INFO (2024-01-15 10:05:00,000) - ended";
expect("get_runtime: computes diff", get_runtime("$s1\n$s2"), 300);

// --- Group: get_runtime_human_format ---
echo_if_wanted("\n--- Testing: get_runtime_human_format ---\n");
expect("get_runtime_human_format: 0 seconds", get_runtime_human_format(0), "0s");
expect("get_runtime_human_format: negative", get_runtime_human_format(-5), "0s");
expect("get_runtime_human_format: 45 seconds", get_runtime_human_format(45), "45s");
expect("get_runtime_human_format: 5 minutes", get_runtime_human_format(300), "5m");
expect("get_runtime_human_format: 1h30m", get_runtime_human_format(5400), "1h:30m");
expect("get_runtime_human_format: 2h15m30s", get_runtime_human_format(8130), "2h:15m:30s");

// --- Group: ends_with_submitit_info ---
echo_if_wanted("\n--- Testing: ends_with_submitit_info ---\n");
expect("ends_with_submitit_info: valid", ends_with_submitit_info("submitit INFO (2024-01-15 10:00:00,123) - Exiting after successful completion"), true);
expect("ends_with_submitit_info: missing timestamp", ends_with_submitit_info("submitit INFO - Exiting after successful completion"), false);
expect("ends_with_submitit_info: wrong ending", ends_with_submitit_info("submitit INFO (2024-01-15 10:00:00,123) - Something else"), false);
expect("ends_with_submitit_info: empty string", ends_with_submitit_info(""), false);

// --- Group: contains_slurm_time_limit_error ---
echo_if_wanted("\n--- Testing: contains_slurm_time_limit_error ---\n");
$slurmErr = "slurmstepd: error: *** JOB 12345 ON taurusi8001 CANCELLED AT 2024-01-15T10:30:00 DUE TO TIME LIMIT ***";
expect("contains_slurm_time_limit_error: valid error", contains_slurm_time_limit_error($slurmErr), true);
expect("contains_slurm_time_limit_error: different error", contains_slurm_time_limit_error("slurmstepd: error: something else"), false);
expect("contains_slurm_time_limit_error: empty string", contains_slurm_time_limit_error(""), false);
expect("contains_slurm_time_limit_error: non-string", contains_slurm_time_limit_error(null), false);

// --- Group: extract_results_dict ---
echo_if_wanted("\n--- Testing: extract_results_dict ---\n");
$dict1 = extract_results_dict("RESULT: 3.14\nLOSS: 0.5");
expect("extract_results_dict: two values", $dict1['RESULT'], '3.14');
expect("extract_results_dict: second value", $dict1['LOSS'], '0.5');

$dict2 = extract_results_dict("no results here");
expect("extract_results_dict: no matches", $dict2, []);

$dict3 = extract_results_dict("X: 10\nX: 20");
expect("extract_results_dict: duplicate overwrites", $dict3['X'], '20');

// --- Group: format_results_from_dict ---
echo_if_wanted("\n--- Testing: format_results_from_dict ---\n");
expect("format_results_from_dict: basic", format_results_from_dict(['RESULT' => '3.14'], ['RESULT']), 'RESULT: 3.14');
expect("format_results_from_dict: missing name", format_results_from_dict(['RESULT' => '3.14'], ['MISSING']), '');
expect("format_results_from_dict: multiple", format_results_from_dict(['A' => '1', 'B' => '2'], ['A', 'B']), 'A: 1, B: 2');
expect("format_results_from_dict: large int formatting", format_results_from_dict(['RESULT' => '1234567'], ['RESULT']), 'RESULT: 1,234,567');
expect("format_results_from_dict: empty dict", format_results_from_dict([], ['RESULT']), '');

// --- Group: extract_trial_index ---
echo_if_wanted("\n--- Testing: extract_trial_index ---\n");
expect("extract_trial_index: finds index", extract_trial_index("Trial-Index: 42 some log", 0), 42);
expect("extract_trial_index: fallback to nr", extract_trial_index("no index here", 5), 5);
expect("extract_trial_index: no index no nr", extract_trial_index("no index", 0), 0);

// --- Group: clean_result_name_lines ---
echo_if_wanted("\n--- Testing: clean_result_name_lines ---\n");
expect("clean_result_name_lines: cleans", clean_result_name_lines(["RESULT=min", "LOSS  =max"]), ["RESULTmin", "LOSSmax"]);
expect("clean_result_name_lines: empty", clean_result_name_lines([]), []);
expect("clean_result_name_lines: preserves alphanumeric", clean_result_name_lines(["abc123_"]), ["abc123_"]);

// --- Group: is_valid_user_or_experiment_name ---
echo_if_wanted("\n--- Testing: is_valid_user_or_experiment_name ---\n");
expect("is_valid_user_or_experiment_name: valid", is_valid_user_or_experiment_name("user123"), true);
expect("is_valid_user_or_experiment_name: with underscore", is_valid_user_or_experiment_name("user_name"), true);
expect("is_valid_user_or_experiment_name: with hyphen", is_valid_user_or_experiment_name("user-name"), true);
expect("is_valid_user_or_experiment_name: with space", is_valid_user_or_experiment_name("user name"), false);
expect("is_valid_user_or_experiment_name: empty", is_valid_user_or_experiment_name(""), false);

// --- Group: string_is_numeric ---
echo_if_wanted("\n--- Testing: string_is_numeric ---\n");
expect("string_is_numeric: digits", string_is_numeric("12345"), true);
expect("string_is_numeric: zero", string_is_numeric("0"), true);
expect("string_is_numeric: alpha", string_is_numeric("abc"), false);
expect("string_is_numeric: mixed", string_is_numeric("12abc"), false);
expect("string_is_numeric: float string", string_is_numeric("12.34"), false);

// --- Group: time_since ---
echo_if_wanted("\n--- Testing: time_since ---\n");
$now = time();
expect("time_since: just now", time_since($now), "just now");
expect("time_since: 2 minutes ago", time_since($now - 120), "2 minutes ago");
expect("time_since: 1 hour ago", time_since($now - 3600), "1 hour ago");
expect("time_since: 3 days ago", time_since($now - 259200), "3 days ago");
expect("time_since: 1 month ago (singular)", time_since($now - 30 * 86400), "1 month ago");
expect("time_since: 2 months ago", time_since($now - 60 * 86400), "2 months ago");

// --- Group: count_subfolders_or_files ---
echo_if_wanted("\n--- Testing: count_subfolders_or_files ---\n");
$testDir = create_test_dir(["file1.txt" => "a", "file2.txt" => "b", "subdir" => []]);
expect("count_subfolders_or_files: counts 3", count_subfolders_or_files($testDir), 3);
rmdir_recursive($testDir);

expect("count_subfolders_or_files: nonexistent dir", count_subfolders_or_files('/tmp/nonexistent_xyz_' . uniqid()), 0);

$emptyDir = create_test_dir([]);
expect("count_subfolders_or_files: empty dir", count_subfolders_or_files($emptyDir), 0);
rmdir_recursive($emptyDir);

// --- Group: get_csv_data_as_array ---
echo_if_wanted("\n--- Testing: get_csv_data_as_array ---\n");
$tmpCsvArr = tempnam(sys_get_temp_dir(), 'php_csva_');
file_put_contents($tmpCsvArr, "name,age\nAlice,30\nBob,25");
$csvArr = get_csv_data_as_array($tmpCsvArr);
expect("get_csv_data_as_array: row count", count($csvArr), 3);
expect("get_csv_data_as_array: header preserved", $csvArr[0][0], "name");
expect("get_csv_data_as_array: numeric conversion", $csvArr[1][1], 30);
unlink($tmpCsvArr);

$tmpCsvCustom = tempnam(sys_get_temp_dir(), 'php_csvc_');
file_put_contents($tmpCsvCustom, "a|b\n1|2");
$csvCustom = get_csv_data_as_array($tmpCsvCustom, "|");
expect("get_csv_data_as_array: custom delimiter", $csvCustom[1][0], 1);
unlink($tmpCsvCustom);

$tmpCsvHeader = tempnam(sys_get_temp_dir(), 'php_csvh_');
file_put_contents($tmpCsvHeader, "1,2,3");
$csvHeader = get_csv_data_as_array($tmpCsvHeader, ",", ["col_a", "col_b", "col_c"]);
expect("get_csv_data_as_array: custom header", $csvHeader[0][0], "col_a");
unlink($tmpCsvHeader);

expect("get_csv_data_as_array: nonexistent file", get_csv_data_as_array('/tmp/nonexistent_xyz_' . uniqid()), []);

// --- Group: collapse_runs_keep_first_last ---
echo_if_wanted("\n--- Testing: collapse_runs_keep_first_last ---\n");
$rows1 = [
    ["1", "A", "same"],
    ["2", "A", "same"],
    ["3", "A", "same"],
    ["4", "B", "diff"],
    ["5", "B", "diff"],
];
$collapsed1 = collapse_runs_keep_first_last($rows1);
expect("collapse_runs_keep_first_last: keeps first+last of each run", count($collapsed1), 4);
expect("collapse_runs_keep_first_last: first A kept", $collapsed1[0][2], "same");
expect("collapse_runs_keep_first_last: last A kept", $collapsed1[1][2], "same");
expect("collapse_runs_keep_first_last: first B kept", $collapsed1[2][2], "diff");
expect("collapse_runs_keep_first_last: last B kept", $collapsed1[3][2], "diff");

$singleRow = [["1", "X", "solo"]];
$collapsed2 = collapse_runs_keep_first_last($singleRow);
expect("collapse_runs_keep_first_last: single row", count($collapsed2), 1);

$emptyRows = [];
$collapsed3 = collapse_runs_keep_first_last($emptyRows);
expect("collapse_runs_keep_first_last: empty", count($collapsed3), 0);

// --- Group: remove_ansi_escape_sequences ---
echo_if_wanted("\n--- Testing: remove_ansi_escape_sequences ---\n");
expect("remove_ansi_escape_sequences: strips CSI sequences", remove_ansi_escape_sequences("\x1b[31mRed\x1b[0m"), "Red");
expect("remove_ansi_escape_sequences: plain text", remove_ansi_escape_sequences("hello world"), "hello world");
expect("remove_ansi_escape_sequences: preserves sixel", remove_ansi_escape_sequences("\x1bPqSIXEL\x1b\\"), "\x1bPqSIXEL\x1b\\");
expect("remove_ansi_escape_sequences: empty", remove_ansi_escape_sequences(""), "");

// --- Group: highlight_debug_info ---
echo_if_wanted("\n--- Testing: highlight_debug_info ---\n");
expect("highlight_debug_info: wraps errors", str_contains(highlight_debug_info("E1234 some error"), "<span"), true);
expect("highlight_debug_info: wraps WARNING", str_contains(highlight_debug_info("WARNING: test"), "<span"), true);
expect("highlight_debug_info: wraps INFO", str_contains(highlight_debug_info("INFO stuff"), "<span"), true);
expect("highlight_debug_info: wraps DEBUG block", str_contains(highlight_debug_info("DEBUG INFOS START\nFile: test\nDEBUG INFOS END"), "background-color"), true);

// --- Group: ascii_table_to_html ---
echo_if_wanted("\n--- Testing: ascii_table_to_html ---\n");
$asciiTable = "Name│Age\nAlice│30\nBob│25";
$htmlTable = ascii_table_to_html($asciiTable);
expect("ascii_table_to_html: contains table", str_contains($htmlTable, "<table"), true);
expect("ascii_table_to_html: contains header", str_contains($htmlTable, "Name"), true);
expect("ascii_table_to_html: contains data", str_contains($htmlTable, "Alice"), true);
expect("ascii_table_to_html: contains data", str_contains($htmlTable, "Bob"), true);

// --- Group: is_valid_zip_file ---
echo_if_wanted("\n--- Testing: is_valid_zip_file ---\n");
$tmpZip = tempnam(sys_get_temp_dir(), 'php_zip_');
file_put_contents($tmpZip, "PK\x03\x04random content");
expect("is_valid_zip_file: valid signature", is_valid_zip_file($tmpZip), true);
unlink($tmpZip);

$tmpNotZip = tempnam(sys_get_temp_dir(), 'php_nzip_');
file_put_contents($tmpNotZip, "Not a zip file at all");
expect("is_valid_zip_file: invalid signature", is_valid_zip_file($tmpNotZip), false);
unlink($tmpNotZip);

expect("is_valid_zip_file: nonexistent", is_valid_zip_file('/tmp/nonexistent_xyz_' . uniqid()), false);

// --- Group: get_log_files ---
echo_if_wanted("\n--- Testing: get_log_files ---\n");
$logDir = create_test_dir(["0_0_log.out" => "data", "1_0_log.out" => "data", "notalog.txt" => "data"]);
$logFiles = get_log_files($logDir);
expect("get_log_files: finds log files", count($logFiles), 2);
expect("get_log_files: correct keys", array_key_exists('0', $logFiles) && array_key_exists('1', $logFiles), true);
rmdir_recursive($logDir);

expect("get_log_files: nonexistent dir", get_log_files('/tmp/nonexistent_xyz_' . uniqid()), []);

// --- Group: remove_extra_slashes_from_url (additional) ---
echo_if_wanted("\n--- Testing: remove_extra_slashes_from_url (additional) ---\n");
expect("remove_extra_slashes_from_url: no change needed", remove_extra_slashes_from_url("http://example.com/path"), "http://example.com/path");
expect("remove_extra_slashes_from_url: doubles", remove_extra_slashes_from_url("http://example.com//path"), "http://example.com/path");
expect("remove_extra_slashes_from_url: protocol preserved", remove_extra_slashes_from_url("https://x.com//a//b"), "https://x.com/a/b");

// --- Group: analyze_column_types ---
echo_if_wanted("\n--- Testing: analyze_column_types ---\n");
$colData = [["1", "hello"], ["2", "world"], ["3", "test"]];
$colAnalysis = analyze_column_types($colData, [0 => "num_col", 1 => "str_col"], []);
expect("analyze_column_types: numeric detected", $colAnalysis["num_col"]["numeric"], true);
expect("analyze_column_types: string detected", $colAnalysis["num_col"]["string"], false);
expect("analyze_column_types: string column", $colAnalysis["str_col"]["string"], true);
expect("analyze_column_types: string not numeric", $colAnalysis["str_col"]["numeric"], false);

$specialColAnalysis = analyze_column_types($colData, [0 => "col0"], ["col0"]);
expect("analyze_column_types: skips special col", isset($specialColAnalysis["col0"]), false);

// --- Group: count_column_types ---
echo_if_wanted("\n--- Testing: count_column_types ---\n");
$analysis1 = ["a" => ["numeric" => true, "string" => false], "b" => ["numeric" => false, "string" => true]];
$counts1 = count_column_types($analysis1);
expect("count_column_types: counts numerical", $counts1[0], 1);
expect("count_column_types: counts string", $counts1[1], 1);

$analysis2 = ["a" => ["numeric" => true, "string" => true]];
$counts2 = count_column_types($analysis2);
expect("count_column_types: mixed type counts as string", $counts2[1], 1);

// --- Group: remove_extra_slashes_from_url ---
echo_if_wanted("\n--- Testing: remove_extra_slashes_from_url ---\n");
expect("remove_extra_slashes_from_url: trailing slashes", remove_extra_slashes_from_url("http://example.com///"), "http://example.com/");
expect("remove_extra_slashes_from_url: no slashes to remove", remove_extra_slashes_from_url("http://example.com"), "http://example.com");

// --- Group: clean_result_name_lines (edge cases) ---
echo_if_wanted("\n--- Testing: clean_result_name_lines (edge cases) ---\n");
expect("clean_result_name_lines: removes spaces", clean_result_name_lines(["RESULT = min"]), ["RESULTmin"]);
expect("clean_result_name_lines: removes special chars", clean_result_name_lines(["a@b#c"]), ["abc"]);

// --- Group: get_runtime_human_format (edge cases) ---
echo_if_wanted("\n--- Testing: get_runtime_human_format (edge cases) ---\n");
expect("get_runtime_human_format: exactly 1 minute", get_runtime_human_format(60), "1m:0s");
expect("get_runtime_human_format: exactly 1 hour", get_runtime_human_format(3600), "1h:0s");
expect("get_runtime_human_format: 59 seconds", get_runtime_human_format(59), "59s");
expect("get_runtime_human_format: large number", get_runtime_human_format(3661), "1h:1m:1s");

// --- Group: extract_results_dict (edge cases) ---
echo_if_wanted("\n--- Testing: extract_results_dict (edge cases) ---\n");
$dictNeg = extract_results_dict("RESULT: -3.14");
expect("extract_results_dict: negative number", $dictNeg['RESULT'], '-3.14');
$dictInt = extract_results_dict("SCORE: 42");
expect("extract_results_dict: integer", $dictInt['SCORE'], '42');

// --- Group: file_string_contains_results ---
echo_if_wanted("\n--- Testing: file_string_contains_results ---\n");
expect("file_string_contains_results: found", file_string_contains_results("RESULT: 3.14\nmore stuff", ["result"]), true);
expect("file_string_contains_results: not found", file_string_contains_results("no results here", ["result"]), false);
expect("file_string_contains_results: multiple names", file_string_contains_results("RESULT: 3.14\nLOSS: 0.5", ["result", "loss"]), true);
expect("file_string_contains_results: empty string", file_string_contains_results("", ["result"]), false);
expect("file_string_contains_results: partial name match rejected", file_string_contains_results("RESULTX: 5", ["result"]), false);

// --- Group: format_results_from_dict (edge cases) ---
echo_if_wanted("\n--- Testing: format_results_from_dict (edge cases) ---\n");
expect("format_results_from_dict: float value", format_results_from_dict(['LOSS' => '0.123'], ['LOSS']), 'LOSS: 0.123');
expect("format_results_from_dict: negative large int", format_results_from_dict(['SCORE' => '-1234567'], ['SCORE']), 'SCORE: -1,234,567');

// --- Group: convertNewlinesToBr (additional) ---
echo_if_wanted("\n--- Testing: convertNewlinesToBr (additional) ---\n");
expect("convertNewlinesToBr: no newlines", convertNewlinesToBr("hello"), "hello");
expect("convertNewlinesToBr: single newline", convertNewlinesToBr("line1\nline2"), "line1\nline2");
expect("convertNewlinesToBr: triple newline", convertNewlinesToBr("a\n\n\nb"), "a\n<br><br>b");

// --- Group: generate_argparse_html_table ---
echo_if_wanted("\n--- Testing: generate_argparse_html_table ---\n");
$emptyArgs = generate_argparse_html_table([], true);
expect("generate_argparse_html_table: empty with no_msg", $emptyArgs, "");
$emptyArgsNoMsg = generate_argparse_html_table([], false);
expect("generate_argparse_html_table: empty without no_msg", str_contains($emptyArgsNoMsg, "No arguments found"), true);

$validArgs = ["Group" => ["desc" => "Test group", "args" => [["--test", "Test arg", "default_val", "type: str"]]]];
$htmlArgs = generate_argparse_html_table($validArgs, false);
expect("generate_argparse_html_table: has table", str_contains($htmlArgs, "<table"), true);
expect("generate_argparse_html_table: has arg name", str_contains($htmlArgs, "--test"), true);

// --- Group: add_tabs_to_string ---
echo_if_wanted("\n--- Testing: add_tabs_to_string ---\n");
expect("add_tabs_to_string: adds tabs", add_tabs_to_string("line1\nline2", 1), "\tline1\n\tline2");
expect("add_tabs_to_string: preserves pre blocks", add_tabs_to_string("<pre>\ncode\n</pre>", 2), "<pre>\ncode\n</pre>");
expect("add_tabs_to_string: zero tabs", add_tabs_to_string("line1", 0), "line1");

// --- Group: remove_font_face_rules ---
echo_if_wanted("\n--- Testing: remove_font_face_rules ---\n");
$css1 = "@font-face { font-family: test; }\n.body { color: red; }";
expect("remove_font_face_rules: removes font-face", remove_font_face_rules($css1), "\n.body { color: red; }");
expect("remove_font_face_rules: no font-face", remove_font_face_rules(".body { color: red; }"), ".body { color: red; }");

// --- Group: remove_excessive_newlines ---
echo_if_wanted("\n--- Testing: remove_excessive_newlines ---\n");
expect("remove_excessive_newlines: reduces triple", remove_excessive_newlines("a\n\n\nb"), "a\nb");
expect("remove_excessive_newlines: keeps double", remove_excessive_newlines("a\n\nb"), "a\n\nb");
expect("remove_excessive_newlines: reduces quad", remove_excessive_newlines("a\n\n\n\nb"), "a\nb");

// --- Group: check_and_filter_tabs ---
echo_if_wanted("\n--- Testing: check_and_filter_tabs ---\n");
$tabs1 = ["Results" => [], "Logs" => [], "Errors" => []];
[$filteredTabs1, $warn1] = check_and_filter_tabs("Logs", $tabs1, []);
expect("check_and_filter_tabs: filters matching tab", isset($filteredTabs1["Logs"]), false);
expect("check_and_filter_tabs: keeps non-matching", isset($filteredTabs1["Results"]), true);

$tabs2 = ["Tab A" => [], "Tab B" => []];
[$filteredTabs2, $warn2] = check_and_filter_tabs("(A|B)", $tabs2, []);
expect("check_and_filter_tabs: regex filter", count($filteredTabs2), 0);

expect_throws("check_and_filter_tabs: invalid regex", function() {
	check_and_filter_tabs("!!!invalid!!!", [], []);
});

// =================================================================
// FINISH
// =================================================================
echo_if_wanted("\n---------------------------------\n");
if ($failedTests === 0) {
	echo_if_wanted("SUMMARY: All extended tests passed successfully.\n");
	exit(0);
} else {
	echo_if_wanted("SUMMARY: $failedTests extended test(s) failed.\n");
	exit(1);
}
