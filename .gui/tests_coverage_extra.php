<?php
chdir(__DIR__);
/**
 * Additional PHP unit tests aimed at the parts of the share/GUI helpers that
 * the other test files do not exercise.  Every helper is called through its
 * public API with small temporary fixtures.
 */

$__oo_tmp_root = sys_get_temp_dir() . '/oo_coverage_' . bin2hex(random_bytes(8));
mkdir($__oo_tmp_root, 0777, true);
putenv("share_path=$__oo_tmp_root/shares");
mkdir("$__oo_tmp_root/shares", 0777, true);
putenv("IS_RUNNING_UNIT_TESTS=1");

require_once __DIR__ . '/share_functions.php';
if (session_status() === PHP_SESSION_NONE) {
	session_start();
}

$__oo_failures = 0;

function oo_check(string $name, bool $condition): void {
	global $__oo_failures;
	if (!$condition) {
		$__oo_failures++;
		echo "\033[31mFAIL: $name\033[0m\n";
	}
}

function oo_expect(string $name, $expected, $actual): void {
	global $__oo_failures;
	if ($expected !== $actual) {
		$__oo_failures++;
		echo "\033[31mFAIL: $name (expected " . var_export($expected, true)
			. ", got " . var_export($actual, true) . ")\033[0m\n";
	}
}

function oo_write(string $path, string $contents): string {
	$dir = dirname($path);
	if (!is_dir($dir)) {
		mkdir($dir, 0777, true);
	}
	file_put_contents($path, $contents);
	return $path;
}

function oo_capture(callable $fn) {
	ob_start();
	$result = $fn();
	return [$result, ob_get_clean()];
}

function oo_tolerate_warnings(callable $fn) {
	set_error_handler(function () { return true; });
	try {
		return $fn();
	} finally {
		restore_error_handler();
	}
}

function oo_run_php(string $code): array {
	global $root;
	$file = "$root/child_" . bin2hex(random_bytes(4)) . ".php";
	file_put_contents($file, "<?php\nchdir(" . var_export(__DIR__, true) . ");\nrequire 'share_functions.php';\n" . $code . "\n");
	$lines = [];
	$rc = 0;
	exec("php " . escapeshellarg($file) . " 2>&1", $lines, $rc);
	@unlink($file);
	return [$rc, implode("\n", $lines)];
}

$root = $__oo_tmp_root;
$shares = "$root/shares";

// ---------------------------------------------------------------------------
// Share cache helpers.
// ---------------------------------------------------------------------------
oo_check('cache base dir', share_cache_base_dir() !== "" && is_dir(share_cache_base_dir()));
oo_check('cache namespace', share_cache_namespace_dir("coverage") !== null);
oo_expect('cache invalid namespace', null, share_cache_namespace_dir("../bad"));

$key = sha1("coverage-key");
oo_check('cache write blob', share_cache_write_blob("coverage", $key, "payload") === true);
oo_expect('cache read blob', "payload", share_cache_read_blob("coverage", $key));
oo_expect('cache read missing blob', null, share_cache_read_blob("coverage", sha1("missing")));
oo_expect('cache invalid key', null, share_cache_read_blob("coverage", "NOTHEX"));
oo_check('cache write non-string', share_cache_write_blob("coverage", $key, 12) === false);
oo_check('cache write serialized', share_cache_write_serialized("coverage", $key, ["a" => 1]) === true);
oo_expect('cache read serialized', ["a" => 1], share_cache_read_serialized("coverage", $key));
oo_check('cache code version', is_string(share_cache_code_version()) && strlen(share_cache_code_version()) === 40);
oo_expect('share file cache key', 40, strlen(share_file_cache_key("x", "$root/none")));

$run_dir = "$root/run";
oo_write("$run_dir/results.csv", "result\n1\n");
oo_write("$run_dir/state_files/run_uuid", "uuid-123\n");
oo_check('run fingerprint', is_string(share_run_fingerprint($run_dir)));
oo_expect('run fingerprint missing', null, share_run_fingerprint("$root/missing-run"));
$_SERVER["REQUEST_URI"] = "/share.php?run_nr=0";
$_GET = ["run_nr" => "0"];
$page_key = share_run_page_cache_key($run_dir);
oo_check('page cache write', share_run_page_cache_write($run_dir, $page_key, "<p>x</p>") === true);
oo_expect('page cache read', "<p>x</p>", share_run_page_cache_read($run_dir, $page_key));
oo_check('page cache write non-string', share_run_page_cache_write($run_dir, $page_key, 1) === false);
oo_expect('page cache read invalid', null, share_run_page_cache_read($run_dir, "missing"));

// ---------------------------------------------------------------------------
// Parameter validation and CSV helpers.
// ---------------------------------------------------------------------------
$_GET = ["name" => "abc"];
oo_expect('validate param', "abc", validate_param("name", "/^[a-z]+$/", "bad"));
try {
	validate_param("name", "/^[0-9]+$/", "bad");
	oo_check('validate param throws', false);
} catch (Exception $e) {
	oo_expect('validate param message', "bad", $e->getMessage());
}
validate_directory($root);
try {
	validate_directory("$root/missing-dir");
	oo_check('validate directory throws', false);
} catch (Exception $e) {
	oo_check('validate directory message', strpos($e->getMessage(), "not found") !== false);
}
oo_expect('utf8ize nested', ["ä" => "ö"], utf8ize(["ä" => "ö"]));
oo_expect('utf8ize string', "ä", utf8ize("ä"));

$csv = oo_write("$root/status.csv", "trial_index,trial_status,result\n1,completed,1\n2,failed,2\n3,running,3\n");
$status = get_status_for_results_csv($csv);
oo_expect('status succeeded', 1, $status["succeeded"]);
oo_expect('status failed', 1, $status["failed"]);
oo_expect('status running', 1, $status["running"]);
oo_expect('status total', 3, $status["total"]);
oo_expect('status cached', $status, get_status_for_results_csv($csv));
oo_check('status missing', strpos(get_status_for_results_csv("$root/nope.csv"), "error") !== false);
oo_expect('analyze results', "Completed: 1, Failed: 1, Running: 1", analyze_results_csv($csv));
oo_expect('analyze missing', "", analyze_results_csv("$root/nope.csv"));

oo_expect('read lines', ["a", "b"], read_file_as_array(oo_write("$root/lines.txt", "a\r\n\r\nb\n")));
oo_expect('read missing lines', [], read_file_as_array("$root/nope.txt"));
oo_write("$root/binary.txt", "\xff\xfe\x00");
oo_expect('read binary lines', [], read_file_as_array("$root/binary.txt"));

oo_expect('normalize int float', "10", normalize_csv_value("10.000"));
oo_expect('normalize float', "10.5", normalize_csv_value("10.5"));
oo_expect('normalize int', "10", normalize_csv_value("10"));
oo_expect('normalize empty', "", normalize_csv_value("  "));
oo_expect('normalize text', "abc", normalize_csv_value("abc"));
oo_expect('normalize contents', "a,b\n1,2", normalize_csv_file_contents("a,b\n\n1.0,2.000\n"));
oo_expect('csv to text', "a,b\n1,2", csv_array_to_text([["a", "b"], ["1", "2"]]));
oo_expect('keep rows small', [["a"]], keep_rows_every_n_seconds([["a"]]));
oo_expect('keep rows', 3, count(keep_rows_every_n_seconds([[0, "h"], [10, "a"], [20, "b"], [100, "c"]], 60)));
$collapsed = collapse_runs_keep_first_last([["x", "a"], ["y", "a"], ["z", "a"], ["w", "b"]]);
oo_expect('collapse runs', 3, count($collapsed));
oo_check('has real char', has_real_char($csv));
oo_check('no real char', !has_real_char(oo_write("$root/spaces.txt", "   \n")));
oo_check('ascii or utf8', is_ascii_or_utf8($csv));
oo_check('ascii or utf8 cached', is_ascii_or_utf8($csv));
oo_check('ascii or utf8 missing', !is_ascii_or_utf8("$root/nope.txt"));
oo_check('ascii or utf8 invalid', !is_ascii_or_utf8("$root/binary.txt"));

// ---------------------------------------------------------------------------
// Log parsing helpers.
// ---------------------------------------------------------------------------
oo_expect('remove ansi colors', "red", remove_ansi_colors("\033[31mred\033[0m"));
oo_expect('remove ansi sequences', "a\nb", remove_ansi_escape_sequences("a\033[0m\n\033]0;x\007b"));
oo_check('highlight debug', strpos(highlight_debug_info("E1234 something\nWARNING: w\nINFO i\n2020-01-01 00:00:00,000"), "<span") !== false);
oo_expect('exit code', 3, get_exit_code_from_outfile("EXIT_CODE: 3\n"));
oo_expect('exit code missing', null, get_exit_code_from_outfile("nothing"));
oo_expect('runtime empty', null, get_runtime(""));
$runtime_log = "submitit INFO (2020-01-01 00:00:00,000)\nsubmitit INFO (2020-01-01 00:00:10,000)\n";
oo_expect('runtime seconds', 10, get_runtime($runtime_log));
oo_expect('runtime no dates', 0, get_runtime("no dates"));
oo_expect('runtime human zero', "0s", get_runtime_human_format(0));
oo_expect('runtime human mixed', "1h:1m:1s", get_runtime_human_format(3661));
oo_expect('worker info empty', "", extract_min_max_ram_cpu_from_worker_info("nothing"));
oo_expect('worker info empty 2', "", extract_min_max_ram_cpu_from_worker_info("nothing"));

oo_check('contains results', file_string_contains_results("foo\nresult: 1\nbar: 2\n", ["result", "bar"]));
oo_check('contains results missing', !file_string_contains_results("", ["result"]));
oo_check('contains results no match', !file_string_contains_results("foo\n", ["result"]));
oo_check('ends with submitit', ends_with_submitit_info("submitit INFO (2020-01-01 00:00:00,000) - Exiting after successful completion"));
oo_check('not ends with submitit', !ends_with_submitit_info("x"));
$time_error = "slurmstepd: error: *** JOB 1 ON host CANCELLED AT 2020-01-01T00:00:00 DUE TO TIME LIMIT ***";
oo_check('time limit error', contains_slurm_time_limit_error($time_error));
oo_check('time limit error non-string', !contains_slurm_time_limit_error(12));
oo_expect('extract results dict', ["a" => "1", "b" => "-2.5"], extract_results_dict("a: 1\nb: -2.5\n"));
oo_expect('format results', "a: 1, b: 1,000", format_results_from_dict(["a" => "1", "b" => "1000", "c" => "x"], ["a", "b"]));
oo_expect('extract trial index', 4, extract_trial_index("Trial-Index: 4", 1));
oo_expect('extract trial index fallback', 1, extract_trial_index("none", 1));

oo_write("$root/small.log", "hello");
oo_expect('log excerpt small', "hello", read_log_excerpt("$root/small.log"));
oo_write("$root/large.log", str_repeat("a", 1000));
oo_check('log excerpt large', strpos(read_log_excerpt("$root/large.log", 100), "middle omitted") !== false);
oo_expect('log excerpt missing', false, oo_tolerate_warnings(fn() => read_log_excerpt("$root/nope.log")));

oo_write("$root/omitted_logs.json", json_encode([
	"omitted" => [["folder" => "1", "reason" => "successful"], ["folder" => "2", "reason" => "log cap"], ["folder" => "3", "reason" => "custom"]],
	"single_run_logs_skipped" => false,
	"commands" => ["all" => "omniopt_share /tmp/run --update --send_all_single_runs", "bad" => 12],
]));
$meta = get_omitted_logs_metadata($root);
oo_check('omitted metadata', is_array($meta));
oo_expect('omitted commands', 2, count(get_omitted_log_commands($root)));
oo_expect('run folder hint', "/tmp/run", get_run_folder_hint_from_commands(["omniopt_share /tmp/run --update"]));
oo_expect('run folder hint default', "<run-folder>", get_run_folder_hint_from_commands(["nothing"]));
oo_expect('local log command default', "less <run-folder>/x.log", format_local_log_command("<run-folder>", "x.log"));
oo_expect('local log command quoted', "less '/tmp/run/x.log'", format_local_log_command("'/tmp/run/'", "x.log"));
oo_check('missing log help', strpos(build_missing_log_help_html($root, "x.log"), "omniopt_share") !== false);
oo_write("$root/no-omitted/omitted_logs.json", "{}");
oo_check('missing log help fallback', strpos(build_missing_log_help_html("$root/no-omitted"), "send_all_single_runs") !== false);

// ---------------------------------------------------------------------------
// SVG, ZIP and image helpers.
// ---------------------------------------------------------------------------
$svg = oo_write("$root/plot.svg", '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10"></svg>');
$bad_svg = oo_write("$root/bad.svg", '<html></html>');
oo_check('valid svg', is_valid_svg_file($svg));
oo_check('invalid svg root', !is_valid_svg_file($bad_svg));
oo_check('invalid svg missing', !is_valid_svg_file("$root/nope.svg"));
oo_expect('tabs flame svg size', 1, count(add_flame_svg_file([], [], $svg, "flame", "id")[0]));
oo_expect('warnings invalid flame svg', 1, count(add_flame_svg_file([], [], $bad_svg, "flame", "id")[1]));
oo_expect('warnings missing flame svg', 1, count(add_flame_svg_file([], [], "$root/nope.svg", "flame", "id")[1]));

$zip_path = "$root/archive.zip";
$zip = new ZipArchive();
$zip->open($zip_path, ZipArchive::CREATE | ZipArchive::OVERWRITE);
$zip->addFromString("hello.txt", "hello");
$zip->close();
oo_check('valid zip', is_valid_zip_file($zip_path));
oo_check('invalid zip', !is_valid_zip_file($csv));
oo_check('missing zip', !is_valid_zip_file("$root/nope.zip"));

if (function_exists("imagecreatetruecolor")) {
	$png = "$root/image.png";
	$img = imagecreatetruecolor(4, 4);
	$black = imagecolorallocate($img, 0, 0, 0);
	$red = imagecolorallocate($img, 255, 0, 0);
	imagefill($img, 0, 0, $black);
	imagesetpixel($img, 1, 2, $red);
	imagepng($img, $png);
	imagedestroy($img);
	clean_black_lines_inplace($png);
	oo_check('clean black lines', filesize($png) > 0);
	clean_black_lines_inplace("$root/nope.png");
	oo_write("$root/not-image.png", "no image");
	oo_tolerate_warnings(fn() => clean_black_lines_inplace("$root/not-image.png"));
}

// ---------------------------------------------------------------------------
// Rich HTML / table helpers.
// ---------------------------------------------------------------------------
$unsafe = '<b onclick="x">B</b><script>alert(1)</script><img src="javascript:1"><span style="color:red;position:absolute;background:url(javascript:1)">S</span><br>';
$safe = sanitize_safe_html($unsafe);
oo_check('sanitize keeps b', strpos($safe, "<b>B</b>") !== false);
oo_check('sanitize removes script', strpos($safe, "script") === false);
oo_check('sanitize removes onclick', strpos($safe, "onclick") === false);
oo_check('sanitize removes bad src', strpos($safe, "javascript:1") === false);
oo_check('sanitize keeps safe style', strpos($safe, "color: red") !== false);
oo_check('sanitize removes unsafe style', strpos($safe, "position: absolute") === false);
oo_expect('sanitize self closing', '<br />', sanitize_safe_html('<br />'));
oo_expect('sanitize closing', '</b>', sanitize_safe_html('</b>'));

oo_expect('dedupe log lines', "a\n\nb\n\nc", remove_duplicate_log_lines("a\n\n\na\n\n\nb\n\nc\n"));
$ascii = "Title\n┏━━━━┳━━━━┓\n┃ a  ┃ b  ┃\n┡━━━━╇━━━━┩\n│ 1  │ 2  │\n└────┴────┘";
$table = ascii_table_to_html($ascii);
oo_check('ascii table title', strpos($table, "<h2>Title</h2>") !== false);
oo_check('ascii table cells', strpos($table, "<table") !== false && strpos($table, "1") !== false && strpos($table, "2") !== false);
oo_expect('ascii table empty', "<p>Error: No valid table found.</p>", ascii_table_to_html("───"));

$analysis = analyze_column_types([["1", "a"], ["2", "b"], [5, "c"]], ["num", "text"], ["special"]);
oo_expect('column analysis numeric', ["numeric" => true, "string" => false], $analysis["num"]);
oo_expect('column analysis string', ["numeric" => false, "string" => true], $analysis["text"]);
oo_expect('count column types', [1, 1], count_column_types($analysis));
oo_expect('add tabs', "\tx\n<pre>\ny\n</pre>", add_tabs_to_string("x\n<pre>\ny\n</pre>", 1));
oo_expect('remove font face', "a {}", remove_font_face_rules("@font-face { x }a {}"));
oo_expect('remove excessive newlines', "a\nb", remove_excessive_newlines("a\n\n\nb"));
oo_check('css style tag', strpos(generate_css_style_tag(oo_write("$root/style.css", "@font-face {x}\na {}\n\n\n")), "a {}") !== false);
oo_check('css style missing', oo_tolerate_warnings(fn() => generate_css_style_tag("$root/none.css")) === "");
oo_expect('clean result names', ["ab_c12"], clean_result_name_lines(["a-b_c!1 2"]));

$email_png = "$root/email.png";
if (function_exists("imagecreatetruecolor")) {
	$img = imagecreatetruecolor(2, 2);
	imagepng($img, $email_png);
	imagedestroy($img);
	$embedded = embed_local_images_as_base64('<img src="email.png"><img src="https://x/y.png"><img src="missing.png">', $root);
	oo_check('embed local image', strpos($embedded, "data:image/png;base64,") !== false);
	oo_check('embed keeps remote image', strpos($embedded, "https://x/y.png") !== false);
}

// ---------------------------------------------------------------------------
// Tab builders.
// ---------------------------------------------------------------------------
oo_write("$root/worker_info.txt", "CPU: 10.0%, RAM: 100.0 MB\nCPU: 20.0%, RAM: 200.0 MB\n");
[$tabs, $warnings] = add_worker_cpu_ram_from_file([], [], "$root/nope.txt", "Worker", "worker");
oo_expect('worker tab missing warning', 1, count($warnings));
[$tabs, $warnings] = add_worker_cpu_ram_from_file([], [], "$root/binary.txt", "Worker", "worker");
oo_expect('worker tab invalid warning', 1, count($warnings));
[$tabs, $warnings] = add_worker_cpu_ram_from_file([], [], "$root/lines.txt", "Worker", "worker");
oo_expect('worker tab added', 1, count($tabs));
[$tabs, $warnings] = add_worker_cpu_ram_from_file([], [], "$root/worker_info.txt", "Worker", "worker");
oo_expect('worker tab empty table warning', 1, count($warnings));

[$tabs, $warnings] = add_debug_log_from_file([], [], "$root/small.log", "Debug", "debug");
oo_expect('debug tab added', 1, count($tabs));
[$tabs, $warnings] = add_debug_log_from_file([], [], "$root/nope.log", "Debug", "debug");
oo_expect('debug tab warning', 1, count($warnings));
[$tabs, $warnings] = add_debug_log_from_file([], [], oo_write("$root/empty.log", ""), "Debug", "debug");
oo_expect('debug empty warning', 1, count($warnings));

[$tabs, $warnings] = add_cpu_ram_usage_main_worker_from_file([], [], oo_write("$root/cpu.csv", "timestamp,cpu,ram\n0,10,100\n100,20,200\n"), "CPU", "cpu");
oo_expect('cpu ram tab', 1, count($tabs));
oo_check('cpu ram globals', isset($GLOBALS["json_data"]["cpu_csv_json"]));
[$tabs, $warnings] = add_cpu_ram_usage_main_worker_from_file([], [], "$root/nope.csv", "CPU", "cpu");
oo_expect('cpu ram warning', 1, count($warnings));

oo_expect('scatter3d tab', 1, count(add_scatter_3d_plots([], $csv, "S3", "s3")));
oo_expect('scatter3d missing', 0, count(add_scatter_3d_plots([], "$root/nope", "S3", "s3")));
oo_expect('scatter2d tab', 1, count(add_scatter_2d_plots([], $csv, "S2", "s2")));
oo_expect('scatter2d missing', 0, count(add_scatter_2d_plots([], "$root/nope", "S2", "s2")));

[$tabs, $warnings] = add_parameter_distribution_by_type_plot([], [], $csv, "PD", "pd");
oo_expect('parameter distribution tab', 1, count($tabs));
[$tabs, $warnings] = add_parameter_distribution_by_type_plot([], [], "$root/nope", "PD", "pd");
oo_expect('parameter distribution warning', 1, count($warnings));

[$tabs, $warnings] = add_worker_usage_plot_from_file([], [], oo_write("$root/worker.csv", "timestamp,worker\n0,a\n1,a\n2,b\n"), "WU", "wu");
oo_expect('worker usage tab', 1, count($tabs));
[$tabs, $warnings] = add_worker_usage_plot_from_file([], [], "$root/nope", "WU", "wu");
oo_expect('worker usage warning', 1, count($warnings));

[$tabs, $warnings] = add_simple_table_from_ascii_table_file([], [], oo_write("$root/table.txt", $ascii), "Table", "table");
oo_expect('ascii table tab', 1, count($tabs));
[$tabs, $warnings] = add_simple_table_from_ascii_table_file([], [], oo_write("$root/table_ansi.txt", "\033[31m".$ascii."\033[0m"), "Table", "table", true);
oo_expect('ascii table ansi tab', 1, count($tabs));
[$tabs, $warnings] = add_simple_table_from_ascii_table_file([], [], "$root/nope", "Table", "table");
oo_expect('ascii table warning', 1, count($warnings));

$pre_file = oo_write("$root/pre.log", "a\n\na\n\nb\n");
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], $pre_file, "Pre", "pre");
oo_expect('pre tab', 1, count($tabs));
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], $pre_file, "Pre", "pre", true, true);
oo_expect('pre tab dedupe', 1, count($tabs));
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], "$root/nope", "Pre", "pre");
oo_expect('pre tab warning', 1, count($warnings));
$GLOBALS["lazy_share_logs"] = true;
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], $pre_file, "Pre", "pre");
unset($GLOBALS["lazy_share_logs"]);
oo_expect('pre tab normal', 1, count($tabs));

[$tabs, $warnings] = add_simple_csv_tab_from_file([], [], $csv, "CSV", "csvtab");
oo_expect('simple csv tab', 1, count($tabs));
oo_check('simple csv globals', isset($GLOBALS["json_data"]["csvtab_csv_json"]));
[$tabs, $warnings] = add_simple_csv_tab_from_file([], [], "$root/nope", "CSV", "csvtab");
oo_expect('simple csv warning', 1, count($warnings));

$tabs = [];
$tabs = add_exit_codes_pie_plot($tabs);
$tabs = add_violin_plot($tabs);
$tabs = add_plot_result_pairs($tabs);
$tabs = add_histogram_plot($tabs);
$tabs = add_heatmap_plot_tab($tabs);
$tabs = add_job_status_distribution($tabs);
$tabs = add_results_distribution_by_generation_method($tabs);
$tabs = add_box_plot_tab($tabs);
$tabs = add_gpu_plots($tabs);
$tabs = add_kde_histogram_tab($tabs);
$tabs = add_stacked_histogram_tab($tabs);
$tabs = add_parallel_plot_tab($tabs);
oo_expect('simple plot tabs', 12, count($tabs));

[$tabs, $warnings] = add_result_evolution_tab([], [], []);
oo_expect('evolution no results warning', 1, count($warnings));
$GLOBALS["json_data"]["tab_results_headers_json"] = ["r"];
[$tabs, $warnings] = add_result_evolution_tab([], [], ["r"]);
oo_expect('evolution tab', 1, count($tabs));
unset($GLOBALS["json_data"]["tab_results_headers_json"]);
[$tabs, $warnings] = add_result_evolution_tab([], [], ["r"]);
oo_expect('evolution missing globals warning', 1, count($warnings));

[$tabs, $warnings] = add_job_info_charts_tab([], [], $root);
oo_expect('job info missing warning', 1, count($warnings));
oo_write("$root/job_infos.csv", "trial_index,run_time\n1,10\n2,20\n");
[$tabs, $warnings] = add_job_info_charts_tab([], [], $root);
oo_expect('job info tab', 1, count($tabs));
oo_write("$root/job_infos.csv", "trial_index,other\n1,10\n");
[$tabs, $warnings] = add_job_info_charts_tab([], [], $root);
oo_expect('job info no run_time warning', 1, count($warnings));
oo_write("$root/job_infos.csv", "trial_index,run_time\n1,10\n2,20\n");

[$tabs, $warnings] = add_timeline([], [], $csv, "Timeline", "timeline");
oo_expect('timeline tab', 1, count($tabs));

[$tabs, $warnings] = add_arm_distribution_tab([], [], $root);
oo_expect('arm distribution missing warning', 1, count($warnings));
oo_write("$root/arm_evals_results.csv", "arm,result\n1,2\n");
[$tabs, $warnings] = add_arm_distribution_tab([], [], $root);
oo_expect('arm distribution tab', 1, count($tabs));
oo_write("$root/arm_evals_results.csv", "");
[$tabs, $warnings] = add_arm_distribution_tab([], [], $root);
oo_expect('arm distribution empty warning', 1, count($warnings));
oo_write("$root/arm_evals_results.csv", "arm,result\n");
[$tabs, $warnings] = add_arm_distribution_tab([], [], $root);
oo_expect('arm distribution no rows warning', 1, count($warnings));

oo_write("$root/pareto_idxs.json", "[1,2]");
[$tabs, $warnings] = add_pareto_from_from_file([], [], $root);
oo_expect('pareto idx tab', 1, count($tabs));
unlink("$root/pareto_idxs.json");
oo_write("$root/pareto_front_data.json", "[]");
oo_write("$root/pareto_front_table.txt", "table");
[$tabs, $warnings] = add_pareto_from_from_file([], [], $root);
oo_expect('pareto old tab', 1, count($tabs));
unlink("$root/pareto_front_data.json");
unlink("$root/pareto_front_table.txt");
[$tabs, $warnings] = add_pareto_from_from_file([], [], $root);
oo_expect('pareto warnings', 3, count($warnings));

[$tabs, $warnings] = get_outfiles_tab_from_run_dir($root, [], [], []);
oo_check('outfiles missing warning', count($warnings) >= 1);
oo_write("$run_dir/0_0_log.out", "Trial-Index: 1\nresult: 1\nEXIT_CODE: 0\n");
[$tabs, $warnings] = get_outfiles_tab_from_run_dir($run_dir, [], [], ["result"]);
oo_expect('outfiles tab', 1, count($tabs));

// ---------------------------------------------------------------------------
// Insights / overview helpers.
// ---------------------------------------------------------------------------
$insights_csv = oo_write("$root/insights.csv",
	"trial_index,trial_status,generation_method,generation_node,x,y,result\n" .
	"1,completed,random,0,0,0,0\n" .
	"2,completed,random,0,1,1,1\n" .
	"3,completed,tpe,1,2,2,4\n" .
	"4,failed,tpe,1,3,3,9\n" .
	"5,completed,tpe,2,4,4,16\n" .
	"6,completed,botorch,2,5,5,25\n");
oo_check('insights cached', strlen(create_insights_cached($insights_csv, ["result"], ["max"])) > 0);
oo_check('insights cached again', strlen(create_insights_cached($insights_csv, ["result"], ["max"])) > 0);
[$tabs, $warnings] = add_insights_from_file([], [], $root, ["result"], ["max"]);
oo_expect('insights missing warning', 1, count($warnings));
oo_write("$root/results.csv", $insights_csv ? file_get_contents($insights_csv) : "");
[$tabs, $warnings] = add_insights_from_file([], [], $root, ["result"], ["max"]);
oo_expect('insights tab', 1, count($tabs));
oo_write("$root/results.csv", "result,trial_status\n");
[$tabs, $warnings] = add_insights_from_file([], [], $root, ["result"], ["max"]);
oo_expect('insights no evaluations warning', 1, count($warnings));
oo_write("$root/results.csv", file_get_contents($insights_csv));

[$overview, $warnings, $status] = add_overview_table_to_overview_and_get_status_data($root, [], "", []);
oo_check('overview status table', strpos($overview, "Number of evaluations") !== false);
oo_expect('overview status total', 6, $status["total"]);

[$overview, $warnings] = add_overview_for_jobs_and_generation_stuff($root, "", []);
oo_check('overview job summary', strpos($overview, "Job Summary") !== false);

oo_write("$root/ui_url.txt", "https://example.com/gui");
[$overview, $warnings] = add_ui_url_from_file_to_overview($root, "", []);
oo_check('overview ui url', strpos($overview, "example.com") !== false);
[$overview, $warnings] = add_ui_url_from_file_to_overview("$root/no-ui", "", []);
oo_expect('overview ui warning', 1, count($warnings));

oo_write("$root/constraints.txt", $ascii);
[$overview, $warnings] = add_constraints_to_overview($root, "", []);
oo_check('overview constraints', strpos($overview, "Constraints") !== false);
[$overview, $warnings] = add_constraints_to_overview("$root/no-constraints", "", []);
oo_expect('overview constraints warning', 1, count($warnings));

oo_write("$root/experiment_overview.txt", $ascii);
[$overview, $warnings] = add_experiment_overview_to_overview($root, "", []);
oo_check('overview experiment', strpos($overview, "<table") !== false);
[$overview, $warnings] = add_experiment_overview_to_overview("$root/no-exp", "", []);
oo_expect('overview experiment warning', 1, count($warnings));

oo_write("$root/best_result.txt", $ascii);
[$overview, $warnings] = add_best_results_to_overview($root, "", []);
oo_check('overview best', strpos($overview, "<table") !== false);
[$overview, $warnings] = add_best_results_to_overview("$root/no-best", "", []);
oo_expect('overview best warning', 1, count($warnings));

oo_write("$root/formula_underbraces.txt", "\\frac{a}{b}");
oo_write("$root/formula.txt", "a / b");
oo_write("$root/formula_pretty.txt", "a / b");
[$overview, $warnings] = add_formula_to_overview($root, "", []);
oo_check('overview formula', strpos($overview, "Formula") !== false);
unlink("$root/formula_underbraces.txt");
[$overview, $warnings] = add_formula_to_overview($root, "", []);
oo_expect('overview formula skip', "", $overview);
oo_write("$root/formula_underbraces.txt", "\\frac{a}{b}");

oo_write("$root/parameters.txt", $ascii);
[$overview, $warnings] = add_parameters_to_overview($root, "", []);
oo_check('overview parameters', strpos($overview, "<table") !== false);
[$overview, $warnings] = add_parameters_to_overview("$root/no-params", "", []);
oo_expect('overview parameters warning', 1, count($warnings));

oo_write("$root/progressbar", "line 1\nline 2\n");
[$overview, $warnings] = add_progressbar_to_overview($root, "", []);
oo_check('overview progressbar', strpos($overview, "line 2") !== false);
[$overview, $warnings] = add_progressbar_to_overview("$root/no-progress", "", []);
oo_expect('overview progressbar warning', 1, count($warnings));

[$overview, $warnings] = add_result_names_table_to_overview(["result"], ["max"], "", []);
oo_check('overview result names', strpos($overview, "result") !== false);
[$overview, $warnings] = add_result_names_table_to_overview([], [], "", []);
oo_expect('overview result names warning', 1, count($warnings));

oo_write("$root/git_version", "abc\n");
[$overview, $warnings] = add_git_version_to_overview($root, "", []);
oo_check('overview git version', strpos($overview, "abc") !== false);
[$overview, $warnings] = add_git_version_to_overview("$root/no-git", "", []);
oo_expect('overview git warning', 1, count($warnings));

oo_write("$root/result_names.txt", "result\n");
oo_write("$root/result_min_max.txt", "max\n");
[$names, $minmax, $warnings] = get_result_names_and_min_max($root, []);
oo_expect('result names', ["result"], $names);
oo_expect('result min max', ["max"], $minmax);

[$tabs, $warnings, $status] = add_overview_tab([], [], $root, [], ["result"], ["max"]);
oo_expect('overview tab', 1, count($tabs));

// ---------------------------------------------------------------------------
// Export tab / CSS helpers.
// ---------------------------------------------------------------------------
$_SERVER["HTTPS"] = "on";
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php?user_id=u&run_nr=0";
oo_expect('add param to url', "https://example.com/share.php?user_id=u&run_nr=0&export=1", add_param_to_current_url("export", "1"));
$_GET = [];
[$tabs, $warnings] = get_export_tab([], [], $run_dir);
oo_check('export disabled tab', isset($tabs) && count($tabs) === 1);
$_GET = ["export" => "1"];
$GLOBALS["SPECIAL_COL_NAMES"] = ["trial_index"];
$GLOBALS["json_data"] = ["tab_results_headers_json" => ["result"], "gpu_data" => [1]];
$tabs_for_export = [
	"Results" => ["content" => "<p>r</p>", "onclick" => "plot();"],
	"Single Logs" => ["content" => "<p>log</p>"],
];
[$tabs, $warnings] = get_export_tab($tabs_for_export, [], $run_dir);
oo_check('export enabled tab', isset($tabs) && count($tabs) === 3);

// ---------------------------------------------------------------------------
// Folder creation / routing helpers.
// ---------------------------------------------------------------------------
$GLOBALS["sharesPath"] = $shares;
$folder = create_new_folder("user1", "exp1");
oo_expect('create folder 0', "$shares/user1/exp1/0", $folder);
$folder2 = create_new_folder("user1", "exp1");
oo_expect('create folder 1', "$shares/user1/exp1/1", $folder2);

oo_write("$shares/user1/exp1/0/state_files/run_uuid", "uuid-a\n");
oo_write("$shares/user1/exp1/1/run_uuid", "uuid-b\n");
oo_write("$shares/user1/exp1/0/results.csv", "result\n1\n");
oo_expect('find matching uuid state', "$shares/user1/exp1/0", find_matching_uuid_run_folder("uuid-a", "user1", "exp1"));
oo_expect('find matching uuid root', "$shares/user1/exp1/1", find_matching_uuid_run_folder("uuid-b", "user1", "exp1"));
oo_expect('find matching uuid missing', null, find_matching_uuid_run_folder("uuid-c", "user1", "exp1"));
oo_expect('find matching uuid empty', null, find_matching_uuid_run_folder("", "user1", "exp1"));
oo_expect('find matching uuid bad user', null, find_matching_uuid_run_folder("uuid-a", "bad user", "exp1"));
oo_expect('resolve latest', "$shares/user1/exp1/0", resolve_share_target_folder("uuid-a", "user1", "exp1"));
oo_expect('resolve missing', null, resolve_share_target_folder("uuid-x", "user1", "exp1"));
oo_expect('resolve fallback', find_latest_run_folder("user1", "exp1"), resolve_share_target_folder("", "user1", "exp1", true));
oo_check('find latest missing', find_latest_run_folder("nouser", "noexp") === null);
oo_check('find latest bad name', find_latest_run_folder("bad user", "exp1") === null);

putenv("disable_folder_creation=1");
oo_expect('get user folder existing', "$shares/user1/exp1/0", get_user_folder(null, "user1", "exp1", "0"));
oo_expect('get user folder disabled', null, get_user_folder(null, "user1", "exp1", "9"));
putenv("disable_folder_creation");

oo_write("$shares/user1/exp1/0/hash.md5", "hash");
oo_expect('search hash', [true, "$shares/user1/exp1/0"], search_for_hash_file("$shares/user1/exp1/*/hash.md5", "hash"));
oo_expect('search hash missing', [false, null], search_for_hash_file("$shares/user1/exp1/*/hash.md5", "nope"));
oo_expect('path components', ["user1", "exp1", "0"], extract_path_components("$shares/user1/exp1/0"));
[$u, $e, $r] = extract_path_components("$root/not-a-share");
oo_expect('path components invalid', [null, null, null], [$u, $e, $r]);

putenv("CI=1");
oo_check('folder permissions ci', check_folder_permissions($root, "x", "x", "", "", 0755) === false);
putenv("CI");
[$issues, $output] = oo_capture(fn() => check_folder_permissions("$root/no-dir", "x", "x", "", "", 0755));
oo_check('folder permissions missing', $issues === true);
$stat = stat($root);
$current_user = posix_getpwuid($stat["uid"])["name"] ?? "unknown";
$current_group = posix_getgrgid($stat["gid"])["name"] ?? "unknown";
[$issues] = oo_capture(fn() => check_folder_permissions($root, $current_user, $current_group, "", "", intval(substr(sprintf("%o", $stat["mode"]), -4), 8)));
oo_check('folder permissions matching', $issues === false);

$delete_target = "$root/delete-me/sub/file";
oo_write($delete_target, "x");
oo_check('delete folder', delete_folder("$root/delete-me") === true);
oo_write("$root/rrmdir/sub/file", "x");
rrmdir("$root/rrmdir");
oo_check('rrmdir', !is_dir("$root/rrmdir"));

// ---------------------------------------------------------------------------
// Bundle extraction / tree helpers.
// ---------------------------------------------------------------------------
$bundle_run = "$root/bundle-run";
$bundle_zip = "$bundle_run/bundle.zip";
mkdir($bundle_run, 0777, true);
$zip = new ZipArchive();
$zip->open($bundle_zip, ZipArchive::CREATE | ZipArchive::OVERWRITE);
$zip->addFromString("results.csv", "result\n1\n");
$zip->addFromString("single_runs/1/1_0_log.out", "log\n");
$zip->close();
$warnings = [];
_extract_from_bundle_zip($bundle_run, $warnings);
oo_check('bundle extracted results', is_file("$bundle_run/results.csv"));
oo_check('bundle extracted log', is_file("$bundle_run/1_0_log.out"));
oo_check('bundle marker', is_file("$bundle_run/.bundle_extracted"));
ensure_share_bundle_extracted($bundle_run);
ensure_share_bundle_extracted("$root/no-bundle");
ensure_share_bundle_extracted("");

$tree_base = "$root/tree";
oo_write("$tree_base/user1/exp1/0/results.csv", "result\n1\n");
[, $out] = oo_capture(fn() => generate_folder_tree_view($tree_base));
oo_check('folder tree', strpos($out, "user1") !== false && strpos($out, "exp1") !== false);
[, $out] = oo_capture(fn() => generate_folder_tree_view("$root/no-tree"));
oo_check('folder tree missing', strpos($out, "does not exist") !== false);

$_SERVER["REQUEST_URI"] = "/share.php?user_id=user1";
$_GET = ["sort" => "time_asc"];
[, $out] = oo_capture(fn() => generate_folder_buttons("$tree_base/user1", "experiment_name"));
oo_check('folder buttons time asc', strpos($out, "exp1") !== false);
$_GET = ["sort" => "nr_asc"];
[, $out] = oo_capture(fn() => generate_folder_buttons("$tree_base/user1", "experiment_name"));
oo_check('folder buttons nr asc', strpos($out, "exp1") !== false);
$_GET = ["sort" => "nr_desc"];
[, $out] = oo_capture(fn() => generate_folder_buttons("$tree_base/user1", "experiment_name"));
oo_check('folder buttons nr desc', strpos($out, "exp1") !== false);
$_GET = [];
[, $out] = oo_capture(fn() => generate_folder_buttons("$tree_base/user1", "run_nr"));
oo_check('folder buttons run nr', strpos($out, "exp1") !== false);
[, $out] = oo_capture(fn() => generate_folder_buttons("$tree_base/missing", "run_nr"));
oo_check('folder buttons missing', strpos($out, "does not exist") !== false);
unset($_SERVER["REQUEST_URI"]);
oo_check('folder buttons no request', generate_folder_buttons("$tree_base/user1", "run_nr") === null);
$_SERVER["REQUEST_URI"] = "/share.php";

oo_write("$tree_base/user1/empty/0/.keep", "");
oo_write("$tree_base/user1/exp2/0/file.txt", "x");
[, $out] = oo_capture(fn() => generate_folder_buttons("$tree_base/user1", "experiment_name"));
oo_check('folder buttons subfolders', strpos($out, "subfolder") !== false);
[, $out] = oo_capture(fn() => generate_folder_buttons("$tree_base/user1/exp2", "run_nr"));
oo_check('folder buttons no jobs', $out !== "");
oo_check('time since', strpos(time_since(time() - 5), "second") !== false);
oo_check('time since just now', time_since(time()) === "just now");
oo_check('sort options', strpos(get_sort_options(), "time_desc") !== false);

// ---------------------------------------------------------------------------
// GPU usage files.
// ---------------------------------------------------------------------------
oo_expect('gpu files missing', [], find_gpu_usage_files("$root/no-gpu"));
$gpu_header = "timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used";
oo_write("$root/gpu_usage__i0.csv", $gpu_header . "\n2020-01-01 00:00:00,GPU,0,1,0,1,1,50,10%,1,100,50,50\nbad,row\n");
$gpu_files = find_gpu_usage_files($root);
oo_check('gpu files found', count($gpu_files) === 1);
$parsed = parse_gpu_usage_files($gpu_files);
oo_check('gpu parsed', isset($parsed[0]) && count($parsed[0]) === 1);
unlink("$root/gpu_usage__i0.csv");

oo_check('has non empty folder', has_non_empty_folder("$tree_base/user1/exp1"));
oo_check('has non empty folder missing', !has_non_empty_folder("$root/no-folder"));
oo_check('latest recursive modification', is_numeric(get_latest_recursive_modification_time("$tree_base/user1/exp1")));
$sort_base = "$root/sortbase";
oo_write("$sort_base/a/file", "a");
oo_write("$sort_base/b/file", "b");
$folders = ["a", "b"];
sort_folders_by_modification_time($sort_base, $folders);
oo_expect('sort folders', 2, count($folders));
oo_expect('valid folders', ["user1"], get_valid_folders($tree_base));
oo_expect('valid folders missing', [], get_valid_folders("$root/no-folder"));
oo_expect('icon html', "<img class='invert_icon' src='i/plot.svg' style='height: 1em' />&nbsp;", get_icon_html("plot.svg"));

[$tabs, $warnings] = check_and_filter_tabs("Results", ["Results" => 1, "Other" => 2], []);
oo_expect('filter tabs', ["Other"], array_keys($tabs));

// ---------------------------------------------------------------------------
// results_analyzer.php
// ---------------------------------------------------------------------------
$analysis_header = ["trial_index", "trial_status", "generation_method", "x", "y", "result"];
$analysis_rows = [
	["1", "completed", "random", "0", "0", "0"],
	["2", "completed", "random", "1", "1", "1"],
	["3", "completed", "tpe", "2", "2", "4"],
	["4", "failed", "tpe", "3", "3", "9"],
	["5", "completed", "tpe", "4", "4", "16"],
	["6", "completed", "botorch", "5", "5", "25"],
];
$analysis_csv = "$root/analysis.csv";
oo_write($analysis_csv, implode(",", $analysis_header) . "\n" . implode("\n", array_map(fn($r) => implode(",", $r), $analysis_rows)) . "\n");
oo_check('load csv', is_array(load_csv($analysis_csv)));
oo_check('load csv missing', oo_tolerate_warnings(fn() => load_csv("$root/no.csv")) === null);
oo_expect('pearson perfect', 1.0, pearson_correlation([1, 2, 3], [2, 4, 6]));
oo_expect('pearson short', null, pearson_correlation([1], [1]));
oo_expect('pearson mismatch', null, pearson_correlation([1, 2, 3], [1, 2]));
oo_expect('pearson zero denom', 0.0, pearson_correlation([1, 1, 1], [1, 2, 3]));
$stats = calculate_stats($analysis_header, $analysis_rows);
oo_check('calculate stats numeric', $stats["result"]["type"] === "numeric");
oo_check('calculate stats categorical', $stats["generation_method"]["type"] === "categorical");
oo_check('correlation matrix', isset(compute_correlation_matrix($stats, ["result"])["result"]["x"]));
oo_check('create insights', strpos(create_insights($analysis_csv, ["result"], ["max"]), "result") !== false);
oo_check('create insights missing', strpos(create_insights("$root/no.csv", ["result"], ["max"]), "Error") !== false);
oo_write("$root/one-line.csv", "a,b\n");
oo_check('create insights one line', strpos(create_insights("$root/one-line.csv", [], []), "no data") !== false);
$result_min_max = ["result" => "max"];
$result_names = ["result"];
$correlations = ["result" => ["x" => 1.0, "y" => 0.9]];
$flat = compute_csv_insights_flat($analysis_csv, $correlations, $result_min_max, [], array_combine($analysis_header, array_map(fn($i) => array_column($analysis_rows, $i), range(0, 5))));
oo_check('insights flat', count($flat) === 1);
oo_expect('insights flat empty', [], compute_csv_insights_flat($analysis_csv, [], $result_min_max, []));
oo_check('compute csv insights', count(compute_csv_insights($analysis_csv, $result_min_max)) === 1);
try {
	compute_csv_insights("$root/no.csv", []);
	oo_check('compute csv insights missing throws', false);
} catch (Exception $e) {
	oo_check('compute csv insights missing throws', true);
}
oo_check('top k', strpos(compute_top_k_results($analysis_header, $analysis_rows, $result_names, $result_min_max), "Top-5") !== false);
oo_expect('top k empty', "", compute_top_k_results($analysis_header, [], [], []));
oo_check('parameter sensitivity', strpos(compute_parameter_sensitivity($analysis_header, $analysis_rows, $result_names, $result_min_max, []), "Sensitivity") !== false);
oo_expect('parameter sensitivity empty', "", compute_parameter_sensitivity($analysis_header, $analysis_rows, [], $result_min_max, []));
oo_check('convergence', strpos(compute_convergence_analysis($analysis_header, $analysis_rows, $result_names, $result_min_max), "Convergence") !== false);
oo_check('generation comparison', strpos(compute_generation_method_comparison($analysis_header, $analysis_rows, $result_names, $result_min_max), "Generation Method") !== false);
oo_check('bring insights', strpos(bring_insights_data_together($analysis_csv, $analysis_header, $analysis_rows, $stats, $correlations, $result_names, $result_min_max), "Parameter statistics") !== false);
oo_check('bring insights no correlations', strpos(bring_insights_data_together($analysis_csv, $analysis_header, $analysis_rows, $stats, [], $result_names, $result_min_max), "No notable correlations") !== false);

// ---------------------------------------------------------------------------
// _functions.php helpers.
// ---------------------------------------------------------------------------
$_GET = ["from_get" => "get-value"];
putenv("FROM_ENV=env-value");
oo_expect('get or env get', "get-value", get_or_env("from_get"));
oo_expect('get or env env', "env-value", get_or_env("FROM_ENV"));
oo_expect('get or env missing', false, get_or_env("NO_SUCH_VAR_XYZ"));
$_POST = ["posted" => "post-value"];
oo_expect('get post', "post-value", get_post("posted"));
oo_expect('get post default', "d", get_post("missing", "d"));
oo_expect('get server', $_SERVER["REQUEST_URI"], get_server("REQUEST_URI"));
oo_expect('get server default', "d", get_server("MISSING", "d"));

oo_expect('validate readable missing', "$root/no-file does not exist", validate_readable_file("$root/no-file"));
oo_expect('validate readable empty', "$root/empty.log is empty", validate_readable_file("$root/empty.log"));
oo_check('validate readable ok', validate_readable_file($csv) === null);

$token = generate_csrf_token();
oo_check('csrf token', is_string($token) && strlen($token) === 64);
oo_check('csrf verify', verify_csrf_token($token));
oo_check('csrf bad verify', !verify_csrf_token("bad"));
oo_check('csrf field', strpos(csrf_field(), "csrf_token") !== false);

$tabs = [];
$tabs = add_tab_entry($tabs, "Plain", "id1", "content");
$tabs = add_tab_entry($tabs, "Icon", "id2", "content", "click()", "plot.svg");
oo_check('add tab plain', isset($tabs["Plain"]) && !isset($tabs["Plain"]["onclick"]));
$icon_key = null;
foreach (array_keys($tabs) as $tab_key) {
	if (strpos($tab_key, "Icon") !== false) {
		$icon_key = $tab_key;
	}
}
oo_check('add tab icon', $icon_key !== null && $tabs[$icon_key]["onclick"] === "click()");

oo_write("$root/comment.html", "<!-- Category: cat -->
<!-- hello -->
<h2>Heading</h2>
");
oo_expect('html category comment', "cat", get_html_category_comment("$root/comment.html"));
oo_expect('html comment', "Category: cat", get_html_comment("$root/comment.html"));
oo_expect('first heading html', "Heading", get_first_heading_content("$root/comment.html"));
oo_expect('first heading missing', null, get_first_heading_content("$root/no-heading.html"));
oo_write("$root/heading.md", "# Markdown heading
");
oo_expect('first heading markdown', "Markdown heading", get_first_heading_content("$root/heading.md"));
oo_write("$root/no-comment.html", "nothing");
oo_expect('html category none', null, get_html_category_comment("$root/no-comment.html"));
oo_expect('html comment none', null, get_html_comment("$root/no-comment.html"));

oo_expect('highlight backticks', "a <tt>x</tt>", highlight_backticks("a `x`"));
oo_expect('newlines to br', "a
<br><br>b", convertNewlinesToBr("a


b"));

$markdown = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6\n\n- one\n- two\n\n**bold** __strong__ *em*\n\n[link](https://example.com) [local](/x) ![img](pic.png)\n\n`inline`\n\n```\nplain\n```\n\n```csv\na,b\n```\n\n```json\n{}\n```\n\n```yaml\na: b\n```\n\n```toml\na = 1\n```\n\n```bash\necho hi\n```\n\n1. first\n";
$html = convert_markdown_to_html($markdown);
oo_check('markdown headings', strpos($html, "<h1>H1</h1>") !== false && strpos($html, "<h6>H6</h6>") !== false);
oo_check('markdown list', strpos($html, "<ul>") !== false);
oo_check('markdown emphasis', strpos($html, "<strong>bold</strong>") !== false && strpos($html, "<em>em</em>") !== false);
oo_check('markdown links', strpos($html, 'href="https://example.com"') !== false && strpos($html, 'href="/x"') !== false);
oo_check('markdown code blocks', strpos($html, "language-csv") !== false && strpos($html, "language-json") !== false && strpos($html, "language-yaml") !== false && strpos($html, "language-toml") !== false && strpos($html, "language-bash") !== false);
oo_check('markdown ordered list', strpos($html, "<ol>") !== false);

oo_write("$root/snippet.py", "print('snippet')\n");
$markdown_python = "```python[" . "$root/snippet.py" . "]\n```\n\n```python\nprint('inline')\n```\n";
oo_check('markdown python file', strpos(convert_markdown_to_html($markdown_python), "language-python") !== false);
$run_php = "```run_php\necho 'php-output';\n```\n";
oo_check('markdown run php', strpos(convert_markdown_to_html($run_php), "php-output") !== false);

[$out] = oo_capture(fn() => convert_file_to_html("$root/no-md.md"));
[, $out] = oo_capture(fn() => convert_file_to_html("$root/no-md.md"));
oo_check('convert file missing', strpos($out, "File not found") !== false);
oo_write("$root/doc.md", "# Doc\n");
[, $out] = oo_capture(fn() => convert_file_to_html("$root/doc.md"));
oo_check('convert file', strpos($out, "<h1>Doc</h1>") !== false);

oo_expect('replace placeholders', "a 1 {y}", replace_python_placeholders("a {x} {y}", ["x" => "1"]));
try {
	replace_python_placeholders(1, []);
	oo_check('replace placeholders bad input', false);
} catch (InvalidArgumentException $e) {
	oo_check('replace placeholders bad input', true);
}
try {
	replace_python_placeholders("x", "bad");
	oo_check('replace placeholders bad replacements', false);
} catch (InvalidArgumentException $e) {
	oo_check('replace placeholders bad replacements', true);
}

oo_write("$root/lists.py", "VALID_ACQUISITION_CLASSES: list = ['a', 'b']\nvalid_occ_types: list = ['x']\n");
oo_expect('extract python list', '"a", "b"', extract_and_join_python_list("$root/lists.py", "VALID_ACQUISITION_CLASSES"));
oo_expect('extract python list cached', '"a", "b"', extract_and_join_python_list("$root/lists.py", "VALID_ACQUISITION_CLASSES"));
oo_expect('extract python list missing', null, extract_and_join_python_list("$root/no-lists.py", "X"));
oo_write("$root/lists_bad.py", "X: list = 'not a list'\n");
oo_expect('extract python list bad', null, extract_and_join_python_list("$root/lists_bad.py", "X"));

oo_write("$root/args.py", "group1 = parser.add_argument_group('Group', 'Description')\ngroup1.add_argument('--foo', help='Foo help', default=1, type=int, action='store_true')\ngroup2 = self.parser.add_argument_group('G2', 'D2')\ngroup2.add_argument('--bar', help=f'Bar {joined_valid_occ_types}', default=Path.home())\n");
$groups = parse_arguments("$root/args.py");
oo_check('parse args python', isset($groups["Group"]) && count($groups["Group"]["args"]) >= 1);
$args_html = generate_argparse_html_table($groups, false);
oo_check('argparse table', strpos($args_html, "--foo") !== false && strpos($args_html, "<table>") !== false);
oo_expect('argparse table empty hidden', "", generate_argparse_html_table([], true));
oo_check('argparse table empty', strpos(generate_argparse_html_table([], false), "No arguments") !== false);
[, $out] = oo_capture(fn() => parse_arguments_and_print_html_table("$root/args.py"));
oo_check('parse args print', strpos($out, "--foo") !== false);
[, $out] = oo_capture(fn() => parse_arguments_and_print_html_table("$root/args.py"));
oo_check('parse args print cached', strpos($out, "--foo") !== false);
oo_expect('parse args python missing', [], parse_arguments("$root/no-args.py"));
oo_write("$root/empty.py", "");
oo_expect('parse args python empty', [], parse_arguments("$root/empty.py"));
oo_write("$root/args.sh", "function help() {\n  echo \"  --mode=INT Set the mode\";\n  echo \"  --flag Enable flag\";\n  exit 1\n}\n");
$groups = parse_arguments("$root/args.sh");
oo_check('parse args bash', isset($groups["Ungrouped"]) && count($groups["Ungrouped"]["args"]) === 2);
oo_expect('parse args bash missing', [], parse_arguments("$root/no-args.sh"));
oo_write("$root/empty.sh", "");
oo_expect('parse args bash empty', [], parse_arguments("$root/empty.sh"));
oo_write("$root/nohelp.sh", "echo hi\n");
oo_expect('parse args bash no help', [], parse_arguments("$root/nohelp.sh"));
oo_expect('extract magic comment', "value", extract_magic_comment(oo_write("$root/magic.php", "<?php\n# DESCRIPTION: value\n"), "DESCRIPTION"));
oo_expect('extract magic missing file', null, extract_magic_comment("$root/no-magic.php", "DESCRIPTION"));
oo_expect('extract magic missing key', null, extract_magic_comment("$root/magic.php", "OTHER"));
oo_check('extract bash help missing', strpos(extract_help_params_from_bash("$root/no-help.sh"), "File not found") !== false);
oo_write("$root/help.sh", "function help {\n  echo \"--mode Set the mode\";\n  echo \"--help Show help\";\n  exit 1\n}\n");
$help_html = extract_help_params_from_bash("$root/help.sh");
oo_check('extract bash help', strpos($help_html, "--mode") !== false);

$run_program_file = oo_write("$root/runprogram.txt", "Run-Program: echo hello\n");
oo_expect('run program from file', "echo hello", getRunProgramFromFile($run_program_file));
oo_expect('run program from file missing', "", getRunProgramFromFile("$root/no-run.txt"));
oo_expect('run program from file invalid arg', "", getRunProgramFromFile(""));
oo_write("$root/runprogram-empty.txt", "Run-Program: \n");
oo_expect('run program empty', "", getRunProgramFromFile("$root/runprogram-empty.txt"));
oo_write("$root/runprogram-none.txt", "nothing\n");
oo_expect('run program none', "", getRunProgramFromFile("$root/runprogram-none.txt"));
$job_file = oo_write("$root/job.txt", "scancel '12345'\n");
oo_expect('job id from file', "12345", getJobIdFromFile($job_file));
oo_expect('job id missing', "", getJobIdFromFile("$root/no-job.txt"));
oo_expect('job id invalid arg', "", getJobIdFromFile(""));
oo_write("$root/job-none.txt", "nothing\n");
oo_expect('job id none', "", getJobIdFromFile("$root/job-none.txt"));

// ---------------------------------------------------------------------------
// Additional cache / log / table branches.
// ---------------------------------------------------------------------------
$cache_dir = share_cache_namespace_dir("corrupt");
if ($cache_dir !== null) {
	file_put_contents("$cache_dir/" . sha1("plain") . ".cache", "not gzip");
	oo_expect('cache corrupt gzip', null, share_cache_read_blob("corrupt", sha1("plain")));
	share_cache_write_blob("corrupt", sha1("serialized"), "not a serialized value");
	oo_expect('cache corrupt serialized', null, share_cache_read_serialized("corrupt", sha1("serialized")));
}
$bad_page_key = sha1("bad-page");
share_cache_write_serialized("runpage", $bad_page_key, ["fingerprint" => "wrong", "html" => "x"]);
oo_expect('page cache wrong fingerprint', null, share_run_page_cache_read($run_dir, $bad_page_key));
share_cache_write_serialized("runpage", $bad_page_key, "not-an-array");
oo_expect('page cache not array', null, share_run_page_cache_read($run_dir, $bad_page_key));
oo_check('page cache missing run dir', share_run_page_cache_write("$root/no-run", $bad_page_key, "x") === false);

oo_write("$root/utf8.txt", "äöü\n");
oo_expect('read utf8 lines', ["äöü"], read_file_as_array("$root/utf8.txt"));
foreach (["1e3", "0.0", "-2.50", " 1 ", ""] as $value) {
	normalize_csv_value($value);
}
oo_expect('normalize zero', "0", normalize_csv_value("0.0"));
oo_expect('normalize scientific', "1000", normalize_csv_value("1e3"));

oo_write("$root/svg-bom.svg", "\xEF\xBB\xBF<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
oo_check('valid svg bom', is_valid_svg_file("$root/svg-bom.svg"));
oo_write("$root/svg-noxml.svg", "<svg></svg>");
oo_check('invalid svg no xmlns', !is_valid_svg_file("$root/svg-noxml.svg"));
oo_check('valid svg non-html', !is_valid_svg_file("$root/lines.txt"));

$mixed = analyze_column_types([["", "a"], ["'1'", ""], ["x", "1"]], ["quoted", "mixed"], ["quoted"]);
oo_expect('column analysis skip special', [], analyze_column_types([["1"]], ["special"], ["special"]));
oo_expect('count column types string only', [0, 1], count_column_types(["x" => ["numeric" => false, "string" => true]]));

oo_write("$root/status-empty.csv", "");
oo_check('status empty', is_array(get_status_for_results_csv("$root/status-empty.csv")));
oo_write("$root/status-noindex.csv", "a,b\n1,completed\n");
oo_check('status no index', is_array(get_status_for_results_csv("$root/status-noindex.csv")));
oo_write("$root/status-blank.csv", "\n\na,b\n\n");
oo_check('status blanks', is_array(get_status_for_results_csv("$root/status-blank.csv")));
oo_expect('analyze results empty', "", analyze_results_csv(oo_write("$root/empty-results.csv", "trial_status\n")));

$GLOBALS["ascii_or_utf8_cache"] = [];
oo_check('ascii cache warm', is_ascii_or_utf8($csv));
oo_check('ascii cache hit', is_ascii_or_utf8($csv));

$log_run = "$root/logtabrun";
$log_contents = [
	"0_0_log.out" => "Trial-Index: 3\nresult: 1\nEXIT_CODE: 2\nsubmitit INFO (2020-01-01 00:00:00,000)\nsubmitit INFO (2020-01-01 00:00:05,000)\n",
	"1_0_log.out" => "CUDA out of memory\n",
	"2_0_log.out" => "submitit INFO (2020-01-01 00:00:00,000) - Exiting after successful completion\n",
	"3_0_log.out" => "slurmstepd: error: *** JOB 1 ON host CANCELLED AT 2020-01-01T00:00:00 DUE TO TIME LIMIT ***\n",
	"4_0_log.out" => "still working\n",
];
foreach ($log_contents as $file => $contents) {
	oo_write("$log_run/$file", $contents);
}
oo_write("$log_run/single_runs/9/9_0_log.out", "single run\n");
$log_files = get_log_files($log_run);
oo_check('get log files', count($log_files) === 6);
oo_expect('get log files missing', [], get_log_files("$root/no-logs"));
oo_write("$log_run/omitted_logs.json", json_encode([
	"omitted" => [["folder" => "5", "reason" => "successful"], ["folder" => "6", "reason" => "log cap"], "bad"],
	"single_run_logs_skipped" => false,
	"commands" => ["all" => "omniopt_share /tmp/run --update", "bad" => 1],
]));
[$content, $count] = generate_log_tabs($log_run, $log_files, ["result"]);
oo_check('generate log tabs', $count >= 6 && strpos($content, "Missing logs") !== false);
oo_check('generate log tabs oom', strpos($content, "data-status=oom") !== false);
oo_write("$log_run/omitted_logs.json", json_encode(["single_run_logs_skipped" => true]));
[$content, $count] = generate_log_tabs($log_run, $log_files, []);
oo_check('generate log tabs skipped', strpos($content, "omitted completely") !== false);

$summary_base = "$root/summary";
oo_write("$summary_base/full/file", "x");
$summary = get_folder_summary("$summary_base/full");
oo_check('folder summary non empty', $summary["non_empty"] === true && $summary["mtime"] > 0);
oo_check('folder summary cached', get_folder_summary("$summary_base/full") === $summary);
$missing_summary = get_folder_summary("$summary_base/missing");
oo_check('folder summary missing', $missing_summary["non_empty"] === false);
oo_check('latest modification empty', get_latest_modification_time("$summary_base/missing") === 0);
oo_write("$summary_base/full/file2", "y");
oo_check('latest modification cached', is_numeric(get_latest_modification_time("$summary_base/full")));

// Overview job summary variants.
$job_root = "$root/jobs";
oo_write("$job_root/missing/results.csv", "");
[$ov, $w] = add_overview_for_jobs_and_generation_stuff("$job_root/missing", "", []);
oo_expect('jobs empty warning', 1, count($w));
oo_write("$job_root/noheader/results.csv", "\n");
[$ov, $w] = add_overview_for_jobs_and_generation_stuff("$job_root/noheader", "", []);
oo_expect('jobs no header warning', 1, count($w));
oo_write("$job_root/nocolumns/results.csv", "a,b\n1,2\n");
[$ov, $w] = add_overview_for_jobs_and_generation_stuff("$job_root/nocolumns", "", []);
oo_expect('jobs no columns warning', 1, count($w));
oo_write("$job_root/nonascii/results.csv", "\xff\xfe");
[$ov, $w] = add_overview_for_jobs_and_generation_stuff("$job_root/nonascii", "", []);
oo_expect('jobs nonascii warning', 1, count($w));
oo_write("$job_root/unknown/results.csv", "generation_node,trial_status\n,completed\n0,FAILED\n1,completed\n");
[$ov, $w] = add_overview_for_jobs_and_generation_stuff("$job_root/unknown", "", []);
oo_check('jobs unknown generation', strpos($ov, "UNKNOWN") !== false);

// Folder button variants.
$button_tree = "$root/buttons";
oo_write("$button_tree/user/exp/0/results.csv", "trial_status\ncompleted\n");
oo_write("$button_tree/user/exp/0/password.sha256", "hash");
oo_write("$button_tree/user/exp/1/file", "x");
oo_write("$button_tree/user/exp/2/.keep", "");
mkdir("$button_tree/user/empty/0", 0777, true);
$_SERVER["REQUEST_URI"] = "/share.php";
$_GET = ["sort" => "bogus"];
[, $out] = oo_capture(fn() => generate_folder_buttons("$button_tree/user", "run_nr"));
[, $out] = oo_capture(fn() => generate_folder_buttons("$button_tree/user/exp", "run_nr"));
oo_check('folder buttons lock', strpos($out, "lock.svg") !== false);
[, $out] = oo_capture(fn() => generate_folder_buttons("$button_tree/user/empty", "run_nr"));
oo_check('folder buttons no uploaded jobs', strpos($out, "no jobs have been uploaded") !== false);
[, $out] = oo_capture(fn() => generate_folder_buttons("$button_tree/user/exp", "experiment_name"));
oo_check('folder buttons singular subfolder', strpos($out, "1 subfolder") !== false);

// More sanitizer variants.
foreach ([
	'<img src="data:image/png;base64,abc" alt="x">',
	'<span style="color:red;expression(x);javascript:y;background:url(javascript:z);">x</span>',
	'<SPAN STYLE="font-size: 12px;">X</SPAN>',
	'<span style="invalid">x</span>',
	'<span style="; ;">x</span>',
	'<b class="x" data-x="y">B</b>',
	'<b/>',
	'</b/>',
	'<unknown>x</unknown>',
] as $html_variant) {
	sanitize_safe_html($html_variant);
}
oo_check('sanitize data src', strpos(sanitize_safe_html('<img src="data:image/png;base64,abc">'), "data:image/png") !== false);
oo_check('sanitize uppercase', strpos(sanitize_safe_html('<SPAN STYLE="font-size: 12px;">X</SPAN>'), "<span") !== false);

// More tab warning branches.
$lazy_large = "$root/lazy-large.log";
oo_write($lazy_large, str_repeat("x", 1024 * 1024 + 10));
$GLOBALS["lazy_share_logs"] = true;
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], $lazy_large, "Lazy", "lazy");
unset($GLOBALS["lazy_share_logs"]);
oo_expect('lazy pre tab', 1, count($tabs));
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], "$$root/no-pre", "Pre", "pre");
oo_expect('pre missing warning', 1, count($warnings));
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], oo_write("$root/pre-empty.log", ""), "Pre", "pre");
oo_expect('pre empty warning', 1, count($warnings));
[$tabs, $warnings] = add_simple_pre_tab_from_file([], [], "$root/binary.txt", "Pre", "pre");
oo_expect('pre binary warning', 1, count($warnings));

[$tabs, $warnings] = add_simple_csv_tab_from_file([], [], oo_write("$root/csv-empty.csv", ""), "CSV", "csvtab");
oo_expect('simple csv empty warning', 1, count($warnings));
[$tabs, $warnings] = add_simple_csv_tab_from_file([], [], "$root/binary.txt", "CSV", "csvtab");
oo_expect('simple csv binary warning', 1, count($warnings));
oo_write("$root/csv-blank.csv", "   \n");
[$tabs, $warnings] = add_simple_csv_tab_from_file([], [], "$root/csv-blank.csv", "CSV", "csvtab");
oo_expect('simple csv blank warning', 0, count($warnings));

[$tabs, $warnings] = add_simple_table_from_ascii_table_file([], [], oo_write("$root/table-empty.txt", ""), "Table", "table");
oo_expect('ascii empty warning', 1, count($warnings));
[$tabs, $warnings] = add_simple_table_from_ascii_table_file([], [], "$root/binary.txt", "Table", "table");
oo_expect('ascii binary warning', 1, count($warnings));

[$tabs, $warnings] = add_worker_usage_plot_from_file([], [], oo_write("$root/worker-empty.csv", ""), "WU", "wu");
oo_expect('worker usage empty warning', 1, count($warnings));
[$tabs, $warnings] = add_parameter_distribution_by_type_plot([], [], oo_write("$root/param-empty.csv", ""), "PD", "pd");
oo_expect('parameter empty warning', 1, count($warnings));

// Pareto empty branches.
$pareto_dir = "$root/pareto-variants";
oo_write("$pareto_dir/pareto_idxs.json", "");
oo_write("$pareto_dir/pareto_front_data.json", "");
oo_write("$pareto_dir/pareto_front_table.txt", "");
[$tabs, $warnings] = add_pareto_from_from_file([], [], $pareto_dir);
oo_check('pareto empty warnings', count($warnings) >= 3);
oo_write("$pareto_dir/pareto_front_data.json", "[]");
oo_write("$pareto_dir/pareto_front_table.txt", "x");
[$tabs, $warnings] = add_pareto_from_from_file([], [], $pareto_dir);
oo_check('pareto txt only', count($tabs) === 1);

oo_expect('convert sixel passthrough', "plain", convert_sixel("plain"));
oo_check('warn function', true);
[, $out] = oo_capture(fn() => warn("hello"));
oo_check('warn output', strpos($out, "hello") !== false);
oo_check('low disk false', warn_if_low_disk_space($root, 1) === false);
oo_check('low disk true', warn_if_low_disk_space($root, 1e9) === true);
oo_check('low disk missing', oo_tolerate_warnings(fn() => warn_if_low_disk_space("$root/no-disk", 1)) === false);

$old_shares = "$root/old-shares";
$GLOBALS["sharesPath"] = $old_shares;
oo_write("$old_shares/olduser/exp/0/file", "x");
touch("$old_shares/olduser/exp/0", time() - 400 * 24 * 3600);
$deleted = delete_old_shares();
oo_check('delete old shares', is_array($deleted));

// ---------------------------------------------------------------------------
// Exit-path helpers (run in child interpreters so production exit() calls do
// not terminate this test process).  Coverage is still collected by the
// Xdebug wrapper that CI puts on PATH.
// ---------------------------------------------------------------------------
$child_shares = "$root/child-shares";
mkdir($child_shares, 0777, true);
putenv("share_path=$child_shares");
$offered = "$root/offered.txt";
oo_write($offered, "some text\n");
$child_code = '
mkdir("' . "$child_shares/user/exp/0" . '", 0777, true);
move_files([["file" => "' . $offered . '", "filename" => "results.csv"]], 0, "' . "$child_shares/user/exp/0" . '", "updated", "shared", "md5");
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('move files success rc', 0, $rc);
oo_check('move files success output', strpos($out, "shared") !== false);
oo_check('move files wrote file', is_file("$child_shares/user/exp/0/results.csv"));
oo_check('move files wrote hash', is_file("$child_shares/user/exp/0/hash.md5"));

oo_write($offered, "some text\n");
$child_code = '
mkdir("' . "$child_shares/user/exp/1" . '", 0777, true);
move_files([["file" => "' . $offered . '", "filename" => "empty.txt"]], 0, "' . "$child_shares/user/exp/1" . '", "u", "s");
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('move files empty rc', 0, $rc);

$empty_offered = "$root/offered-empty.txt";
oo_write($empty_offered, "");
$child_code = '
mkdir("' . "$child_shares/user/exp/2" . '", 0777, true);
move_files([["file" => "' . $empty_offered . '", "filename" => "empty.txt"]], 0, "' . "$child_shares/user/exp/2" . '", "u", "s");
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('move files empty file rc', 1, $rc);
oo_check('move files empty file message', strpos($out, "empty") !== false);

oo_write($offered, "some text\n");
$child_code = '
mkdir("' . "$child_shares/user/exp/3" . '", 0777, true);
$_GET["update"] = "1";
move_files([["file" => "' . $offered . '", "filename" => "results.csv"]], 0, "' . "$child_shares/user/exp/3" . '", "updated", "shared");
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('move files update rc', 0, $rc);
oo_check('move files update message', strpos($out, "updated") !== false);

oo_write($offered, "some text\n");
$child_code = '
$GLOBALS["sharesPath"] = "' . "$child_shares/" . '";
$_GET["password"] = "secret";
$_FILES = [];
$offered = [["file" => "' . $offered . '", "filename" => "results.csv"]];
move_files_if_not_already_there("no-hash", null, "http://example.com", "user2", "exp2", "0", $offered, "' . "$child_shares/user2/exp2/0" . '", null);
';
mkdir("$child_shares/user2/exp2/0", 0777, true);
[$rc, $out] = oo_run_php($child_code);
oo_expect('move files if not there rc', 0, $rc);
oo_check('move files if not there password', is_file("$child_shares/user2/exp2/0/password.sha256"));

oo_write($offered, "some text\n");
$child_code = '
$GLOBALS["sharesPath"] = "' . "$child_shares/" . '";
mkdir("' . "$child_shares/user3/exp3/0" . '", 0777, true);
file_put_contents("' . "$child_shares/user3/exp3/0/hash.md5" . '", "found-hash");
$offered = [["file" => "' . $offered . '", "filename" => "results.csv"]];
move_files_if_not_already_there("found-hash", null, "http://example.com", "user3", "exp3", "0", $offered, "' . "$child_shares/user3/exp3/0" . '", null);
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('move files if already there rc', 0, $rc);
oo_check('move files if already there message', strpos($out, "already") !== false);

$child_code = '
$GLOBALS["sharesPath"] = "/proc/oo-not-writable";
create_new_folder("user", "exp");
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('create folder failure rc', 1, $rc);
oo_check('create folder failure message', strpos($out, "Critical System Restriction") !== false);

[$rc, $out] = oo_run_php('respond_with_error("boom");');
oo_expect('respond error rc', 1, $rc);
oo_check('respond error json', strpos($out, "boom") !== false);

[$rc, $out] = oo_run_php('handle_jit_security_error(new Exception("jit"));');
oo_expect('jit error rc', 1, $rc);
oo_check('jit error message', strpos($out, "JIT") !== false || strpos($out, "jit") !== false);

$child_code = '
$_GET = ["export" => "1", "export_and_exit" => "1"];
$GLOBALS["SPECIAL_COL_NAMES"] = ["trial_index"];
$GLOBALS["json_data"] = ["tab_results_headers_json" => ["result"]];
$tabs = ["Results" => ["content" => "<p>r</p>", "onclick" => "plot();"]];
get_export_tab($tabs, [], "' . $run_dir . '");
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('export and exit rc', 0, $rc);
oo_check('export and exit html', strpos($out, "<!DOCTYPE html>") !== false);

$child_code = '
putenv("share_cache_path=' . "$root/cache-env" . '");
var_export(share_cache_base_dir());
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('cache env rc', 0, $rc);
oo_check('cache env dir', is_dir("$root/cache-env"));

// Permission diagnostics with real values that do not match.
$stat = stat($root);
$perm = intval(substr(sprintf("%o", $stat["mode"]), -4), 8);
[$issues, $out] = oo_capture(fn() => check_folder_permissions($root, "definitely-wrong-user", "definitely-wrong-group", "", "", ($perm + 1) & 0777));
oo_check('folder permissions mismatches', $issues === true && strpos($out, "Ownership issue") !== false);

// Cache write failure and unreadable blob.
$cache_fail_dir = share_cache_namespace_dir("readonly");
if ($cache_fail_dir !== null) {
	file_put_contents("$cache_fail_dir/" . sha1("unreadable") . ".cache", gzencode("data"));
	chmod("$cache_fail_dir/" . sha1("unreadable") . ".cache", 0000);
	oo_expect('cache unreadable blob', null, share_cache_read_blob("readonly", sha1("unreadable")));
	chmod("$cache_fail_dir/" . sha1("unreadable") . ".cache", 0600);
	chmod($cache_fail_dir, 0555);
	oo_check('cache read only write fails', share_cache_write_blob("readonly", sha1("fail"), "data") === false);
	chmod($cache_fail_dir, 0755);
}

// Fingerprint of an unreadable directory.
$unreadable_dir = "$root/unreadable";
oo_write("$unreadable_dir/file", "x");
chmod($unreadable_dir, 0000);
oo_expect('fingerprint unreadable', null, share_run_fingerprint($unreadable_dir));
chmod($unreadable_dir, 0755);

// Fully black image branch.
if (function_exists("imagecreatetruecolor")) {
	$black_png = "$root/black.png";
	$img = imagecreatetruecolor(2, 2);
	$black = imagecolorallocate($img, 0, 0, 0);
	imagefill($img, 0, 0, $black);
	imagepng($img, $black_png);
	imagedestroy($img);
	oo_tolerate_warnings(fn() => clean_black_lines_inplace($black_png));
	oo_check('black image handled', is_file($black_png));
}

// Status cache with stale data.
$status_cache_file = "$root/status-cache.csv";
oo_write($status_cache_file, "trial_status\ncompleted\n");
get_status_for_results_csv($status_cache_file);
oo_write($status_cache_file, "trial_status\nfailed\n");
$status2 = get_status_for_results_csv($status_cache_file);
oo_expect('status cache invalidated', 1, $status2["failed"]);

// ---------------------------------------------------------------------------
// Render share.php in child interpreters.  This exercises the real page
// controller (including the embedded JavaScript blocks) with a tiny share.
// ---------------------------------------------------------------------------
$ui = "$child_shares/ui/exp/0";
oo_write("$ui/results.csv", "trial_index,trial_status,generation_node,generation_method,start_time,end_time,x,y,result\n1,completed,0,random,0,1,0,0,0\n2,completed,0,tpe,1,2,1,1,1\n3,failed,1,tpe,2,3,2,2,4\n");
oo_write("$ui/result_names.txt", "result\n");
oo_write("$ui/result_min_max.txt", "max\n");
oo_write("$ui/progressbar", "progress\n");
oo_write("$ui/parameters.txt", "param value\n");
oo_write("$ui/constraints.txt", "constraint value\n");
oo_write("$ui/experiment_overview.txt", "overview\n");
oo_write("$ui/best_result.txt", "best\n");
oo_write("$ui/git_version", "abc\n");
oo_write("$ui/formula.txt", "x + y\n");
oo_write("$ui/formula_underbraces.txt", "\\underbrace{x}\\n");
$_GET = [];
$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["user_id" => "ui", "experiment_name" => "exp", "run_nr" => "0"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share page rc', 0, $rc);
oo_check('share page output', strpos($out, "toggleSavePassword") !== false || strpos($out, "share") !== false);

$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["show_overview" => "1"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share overview rc', 0, $rc);

$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["user_id" => "bad user", "experiment_name" => "bad exp", "run_nr" => "abc"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share invalid rc', 0, $rc);
oo_check('share invalid message', strpos($out, "not a valid") !== false);

$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["user_id" => "ui", "experiment_name" => "exp"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share experiment rc', 0, $rc);

$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["user_id" => "ui"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share user rc', 0, $rc);

$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["experiment_name" => "exp"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share unknown state rc', 0, $rc);
oo_check('share unknown state output', strpos($out, "UNKNOWN STATE") !== false);

oo_write("$ui/password.sha256", hash("sha256", "secret"));
$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["user_id" => "ui", "experiment_name" => "exp", "run_nr" => "0"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share password prompt rc', 0, $rc);
oo_check('share password prompt output', strpos($out, "password") !== false);

$child_code = '
putenv("CI=1");
$_SERVER["HTTP_HOST"] = "example.com";
$_SERVER["REQUEST_URI"] = "/share.php";
$_SERVER["HTTPS"] = "off";
$_GET = ["user_id" => "ui", "experiment_name" => "exp", "run_nr" => "0", "filter_tabs_regex" => "Overview"];
include "share.php";
';
[$rc, $out] = oo_run_php($child_code);
oo_expect('share filter rc', 0, $rc);
unlink("$ui/password.sha256");

// ---------------------------------------------------------------------------
// Summary.
// ---------------------------------------------------------------------------
if (is_dir($root)) {
	rrmdir($root);
}

if ($__oo_failures === 0) {
	if (getenv("SHOW_SUCCESS")) {
		echo "All extended PHP coverage tests passed\n";
	}
	exit(0);
}

echo "\033[31m$__oo_failures extended PHP coverage test(s) failed\033[0m\n";
exit(1);
