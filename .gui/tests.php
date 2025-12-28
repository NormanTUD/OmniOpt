<?php
/**
 * Automated Unit Tests for share_functions.php and _functions.php
 */

require_once 'share_functions.php'; // Includes _functions.php internally

function echo_if_wanted($param) {
	if(getenv("SHOW_SUCCESS")) {
		echo $param;
	}
}

$failedTests = 0;

/**
 * Standard Expectation Handler
 */
function expect($label, $actual, $expected) {
    global $failedTests;
    if ($actual === $expected) {
        echo_if_wanted("✅ PASS: $label\n");
    } else {
        echo "❌ FAIL: $label\n";
        echo "   Expected: " . json_encode($expected) . "\n";
        echo "   Actual:   " . json_encode($actual) . "\n";
        $failedTests++;
    }
}

/**
 * Helper: Creates a temp file, runs a test, then deletes it.
 */
function test_file_helper($content, $callback) {
    $tmp = tempnam(sys_get_temp_dir(), 'php_test_');
    if ($content !== null) file_put_contents($tmp, $content);
    else if (file_exists($tmp)) unlink($tmp); 
    
    $result = $callback($tmp);
    if (file_exists($tmp)) unlink($tmp);
    return $result;
}

/**
 * Helper: Creates a temp directory structure for folder tests.
 */
function create_test_dir($structure) {
    $base = sys_get_temp_dir() . '/php_test_dir_' . uniqid();
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

echo "Starting Comprehensive Unit Tests...\n\n";

// --- Group: get_or_env (from _functions.php) ---
echo_if_wanted("--- Testing: get_or_env ---\n");
$_GET['UNIT_TEST_VAR'] = 'value_from_get';
expect("get_or_env: retrieves from \$_GET", get_or_env('UNIT_TEST_VAR'), 'value_from_get');

putenv("UNIT_TEST_ENV=value_from_env");
expect("get_or_env: retrieves from ENV if GET missing", get_or_env('UNIT_TEST_ENV'), 'value_from_env');

unset($_GET['UNIT_TEST_VAR']);
// In _functions.php get_or_env returns false via getenv() if not found
expect("get_or_env: returns false if both missing", get_or_env('NON_EXISTENT_CONSTANT_XYZ'), false);


// --- Group: extract_help_params_from_bash ---
echo_if_wanted("\n--- Testing: extract_help_params_from_bash ---\n");
$bashHelp = '
function help {
    echo "  --mode      Set the operation mode";
    echo "  --verbose   Enable detailed logging";
    exit 1
}';
$resHelp = test_file_helper($bashHelp, fn($p) => extract_help_params_from_bash($p));
expect("Help: Contains --mode", str_contains($resHelp, "--mode"), true);
expect("Help: Contains --verbose", str_contains($resHelp, "--verbose"), true);
expect("Help: Logic generated HTML table", str_contains($resHelp, "<table"), true);


// --- Group: getRunProgramFromFile (from share_functions.php) ---
echo_if_wanted("\n--- Testing: getRunProgramFromFile ---\n");
$runCases = [
    "Simple program line"      => ["content" => "Run-Program: myapp.sh", "expected" => "myapp.sh"],
    "Program line with spaces" => ["content" => "Run-Program:   /usr/bin/python  ", "expected" => "/usr/bin/python"],
    "Line with ANSI colors"    => ["content" => "\x1b[31mRun-Program: colored_app\x1b[0m", "expected" => "colored_app"],
    "Empty program name"       => ["content" => "Run-Program: ", "expected" => ""],
    "Missing line"             => ["content" => "Just some text", "expected" => ""],
];
foreach ($runCases as $label => $data) {
    $res = test_file_helper($data['content'], fn($p) => getRunProgramFromFile($p));
    expect($label, $res, $data['expected']);
}


// --- Group: getJobIdFromFile (from share_functions.php) ---
echo_if_wanted("\n--- Testing: getJobIdFromFile ---\n");
$jobIdCases = [
    "Valid ID with single quotes" => ["content" => "scancel '12345'", "expected" => "12345"],
    "Valid ID with double quotes" => ["content" => 'scancel "67890"', "expected" => "67890"],
    "Valid ID with ANSI colors"   => ["content" => "\x1b[32mscancel 555\x1b[0m", "expected" => "555"],
    "Invalid: Alpha characters"   => ["content" => "scancel 'abc'", "expected" => ""],
    "Empty file"                  => ["content" => "", "expected" => ""],
];
foreach ($jobIdCases as $label => $data) {
    $result = test_file_helper($data['content'], fn($path) => getJobIdFromFile($path));
    expect($label, $result, $data['expected']);
}


// --- Group: Folder Functions ---
echo_if_wanted("\n--- Testing: Folder Functions ---\n");
$testPath = create_test_dir([
    "valid-folder"    => ["file.txt" => "hello"],
    "invalid_folder!" => [],       
    "99_important"    => ["a.log" => "data"],
    ".hidden"         => []        
]);

if ($testPath) {
    $folders = get_valid_folders($testPath);
    sort($folders);
    expect("get_valid_folders: filters invalid names", $folders, ["99_important", "valid-folder"]);

    $folderPath = "$testPath/99_important";
    $mtime = get_latest_recursive_modification_time($folderPath);
    expect("get_latest_recursive_modification_time: returns timestamp", is_numeric($mtime) && $mtime > 0, true);

    sleep(1); 
    file_put_contents("$testPath/valid-folder/new.txt", "newer");
    clearstatcache();

    $folderList = ["99_important", "valid-folder"];
    sort_folders_by_modification_time($testPath, $folderList);
    expect("sort_folders_by_modification_time: newest first", $folderList[0], "valid-folder");

    rmdir_recursive($testPath);
}


// --- Group: Security & Edge Cases ---
echo_if_wanted("\n--- Testing: Security & Edge Cases ---\n");
expect("getJobIdFromFile: handles non-string", getJobIdFromFile(['not a string']), "");
expect("get_valid_folders: non-existent path", get_valid_folders("/tmp/missing_" . uniqid()), []);

// --- Group: Encoding ---
echo_if_wanted("\n--- Testing: Encoding ---\n");
$inputArray = ["umlaut" => "äöü", "nested" => ["key" => "ß"]];
expect("utf8ize: handles umlauts in arrays", utf8ize($inputArray), $inputArray);
expect("utf8ize: handles plain string", utf8ize("test"), "test");

// --- Group: Path Building ---
echo_if_wanted("\n--- Testing: Path Building ---\n");
expect(
    "build_run_folder_path: generates correct structure", 
    build_run_folder_path("user1", "exp_alpha", "42"), 
    "user1/exp_alpha/42/"
);

// --- Group: CSV Processing ---
echo_if_wanted("\n--- Testing: CSV Normalization ---\n");
expect("normalize_csv_value: converts float-integers to string ints", normalize_csv_value("10.000"), "10");
expect("normalize_csv_value: keeps real floats", normalize_csv_value("10.5"), "10.5");
expect("normalize_csv_value: handles empty string", normalize_csv_value(""), "");

// --- Group: Data Filtering ---
echo_if_wanted("\n--- Testing: CSV Time Filtering ---\n");
$csvData = [
    ["timestamp", "value"],
    ["1000", "A"],
    ["1030", "B"], // Sollte übersprungen werden bei 60s Intervall
    ["1061", "C"]  // Sollte behalten werden (1061 - 1000 > 60)
];
$filtered = keep_rows_every_n_seconds($csvData, 60);
expect("keep_rows_every_n_seconds: filters correctly", count($filtered), 3); // Header + 1000 + 1061
expect("keep_rows_every_n_seconds: keeps last valid row", $filtered[2][1], "C");

// --- Group: File Validation ---
echo_if_wanted("\n--- Testing: SVG Validation ---\n");
$validSvg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>';
$invalidSvg = '<div xmlns="http://www.w3.org/2000/svg">Not an SVG</div>';

expect("is_valid_svg_file: identifies valid SVG", test_file_helper($validSvg, 'is_valid_svg_file'), true);
expect("is_valid_svg_file: rejects invalid root tag", test_file_helper($invalidSvg, 'is_valid_svg_file'), false);

echo_if_wanted("\n--- Testing: highlight_backticks ---\n");
expect("highlight_backticks: wraps text in tt tags", highlight_backticks("Dies ist `code`."), "Dies ist <tt>code</tt>.");
expect("highlight_backticks: handles multiple backticks", highlight_backticks("`a` and `b`"), "<tt>a</tt> and <tt>b</tt>");

echo_if_wanted("\n--- Testing: convertNewlinesToBr ---\n");
$nlText = "Line 1\n\nLine 2\n\n\nLine 3";
$expectedNl = "Line 1\n<br>Line 2\n<br><br>Line 3"; 
expect("convertNewlinesToBr: converts double newlines to br", convertNewlinesToBr($nlText), $expectedNl);

// --- Group: File & Directory Logic ---
echo_if_wanted("\n--- Testing: build_run_folder_path ---\n");
expect("build_run_folder_path: constructs correct path", build_run_folder_path("user1", "expA", 5), "user1/expA/5/");

echo_if_wanted("\n--- Testing: validate_directory ---\n");
$tempDir = sys_get_temp_dir() . '/test_dir_' . uniqid();
mkdir($tempDir);
try {
    validate_directory($tempDir);
    echo_if_wanted("✅ PASS: validate_directory: identifies existing directory\n");
} catch (Exception $e) {
    echo "❌ FAIL: validate_directory: should have found directory\n";
    $failedTests++;
}
rmdir($tempDir);
try {
    validate_directory("/path/to/nonexistent/dir/12345");
    echo "❌ FAIL: validate_directory: should have thrown exception for missing dir\n";
    $failedTests++;
} catch (Exception $e) {
    echo_if_wanted("✅ PASS: validate_directory: throws exception for missing directory\n");
}

echo_if_wanted("\n--- Testing: read_file_as_array ---\n");
$fileContent = "Line 1\n  \nLine 2\r\nLine 3";
$readArray = test_file_helper($fileContent, 'read_file_as_array');
expect("read_file_as_array: filters empty lines and trims", count($readArray), 3);
expect("read_file_as_array: correct first element", $readArray[0], "Line 1");

// --- Group: Specialized Parsing ---
echo_if_wanted("\n--- Testing: get_status_for_results_csv ---\n");
$csvContent = "id,name,trial_status\n1,test,completed\n2,test,failed\n3,test,running";
$statusResult = test_file_helper($csvContent, 'get_status_for_results_csv');
expect("get_status_for_results_csv: counts succeeded", $statusResult["succeeded"], 1);
expect("get_status_for_results_csv: counts failed", $statusResult["failed"], 1);
expect("get_status_for_results_csv: counts running", $statusResult["running"], 1);
expect("get_status_for_results_csv: counts total", $statusResult["total"], 3);

// --- Group: HTML Sanitization ---
echo_if_wanted("\n--- Testing: sanitize_safe_html ---\n");
$unsafeHtml = "<b>Safe</b><script>alert(1)</script><img src='https://example.com' onclick='bad()'>";
$safeHtml = sanitize_safe_html($unsafeHtml);
expect("sanitize_safe_html: removes script tags", strpos($safeHtml, "<script>"), false);
expect("sanitize_safe_html: keeps allowed tags (b)", strpos($safeHtml, "<b>Safe</b>") !== false, true);
expect("sanitize_safe_html: removes dangerous attributes (onclick)", strpos($safeHtml, "onclick"), false);

// =================================================================
// FINISH
// =================================================================
echo "\n---------------------------------\n";
if ($failedTests === 0) {
    echo "SUMMARY: All php-unit-tests tests passed successfully.\n";
    exit(0);
} else {
    echo "SUMMARY: $failedTests php-unit-test(s) failed.\n";
    exit(1);
}
