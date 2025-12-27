<?php
/**
 * Automated Unit Tests for share_functions.php and _functions.php
 */

require_once 'share_functions.php'; // Includes _functions.php internally

$failedTests = 0;

/**
 * Standard Expectation Handler
 */
function expect($label, $actual, $expected) {
    global $failedTests;
    if ($actual === $expected) {
        echo "✅ PASS: $label\n";
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
echo "--- Testing: get_or_env ---\n";
$_GET['UNIT_TEST_VAR'] = 'value_from_get';
expect("get_or_env: retrieves from \$_GET", get_or_env('UNIT_TEST_VAR'), 'value_from_get');

putenv("UNIT_TEST_ENV=value_from_env");
expect("get_or_env: retrieves from ENV if GET missing", get_or_env('UNIT_TEST_ENV'), 'value_from_env');

unset($_GET['UNIT_TEST_VAR']);
// In _functions.php get_or_env returns false via getenv() if not found
expect("get_or_env: returns false if both missing", get_or_env('NON_EXISTENT_CONSTANT_XYZ'), false);


// --- Group: extract_help_params_from_bash ---
echo "\n--- Testing: extract_help_params_from_bash ---\n";
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
echo "\n--- Testing: getRunProgramFromFile ---\n";
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
echo "\n--- Testing: getJobIdFromFile ---\n";
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
echo "\n--- Testing: Folder Functions ---\n";
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
echo "\n--- Testing: Security & Edge Cases ---\n";
expect("getJobIdFromFile: handles non-string", getJobIdFromFile(['not a string']), "");
expect("get_valid_folders: non-existent path", get_valid_folders("/tmp/missing_" . uniqid()), []);


// =================================================================
// FINISH
// =================================================================
echo "\n---------------------------------\n";
if ($failedTests === 0) {
    echo "SUMMARY: All tests passed successfully.\n";
    exit(0);
} else {
    echo "SUMMARY: $failedTests test(s) failed.\n";
    exit(1);
}
