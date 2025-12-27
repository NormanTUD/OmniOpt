<?php
/**
 * Automated Unit Tests for share_functions.php
 */

require_once 'share_functions.php';

$failedTests = 0;

/**
 * Standard Expectation Handler
 */
function expect($label, $actual, $expected) {
    global $failedTests;
    // For arrays, we sort or use a loose comparison if order doesn't matter, 
    // but here we use strict for values.
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
    else if (file_exists($tmp)) unlink($tmp); // Simulate non-existent
    
    $result = $callback($tmp);
    if (file_exists($tmp)) unlink($tmp);
    return $result;
}

/**
 * Helper: Creates a temp directory structure for folder tests.
 */
function create_test_dir($structure) {
    $base = sys_get_temp_dir() . '/php_test_dir_' . uniqid();
    mkdir($base);
    foreach ($structure as $name => $content) {
        $path = "$base/$name";
        if (is_array($content)) {
            mkdir($path);
            foreach ($content as $subName => $subContent) {
                file_put_contents("$path/$subName", $subContent);
            }
        } else {
            mkdir($path); // Create empty dir if not array
        }
    }
    return $base;
}

// Clean up helper for directories
function rmdir_recursive($dir) {
    foreach (scandir($dir) as $file) {
        if ($file === '.' || $file === '..') continue;
        $path = "$dir/$file";
        is_dir($path) ? rmdir_recursive($path) : unlink($path);
    }
    rmdir($dir);
}

// =================================================================
// TESTS
// =================================================================

echo "Starting Tests...\n\n";

// --- Group: getRunProgramFromFile ---
echo "--- Testing: getRunProgramFromFile ---\n";
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

// --- Group: Folder Functions ---
echo "\n--- Testing: Folder Functions ---\n";

// Setup a dummy environment
$testPath = create_test_dir([
    "valid-folder"    => ["file.txt" => "hello"],
    "invalid_folder!" => [],       // Contains special char '!'
    "99_important"    => ["a.log" => "data"],
    ".hidden"         => []        // Starts with dot
]);

// 1. Test get_valid_folders
$folders = get_valid_folders($testPath);
sort($folders); // Sort for comparison stability
expect("get_valid_folders filters invalid names", $folders, ["99_important", "valid-folder"]);

// 2. Test get_latest_recursive_modification_time
$folderPath = "$testPath/99_important";
$mtime = get_latest_recursive_modification_time($folderPath);
expect("Recursive mtime returns a timestamp", is_numeric($mtime) && $mtime > 0, true);

// 3. Test sort_folders_by_modification_time
// Create one folder that is definitely "newer"
sleep(1); 
$newFolderPath = "$testPath/valid-folder/new.txt";
file_put_contents($newFolderPath, "newer content");
clearstatcache();

$folderList = ["99_important", "valid-folder"];
sort_folders_by_modification_time($testPath, $folderList);
// valid-folder should now be first because it was modified last
expect("Sorting folders by time (newest first)", $folderList[0], "valid-folder");

// Cleanup
rmdir_recursive($testPath);

// =================================================================
// FINISH
// =================================================================
echo "\n---------------------------------\n";
if ($failedTests === 0) {
    echo "SUMMARY: All tests passed.\n";
    exit(0);
} else {
    echo "SUMMARY: $failedTests test(s) failed.\n";
    exit(1);
}
