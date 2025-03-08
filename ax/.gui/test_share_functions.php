<?php
	$nr_of_errors = 0;

	include("share_functions.php");

	function is_equal($name, $expected, $got) {
		global $nr_of_errors;
		if ($expected !== $got) {
			$nr_of_errors++;
			echo "\033[31mTest $name failed, expected: $expected, got: $got\033[0m\n";
		}
	}

	function is_not_equal($name, $expected, $got) {
		global $nr_of_errors;
		if ($expected === $got) {
			$nr_of_errors++;
			echo "\033[31mTest $name failed, expected: not $expected, got: $got\033[0m\n";
		}
	}

	is_equal('Test 1', 5, 5);
	is_not_equal('Test 3', 5, 3);

	try {
		$result = validate_param('username', '/^[a-zA-Z0-9_]{3,16}$/', 'Invalid username');
		is_equal('Username validation', $result, 'valid_username');
	} catch (Exception $e) {
		is_not_equal('Username validation', $e->getMessage(), 'Invalid username');
	}

	$result = build_run_folder_path(123, 'experiment1', 5);
	is_equal('Build run folder path', $result, '123/experiment1/5/');

	$ansi_string = "\033[31mThis is red\033[0m and \033[32mthis is green\033[0m.";
	$html_result = ansi_to_html($ansi_string);

	$expected_html = '<span style="color:red;">This is red</span> and <span style="color:green;">this is green</span>.';

	is_equal('Ansi to HTML conversion', $html_result, $expected_html);

	// Test für copy_id_to_clipboard_string
	$id = '123abc';
	$html_result = copy_id_to_clipboard_string($id);

	// Erwartetes HTML: <button onclick='copy_to_clipboard_from_id("123abc")'>Copy raw data to clipboard</button>
	$expected_html = '<button onclick=\'copy_to_clipboard_from_id("123abc")\'>Copy raw data to clipboard</button>';

	is_equal('Copy ID to Clipboard button', $html_result, $expected_html);

	$final_errors = min(255, $nr_of_errors);
	if ($final_errors > 0) {
		echo "\033[31mTotal errors: $final_errors\033[0m\n";
	} else {
		echo "No errors found\n";
	}
?>
