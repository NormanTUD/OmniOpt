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


	$is_valid_user_id_test_cases = [
		['user123', true],
		['user_456', true],
		['user@123', false],
		['123abc', true],
		[null, false],
		['', false],
		['user name', false],
		['user123#', false],
		['user123!', false],
		['user123$', false]
	];

	foreach ($is_valid_user_id_test_cases as $test_case) {
		list($input, $expected) = $test_case;
		$result = is_valid_user_id($input);
		is_equal('is_valid_user_id test', $result, $expected);
	}

	$is_valid_experiment_name_test_cases = [
		['experiment1', true],
		['experiment-2', true],
		['experiment_3', true],
		['exp 4', false],
		[null, false],
		['', false],
		['exp#5', false],
		['exp@6', false],
		['experiment/7', false],
		['exp!8', false]
	];

	foreach ($is_valid_experiment_name_test_cases as $test_case) {
		list($input, $expected) = $test_case;
		$result = is_valid_experiment_name($input);
		is_equal('is_valid_experiment_name test', $result, $expected);
	}

	$is_valid_run_nr_test_cases = [
		['123', true],
		['0001', true],
		['5', true],
		['a123', false],
		[null, false],
		['', false],
		['12.34', false],
		['run123', false],
		['12abc', false],
		['123run', false]
	];

	foreach ($is_valid_run_nr_test_cases as $test_case) {
		list($input, $expected) = $test_case;
		$result = is_valid_run_nr($input);
		is_equal('is_valid_run_nr test', $result, $expected);
	}

	// Test für remove_extra_slashes_from_url
	$remove_extra_slashes_from_url_test_cases = [
		['http://example.com//path///to///file', 'http://example.com/path/to/file'],
		['https://www.example.com//about//us', 'https://www.example.com/about/us'],
		['ftp://example.com///files///docs', 'ftp://example.com/files/docs'],
		['http://example.com', 'http://example.com'], // Keine Änderungen notwendig
		['http://example.com////', 'http://example.com/'], // Überflüssige Schrägstriche am Ende
		['http://example.com//path//to//file/', 'http://example.com/path/to/file/'], // Schrägstriche am Ende entfernt
		['http://example.com////path', 'http://example.com/path'], // Schrägstriche am Anfang
		['http://example.com/path//to//file//', 'http://example.com/path/to/file/'], // Überflüssige Schrägstriche in der Mitte
		['http://example.com////file', 'http://example.com/file'], // Überflüssige Schrägstriche vor einem Ordner
		['https://example.com///', 'https://example.com/'], // Überflüssige Schrägstriche nach dem Protokoll
	];

	foreach ($remove_extra_slashes_from_url_test_cases as $test_case) {
		list($input, $expected) = $test_case;
		$result = remove_extra_slashes_from_url($input);
		is_equal('remove_extra_slashes_from_url test', $result, $expected);
	}

	$final_errors = min(255, $nr_of_errors);

	if ($final_errors > 0) {
		echo "\033[31mTotal errors: $final_errors\033[0m\n";
	} else {
		echo "No errors found\n";
	}
?>
