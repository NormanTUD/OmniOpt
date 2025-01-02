<?php
function myprint($arg) {
	print("$arg<br>\n");
}

// Mindestanforderungen festlegen
$required_post_max_size = 32 * 1024 * 1024; // 100M in Bytes
$required_upload_max_filesize = 32 * 1024 * 1024; // 100M in Bytes
$required_max_file_uploads = 10000;

// Funktion zum Konvertieren von PHP-Einstellungen in Bytes
function convertToBytes($val) {
	$val = trim($val);
	$last = strtolower($val[strlen($val) - 1]);
	$val = (int)$val;

	switch ($last) {
	case 'g':
		$val *= 1024;
	case 'm':
		$val *= 1024;
	case 'k':
		$val *= 1024;
	}

	return $val;
}

// Aktuelle PHP-Einstellungen abfragen
$current_post_max_size = convertToBytes(ini_get('post_max_size'));
$current_upload_max_filesize = convertToBytes(ini_get('upload_max_filesize'));
$current_max_file_uploads = (int)ini_get('max_file_uploads');

$ini_file = php_ini_loaded_file();

if ($current_post_max_size < $required_post_max_size) {
	myprint("Message: post_max_size is too small. Required: 100M, Current: " . ini_get('post_max_size') . ". Config File: " . $ini_file);
}

if ($current_upload_max_filesize < $required_upload_max_filesize) {
	myprint("Message: upload_max_filesize is too small. Required: 100M, Current: " . ini_get('upload_max_filesize') . ". Config File: " . $ini_file);
}

if ($current_max_file_uploads < $required_max_file_uploads) {
	myprint("Message: max_file_uploads is too small. Required: 10000, Current: " . ini_get('max_file_uploads') . ". Config File: " . $ini_file);
}

if(!function_exists("mb_detect_encoding")) {
	myprint("The function 'mb_detect_encoding' was not found. Try installing 'php-mbstring' on the server.");
}
