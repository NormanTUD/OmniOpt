<?php
	$GLOBALS["time_start"] = microtime(true);
	if(!defined('STDERR')) define('STDERR', fopen('php://stderr', 'wb'));

	$GLOBALS["modificationCache"] = [];
	$GLOBALS["recursiveModificationCache"] = [];
	$GLOBALS["ascii_or_utf8_cache"] = [];

	$GLOBALS["sharesPath"] = "shares/";

	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);
?>
