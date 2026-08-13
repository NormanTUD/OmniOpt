<?php
	$GLOBALS["time_start"] = microtime(true);
	if(!defined('STDERR')) define('STDERR', fopen('php://stderr', 'wb'));

	$GLOBALS["modificationCache"] = [];
	$GLOBALS["recursiveModificationCache"] = [];
	$GLOBALS["ascii_or_utf8_cache"] = [];

	$GLOBALS["sharesPath"] = "shares/";
	$env_share_path = getenv("share_path");
	if ($env_share_path && is_dir($env_share_path) && !preg_match("/\.\./", $env_share_path)) {
		$GLOBALS["sharesPath"] = rtrim($env_share_path, "/") . "/";
	}

	error_reporting(E_ALL);
	set_error_handler(
		function ($severity, $message, $file, $line) {
			throw new \ErrorException($message, $severity, $severity, $file, $line);
		}
	);

	ini_set('display_errors', 1);
	set_time_limit(300);
?>
