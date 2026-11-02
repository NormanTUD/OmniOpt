<?php
/**
 * Xdebug coverage bootstrap for the OmniOpt test-suite.
 *
 * This file is installed as PHP's auto_prepend_file by
 * _framework/coverage_utils.enable_php_coverage().  It starts line coverage
 * for the current CLI process or PHP built-in-server request and writes one
 * JSON file per process/request into OMNIOPT_PHP_COVERAGE_DIR.  These files
 * are combined by .tests/coverage_report.
 */

$__oo_coverage_dir = getenv("OMNIOPT_PHP_COVERAGE_DIR");

if ($__oo_coverage_dir && function_exists("xdebug_start_code_coverage")) {
    xdebug_start_code_coverage(XDEBUG_CC_UNUSED | XDEBUG_CC_DEAD_CODE);

    register_shutdown_function(function () use ($__oo_coverage_dir) {
        if (!function_exists("xdebug_get_code_coverage")) {
            return;
        }

        $data = xdebug_get_code_coverage();
        if (!is_array($data) || !$data) {
            return;
        }

        if (!is_dir($__oo_coverage_dir)) {
            @mkdir($__oo_coverage_dir, 0777, true);
        }

        $name = "php-coverage-" . getmypid() . "-" . str_replace(".", "", uniqid("", true)) . ".json";
        $payload = array(
            "pid" => getmypid(),
            "files" => $data,
        );
        @file_put_contents($__oo_coverage_dir . DIRECTORY_SEPARATOR . $name, json_encode($payload));
    });
}
