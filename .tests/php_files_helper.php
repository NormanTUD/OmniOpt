<?php
/**
 * Helper for `.tests/php_files.py`: syntax-check every *.php file
 * under the directory passed as $argv[1] in a single PHP process.
 *
 * Uses `token_get_all()` directly so we don't have to pay PHP's
 * startup cost once per file (the original `php -l` loop took ~9 s
 * for ~40 files).
 */

$dir = $argv[1];

$rii = new RecursiveIteratorIterator(
    new RecursiveDirectoryIterator($dir, FilesystemIterator::SKIP_DOTS)
);
$bad = 0;
foreach ($rii as $f) {
    if ($f->getExtension() !== "php") {
        continue;
    }
    $path = $f->getPathname();
    $code = @file_get_contents($path);
    if ($code === false) {
        continue;
    }
    // Suppress notices from token_get_all() on bad code so we can
    // surface our own error message.
    $prev = error_reporting(0);
    try {
        $tokens = @token_get_all($code, TOKEN_PARSE);
    } finally {
        error_reporting($prev);
    }
    if ($tokens === false) {
        $err = error_get_last();
        $msg = $err["message"] ?? "unknown syntax error";
        // Strip the filename prefix that PHP prepends.
        $msg = preg_replace("/^.*?:\s*/", "", $msg);
        fwrite(STDERR, $path . "\n" . $msg . "\n");
        $bad++;
    }
}
exit($bad === 0 ? 0 : 1);

