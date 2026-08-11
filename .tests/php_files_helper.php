<?php
/**
 * Helper for `.tests/php_files.py`: syntax-check every *.php file
 * under the directory passed as $argv[1] in a single PHP process.
 *
 * Prints "<path>\n<stdout of php -l>" for every failing file to
 * STDERR and exits 0 when everything is fine.
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
    $out = [];
    $rc = 0;
    exec("php -l " . escapeshellarg($path) . " 2>&1", $out, $rc);
    if ($rc !== 0) {
        fwrite(STDERR, $path . "\n" . implode("\n", $out) . "\n");
        $bad++;
    }
}
exit($bad === 0 ? 0 : 1);
