<?php
/**
 * Test-only router for the PHP built-in server.
 *
 * The built-in CLI server does not populate ``$_FILES`` for multipart
 * requests, so we have to parse the multipart body ourselves before
 * delegating to ``share_internal.php``.
 *
 * In a production deployment (Apache + mod_php, Nginx + PHP-FPM, ...)
 * ``$_FILES`` is populated automatically and this router is NOT used.
 *
 * Pass the path to ``share_internal.php`` via the ``OO_TARGET_FILE``
 * environment variable.
 */

$OO_TARGET_FILE = getenv("OO_TARGET_FILE") ?: (__DIR__ . "/share_internal.php");

if (!file_exists($OO_TARGET_FILE)) {
    fwrite(STDERR, "OO_TARGET_FILE does not exist: $OO_TARGET_FILE\n");
    http_response_code(500);
    exit(1);
}

$contentType = $_SERVER["CONTENT_TYPE"] ?? "";

if (strpos($contentType, "multipart/form-data") !== false) {
    preg_match('/boundary=(?:"([^"]+)"|([^;]+))/i', $contentType, $matches);
    if (!empty($matches[1] ?? $matches[2] ?? "")) {
        $boundary = $matches[1] ?? $matches[2];
        $body = file_get_contents("php://input");
        $parts = preg_split(
            '/\\r?\\n?--' . preg_quote($boundary, '/') . '(?:--)?(?:\\r?\\n)?/',
            $body
        );
        foreach ($parts as $part) {
            if (trim($part) === "") {
                continue;
            }
            $hdrEnd = strpos($part, "\r\n\r\n");
            if ($hdrEnd === false) {
                continue;
            }
            $hdrs = substr($part, 0, $hdrEnd);
            $data = substr($part, $hdrEnd + 4);
            $data = preg_replace('/\\r\\n$/', '', $data);
            if (preg_match(
                '/Content-Disposition:\\s*form-data;\\s*name="([^"]+)"(?:;\\s*filename="([^"]+)")?/i',
                $hdrs,
                $m
            )) {
                $name = $m[1];
                $filename = $m[2] ?? null;
                if ($filename) {
                    $tmp = tempnam(sys_get_temp_dir(), "oo_share_test_");
                    file_put_contents($tmp, $data);
                    $_FILES[$name] = [
                        "name" => $filename,
                        "type" => "application/octet-stream",
                        "tmp_name" => $tmp,
                        "error" => 0,
                        "size" => strlen($data),
                    ];
                } else {
                    $_POST[$name] = $data;
                }
            }
        }
    }
}

require $OO_TARGET_FILE;
