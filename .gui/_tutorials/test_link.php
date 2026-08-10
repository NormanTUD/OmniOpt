<?php
$file_path = "../omniopt";
echo "file_exists $file_path: " . (file_exists($file_path) ? "yes" : "no") . "\n";
echo "file_exists $file_path.py: " . (file_exists($file_path . ".py") ? "yes" : "no") . "\n";
$result = @symlink($file_path, $file_path . ".py");
echo "symlink result: " . ($result ? "ok" : "FAILED") . "\n";
echo "Last error: " . error_get_last()["message"] . "\n";
