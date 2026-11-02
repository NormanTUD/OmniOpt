<?php
	header("Content-Type: application/json; charset=utf-8");
	header("Cache-Control: no-store");

	$root = __DIR__;
	$urls = [];

	function add_offline_file(array &$urls, string $root, string $path): void {
		if (is_file($root . "/" . $path)) {
			$urls[$path] = true;
		}
	}

	function add_offline_directory(array &$urls, string $root, string $directory, array $extensions): void {
		$full_path = $root . "/" . $directory;

		if (!is_dir($full_path)) {
			return;
		}

		$iterator = new RecursiveIteratorIterator(
			new RecursiveDirectoryIterator($full_path, FilesystemIterator::SKIP_DOTS)
		);

		foreach ($iterator as $file) {
			if (!$file->isFile()) {
				continue;
			}

			if (!in_array(strtolower($file->getExtension()), $extensions, true)) {
				continue;
			}

			$path = str_replace("\\", "/", substr($file->getPathname(), strlen($root) + 1));
			$urls[$path] = true;
		}
	}

	$top_level_extensions = ["css", "js", "svg", "png", "jpg", "jpeg", "gif", "ico", "webp"];

	foreach (glob($root . "/*") as $file) {
		if (!is_file($file)) {
			continue;
		}

		$name = basename($file);
		$extension = strtolower(pathinfo($name, PATHINFO_EXTENSION));

		if (str_contains($name, "_tests") || $name === "service-worker.js") {
			continue;
		}

		if (in_array($extension, $top_level_extensions, true)) {
			$urls[$name] = true;
		}
	}

	add_offline_directory($urls, $root, "css", ["css", "js", "woff", "woff2", "ttf", "otf", "eot", "svg", "png", "gif", "webp"]);
	add_offline_directory($urls, $root, "js", ["js"]);
	add_offline_directory($urls, $root, "libs", ["js", "css", "woff", "woff2", "ttf", "otf", "eot", "svg", "png", "gif", "webp", "json", "wasm"]);
	add_offline_directory($urls, $root, "emojis", ["svg", "png", "gif", "webp"]);
	add_offline_directory($urls, $root, "imgs", ["svg", "png", "jpg", "jpeg", "gif", "webp"]);
	add_offline_directory($urls, $root, "images", ["svg", "png", "jpg", "jpeg", "gif", "webp"]);
	add_offline_directory($urls, $root, "logos", ["svg", "png", "jpg", "jpeg", "gif", "webp"]);
	add_offline_directory($urls, $root, "i", ["svg", "png", "gif", "webp"]);
	add_offline_directory($urls, $root, "icons", ["svg", "png"]);
	add_offline_directory($urls, $root, "documentation", ["svg", "png", "jpg", "jpeg", "gif", "webp"]);

	$pages = [
		"index.php",
		"gui.php",
		"tutorials.php",
		"share.php",
		"usage_stats.php",
		"search.php",
		"404.php",
		"offline.html",
		"manifest.webmanifest"
	];

	foreach ($pages as $page) {
		add_offline_file($urls, $root, $page);
	}

	$rewrite_pages = ["index", "gui", "tutorials", "share", "usage_stats", "search"];

	foreach ($rewrite_pages as $page) {
		if (is_file($root . "/" . $page . ".php")) {
			$urls[$page] = true;
		}
	}

	$tutorial_files = glob($root . "/_tutorials/*.{md,php}", GLOB_BRACE);

	foreach ($tutorial_files as $file) {
		$tutorial_name = pathinfo($file, PATHINFO_FILENAME);

		if ($tutorial_name === "favicon") {
			continue;
		}

		$tutorial_name = rawurlencode($tutorial_name);
		$urls["tutorials.php?tutorial=" . $tutorial_name] = true;
		$urls["tutorials?tutorial=" . $tutorial_name] = true;
	}

	$asset_urls = array_keys($urls);
	sort($asset_urls);

	echo json_encode($asset_urls, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
?>
