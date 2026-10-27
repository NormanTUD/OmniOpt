<?php
	/**
	 * Unit tests for the UUID -> run-folder routing rules used by
	 * share_internal.php.
	 *
	 * Invariants under test:
	 *
	 *   1. A UUID that exists for the same user + experiment updates that
	 *      exact folder.
	 *   2. A UUID that does not exist (or belongs to another user /
	 *      experiment) starts a new job in a new folder.
	 *   3. A missing/empty UUID always starts a new job -- it must never be
	 *      merged into the newest existing share.
	 *   4. Creating a new job never modifies an old job's folder.
	 */

	$nr_of_errors = 0;

	$GLOBALS["sharesPath"] = rtrim(sys_get_temp_dir(), "/")
		. "/oo_share_uuid_routing_" . bin2hex(random_bytes(6)) . "/";

	if (!is_dir($GLOBALS["sharesPath"])) {
		mkdir($GLOBALS["sharesPath"], 0777, true);
	}

	// globals.php honors $share_path; keep the tests out of the real
	// .gui/shares directory even when the caller has that env var set.
	putenv("share_path=" . $GLOBALS["sharesPath"]);

	include("share_functions.php");

	// globals.php normalizes the trailing slash; keep both in sync.
	$GLOBALS["sharesPath"] = rtrim($GLOBALS["sharesPath"], "/") . "/";

	function check_true($name, $condition, $detail = "") {
		global $nr_of_errors;
		if (!$condition) {
			$nr_of_errors++;
			echo "\033[31mFAIL: $name" . ($detail !== "" ? " ($detail)" : "") . "\033[0m\n";
		}
	}

	function check_equal($name, $expected, $got) {
		global $nr_of_errors;
		if ($expected !== $got) {
			$nr_of_errors++;
			echo "\033[31mFAIL: $name, expected: " . var_export($expected, true)
				. ", got: " . var_export($got, true) . "\033[0m\n";
		}
	}

	function check_path($name, $expected, $got) {
		// The production code appends "/" to $sharesPath (which already has
		// one) when globbing, so returned paths can contain "//".  Compare
		// normalized paths instead of raw strings.
		$norm = function ($p) {
			return is_string($p) ? preg_replace('#/+#', '/', $p) : $p;
		};
		check_equal($name, $norm($expected), $norm($got));
	}

	function remove_tree($path) {
		if (!is_dir($path)) {
			if (file_exists($path)) {
				@unlink($path);
			}
			return;
		}
		foreach (scandir($path) as $entry) {
			if ($entry === "." || $entry === "..") {
				continue;
			}
			remove_tree("$path/$entry");
		}
		@rmdir($path);
	}

	function reset_shares() {
		$path = $GLOBALS["sharesPath"];
		if (is_dir($path)) {
			remove_tree($path);
		}
		mkdir($path, 0777, true);
	}

	function make_run($user_id, $experiment_name, $run_nr, $uuid, $content = "content\n", $legacy_uuid = false) {
		$dir = $GLOBALS["sharesPath"] . "$user_id/$experiment_name/$run_nr";
		if (!is_dir($dir)) {
			mkdir($dir, 0777, true);
		}
		file_put_contents("$dir/results.csv", $content);
		if ($uuid !== null) {
			if ($legacy_uuid) {
				file_put_contents("$dir/run_uuid", $uuid . "\n");
			} else {
				if (!is_dir("$dir/state_files")) {
					mkdir("$dir/state_files", 0777, true);
				}
				file_put_contents("$dir/state_files/run_uuid", $uuid . "\n");
			}
		}
		return $dir;
	}

	function run_folders($user_id, $experiment_name) {
		$base = $GLOBALS["sharesPath"] . "$user_id/$experiment_name";
		if (!is_dir($base)) {
			return [];
		}
		$out = [];
		foreach (scandir($base) as $entry) {
			if ($entry === "." || $entry === "..") {
				continue;
			}
			if (ctype_digit($entry) && is_dir("$base/$entry")) {
				$out[] = $entry;
			}
		}
		sort($out, SORT_NUMERIC);
		return $out;
	}

	// -----------------------------------------------------------------
	// find_matching_uuid_run_folder
	// -----------------------------------------------------------------

	reset_shares();
	$run0 = make_run("alice", "exp", "0", "uuid-old", "old\n");
	check_path(
		"find_matching finds state_files/run_uuid",
		$run0,
		find_matching_uuid_run_folder("uuid-old", "alice", "exp")
	);

	reset_shares();
	$run0 = make_run("alice", "exp", "0", "uuid-old", "old\n", true);
	check_path(
		"find_matching finds legacy top-level run_uuid",
		$run0,
		find_matching_uuid_run_folder("uuid-old", "alice", "exp")
	);

	reset_shares();
	make_run("alice", "exp", "0", "uuid-old");
	check_equal(
		"find_matching returns null for unknown UUID",
		null,
		find_matching_uuid_run_folder("uuid-new", "alice", "exp")
	);

	reset_shares();
	$run0 = make_run("alice", "exp", "0", "uuid-old");
	check_path(
		"find_matching trims whitespace in target UUID",
		$run0,
		find_matching_uuid_run_folder("  uuid-old \n", "alice", "exp")
	);

	reset_shares();
	make_run("alice", "exp", "0", "  uuid-old  \n");
	check_path(
		"find_matching trims whitespace in stored UUID",
		$GLOBALS["sharesPath"] . "alice/exp/0",
		find_matching_uuid_run_folder("uuid-old", "alice", "exp")
	);

	reset_shares();
	make_run("alice", "exp", "0", "uuid-shared");
	check_equal(
		"find_matching does not match another experiment",
		null,
		find_matching_uuid_run_folder("uuid-shared", "alice", "other-exp")
	);
	check_equal(
		"find_matching does not match another user",
		null,
		find_matching_uuid_run_folder("uuid-shared", "bob", "exp")
	);

	reset_shares();
	make_run("alice", "exp", "0", "", "empty uuid\n");
	check_equal(
		"find_matching rejects an empty target UUID",
		null,
		find_matching_uuid_run_folder("", "alice", "exp")
	);

	// -----------------------------------------------------------------
	// resolve_share_target_folder
	// -----------------------------------------------------------------

	reset_shares();
	$run0 = make_run("alice", "exp", "0", "uuid-old");
	check_path(
		"resolve existing UUID -> exact folder",
		$run0,
		resolve_share_target_folder("uuid-old", "alice", "exp")
	);

	check_equal(
		"resolve unknown UUID -> null (new job)",
		null,
		resolve_share_target_folder("uuid-new", "alice", "exp")
	);

	check_equal(
		"resolve empty UUID without fallback -> null even if an old job exists",
		null,
		resolve_share_target_folder("", "alice", "exp", false)
	);

	check_equal(
		"resolve null UUID without fallback -> null even if an old job exists",
		null,
		resolve_share_target_folder(null, "alice", "exp", false)
	);

	check_equal(
		"resolve whitespace-only UUID without fallback -> null",
		null,
		resolve_share_target_folder("   \n", "alice", "exp", false)
	);

	check_path(
		"resolve empty UUID with explicit legacy fallback -> latest folder",
		$run0,
		resolve_share_target_folder("", "alice", "exp", true)
	);

	$folders_before = run_folders("alice", "exp");
	resolve_share_target_folder("uuid-does-not-exist", "alice", "exp");
	$folders_after = run_folders("alice", "exp");
	check_equal(
		"resolve is read-only (does not create folders)",
		$folders_before,
		$folders_after
	);

	// -----------------------------------------------------------------
	// get_user_folder
	// -----------------------------------------------------------------

	reset_shares();
	$new0 = get_user_folder(null, "alice", "exp");
	check_path(
		"get_user_folder(null) creates run 0",
		$GLOBALS["sharesPath"] . "alice/exp/0",
		$new0
	);
	check_true("run 0 was really created", is_dir($new0));

	$new1 = get_user_folder(null, "alice", "exp");
	check_path(
		"get_user_folder(null) creates run 1 next",
		$GLOBALS["sharesPath"] . "alice/exp/1",
		$new1
	);
	check_equal(
		"run numbering is dense and monotonic",
		["0", "1"],
		run_folders("alice", "exp")
	);

	check_path(
		"get_user_folder(existing) returns the existing folder",
		$new0,
		get_user_folder($new0, "alice", "exp")
	);

	check_path(
		"get_user_folder(run_nr=1) probes the numeric folder",
		$new1,
		get_user_folder(null, "alice", "exp", "1")
	);

	// Creating a new job must not touch the old one.
	reset_shares();
	$old = make_run("alice", "exp", "0", "uuid-old", "OLD-CONTENT\n");
	$new = get_user_folder(null, "alice", "exp");
	file_put_contents("$new/results.csv", "NEW-CONTENT\n");
	check_equal(
		"creating a new job leaves the old folder at run 0",
		"OLD-CONTENT\n",
		file_get_contents("$old/results.csv")
	);
	check_equal(
		"new job gets its own folder at run 1",
		"NEW-CONTENT\n",
		file_get_contents("$new/results.csv")
	);

	// The new folder is the one the new UUID must resolve to once stored.
	make_run("alice", "exp", "1", "uuid-new", "NEW-CONTENT\n");
	check_path(
		"old UUID still resolves to the old folder",
		$old,
		resolve_share_target_folder("uuid-old", "alice", "exp")
	);
	check_path(
		"new UUID resolves to the new folder",
		$new,
		resolve_share_target_folder("uuid-new", "alice", "exp")
	);

	// -----------------------------------------------------------------
	// find_latest_run_folder (only used by the explicit legacy fallback)
	// -----------------------------------------------------------------

	reset_shares();
	$older = make_run("alice", "exp", "0", "uuid-0");
	$newer = make_run("alice", "exp", "1", "uuid-1");
	touch($older, time() - 100);
	touch($newer, time());
	check_path(
		"find_latest_run_folder selects the most recently modified run",
		$newer,
		find_latest_run_folder("alice", "exp")
	);

	// -----------------------------------------------------------------
	// cleanup / summary
	// -----------------------------------------------------------------

	if (is_dir($GLOBALS["sharesPath"])) {
		remove_tree($GLOBALS["sharesPath"]);
	}

	if ($nr_of_errors > 0) {
		echo "\033[31mUUID routing: $nr_of_errors error(s)\033[0m\n";
	} else {
		echo "UUID routing unit tests: all checks passed\n";
	}

	exit(min(255, $nr_of_errors));
?>
