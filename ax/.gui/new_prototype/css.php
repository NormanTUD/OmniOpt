<?php
	$theme = "https://unpkg.com/xp.css";

	if($_GET["theme"] == 98) {
		$theme = "https://unpkg.com/98.css";
	} else if($_GET["theme"] == 7) {
		$theme = "7.css";
	}
?>
<link rel="stylesheet" href="<?php print $theme; ?>">
