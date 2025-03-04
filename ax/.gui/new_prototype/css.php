<?php
	$theme = "xp";

	if($_GET["theme"] == 98) {
		$theme = "98";
	}
?>
<link rel="stylesheet" href="https://unpkg.com/<?php print $theme; ?>.css">
