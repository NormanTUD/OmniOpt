<style>
	body.dark-mode {
		background-color: #1e1e1e; color: #fff;
	}

	.plot-container {
		margin-bottom: 2rem;
	}

	.spinner {
		border: 4px solid #f3f3f3;
		border-top: 4px solid #3498db;
		border-radius: 50%;
		width: 40px;
		height: 40px;
		animation: spin 2s linear infinite;
		margin: auto;
	}

	@keyframes spin {
		0% { transform: rotate(0deg); }
		100% { transform: rotate(360deg); }
	}

	.tabs {
		margin-bottom: 20px;
	}

	.tab-content {
		display: none;
	}

	.tab-content.active {
		display: block;
	}

	.tab-button.active {
		background-color: #3498db;
		color: white;
	}

	pre {
		color: green !important;
		background-color: black !important;
	}
</style>
<?php
	$theme = "https://unpkg.com/xp.css";

	if($_GET["theme"] == 98) {
		$theme = "https://unpkg.com/98.css";
	} else if($_GET["theme"] == 7) {
		$theme = "7.css";
	} else if($_GET["theme"] == "none") {
		$theme = "none";
	}

	if ($theme != "none") {
?>
		<link rel="stylesheet" href="<?php print $theme; ?>">
<?php
	}
?>
