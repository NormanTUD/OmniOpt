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

	pre {
		color: #00CC00 !important;
		background-color: black !important;
		font-family: monospace !important;
	}

	menu[role="tablist"] {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		max-width: 100%;
		max-height: 100px;
		overflow: scroll;
	}

	menu[role="tablist"] button {
		white-space: nowrap;
		min-width: 100px;
	}

	.container {
		max-width: 100% !important;
	}

	.gridjs-sort {
		min-width: 1px !important;
	}

	td.gridjs-td {
		overflow: clip;
	}

	.gridjs-input {
		font-family: unset;
	}

	.gridjs-pages>button {
		font-family: unset;
	}

	.title-bar-text {
		font-size: 22px;
	}

	.title-bar {
		height: fit-content;
	}

	.window {
		width: fit-content;
		min-width: 100%;
	}

	.top_link {
		display: inline-block;
		padding: 5px 5px;
		background-color: #007bff; /* Blau, kannst du anpassen */
		color: white;
		text-decoration: none;
		font-size: 16px;
		font-weight: bold;
		border-radius: 6px;
		border: 2px solid #0056b3;
		text-align: center;
		transition: all 0.3s ease-in-out;
	}

	.top_link:hover {
		background-color: #0056b3;
		border-color: #004494;
	}

	.top_link:active {
		background-color: #003366;
		border-color: #002244;
	}
</style>
<?php
	$theme = "xp.css";


	if($_GET["theme"] == "none") {
		$theme = "none";
	}

	if ($theme != "none") {
?>
		<link rel="stylesheet" href="<?php print $theme; ?>">
<?php
	}
?>
