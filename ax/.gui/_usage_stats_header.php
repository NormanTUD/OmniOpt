<?php
	require "_header_base.php";
?>
	<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
	<style>
		table {
			width: 100%;
			border-collapse: collapse;
		}
		th, td {
			border: 1px solid #ddd;
			padding: 8px;
		}
		th {
			padding-top: 12px;
			padding-bottom: 12px;
			text-align: left;
			background-color: #4CAF50;
			color: white;
		}
		tr:nth-child(even) {
			background-color: #f2f2f2;
		}
	</style>
	<link rel="stylesheet" href="jquery-ui.css">
	<script src="jquery-ui.js"></script>
	<script>
		$(function() {
			$("#tabs").tabs();
		});
	</script>
