<style>
	#share_path {
		color: black;
	}

	.debug_log_pre {
		min-width: 300px;
	}

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
		line-break: anywhere;
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
		display: block ruby;
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

	button {
		color: black;
		font-family: -apple-system, ".SFNSDisplay-Regular", "Helvetica Neue", Helvetica, Arial, sans-serif
	}

	.share_folder_buttons {
		width: fit-content;
	}

	button {
		background: #fcfcfe;
		border-color: #919b9c;
		border-top-color: rgb(145, 155, 156);
		border-bottom-color: rgb(145, 155, 156);
		margin-right: -1px;
		border-bottom: 1px solid transparent;
		border-top: 1px solid #e68b2c;
		box-shadow: inset 0 2px #ffc73c;
	}

	button {
		padding-bottom: 2px;
		margin-top: -2px;
		background-color: #ece9d8;
		position: relative;
		z-index: 8;
		margin-left: -3px;
		margin-bottom: 1px;
	}

	.window {
		min-width: 1100px;
	}

	.error_text {
		color: red;
	}

	[role="tab"] {
		padding: 10px !important;
	}

	[role="tabpanel"] {
		min-width: fit-content;
	}

	select {
		border: 1px solid #7f9db9;
		background-image: url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 -0.5 15 17' shape-rendering='crispEdges'%3E%3Cpath stroke='%23e6eefc' d='M0 0h1'/%3E%3Cpath stroke='%23d1e0fd' d='M1 0h1M0 1h1m3 0h2M2 3h1M2 4h1'/%3E%3Cpath stroke='%23cad8f9' d='M2 0h1M0 2h1'/%3E%3Cpath stroke='%23c4d3f7' d='M3 0h1M0 3h1M0 4h1'/%3E%3Cpath stroke='%23bfd0f8' d='M4 0h2M0 5h1'/%3E%3Cpath stroke='%23bdcef7' d='M6 0h1M0 6h1'/%3E%3Cpath stroke='%23baccf4' d='M7 0h1m6 2h1m-1 5h1m-1 1h1'/%3E%3Cpath stroke='%23b8cbf6' d='M8 0h1M0 7h1M0 8h1'/%3E%3Cpath stroke='%23b7caf5' d='M9 0h2M0 9h1'/%3E%3Cpath stroke='%23b5c8f7' d='M11 0h1'/%3E%3Cpath stroke='%23b3c7f5' d='M12 0h1'/%3E%3Cpath stroke='%23afc5f4' d='M13 0h1'/%3E%3Cpath stroke='%23dce6f9' d='M14 0h1'/%3E%3Cpath stroke='%23e1eafe' d='M1 1h1'/%3E%3Cpath stroke='%23dae6fe' d='M2 1h1M1 2h1'/%3E%3Cpath stroke='%23d4e1fc' d='M3 1h1M1 3h1M1 4h1'/%3E%3Cpath stroke='%23d0ddfc' d='M6 1h1M1 5h1'/%3E%3Cpath stroke='%23cedbfd' d='M7 1h1M4 2h2'/%3E%3Cpath stroke='%23cad9fd' d='M8 1h1M6 2h1M3 5h1'/%3E%3Cpath stroke='%23c8d8fb' d='M9 1h2'/%3E%3Cpath stroke='%23c5d6fc' d='M11 1h1M2 11h4'/%3E%3Cpath stroke='%23c2d3fc' d='M12 1h1m-2 1h1M1 11h1m0 1h2m-2 1h2'/%3E%3Cpath stroke='%23bccefa' d='M13 1h1m-1 1h1m-1 1h1m-1 1h1M3 15h4'/%3E%3Cpath stroke='%23b9c9f3' d='M14 1h1M3 16h4'/%3E%3Cpath stroke='%23d8e3fc' d='M2 2h1'/%3E%3Cpath stroke='%23d1defd' d='M3 2h1'/%3E%3Cpath stroke='%23c9d8fc' d='M7 2h1M4 3h3M4 4h3M3 6h1m1 0h2M1 7h1M1 8h1'/%3E%3Cpath stroke='%23c5d5fc' d='M8 2h1m-8 8h5'/%3E%3Cpath stroke='%23c5d3fc' d='M9 2h2'/%3E%3Cpath stroke='%23bed0fc' d='M12 2h1M8 3h1M8 4h1m-8 8h1m-1 1h1m0 1h1m1 0h3'/%3E%3Cpath stroke='%23cddbfc' d='M3 3h1M3 4h1M1 6h2'/%3E%3Cpath stroke='%23c8d5fb' d='M7 3h1M7 4h1'/%3E%3Cpath stroke='%23bbcefd' d='M9 3h4M9 4h4M8 5h1M7 6h1'/%3E%3Cpath stroke='%23bcccf3' d='M14 3h1m-1 1h1m-1 1h1m-1 1h1'/%3E%3Cpath stroke='%23ceddfd' d='M2 5h1'/%3E%3Cpath stroke='%23c8d6fb' d='M4 5h4M1 9h3'/%3E%3Cpath stroke='%23bacdfc' d='M9 5h2m1 0h2M1 14h1'/%3E%3Cpath stroke='%23b9cdfb' d='M11 5h1M8 6h2m2 0h2m-1 1h1m-1 1h1'/%3E%3Cpath stroke='%234d6185' d='M4 6h1m5 0h1M3 7h3m3 0h3M4 8h3m1 0h3M5 9h5m-4 1h3m-2 1h1'/%3E%3Cpath stroke='%23b7cdfc' d='M11 6h1m0 1h1m-1 1h1'/%3E%3Cpath stroke='%23cad8fd' d='M2 7h1M2 8h2'/%3E%3Cpath stroke='%23c1d3fb' d='M6 7h2M7 8h1M4 9h1'/%3E%3Cpath stroke='%23b6cefb' d='M8 7h1m2 1h1m-2 1h3m-2 1h2'/%3E%3Cpath stroke='%23b6cdfb' d='M13 9h1m-6 6h1'/%3E%3Cpath stroke='%23b9cbf3' d='M14 9h1'/%3E%3Cpath stroke='%23b4c8f6' d='M0 10h1'/%3E%3Cpath stroke='%23bdd3fb' d='M9 10h2m-4 4h1'/%3E%3Cpath stroke='%23b5cdfa' d='M13 10h1'/%3E%3Cpath stroke='%23b5c9f3' d='M14 10h1'/%3E%3Cpath stroke='%23b1c7f6' d='M0 11h1'/%3E%3Cpath stroke='%23c3d5fd' d='M6 11h1'/%3E%3Cpath stroke='%23bad4fc' d='M8 11h1m-1 1h1m-1 1h1'/%3E%3Cpath stroke='%23b2cffb' d='M9 11h4m-2 3h1'/%3E%3Cpath stroke='%23b1cbfa' d='M13 11h1m-3 4h1'/%3E%3Cpath stroke='%23b3c8f5' d='M14 11h1m-7 5h3'/%3E%3Cpath stroke='%23adc3f6' d='M0 12h1m-1 1h1m-1 1h1'/%3E%3Cpath stroke='%23c2d5fc' d='M4 12h4m-4 1h4'/%3E%3Cpath stroke='%23b7d3fc' d='M9 12h2m-2 1h2m-3 1h1'/%3E%3Cpath stroke='%23b3d1fc' d='M11 12h1m-1 1h1'/%3E%3Cpath stroke='%23afcdfb' d='M12 12h1m-1 1h1m-1 1h1'/%3E%3Cpath stroke='%23afcbfa' d='M13 12h1m-1 1h1'/%3E%3Cpath stroke='%23b2c8f4' d='M14 12h1m-1 1h1m-4 3h1'/%3E%3Cpath stroke='%23c1d2fb' d='M3 14h1'/%3E%3Cpath stroke='%23b6d1fb' d='M9 14h2'/%3E%3Cpath stroke='%23adc9f9' d='M13 14h1m-2 1h1'/%3E%3Cpath stroke='%23b1c6f3' d='M14 14h1m-3 2h1'/%3E%3Cpath stroke='%23abc1f4' d='M0 15h1'/%3E%3Cpath stroke='%23b7cbf9' d='M1 15h1'/%3E%3Cpath stroke='%23b9cefb' d='M2 15h1'/%3E%3Cpath stroke='%23b9cffb' d='M7 15h1'/%3E%3Cpath stroke='%23b2cdfb' d='M9 15h2'/%3E%3Cpath stroke='%23aec8f7' d='M13 15h1'/%3E%3Cpath stroke='%23b0c5f2' d='M14 15h1m-2 1h1'/%3E%3Cpath stroke='%23dbe3f8' d='M0 16h1'/%3E%3Cpath stroke='%23b7c6f1' d='M1 16h1'/%3E%3Cpath stroke='%23b8c9f2' d='M2 16h1m4 0h1'/%3E%3Cpath stroke='%23d9e3f6' d='M14 16h1'/%3E%3C/svg%3E");
		background-size: 15px;
		font-size: 11px;
		border: none;
		background-color: #fff;
		box-sizing: border-box;
		height: 21px;
		appearance: none;
		-webkit-appearance: none;
		-moz-appearance: none;
		position: relative;
		padding: 5px 32px 32px 5px;
		background-position: top 50% right 2px;
		background-repeat: no-repeat;
		border-radius: 0;
		border: 1px solid black;
	}
</style>
<link rel="stylesheet" href="css/xp.css">
