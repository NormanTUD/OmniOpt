<?php
	require "_header_base.php";
?>
        <script>
		var partition_data = <?php include("partition_data.json"); ?>;
	</script>
<?php
        js("gui_data.js");
        js("gui.js");
?>
	<script>
		$(document).ready(run_when_document_ready);
	</script>
	<style>
		/* Smooth transitions on all inputs */
		input, select, textarea {
		    transition: border-color 0.2s, box-shadow 0.2s;
		    border: 2px solid #ddd;
		    border-radius: 6px;
		}

		input:focus, select:focus, textarea:focus {
		    border-color: #4a90d9;
		    box-shadow: 0 0 0 3px rgba(74, 144, 217, 0.15);
		    outline: none;
		}

		/* Required-but-empty field highlight (red border).  Used for
		   the run_program textarea and the formula editor when neither
		   is filled in. */
		.field_missing {
			border-color: #d32f2f !important;
			box-shadow: 0 0 0 3px rgba(211, 47, 47, 0.25) !important;
			background-color: #fff5f5 !important;
			animation: shake 0.3s ease-in-out;
		}

		/* Error states with animation */
		.error_element {
		    animation: shake 0.3s ease-in-out;
		    font-size: 0.85em;
		    margin-top: 4px;
		}

		.run_program_suggestions {
			background-color: #fff8e1;
			border: 2px solid #f0c36d;
			border-radius: 10px;
			padding: 10px 12px;
			margin-top: 8px;
			color: #6d4c00;
		}

		.run_program_suggestions_title {
			font-size: 0.9em;
			font-weight: 600;
			margin-bottom: 6px;
		}

		.run_program_suggestion_list {
			display: flex;
			flex-wrap: wrap;
			gap: 6px;
		}

		.run_program_suggestion_chip {
			display: flex;
			flex-direction: column;
			align-items: stretch;
			gap: 4px;
			padding: 6px 8px;
			background: #fffdf5;
			border: 1px solid #e0b34d;
			border-radius: 8px;
			box-shadow: none;
			color: #5a4300;
			line-height: 1.2;
			text-align: left;
			transform: none;
			transition: background-color 0.15s ease, box-shadow 0.15s ease;
		}

		.run_program_suggestion_chip:hover {
			background: #fff8e6;
			box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
			transform: none;
		}

		.run_program_suggestion_head {
			display: flex;
			align-items: center;
			justify-content: flex-start;
			gap: 6px;
		}

		.run_program_suggestion_name {
			font-family: monospace;
			font-size: 0.95em;
			font-weight: 700;
			margin-right: auto;
		}

		.run_program_suggestion_add {
			display: inline-flex;
			align-items: center;
			justify-content: center;
			width: 22px;
			height: 22px;
			box-sizing: border-box;
			padding: 0;
			background: #2e9e44;
			border: 1px solid #1f7a31;
			border-radius: 5px;
			box-shadow: none;
			color: #fff;
			cursor: pointer;
			font-size: 14px;
			font-weight: 700;
			line-height: 1;
			transform: none;
			transition: background-color 0.15s ease;
		}

		.run_program_suggestion_add:hover {
			background: #37b850;
			box-shadow: none;
			transform: none;
		}

		.run_program_suggestion_add_new {
			display: inline-flex;
			align-items: center;
			justify-content: center;
			height: 22px;
			box-sizing: border-box;
			padding: 0 7px;
			background: #fff;
			border: 1px solid #2e9e44;
			border-radius: 5px;
			box-shadow: none;
			color: #1f7a31;
			cursor: pointer;
			font-size: 0.72em;
			font-weight: 700;
			line-height: 1;
			transform: none;
			transition: background-color 0.15s ease;
		}

		.run_program_suggestion_add_new:hover {
			background: #e8f7eb;
			box-shadow: none;
			transform: none;
		}

		.run_program_suggestion_fields {
			display: flex;
			flex-wrap: wrap;
			align-items: center;
			gap: 5px;
		}

		.run_program_suggestion_fields label {
			display: flex;
			align-items: center;
			gap: 3px;
			color: #8a6d3b;
			font-size: 0.78em;
		}

		.run_program_suggestion_fields input[type="number"],
		.run_program_suggestion_fields input[type="text"],
		.run_program_suggestion_fields select {
			width: 5.2em;
			box-sizing: border-box;
			padding: 2px 4px;
			background: #fff;
			border: 1px solid #d8b45a;
			border-radius: 4px;
			color: #5a4300;
			font-size: 0.9em;
		}

		.run_program_suggestion_fields input[type="text"] {
			width: 15em;
		}

		.run_program_suggestion_fields input[type="checkbox"] {
			width: auto;
			padding: 0;
		}

		.run_program_suggestion_fields input[type="number"]:focus,
		.run_program_suggestion_fields input[type="text"]:focus,
		.run_program_suggestion_fields select:focus {
			border-color: #c4972f;
			box-shadow: 0 0 0 2px rgba(240, 195, 109, 0.35);
			outline: none;
		}

		.run_program_error_add_parameter {
			display: inline-block;
			margin-left: 6px;
			padding: 2px 8px;
			background: #fff;
			border: 1px solid rgba(0, 0, 0, 0.25);
			border-radius: 6px;
			box-shadow: none;
			color: #7a1f1f;
			cursor: pointer;
			font-size: 0.85em;
			font-weight: 600;
			transform: none;
		}

		.run_program_error_add_parameter:hover {
			background: #ffecec;
			box-shadow: none;
			transform: none;
		}

		@keyframes shake {
		    0%, 100% { transform: translateX(0); }
		    25% { transform: translateX(-5px); }
		    75% { transform: translateX(5px); }
		}

		/* Card-style sections */
		.config-section {
		    background: #fff;
		    border-radius: 12px;
		    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
		    padding: 24px;
		    margin-bottom: 20px;
		}

		/* Better buttons */
		button, .add_parameter {
		    border-radius: 8px;
		    padding: 10px 18px;
		    font-weight: 600;
		    transition: transform 0.1s, box-shadow 0.2s;
		}

		button:hover {
		    transform: translateY(-1px);
		    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
		}

		#commands {
		    top: 20px;
		    border-radius: 12px;
		    padding: 20px;
		    overflow-y: auto;
		}

		#commands code {
			font-size: 0.9em;
			line-height: 1.6;
			word-break: break-all;
		}

		/* Formula editor */
		.formula_tab {
			background: #fff;
			border: 1px solid #c0c0d0;
			border-radius: 6px;
			padding: 4px 10px;
			cursor: pointer;
			font-size: 0.85em;
		}
		.formula_tab.active {
			background: #4a90d9;
			color: #fff;
			border-color: #4a90d9;
		}
		.formula_tab:hover:not(.active):not(.rp_tab_active) {
			background: #eef3fa;
		}
		.formula_tab.active:hover,
		.rp_tab_active:hover {
			background: #4a90d9;
			color: #fff;
			border-color: #4a90d9;
		}
		#formula_card textarea {
			font-size: 0.95em;
		}
		#formula_card {
			width: 100%;
		}
		#formula_card_right table {
			font-size: 0.85em;
		}
		#formula_card_right th,
		#formula_card_right td {
			padding: 3px 6px;
			border-bottom: 1px solid #eee;
		}
		#formula_preview {
			overflow: visible !important;
		}
		#formula_preview mjx-container,
		#formula_preview .mjx-container,
		#formula_preview .MJx-Container,
		#formula_preview .MJXc-Display {
			overflow: visible !important;
		}
		#formula_preview mjx-container mjx-munder,
		#formula_preview .mjx-container .mjx-munder {
			overflow: visible !important;
		}

		.rp_tab {
			font-size: 0.9em;
			padding: 5px 14px;
		}
		.rp_tab_active {
			background: #4a90d9;
			color: #fff;
			border-color: #4a90d9;
			font-weight: 600;
		}

		.omniopt_bound_overlay {
			position: absolute;
			box-sizing: border-box;
			cursor: cell;
			z-index: 5;
			border-radius: 3px;
			transition: background 0.12s ease, box-shadow 0.12s ease;
		}
		.omniopt_bound_overlay:hover {
			background: rgba(74, 144, 217, 0.16);
			box-shadow: 0 0 0 1px rgba(74, 144, 217, 0.75);
		}
		.omniopt_bound_overlay::after {
			content: "\270E";
			position: absolute;
			top: -7px;
			right: -7px;
			width: 14px;
			height: 14px;
			line-height: 13px;
			font-size: 9px;
			text-align: center;
			color: #fff;
			background: #4a90d9;
			border-radius: 50%;
			box-shadow: 0 0 2px rgba(0, 0, 0, 0.35);
			opacity: 0;
			pointer-events: none;
			transition: opacity 0.12s ease;
		}
		.omniopt_bound_overlay:hover::after {
			opacity: 1;
		}
		.omniopt_bound_overlay input {
			display: none;
		}
		.omniopt_edit_pop {
			position: fixed;
			z-index: 9999;
			box-sizing: border-box;
			width: 188px;
			padding: 8px 10px;
			background: #fff;
			border: 1px solid #c6cdd6;
			border-radius: 8px;
			box-shadow: 0 4px 16px rgba(20, 40, 70, 0.22);
			font-size: 12px;
			line-height: 1.35;
			color: #1a1a1a;
		}
		.omniopt_edit_head {
			display: flex;
			align-items: center;
			justify-content: space-between;
			gap: 6px;
			margin-bottom: 6px;
		}
		.omniopt_edit_name {
			font-family: monospace;
			font-size: 13px;
			font-weight: bold;
			color: #1d3d7a;
			background: rgba(74, 144, 217, 0.12);
			border-radius: 4px;
			padding: 1px 5px;
			white-space: nowrap;
		}
		.omniopt_edit_chip {
			font-size: 10px;
			font-style: italic;
			color: #5b6472;
			white-space: nowrap;
		}
		.omniopt_edit_inputrow {
			display: flex;
			align-items: center;
			gap: 6px;
			margin-bottom: 6px;
		}
		.omniopt_edit_inputrow label {
			font-size: 11px;
			color: #5b6472;
			min-width: 34px;
			text-align: right;
			white-space: nowrap;
		}
		.omniopt_edit_inputrow input {
			flex: 1;
			min-width: 0;
			box-sizing: border-box;
			padding: 3px 6px;
			font-family: monospace;
			font-size: 12px;
			color: #1a1a1a;
			background: #fff;
			border: 1px solid #9db3cf;
			border-radius: 4px;
		}
		.omniopt_edit_inputrow input:focus {
			outline: none;
			border-color: #4a90d9;
			box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.28);
		}
		.omniopt_edit_meta {
			font-size: 10px;
			color: #5b6472;
			margin-bottom: 4px;
			overflow: hidden;
			text-overflow: ellipsis;
			white-space: nowrap;
		}
		.omniopt_edit_hint {
			font-size: 10px;
			color: #8a93a0;
			border-top: 1px solid #eef1f5;
			padding-top: 4px;
			margin-top: 2px;
		}

		.omniopt_edit_btn {
			position: absolute;
			box-sizing: border-box;
			width: 15px;
			height: 15px;
			padding: 0;
			display: flex;
			align-items: center;
			justify-content: center;
			border: none;
			border-radius: 50%;
			background: transparent;
			cursor: cell;
			z-index: 8;
			opacity: 0.55;
			transition: opacity 0.12s ease, background 0.12s ease, box-shadow 0.12s ease;
		}
		.omniopt_edit_btn:hover {
			opacity: 1;
			background: rgba(74, 144, 217, 0.18);
			box-shadow: 0 0 0 1px rgba(74, 144, 217, 0.55);
		}
	</style>
<div id="loader">
	<div class="spinner"></div> Loading...
</div>

<div id="site" style="display: none">
	<div class="responsive-container">
		<div class="half">
			<div class="config-section">
				<table id="config_table">
					<thead class="invert_in_dark_mode">
						<tr>
							<th>Option</th>
							<th colspan="2">Value</th>
						</tr>
					</thead>
					<tbody></tbody>
				</table>

				<textarea id="formula" style="display: none"></textarea>
				<select id="formula_mode" style="display: none">
					<option value="auto">auto</option>
					<option value="latex">latex</option>
					<option value="infix">infix</option>
				</select>

				<br>

				<button onclick='$("#hidden_config_table").toggle()' class='add_parameter' id='main_add_row_button'>
					<img src='i/cogwheels.svg' class="invert_in_dark_mode" style='height: 1em' />&nbsp;Show additional parameters
				</button>

				<table id="hidden_config_table" style="display: none">
					<thead class="invert_in_dark_mode">
						<tr>
							<th>Option</th>
							<th colspan="2">Value</th>
						</tr>
					</thead>
					<tbody></tbody>
				</table>
			</div>
		</div>

		<div class="half">
			<div id="commands">
				<div id="install_and_run">
					<h2>Install and run</h2>
					<p class="no_linebreak">Run this to install OmniOpt2 and run this command. First time installation may take up to 30 minutes.</p>
					<div class="dark_code_bg invert_in_dark_mode">
						<code id="curl_command_highlighted"></code>
						<code style="display: none" id="curl_command"></code>
					</div>
					<div id="copytoclipboard_curl">
						<button type="button" id="copytoclipboardbutton_curl">
							<img src='i/clipboard.svg' style='height: 1em' /> Copy to clipboard
						</button>
					</div>
					<div class="invert_in_dark_mode" id="copied_curl" style="display: none">
						<img src='i/clipboard.svg' style='height: 1em' /> <b>Copied bash command to the clipboard</b>
					</div>

					<br><br>
				</div>

				<div id="only_run">
					<h2>Run</h2>
					<p class="no_linebreak">Run this command in the <code>ax</code>-folder when you already have OmniOpt2 installed.</p>
					<div class="dark_code_bg invert_in_dark_mode">
						<code id="command_element_highlighted"></code>
						<code style="display: none" id="command_element"></code>
					</div>
					<div id="copytoclipboard_main">
						<button type="button" id="copytoclipboardbutton_main">
							<img src='i/clipboard.svg' style='height: 1em' /> Copy to clipboard
						</button>
					</div>
					<div class="invert_in_dark_mode" id="copied_main" style="display: none">
						<img src='i/clipboard.svg' style='height: 1em' /> <b>Copied bash command to the clipboard</b>
					</div>
				</div>
			</div>
			<div id="warnings" style="display: none"></div>
		</div>
	</div>
</div>
<?php
	include("footer.php");
?>
