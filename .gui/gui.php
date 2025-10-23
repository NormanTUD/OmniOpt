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
			<div id="warnings" style="display: none"></div>
		</div>
	</div>
</div>
<?php
	include("footer.php");
?>
