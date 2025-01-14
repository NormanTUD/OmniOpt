<?php
        require "_header_base.php";
?>
        <script>
	    var partition_data = <?php include("partition_data.json"); ?>;
        </script>
<?php
        js("gui.js");
?>
	<script>
		$(document).ready(function() {
			create_tables();
			update_partition_options();

			var urlParams = new URLSearchParams(window.location.search);
			tableData.forEach(function(item) {
				var paramValue = urlParams.get(item.id);
				if (paramValue !== null) {
					$("#" + item.id).val(paramValue).trigger('change');
				}
			});

			hiddenTableData.forEach(function(item) {
				var paramValue = urlParams.get(item.id);
				if (paramValue !== null) {
					$("#" + item.id).val(paramValue).trigger('change');
				}
			});

			var num_parameters = urlParams.get("num_parameters");
			if (num_parameters) {
				for (var k = 0; k < num_parameters; k++) {
					$("#main_add_row_button").click();
				}
			} else {
				$("#main_add_row_button").click();
			}

			var parameterIndex = 0;
			$(".parameterRow").each(function(index) {
				var parameterName = urlParams.get("parameter_" + parameterIndex + "_name");
				var option = urlParams.get("parameter_" + parameterIndex + "_type");

				if (parameterName && option) {
					$(this).find(".parameterName").val(parameterName);
					$(this).find(".optionSelect").val(option).trigger('change');
					if (option === 'range') {
						$(this).find(".minValue").val(urlParams.get("parameter_" + parameterIndex + "_min"));
						$(this).find(".maxValue").val(urlParams.get("parameter_" + parameterIndex + "_max"));
						$(this).find(".numberTypeSelect").val(urlParams.get("parameter_" + parameterIndex + "_number_type"));
						$(this).find(".log_scale").prop(urlParams.get("parameter_" + parameterIndex + "_log_scale") == "true" ? true : false);
					} else if (option === 'choice') {
						$(this).find(".choiceValues").val(urlParams.get("parameter_" + parameterIndex + "_values"));
					} else if (option === 'fixed') {
						$(this).find(".fixedValue").val(urlParams.get("parameter_" + parameterIndex + "_value"));
					}
				}
				parameterIndex++;
			});

			update_command();

			initialized = true;

			update_url();

			document.getElementById("copytoclipboardbutton_curl").addEventListener(
				"click",
				copy_bashcommand_to_clipboard_curl,
				false
			);

			document.getElementById("copytoclipboardbutton_main").addEventListener(
				"click",
				copy_bashcommand_to_clipboard_main,
				false
			);

			input_to_time_picker("time")
			input_to_time_picker("worker_timeout")

			$('.tooltip').tooltipster();

			apply_theme_based_on_system_preferences();
		});
	</script>
        <div id="loader">
            <div class="spinner"></div>
            <br>
            <h2>Loading...</h2>
        </div>
        <div id="site" style="display: none">
            <table>
                <tr>
                    <td class='half_width_td'>
                        <table id="config_table">
                            <thead class="invert_in_dark_mode">
                                <tr>
                                    <th>Option</th>
                                    <th colspan="2">Value</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                        <button onclick='$("#hidden_config_table").toggle()' class='add_parameter invert_in_dark_mode' id='main_add_row_button'>Show additional parameters</button>
                        <table id="hidden_config_table" style="display: none">
                            <thead class="invert_in_dark_mode">
                                <tr>
                                    <th>Option</th>
                                    <th colspan="2">Value</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </td>
                    <td class='half_width_td'>
                        <div id="commands">
                            <h2>Install and run</h2>

                            <p class="no_linebreak">Run this to install OmniOpt2 and run this command. First time installation may take up to 30 minutes.</p>

                            <div class="dark_code_bg invert_in_dark_mode">
                                <code id="curl_command_highlighted"></code>
                                <code style="display: none" id="curl_command"></code>
                            </div>
                            <div class="invert_in_dark_mode" id="copytoclipboard_curl"><button type="button" id="copytoclipboardbutton_curl">&#128203; Copy to clipboard</button></div>
                            <div class="invert_in_dark_mode" id="copied_curl" style="display: none">&#128203; <b>Copied bash command to the clipboard</b></div>

                            <br>
                            <br>

                            <h2>Run</h2>

                            <p class="no_linebreak">Run this command in the <code>ax</code>-folder when you already have OmniOpt2 installed.</p>

                            <div class="dark_code_bg invert_in_dark_mode">
                                <code id="command_element_highlighted"></code>
                                <code style="display: none" id="command_element"></code>
                            </div>
                            <div class="invert_in_dark_mode" id="copytoclipboard_main"><button type="button" id="copytoclipboardbutton_main">&#128203; Copy to clipboard</button></div>
                            <div class="invert_in_dark_mode" id="copied_main" style="display: none">&#128203; <b>Copied bash command to the clipboard</b></div>
                        </div>
                        <div id="warnings" style="display: none"></div>
                    </td>
                </tr>
            </table>
        </div>
        </div>
<?php
	include("footer.php");
?>
