var invalid_names = ["generation_node", "start_time", "end_time", "hostname", "signal", "exit_code", "run_time", "program_string", "arm_name", "trial_index", "generation_method", "trial_status", "idxs", "submit_time", "queue_time", "worker_generator_uuid"];

var fadeTime = 0;
var fadeTimeAfterLoading = 300;

function normalizeFloat(value) {
	if (!isFinite(value)) {
		return '';
	}

	var str = value.toString();

	if (str.includes('e')) {
		var fixed = value.toFixed(20);
		fixed = fixed.replace(/(\.\d*?[1-9])0+$/, '$1');
		fixed = fixed.replace(/\.0+$/, '');
		return fixed;
	}

	return str;
}

function get_invalid_names () {
	var gin = JSON.parse(JSON.stringify(invalid_names));

	let _element = document.getElementById("result_names");

	if (_element) {
		let content = $(_element).val().trim();
		let parts = content.split(/\s+/);
		let new_gin = parts.map(item => item.includes('=') ? item.split('=')[0] : item);

		for (var i = 0; i < new_gin.length; i++) {
			gin.push(new_gin[i]);
		}
	}

	return gin;
}

var initialized = false;
var shown_operation_insecure_without_server = false;

var l = typeof log === "function" ? log : console.log;

var useSmoothFade = false;

function smoothShow($elem) {
	if (useSmoothFade) {
		$elem.fadeIn(fadeTime);
	} else {
		$elem.show();
	}
}

function smoothHide($elem) {
	if (useSmoothFade) {
		$elem.fadeOut(fadeTime);
	} else {
		$elem.hide();
	}
}

function smoothToggle($elem) {
	if($elem.is(":visible")) {
		smoothHide($elem);
	} else {
		smoothShow($elem);
	}
}

function input_to_time_picker (input_id) {
	var $input = $("#" + input_id);
	var $parent = $($input).parent();

	if (
		$parent.find(".time_picker_container").length ||
		$parent.find(".time_picker_minutes").length ||
		$parent.find(".time_picker_hours").length
	) {
		log(".time_picker_minutes or .time_picker_hours already found. Not reinstantiating for id " + input_id);
		return;
	}

	var minutes = $input.val();
	var _hours = 0;
	var _minutes = 0;

	if (minutes) {
		_hours = Math.floor(minutes / 60);
		_minutes = minutes % 60;
	}

	var $div = $(`
		    <div class='time_picker_container'>
			<input type='number' min=-1 max=159 class="time_picker_hours" value='${_hours}' onchange='update_original_time_element("${input_id}", this)'></input> Hours,
			<input type='number' min=-1 step=31 class="time_picker_minutes" value='${_minutes}' onchange='update_original_time_element("${input_id}", this)'></input> Minutes
		    </div>
		`);

	$parent.prepend($div);

	$input.hide();
}

function update_original_time_element (original_element_id, new_element) {
	var $parent = $(new_element).parent();
	var $time_picker_minutes = $parent.find(".time_picker_minutes");
	var $time_picker_hours = $parent.find(".time_picker_hours");

	var _minutes = parseInt($time_picker_minutes.val());
	var _hours = parseInt($time_picker_hours.val());

	if (_minutes == -1) {
		if (_hours > 0) {
			_hours = _hours - 1;
			_minutes = 55;
		} else {
			_hours = 0;
			_minutes = 5;
		}
	} else if (_minutes >= 60) {
		_hours = _hours + 1;
		_minutes = 0;
	}

	if (_hours == -1) {
		if (_hours > 1) {
			_hours = _hours - 1;
		} else {
			_hours = 0;
		}
	}

	$time_picker_hours.val(_hours);
	$time_picker_minutes.val(_minutes);

	var new_val = (parseInt(_hours) * 60) + parseInt(_minutes);

	$("#" + original_element_id).val(new_val).trigger("change");
}

function highlight_bash (code) {
	return Prism.highlight(code, Prism.languages.bash, "bash");
}

function highlight_all_bash () {
	$(".highlight_me").each(function (i, e) {
		$(e).html(highlight_bash($(e).text()));
	});
}

function update_partition_options() {
	var partitionSelect = $("#partition");
	partitionSelect.empty();

	$.each(partition_data, function(key, value) {
		partitionSelect.append($("<option></option>")
			.attr("value", key)
			.text(value.name));
	});

	partitionSelect.on("change", function() {
		var partition = $(this).val();
		if(Object.keys(partition_data).includes(partition)) {
			var partitionInfo = partition_data[partition];

			if (partitionInfo) {
				$("#mem_gb").attr("max", Math.floor(partitionInfo.max_mem_per_core / 1000)).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#time").attr("max", partitionInfo.computation_time).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#worker_timeout").attr("max", partitionInfo.computation_time).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#max_eval").attr("min", 1).each(function() {
					if ($(this).val() < $(this).attr("min")) {
						$(this).val($(this).attr("min"));
					}
				});

				$("#num_parallel_jobs").attr("min", 1).each(function() {
					if ($(this).val() < $(this).attr("min")) {
						$(this).val($(this).attr("min"));
					}
				});

				$("#num_parallel_jobs").attr("max", partitionInfo.number_of_workers).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#gpus").attr("max", partitionInfo.max_number_of_gpus).each(function() {
					if ($(this).val() > $(this).attr("max")) {
						$(this).val($(this).attr("max"));
					}
				});

				$("#gpus").attr("min", partitionInfo.min_number_of_gpus).each(function() {
					if ($(this).val() < $(this).attr("min")) {
						$(this).val($(this).attr("min"));
					}
				});
			} else {
				error("No partition info");
			}
		} else {
			error(`Cannot find ${partition} in partition_data.`);
		}

		update_url();
	});

	update_url();
}

function set_min_max () {
	document.querySelectorAll("input").forEach(input => {
		if (input.hasAttribute("min") || input.hasAttribute("max")) {
			var _min = input.hasAttribute("min") ? parseFloat(input.getAttribute("min")) : null;
			var _max = input.hasAttribute("max") ? parseFloat(input.getAttribute("max")) : null;

			let value = parseFloat(input.value);

			let red = "#FFE2DE";

			if (_min !== null && (isNaN(value) || value < _min)) {
				if(isNaN(value)) {
					$(input).parent().find("[id$='_error']").html("Value is empty or invalid");
				}

				if(value < _min) {
					$(input).val(_min);
				}
			} else if (_max !== null && value > _max) {
				$(input).val(_max);
			} else {
				$(input).parent().find("[id$='_error']").html("");
			}
		}
	});
}

function quote_variables(input) {
	return input.replace(/(["'])(.*?)\1|%(\((\w+)\)|(\w+))/g, function(match, quotes, insideQuotes, p1, p2, p3) {
		if (quotes) {
			return match;
		} else {
			var variable = p2 || p3;
			return "'%(" + variable + ")'";
		}
	});
}

function get_var_names_from_run_program(run_program_string) {
	const pattern = /(?:\$|\%)?\([a-zA-Z_]+\)|(?:\$|%)[a-zA-Z_]+/g;
	const variableNames = [];

	let match;
	while ((match = pattern.exec(run_program_string)) !== null) {
		let varName = match[0];
		varName = varName.replace(/^(\$|%)/, "");
		varName = varName.replace(/^(\$|%)?\(|\)$/g, "");
		if (/^[a-zA-Z_]+$/.test(varName)) {
			variableNames.push(varName);
		}
	}

	return variableNames;
}

function update_table_row (item, errors, warnings, command) {
	var value = $("#" + item.id).val();

	if(item.regex) {
		var re = new RegExp(item.regex, "i");

		var text = $("#" + item.id).val();

		if(!text.match(re)) {
			var this_error = `The element "${item.id}" does not match regex /${item.regex}/.`;
			if(Object.keys(item).includes("regex_does_not_match_text")) {
				this_error = item.regex_does_not_match_text;
			}
			errors.push(this_error);
			smoothShow($("#" + item.id + "_error").html(`<img src='i/warning.svg' style='height: 1em' /> ${this_error}`));
		} else {
			smoothHide($("#" + item.id + "_error").html(""));
		}
	}

	if (item.type === "checkbox") {
		value = $("#" + item.id).is(":checked") ? "1" : "0";
		if (value === "1") {
			command += " --" + item.id;
		}
	} else if ((item.type === "textarea" || item.type === "text") && value === "") {
		if(item.required) {
			var this_error = "<img src='i/warning.svg' style='height: 1em' /> Field '" + item.label + "' is required.";
			smoothShow($("#" + item.id + "_error").html(this_error));
			$("#" + item.id).css("background-color", "#FFCCCC !important");

			errors.push(this_error);
		}
	} else if (item.id == "time") {
		var worker_timeout_larger_than_global_timeout = parseInt($("#worker_timeout").val()) > parseInt($("#time").val());
		var new_errors = [];
		var numValue = parseFloat(value);

		if (worker_timeout_larger_than_global_timeout) {
			new_errors.push("Worker timeout is larger than global time. Increase global time or decrease worker time.");
		} else if (isNaN(numValue) && ((Object.keys(item).includes("required") && item.required) || !Object.keys(item).includes("required"))) {
			new_errors.push("Invalid value for '" + item.label + "'. Must be a number.");
		} else if (item.min && item.max && (numValue < item.min || numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be between " + item.min + " and " + item.max + ".");
		} else if (item.min && (numValue < item.min)) {
			new_errors.push("Value for '" + item.label + "' must be larger than " + item.min + ".");
		} else if (item.max && (numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be smaller than" + item.max + ".");
		}

		if(new_errors.length) {
			smoothShow($("#time_error").html(string_or_array_to_list(new_errors)));
			errors.push(...new_errors);
		} else {
			smoothHide($("#time_error").html(""));
			command += " --" + item.id + "=" + value;
		}
	} else if (item.id == "max_eval") {
		var parallel_evaluations = parseInt($("#num_parallel_jobs").val());
		var max_eval = parseInt($("#max_eval").val());
		var num_random_steps = parseInt($("#num_random_steps").val());

		if (parallel_evaluations <= 0) {
			$("#num_parallel_jobs").val(1);
		}

		if (max_eval < parallel_evaluations) {
			$("#num_parallel_jobs").val(max_eval);
		}

		if (max_eval < num_random_steps) {
			$("#num_random_steps").val(max_eval);
		}

		if (num_random_steps <= 0) {
			$("#num_random_steps").val(1);
		}

		command += " --" + item.id + "=" + value;
	} else if (item.id == "worker_timeout") {
		var worker_timeout_larger_than_global_timeout = parseInt($("#worker_timeout").val()) > parseInt($("#time").val());
		var new_errors = [];
		var numValue = parseFloat(value);

		if (worker_timeout_larger_than_global_timeout) {
			new_errors.push("Worker timeout is larger than global time. Increase global time or decrease worker time.");
		} else if (isNaN(numValue) && ((Object.keys(item).includes("required") && item.required) || !Object.keys(item).includes("required"))) {
			new_errors.push("Invalid value for '" + item.label + "'. Must be a number.");
		} else if (item.min && item.max && (numValue < item.min || numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be between " + item.min + " and " + item.max + ".");
		} else if (item.min && (numValue < item.min)) {
			new_errors.push("Value for '" + item.label + "' must be larger than " + item.min + ".");
		} else if (item.max && (numValue > item.max)) {
			new_errors.push("Value for '" + item.label + "' must be smaller than" + item.max + ".");
		}

		if(new_errors.length) {
			smoothShow($("#worker_timeout_error").html(string_or_array_to_list(new_errors)));
			errors.push(...new_errors);
		} else {
			smoothHide($("#worker_timeout_error").html(""));
			command += " --" + item.id + "=" + value;
		}
	} else if (item.type === "number") {
		var numValue = parseFloat(value);

		if (isNaN(numValue) && ((Object.keys(item).includes("required") && item.required) || !Object.keys(item).includes("required"))) {
			errors.push("Invalid value for '" + item.label + "'. Must be a number.");
		} else if (item.min && item.max && (numValue < item.min || numValue > item.max)) {
			errors.push("Value for '" + item.label + "' must be between " + item.min + " and " + item.max + ", is " + numValue + ".");
		} else if (item.min && (numValue < item.min)) {
			errors.push("Value for '" + item.label + "' must be larger than " + item.min + ".");
		} else if (item.max && (numValue > item.max)) {
			errors.push("Value for '" + item.label + "' must be smaller than" + item.max + ".");
		} else {
			value = numValue.toString();
			if(value != "NaN") {
				if (item.type == "number" || value.matches(/^[a-zA-Z0-9=_]+$/)) {
					command += " --" + item.id + "=" + value;
				} else {
					command += " --" + item.id + "='" + value + "'";
				}
			}
		}
	} else if (item.id == "external_generator") {
		command += " --" + item.id + "='" + btoa(value) + "'";
	} else if (item.id == "run_program") {
		var variables_in_run_program = get_var_names_from_run_program(value);
		//value = quote_variables(value);

		var existing_parameter_names = $(".parameterName").map(function() {
			const val = $(this).val();
			var ret = null;
			if(!/^\s*$/.test(val) && /^[a-zA-Z_]+$/.test(val)) {
				ret = val;
			}
			return ret;
		}).get().filter(Boolean);

		var new_errors = [];

		for (var k = 0; k < variables_in_run_program.length; k++) {
			var test_this_var_name = variables_in_run_program[k];

			if(!existing_parameter_names.includes(test_this_var_name)) {
				var err_msg = `<img src='i/warning.svg' style='height: 1em' /> <code>%(${test_this_var_name})</code> not in existing defined parameters.`;
				new_errors.push(err_msg);
			}
		}

		for (var k = 0; k < existing_parameter_names.length; k++) {
			var test_this_var_name = existing_parameter_names[k];

			if(!variables_in_run_program.includes(test_this_var_name)) {
				var err_msg = `<img src='i/warning.svg' style='height: 1em' /> <code>%(${test_this_var_name})</code> is defined but not used.`;
				new_errors.push(err_msg);
			}
		}

		if(new_errors.length) {
			smoothShow($("#run_program_error").html(string_or_array_to_list(new_errors)));
			errors.push(...new_errors);
		} else {
			smoothHide($("#run_program_error").html(""));
		}

		value = btoa(value);

		command += " --" + item.id + "='" + value + "'";
		$("#" + item.id).css("background-color", "");
	} else {
		if(!errors.length) {
			if (item.id != "constraints") {
				if (item.id == "result_names") {
					command += " --" + item.id + " " + value;
				} else {
					command += " --" + item.id + "=" + value;
				}
				smoothHide($("#" + item.id + "_error").html(""));
				$("#" + item.id).css("background-color", "");
			}
		}
	}

	return [command, errors, warnings];
}

function set_row_background_color_red_color(_row) {
	log("_row:", _row);
	console.trace();
	$(_row).css("background-color", "#ffabab").addClass("invert_in_dark_mode");
}

function is_invalid_parameter_name(name) {
	if(name.startsWith("OO_Info_")) {
		return true;
	}

	var gin = get_invalid_names();

	if (gin.includes(name)) {
		return true;
	}

	return false;
}

function show_warnings_and_errors(warnings, errors) {
	var warnings_html = "";
	var errors_html = "";

	var img_warning = "<img src='i/warning.svg' width=16 />";

	function formatMessages(messages, title, color) {
		if (!Array.isArray(messages) || messages.length === 0) return '';
		return `<h2 class="invert_in_dark_mode" style="color: ${color};">${title}:</h2><ul>` + 
			messages.map(msg => {
				if (msg.startsWith('<img')) {
					msg = msg.replace(/<img[^>]*>/, '');
				}
				return `<li style="color: ${color};">${img_warning} <span class="invert_in_dark_mode">${msg}</span></li>`;
			}).join('') +
			'</ul>';
	}

	warnings_html = formatMessages(warnings, 'Warnings', 'orange');
	errors_html = formatMessages(errors, 'Errors', 'red');

	var content = warnings_html + errors_html;
	var warnings_element = $('#warnings');

	warnings_element.html(content);

	if (content.length > 0) {
		warnings_element.show();
	} else {
		warnings_element.hide();
	}

	apply_theme_based_on_system_preferences();
}

function update_command() {
	set_min_max();

	var errors = [];
	var warnings = [];
	var command = "./omniopt";

	if($("#installation_method").val() == "pip") {
		command = "omniopt";
	}

	var curl_options = "";

	if ($("#run_mode").val() == "docker") {
		command = "bash omniopt_docker omniopt";
	}

	function processTableData(_tableData) {
		_tableData.forEach(function(item) {
			if (!item.use_in_curl_bash) {
				var command_error_and_warning = update_table_row(item, errors, warnings, command);
				command = command_error_and_warning[0];
				errors = command_error_and_warning[1];
				warnings = command_error_and_warning[2];
			} else {
				if (item.type == "select") {
					curl_options = ` --${item.id}=${$(`#${item.id}`).val()} `;
				} else if (item.type == "checkbox" && $(`#${item.id}`).is(":checked")) {
					curl_options = ` --${item.id} `;
				} else {
					error("use_in_curl_bash currently only supports select and checkbox");
				}
			}
		});
	}

	processTableData(tableData);
	processTableData(hiddenTableData);

	var parameters = [];

	var i = 0;
	var parameter_names = [];

	$(".parameterRow").each(function() {
		var option = $(this).find(".optionSelect").val();
		var parameterName = $(this).find(".parameterName").val().trim();
		var _value;

		var warn_msg = [];

		if(parameter_names.includes(parameterName)) {
			var err_msg = `Parameter name "${parameterName}" already exists. Can only be defined once!`;
			warn_msg.push(err_msg);

			$($(".parameterRow")[i]).css("background-color", "#e57373");
		} else if(parameterName && !parameterName.match(/^[a-zA-Z_]+$/)) {
			warn_msg.push("Name contains invalid characters. Must be all-letters.");
		} else if(is_invalid_parameter_name(parameterName)) {
			warn_msg.push(`Name is or contains a reserved keyword, cannot be any of those: <tt>${invalid_names.join(', ')}</tt>, or any of the names specified in the results-names.`);
		} else if(parameterName.match(/^[a-zA-Z_]+$/)) {
			if (option === "range") {
				var $this = $(this);
				//log("$this.find('.minValue').val():", $this.find(".minValue").val());

				var minValue = normalizeFloat(parseFloat($this.find(".minValue").val()));
				var maxValue = normalizeFloat(parseFloat($this.find(".maxValue").val()));

				var numberType = $($(".parameterRow")[i]).find(".numberTypeSelect").val();

				if (minValue === maxValue) {
					warn_msg.push("Warning: The minimum and maximum values for parameter " + parameterName + " are equal.");
				}

				var is_ok = true;

				//log("minValue:", minValue);

				if(isNaN(minValue)) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>minValue</i> for parameter <i>" + parameterName + "</i> is not a number.");
					is_ok = false;
				}

				if(isNaN(maxValue)) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>maxValue</i> for parameter <i>" + parameterName + "</i> is not a number.");
					is_ok = false;
				}

				if(maxValue == "") {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>maxValue</i> for parameter <i>" + parameterName + "</i> is empty.");
					is_ok = false;
				}

				if(minValue == "") {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>minValue</i> for parameter <i>" + parameterName + "</i> is empty.");
					is_ok = false;
				}

				if (numberType == "int") {
					var parsed_int_max = parseInt(maxValue);
					if (parsed_int_max != maxValue) {
						warn_msg.push("maxValue is not an integer");
					}

					var parsed_int_min = parseInt(minValue);
					if (parsed_int_min != minValue) {
						warn_msg.push("minValue is not an integer");
					}
				}

				var log_scale = $($(".parameterRow")[i]).find(".log_scale").is(":checked") ? "true" : "false";

				if(is_ok) {
					_value = `${parameterName} range ${minValue} ${maxValue} ${numberType} ${log_scale}`;
				}
			} else if (option === "choice") {
				var choiceValues = $(this).find(".choiceValues").val();

				if(choiceValues !== undefined) {
					choiceValues = choiceValues.replaceAll(/\s/g, ",");
					choiceValues = choiceValues.replaceAll(/,,*/g, ",");
					choiceValues = choiceValues.replaceAll(/,,*$/g, "");
					choiceValues = choiceValues.replaceAll(/^,,*/g, "");

					choiceValues = [...new Set(choiceValues.split(","))].join(",");

					_value = `${parameterName} choice ${choiceValues}`;

					if(!choiceValues.match(/./)) {
						warn_msg.push("Values are missing.");
					}
				} else {
					warn_msg.push("choiceValues not defined.");
				}
			} else if (option === "fixed") {
				var fixedValue = $(this).find(".fixedValue").val();

				if(typeof(fixedValue) == "string") {
					fixedValue = fixedValue.replace(/,*$/g, "");
					fixedValue = fixedValue.replace(/,+/g, ",");

					fixedValue = Array.from(new Set(fixedValue.split(","))).join(",");

					_value = `${parameterName} fixed ${fixedValue}`;
				}

				if(fixedValue === undefined) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /> <i>Value</i> is missing.");
				} else if(!fixedValue.match(/./)) {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /> <i>Value</i> is missing.");
				} else if(!fixedValue.match(/^[a-zA-Z0-9\.,_]+$/)) {
					warn_msg.push("Invalid values. Must match Regex /[a-zA-Z0-9,_\.]/.");

				}
			}

			if (parameterName && _value) {
				parameters.push(_value);
				parameter_names.push(parameterName);

				if(!warn_msg.length) {
					$($(".parameterRow")[i]).css("background-color", "");
				}
			} else {
				if(!parameterName) {
					warn_msg.push("No parameter name");
				}
			}
		} else {
			warn_msg.push("<img src='i/warning.svg' style='height: 1em' /> Parameter option <i>Name</i> is missing.");
		}

		if(warn_msg.length) {
			smoothShow($($(".parameterError")[i]).html(string_or_array_to_list(warn_msg)));
		} else {
			$($(".parameterError")[i]).html("").hide();
		}

		errors.push(...warn_msg);

		i++;
	});

	if (parameters.length > 0) {
		command += " --parameter " + parameters.join(" --parameter ");
	}

	if ($("#constraints").val()) {
		var constraints_string = $("#constraints").val();

		constraints_string = constraints_string.replaceAll(/;;*/g, ";");
		constraints_string = constraints_string.replace(/;;*$/, "");
		constraints_string = constraints_string.replace(/^;;*/, "");

		var _constraints = constraints_string.split(";");
		_constraints = _constraints.filter(function(entry) { return entry.trim() != ""; }).map(function (el) {
			return el.trim();
		});
		for (var r = 0; r < _constraints.length; r++) {
			command += " --experiment_constraints '" + btoa(add_equation_spaces(_constraints[r])) + "'";
		}

		var constraints_string = $("#constraints").val();
		var errors_string = is_valid_constraints_string(constraints_string);

		if(isAnyLogScaleSet()) {
			errors_string += "Cannot set constraints if one or more log-scale parameters are there."
		}

		if (errors_string != "") {
			errors.push("Something was wrong in the constraints parameter");
			smoothShow($("#constraints_error").html(errors_string));
		} else {
			smoothHide($("#constraints_error").html(""));
		}
	} else {
		$("#constraints_error").html("");
	}

	var errors_visible = false;
	$(".parameterError").each(function (i, e) {
		if($(e).is(":visible")) {
			errors_visible = true;
		}
	});

	if (!errors.length && $(".optionSelect").length && !errors_visible) {
		var base_url = location.protocol + "//" + location.host + "/" + location.pathname + "/";

		base_url = base_url.replaceAll(/\/+/g, "/");

		base_url = base_url.replace(/^http:\//, "http://");
		base_url = base_url.replace(/^https:\//, "https://");
		base_url = base_url.replace(/^file:\//, "file://");

		var base_64_string = btoa(command);

		var curl_or_cat = "curl";

		if (base_url.startsWith("file://")) {
			curl_or_cat = "cat";

			var filename = location.pathname.substring(location.pathname.lastIndexOf("/")+1);

			var _re_ = new RegExp(`${filename}/?$`);

			base_url = base_url.replace(_re_, "");

			base_url = base_url.replace(/^file:\//, "/");
			base_url = base_url.replace(/^\/\//, "/");
		}

		base_url = base_url.replace(/\/index(?:.php)?/, "");
		base_url = base_url.replace(/\/gui(?:.php)?/, "");

		var curl_command = "";
		var command_end = ` | bash -l -s -- "${base_64_string}"${curl_options}`;

		if(curl_or_cat == "curl") {
			curl_command = `${curl_or_cat} ${base_url}install_omniax.sh 2>/dev/null${command_end}`;
		} else {
			curl_command = `${curl_or_cat} ${base_url}install_omniax.sh${command_end}`;
		}

		nicer_command = addBase64DecodedVersions(command);

		toggleElementVisibility("#curl_command_highlighted", curl_command, true);
		toggleElementVisibility("#command_element_highlighted", nicer_command, true);

		$("#curl_command").text(curl_command);
		$("#command_element").text(nicer_command);
	} else {
		toggleElementVisibility("#command_element_highlighted", "", false);
		toggleElementVisibility("#curl_command_highlighted", "", false);

		$("#curl_command").text("");
		$("#command_element").text("");
	}

	show_warnings_and_errors(warnings, errors);

	update_url();

	toggleHiddenConfigTableIfError();
}

function addBase64DecodedVersions(cmdString) {
    return cmdString.replace(/(--[a-zA-Z0-9_]+)=('([^']+)'|"([^"]+)"|([^\s]+))/g, (match, key, _, singleQuoted, doubleQuoted, bare) => {
        const value = singleQuoted || doubleQuoted || bare;

        let decoded = null;
        try {
            if (key === "--run_program") {
                decoded = atob(value);
            }
        } catch (e) {
            console.error(e);
        }

        if (decoded) {
            const safeDecoded = decoded.replace(/\x27/g, `'\\''`);
            return ` ${key}=$(echo '${safeDecoded}' | base64 -w0)`;
        } else {
            return match;
        }
    });
}

async function toggleElementVisibility(selector, content, show) {
	let element = $(selector);

	if (show) {
		smoothShow(element.html(highlight_bash(content)));
		smoothShow(element.parent());
		smoothShow(element.parent().parent());
	} else {
		await Promise.all([
			new Promise(resolve => element.fadeOut(fadeTime, resolve)),
			new Promise(resolve => element.parent().fadeOut(fadeTime, resolve)),
			new Promise(resolve => element.parent().parent().fadeOut(fadeTime, resolve))
		]);
		element.html("");
	}
}

function updateOptions(select) {
	var selectedOption = select.value;
	var valueCell = select.parentNode.nextSibling;
	var paramName = $(select).parent().parent().find(".parameterName").val();

	if(paramName === undefined) {
		paramName = "";
	}

	if (selectedOption === "range") {
		valueCell.innerHTML = `
			<table class='parameter_config_table'>
				<tr>
					<td>Name:</td>
					<td><input placeholder="Parameter name" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='parameterName'></td>
				</tr>
				<tr>
					<td>Min:</td>
					<td><input placeholder="Minimum value" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='number' class='minValue'></td>
				</tr>
				<tr>
					<td>Max:</td>
					<td><input placeholder="Maximum value" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='number' class='maxValue'></td>
				</tr>
				<tr>
					<td>Type:</td>
					<td>
						<select onchange="update_command()" onkeyup="update_command()" onclick="update_command()" class="numberTypeSelect">
							<option value="float">Float</option>
							<option value="int">Integer</option>
						</select>
					</td>
				</tr>
				<tr>
					<td>Log-Scale:</td>
					<td>
						<input onchange="update_command()" type="checkbox" class="log_scale" />
					</td>
				</tr>
			</table>
		    `;
	} else if (selectedOption === "choice") {
		valueCell.innerHTML = `
			<table class='parameter_config_table'>
				<tr>
					<td>Name:</td>
					<td><input placeholder="Parameter name" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='parameterName'></td>
				</tr>
				<tr>
					<td>Values (comma separated):</td>
					<td><input placeholder="Comma-Seperated Values" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='text' class='choiceValues'></td>
				</tr>
			</table>
		    `;
	} else if (selectedOption === "fixed") {
		valueCell.innerHTML = `
			<table class='parameter_config_table'>
				<tr>
					<td>Name:</td>
					<td><input placeholder="Parameter name" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='parameterName'></td>
				</tr>
				<tr>
					<td>Value:</td>
					<td><input placeholder="Value" onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='text' class='fixedValue'></td>
				</tr>
			</table>
		    `;
	}

	valueCell.innerHTML += "<div style='display: none' class='error_element parameterError invert_in_dark_mode'></div>";

	update_command();

	apply_theme_based_on_system_preferences();
}

function add_parameter_row(button) {
	var table = document.getElementById("config_table");
	var rowIndex = button.parentNode.parentNode.rowIndex;
	var numberOfParams = $(".parameterRow").length;
	var newRow = table.insertRow(rowIndex + numberOfParams + 1);

	$(newRow).css("display", "none");

	var optionCell = newRow.insertCell(0);
	var valueCell = newRow.insertCell(1);
	var buttonCell = newRow.insertCell(2);

	optionCell.innerHTML = `<select onchange='updateOptions(this)' class='optionSelect'>
		<option value='range'>Range</option>
		<option value='choice'>Choice</option>
		<option value='fixed'>Fixed</option>
		</select>`;

	valueCell.innerHTML = "";

	buttonCell.innerHTML = "<button class='remove_parameter' onclick='remove_parameter_row(this)'><img class='invert_in_dark_mode' style='height: 1em' src='i/red_x.svg' />&nbsp;Remove</button>";

	updateOptions(optionCell.firstChild);

	newRow.classList.add("parameterRow");
	optionCell.firstChild.classList.add("optionSelect");

	smoothShow($(newRow));

	update_command();

	toggle_disabled_status_of_remove_parameters_depending_on_if_there_are_more_than_one();
}

function remove_parameter_row(button) {
	var table = document.getElementById("config_table");
	var rowIndex = button.parentNode.parentNode.rowIndex;
	var rowCount = table.rows.length;
	if (rowCount > 2) {
		table.deleteRow(rowIndex);
		update_command();
	}

	toggle_disabled_status_of_remove_parameters_depending_on_if_there_are_more_than_one();
}

function string_or_array_to_list (input) {
	if (typeof input === "string") {
		return input;
	} else if (Array.isArray(input)) {
		if (input.length === 1) {
			return input[0];
		} else {
			const listItems = input.map(item => `<li>${item}</li>`);
			return `<ul>${listItems.join("")}</ul>`;
		}
	} else {
		throw new Error("Invalid input type. Only strings or arrays are allowed.");
	}
}

function create_table_row (table, tbody, item) {
	var row = $("<tr>");

	var left_side_content = item.label;

	if ("help" in item && item.help.length > 0) {
		function escapeQuotes(str) {
			return str.replace(/'/g, "&#039;");
		}

		left_side_content += `<a class='tooltip invert_in_dark_mode' title='${escapeQuotes(item.help)}'>&#10067;</a>`;
	}

	var labelCell = $("<td class='left_side'>").html(left_side_content);
	var valueCell = $("<td class='right_side'>").attr("colspan", "2");

	if (item.type === "select") {
		var $select = $("<select>").attr("id", item.id);

		$.each(item.options, function(index, option) {
			var $option = $("<option></option>")
				.attr("value", option.value)
				.text(option.text);

			if (index == 0) {
				$option.prop("selected", "selected");
			}

			$select.append($option);
		});

		$select.change(update_command);

		if (Object.keys(item).includes("onchange")) {
			$select.change(item.onchange);
		}

		valueCell.append($select);
	} else if (item.type === "textarea") {
		var input = $("<textarea>").attr({ id: item.id, type: item.type, value: item.value, placeholder: item.placeholder, min: item.min, max: item.max });
		$(input).css({"width": "95%", "height": "95%"});

		input.on({
			change: update_command,
			keyup: update_command,
			click: update_command
		});

		valueCell.append(input);

		if (Object.keys(item).includes("onchange")) {
			$(input).change(item.onchange);
		}
	} else {
		var input = $("<input>").attr({ id: item.id, type: item.type, value: item.value, placeholder: item.placeholder, min: item.min, max: item.max, step: item.step }).css("width", "95%");

		if (item.type === "checkbox") {
			input.prop("checked", item.value);
		}

		input.on({
			change: update_command,
			keyup: update_command,
			click: update_command
		});

		valueCell.append(input);

		if (Object.keys(item).includes("onchange")) {
			$(input).change(item.onchange);
		}

		if (Object.keys(item).includes("onkeypress")) {
			$(input).keypress(item.onkeypress);
		}

		if (Object.keys(item).includes("onblur")) {
			$(input).blur(item.onblur);
		}

		if (Object.keys(item).includes("onfocus")) {
			$(input).focus(item.onchange);
		}
	}

	if (item.id !== "partition") {
		valueCell.append($(`<div class='error_element invert_in_dark_mode' id="${item.id}_error"></div>`));
	}

	if (item.info) {
		valueCell.append($(`<div class='info_element' id="${item.id}_info">${item.info}</div>`));
	}

	row.append(labelCell, valueCell);
	tbody.append(row);
}

function create_tables() {
	var table = $("#config_table");
	var tbody = table.find("tbody");

	tableData.forEach(function(item) {
		create_table_row(table, tbody, item);
	});

	tbody.append("<tr><td colspan=3><button onclick='add_parameter_row(this)' class='add_parameter' id='main_add_row_button'><img class='invert_in_dark_mode 'src='i/green_plus.svg' style='height: 1em' />&nbsp;Add variable</button></td></tr>");

	var hidden_table = $("#hidden_config_table");
	var hidden_tbody = hidden_table.find("tbody");

	hiddenTableData.forEach(function(item) {
		create_table_row(hidden_table, hidden_tbody, item);
	});

	highlight_all_bash();

	useSmoothFade = true;

	$("#site").show();
	$("#loader").remove();
}

function update_url() {
	var url = window.location.href;

	var index = url.indexOf("no_update_url");

	if (index !== -1) {
		return;
	}

	var params = [];

	function push_value(item) {
		var element = $("#" + item.id);
		var value;

		if (element.is(":checkbox")) {
			value = element.is(":checked") ? 1 : 0;
		} else {
			value = encodeURIComponent(element.val());
		}

		params.push(item.id + "=" + value);
	}

	tableData.forEach(function(item) {
		push_value(item);
	});

	hiddenTableData.forEach(function(item) {
		push_value(item);
	});

	var parameterIndex = 0;
	$(".parameterRow").each(function() {
		var option = $(this).find(".optionSelect").val();
		var parameterName = $(this).find(".parameterName").val();

		if(parameterName && !parameterName.match(/^\w+$/)) {
			//error(`Parameter name "${parameterName}" does have invalid characters. Must be all letters.`)
		} else if (parameterName) {
			var param_base = "parameter_" + parameterIndex;
			if (option === "range") {
				var minValue = $(this).find(".minValue").val();
				var maxValue = $(this).find(".maxValue").val();
				var numberType = $(this).find(".numberTypeSelect").val();
				var log_scale = $(this).find(".log_scale").is(":checked") ? "true" : "false";

				params.push(param_base + "_name=" + encodeURIComponent(parameterName));
				params.push(param_base + "_type=" + encodeURIComponent(option));
				params.push(param_base + "_min=" + encodeURIComponent(minValue));
				params.push(param_base + "_max=" + encodeURIComponent(maxValue));
				params.push(param_base + "_number_type=" + encodeURIComponent(numberType));
				params.push(param_base + "_log_scale=" + encodeURIComponent(log_scale));
			} else if (option === "choice") {
				var choiceValues = $(this).find(".choiceValues").val();

				params.push(param_base + "_name=" + encodeURIComponent(parameterName));
				params.push(param_base + "_type=" + encodeURIComponent(option));
				params.push(param_base + "_values=" + encodeURIComponent(choiceValues));
			} else if (option === "fixed") {
				var fixedValue = $(this).find(".fixedValue").val();

				params.push(param_base + "_name=" + encodeURIComponent(parameterName));
				params.push(param_base + "_type=" + encodeURIComponent(option));
				params.push(param_base + "_value=" + encodeURIComponent(fixedValue));
			}
		}
		parameterIndex++;
	});

	params.push("partition=" + encodeURIComponent($("#partition").val()));

	if (initialized) {
		var url = window.location.origin + window.location.pathname + "?" + params.join("&") + "&num_parameters=" + $(".parameterRow").length;

		try {
			window.history.replaceState(null, null, url);
		} catch (err) {
			err = "" + err;

			if(err.includes("The operation is insecure") && !shown_operation_insecure_without_server) {
				log(err);
				shown_operation_insecure_without_server = true;
			} else if (!err.includes("The operation is insecure")) {
				error(err);
			}
		}
	}
}

function copy_bashcommand_to_clipboard_main () {
	var serialized = $("#command_element").text();
	copy_to_clipboard(serialized);

	$("#copied_main").show();
	setTimeout(function() {
		$("#copied_main").fadeOut();
	}, 5000);
}

function copy_bashcommand_to_clipboard_curl () {
	var serialized = $("#curl_command").text();
	copy_to_clipboard(serialized);

	$("#copied_curl").show();
	setTimeout(function() {
		$("#copied_curl").fadeOut();
	}, 5000);
}

function get_parameter_names(only_these_types = []) {
	var values = $(".parameterName").map(function() {
		var parameterValue = $(this).val();
		var parameterType = $(this).closest('.parameterRow')
			.find(".optionSelect")
			.val();

		if (only_these_types.length > 0 && only_these_types.includes(parameterType)) {
			return parameterValue;
		} else if (only_these_types.length === 0) {
			return parameterValue;
		}
	}).get().filter(Boolean);

	return values;
}

function is_valid_constraints_string(input) {
	const parameter_names = get_parameter_names(["range"]);

	input = input.replaceAll(/;;*/g, ";");
	input = input.replace(/;;*\s*$/, "");
	input = input.replace(/^\s*;;*/, "");

	return input.split(";").map(part => test_if_equation_is_valid(part, parameter_names)).join("");
}

function isAnyLogScaleSet() {
    return $(".log_scale:checked").length > 0;
}

function run_when_document_ready () {
	create_tables();
	update_partition_options();

	var urlParams = new URLSearchParams(window.location.search);
	tableData.forEach(function(item) {
		var paramValue = urlParams.get(item.id);
		if (paramValue !== null) {
			var $element = $("#" + item.id);
			if ($element.is(":checkbox")) {
				var boolValue = /^(1|true)$/i.test(paramValue);
				$element.prop("checked", boolValue).trigger("change");
			} else {
				$element.val(paramValue).trigger("change");
			}
		}
	});

	hiddenTableData.forEach(function(item) {
		var paramValue = urlParams.get(item.id);
		if (paramValue !== null) {
			var $element = $("#" + item.id);
			if ($element.is(":checkbox")) {
				var boolValue = /^(1|true)$/i.test(paramValue);
				$element.prop("checked", boolValue).trigger("change");
			} else {
				$element.val(paramValue).trigger("change");
			}
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
		var param_base = "parameter_" + parameterIndex
		var parameterName = urlParams.get(param_base + "_name");
		var option = urlParams.get(param_base + "_type");

		if (parameterName && option) {
			$(this).find(".parameterName").val(parameterName);
			$(this).find(".optionSelect").val(option).trigger('change');

			if (option === 'range') {
				$(this).find(".minValue").val(urlParams.get(param_base + "_min"));
				$(this).find(".maxValue").val(urlParams.get(param_base + "_max"));
				$(this).find(".numberTypeSelect").val(urlParams.get(param_base + "_number_type"));

				var log_scale_value = urlParams.get(param_base + "_log_scale") == "true" ? true : false;
				$(this).find(".log_scale").prop("checked", log_scale_value);
			} else if (option === 'choice') {
				$(this).find(".choiceValues").val(urlParams.get(param_base + "_values"));
			} else if (option === 'fixed') {
				$(this).find(".fixedValue").val(urlParams.get(param_base + "_value"));
			}
		}
		parameterIndex++;
	});

	update_command();

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

	initialized = true;

	fadeTime = fadeTimeAfterLoading;

	setTimeout(update_command, 200);

	update_url();

	update_command();
}

function test_if_equation_is_valid(str, names) {
	var errors = [];
	var isValid = true;

	if (!str.includes(">=") && !str.includes("<=")) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Missing '>=' or '<=' operator. The equation should include a comparison operator.");
		isValid = false;
	}

	var splitted = str.includes(">=") ? str.split(">=") : str.split("<=");
	if (splitted.length !== 2) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Equation format is incorrect. There should be exactly one comparison operator.");
		isValid = false;
	}

	var left_side = splitted[0].replace(/\s+/g, "");
	if (!left_side) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Left side is empty or contains only whitespace. Please provide an expression on the left side.");
		isValid = false;
	}

	if (isValid) {
		var right_side = splitted[1].trim();

		if (names.includes(left_side) && names.includes(right_side)) {
			return "";
		}

		if (!/^[+-]?\d+(\.\d+)?$/.test(right_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> The right side does not look like a constant. The right side should be a valid number.");
			isValid = false;
		}

		// Escape variable names for regex usage
		var escapedNames = names.map(n => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
		var namePattern = `(?:${escapedNames.join("|")})`;

		var numberPattern = "\\d+(?:\\.\\d+)?";
		var factorPattern = `(?:${numberPattern}|${namePattern})`;
		var productPattern = `${factorPattern}(?:\\*${factorPattern})*`;
		var termPattern = `[+-]?${productPattern}`;
		var fullPattern = `^${termPattern}(?:[+-]${termPattern})*$`;

		var regex = new RegExp(fullPattern);
		if (!regex.test(left_side)) {
			errors.push(`<img src='i/warning.svg' style='height: 1em' /> Left side does not match expected pattern. Invalid term or parameter format detected in '${left_side}'`);
			isValid = false;
		}

		// Check for multiple operators in a row (e.g., ++, --, **)
		if (/[*+-]{2,}/.test(left_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> The left side contains multiple operators directly in a row. Ensure that operators are used correctly.");
			isValid = false;
		}

		// Check for number directly followed by variable without *
		var nr_re = "([+-]?\\d+(\\.\\d+)?)";
		var number_followed_by_varname = new RegExp(`${nr_re}(${namePattern})`);
		if (number_followed_by_varname.test(left_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> A number is followed directly by a variable name without an operator. Example: '3x' is not valid, use '3*x' instead.");
			isValid = false;
		}

		// Check for starting with invalid operator
		if (/^[*+]/.test(left_side)) {
			errors.push("<img src='i/warning.svg' style='height: 1em' /> Left side starts with an operator. The equation cannot start with an operator.");
			isValid = false;
		}
	}

	function errorsToHtml(_errors) {
		if (_errors.length) {
			_errors.unshift(`<b>Equation: ${str}</b>`);
			return "<ul>" + _errors.map(error => `<li>${error}</li>`).join('') + "</ul>";
		}
		return "";
	}

	var ret_str = errorsToHtml(errors);
	return ret_str;
}

function equation_validation_test () {
	var param_names = ["hallo", "welt", "x", "y"];

	var failed = 0;
	var test_counter = 0;

	function internal_equation_checker(code, should_be) {
		var ret_str = test_if_equation_is_valid(code, param_names);

		if (should_be === true) {
			if (ret_str !== "") {
				console.error(`Error: ${code} failed. Should be: ${should_be}, is: ${ret_str}`);
				failed = failed + 1;
			}
		} else {
			if (ret_str === "") {
				console.error(`Error: ${code} failed. Should be: ${should_be}, is: ${ret_str}`);
				failed = failed + 1;
			}
		}

		test_counter++;
	}

	internal_equation_checker("x >= y", true);
	internal_equation_checker("x + y >= 5", true);
	internal_equation_checker("x + y <= 5", true);
	internal_equation_checker("2*x + 231*y <= 5", true);
	internal_equation_checker("x + 231*y <= 5", true);
	internal_equation_checker("2*x + y <= 5", true);
	internal_equation_checker("2*x+y<=5", true);
	internal_equation_checker("2*x+y>=5", true);
	internal_equation_checker("2+y >= 10", true);
	internal_equation_checker("3*x + 5*y >= 10", true);
	internal_equation_checker("10*hallo + 2*welt <= 100", true);
	internal_equation_checker("x - y >= 0", true);
	internal_equation_checker("5*x + 7*y + 9*hallo - 3*welt <= 50", true);
	internal_equation_checker("y + 10 >= 20", true);
	internal_equation_checker("x - 10 <= 5", true);
	internal_equation_checker("100*x + 50*y >= 1000", true);
	internal_equation_checker("2*x - 3*y + 4*hallo + 5*welt <= 42", true);
	internal_equation_checker("welt + 2*x + 3*y >= 0", true);
	internal_equation_checker("x + y + hallo + welt <= 99", true);
	internal_equation_checker("x + 2*y - 3*hallo + 4*welt >= -50", true);
	internal_equation_checker("x + 2*y + 3*hallo + 4*welt <= 1000000", true);
	internal_equation_checker("2*hallo - 3*welt + 4*x - 5*y >= -100", true);
	internal_equation_checker("100000*x + 200000*y <= 500000", true);
	internal_equation_checker("x - y + hallo - welt >= 1", true);
	internal_equation_checker("3*x - 5*y + 7*hallo - 9*welt <= 0", true);
	internal_equation_checker("hallo + welt + x + y >= 12345", true);
	internal_equation_checker("2*x - 2*y + 2*hallo - 2*welt >= -2", true);
	internal_equation_checker("x + y + hallo + welt <= -10", true);
	internal_equation_checker("3*x - 4*y + 5*hallo <= 30", true);
	internal_equation_checker("10*x + 2*y - 3*hallo + 4*welt >= -15", true);
	internal_equation_checker("x + y + 3*hallo - 2*welt <= 200", true);
	internal_equation_checker("2*x + 3*y - 4*hallo + 5*welt >= 10", true);
	internal_equation_checker("100*x + 50*y + 20*hallo - 30*welt <= 5000", true);
	internal_equation_checker("x - 2*y + 3*hallo - 4*welt >= -25", true);
	internal_equation_checker("4*x + 5*y + 6*hallo + 7*welt <= 1000", true);
	internal_equation_checker("2*hallo + 3*welt + 4*x + 5*y >= -500", true);
	internal_equation_checker("1*x + 2*y + 3*hallo + 4*welt <= 99999", true);
	internal_equation_checker("50*x - 25*y + 75*hallo - 125*welt >= 250", true);
	internal_equation_checker("3 * x + y >= 10", true);
	internal_equation_checker("999999*x + 888888*y - 777777*hallo + 666666*welt <= 555555", true);
	internal_equation_checker("0*x + 0*y + 0*hallo + 0*welt >= 0", true);
	internal_equation_checker("-3*x - 4*y + 5*hallo - 6*welt <= -100", true);
	internal_equation_checker("0002*x + 0003*y + 0004*hallo - 0005*welt >= 0006", true);
	internal_equation_checker("1*x + 2*y + 3*hallo + 4*welt <= 0", true);
	internal_equation_checker("1*x - 1*y + 1*hallo - 1*welt <= -1", true);
	internal_equation_checker("x + 2*y - 3*hallo + 4*welt >= -999999", true);
	internal_equation_checker("500000*x - 250000*y + 125000*hallo - 62500*welt <= 10", true);
	internal_equation_checker("123456789*x + 987654321*y >= 111111111", true);
	internal_equation_checker("x * y >= 10", true);
	internal_equation_checker("1000000*x + 1000000*y + 1000000*hallo + 1000000*welt >= 1000000", true);
	internal_equation_checker("-1*x + 2*y - 3*hallo + 4*welt >= -5", true);
	internal_equation_checker("x + y - 2*hallo + 3*welt <= 50", true);
	internal_equation_checker("1000*x + 999*y - 1234*hallo + 5555*welt >= 99999", true);
	internal_equation_checker("x + 1*y + hallo - 5*welt <= 20", true);
	internal_equation_checker("999999*x + 123456*y + 789101*hallo - 654321*welt <= 1000000000", true);
	internal_equation_checker("x*y + hallo - welt >= 100", true);
	internal_equation_checker("x + 2*welt - 3*hallo + 4*y >= -50", true);
	internal_equation_checker("0.0001*x + 0.0002*y >= 0.0003", true);
	internal_equation_checker("x + y + 5*hallo - 6*welt >= 100", true);
	internal_equation_checker("5*x + 3*y >= 15", true);
	internal_equation_checker("x + 2*y - 3*hallo + 4*welt <= 500", true);
	internal_equation_checker("100*x - 50*y + 75*hallo - 25*welt >= 1000", true);
	internal_equation_checker("1*x + 1*y + 1*hallo + 1*welt <= 10", true);
	internal_equation_checker("10*x + 20*y + 30*hallo - 40*welt >= -100", true);
	internal_equation_checker("123*x + 456*y - 789*hallo + 987*welt <= 654", true);
	internal_equation_checker("2*x - 3*y + 4*hallo + 5*welt >= -1000", true);
	internal_equation_checker("9999*x + 8888*y + 7777*hallo - 6666*welt <= 5555", true);
	internal_equation_checker("3*x - 5*y + 7*hallo - 9*welt >= -500", true);
	internal_equation_checker("0.5*x + 0.25*y - 0.75*hallo + 1.5*welt <= 2.5", true);
	internal_equation_checker("2*x+y>=5*4", false);
	internal_equation_checker("2*x+y", false);
	internal_equation_checker("2*x+y > 10", false);
	internal_equation_checker("2*x+y >= abc", false);
	internal_equation_checker("2*x+y >= welt", false);
	internal_equation_checker("2/x+y >= 10", false);
	internal_equation_checker("2+ASD >= 10", false);
	internal_equation_checker("x + y + 5*hallo - 2* >= 10", false);
	internal_equation_checker("3*x + y <= 10 + 5", false);
	internal_equation_checker("2*x ++ y >= 10", false);
	internal_equation_checker("x + y >= ", false);
	internal_equation_checker("10*x + y >= abc", false);
	internal_equation_checker("x + y ==> 10", false);
	internal_equation_checker("x * 2 >= 10", true);
	internal_equation_checker("3*x + y => 10", false);
	internal_equation_checker("2*x + y >= ", false);
	internal_equation_checker("x+y > 10", false);
	internal_equation_checker("x+abc >= 10", false);
	internal_equation_checker("x / 2 >= 5", false);
	internal_equation_checker("2*x+y>=5*4", false);
	internal_equation_checker("2**x + y >= 10", false);
	internal_equation_checker(">= 10", false);
	internal_equation_checker("x + y = 10", false);
	internal_equation_checker("x + y >== 10", false);
	internal_equation_checker("3x + y >= 10", false);
	internal_equation_checker("2* + y >= 10", false);
	internal_equation_checker("*x + y >= 10", false);
	internal_equation_checker("2**x + y >= 10", false);
	internal_equation_checker("2*x+y=10", false);
	internal_equation_checker("2*x+y >= ", false);
	internal_equation_checker(">= 10", false);
	internal_equation_checker("x + y >== 10", false);
	internal_equation_checker("3*x + + 5*y >= 10", false);
	internal_equation_checker("2*x / 3*y >= 10", false);
	internal_equation_checker("2*x + 5..y >= 10", false);
	internal_equation_checker("x + y + hallo*4 >= 20", true);
	internal_equation_checker("x + = y >= 5", false);
	internal_equation_checker("x + 2* + y >= 10", false);
	internal_equation_checker("x + 2*5*y >= ", false);
	internal_equation_checker("x + y +- 10 >= 10", false);
	internal_equation_checker("2*x / y + 5 >= 10", false);
	internal_equation_checker("2*x + y ** 3 >= 10", false);
	internal_equation_checker("x + + 2*y >= 5", false);
	internal_equation_checker("x + + 2 >= 5", false);
	internal_equation_checker("2*x+y >= 5 4", false);
	internal_equation_checker("x + 2*world - 3*hallo + 4*y >= -50", false);
	internal_equation_checker("x + y + z + 5*hallo - 6*welt >= 100", false);
	internal_equation_checker("5x + 3y >= 15", false);
	internal_equation_checker("2*x + y =>= 10", false);
	internal_equation_checker("x + y >== 5", false);
	internal_equation_checker("3x * y >= 10", false);
	internal_equation_checker("2**x + y <= 5", false);
	internal_equation_checker(">= 10", false);
	internal_equation_checker("5*x + 3*y =", false);
	internal_equation_checker("10*x + abc >= 50", false);
	internal_equation_checker("3*x + y 10", false);
	internal_equation_checker("x + y >> 10", false);
	internal_equation_checker("2*x + 3*y - 4*hallo + 5*welt <= 100", true);
	internal_equation_checker("1.5*hallo + 2.5*welt >= -3.14", true);
	internal_equation_checker("hallo + welt <= 10", true);
	internal_equation_checker("hallo - welt >= -5", true);
	internal_equation_checker("hallo + welt <= 𝟜𝟚", false);
	internal_equation_checker("𝟏𝟎*hallo + 𝟐𝟎*welt - 𝟑𝟎*x + 𝟒𝟎*y <= 𝟓𝟎", false);
	internal_equation_checker("hallo\t+\twelt \t<= 42", true);
	internal_equation_checker("1.2*x + 2.3*y - 3.4*hallo + 4.5*welt >= 6.7", true);
	internal_equation_checker("0*x + 0*y + 0*hallo + 0*welt <= 0", true);
	internal_equation_checker("1000000*x - 999999*y + 888888*hallo - 777777*welt >= 666666", true);
	internal_equation_checker("hallo + 0.0*welt - 0.0*x + 0.0*y <= 7", true);
	internal_equation_checker("hallo - welt - x - y >= -hallo", false);
	internal_equation_checker("𝒙 + 𝒚 - 𝒉𝒂𝒍𝒍𝒐 + 𝒘𝒆𝒍𝒕 <= 𝟏𝟎", false);
	internal_equation_checker("10**hallo + 20**welt - 30**x + 40**y <= 50", false);
	internal_equation_checker("hallo+welt <=+10", true);
	internal_equation_checker("  x  +   y  <=  15  ", true);
	internal_equation_checker("5 * hallo + 6 * welt - 7 * x + 8 * y >= 9", true);
	internal_equation_checker("hallo + 0.0000001*welt <= 42", true);
	internal_equation_checker("0.00000000000000001*x + 0.00000000000000002*y - 0.00000000000000003*hallo <= 0", true);
	internal_equation_checker("1000000000000000000000*hallo + 2000000000000000000000*welt >= 3000000000000000000000", true);
	internal_equation_checker("hallo - (welt) + x - (y) <= 10", false);
	internal_equation_checker("x/2 + y/3 - hallo/4 + welt/5 <= 1", false);
	internal_equation_checker("𝕙𝕒𝕝𝕝𝕠 + 𝕨𝕖𝕝𝕥 - 𝕩 + 𝕪 <= 𝟙𝟘", false);
	internal_equation_checker("hallo + welt + x + y <= 1_000_000", false);


	internal_equation_checker("hällo + welt <= 10", false);
	internal_equation_checker("hallo + welt <= ", false);
	internal_equation_checker("x + y = 10", false);
	internal_equation_checker("x + y << 10", false);
	internal_equation_checker("hallo + welt => 20", false);
	internal_equation_checker("𝑥 + 𝑦 ≤ 10", false);
	internal_equation_checker("x++y <= 10", false);
	internal_equation_checker("hallo / 0 <= 10", false);
	internal_equation_checker("x + y <= '10'", false);
	internal_equation_checker("hallo + welt + <= 10", false);
	internal_equation_checker("2hallo + 3welt <= 10", false);
	internal_equation_checker("hallo + welt <== 10", false);
	internal_equation_checker("x + y <= (10", false);
	internal_equation_checker("hallo ** welt <= 10", false);
	internal_equation_checker("hallo + €welt <= 10", false);
	internal_equation_checker("hallo, welt <= 10", false);
	internal_equation_checker("hallo + 1.2.3 <= 10", false);
	internal_equation_checker("hallo + 'welt' <= 10", false);
	internal_equation_checker("hallo + !welt <= 10", false);
	internal_equation_checker("hallo + x +- y <= 10", false);
	internal_equation_checker("hallo + --welt <= 10", false);
	internal_equation_checker("hallo x + welt y <= 10", false);
	internal_equation_checker("hallo+welt=<10", false);
	internal_equation_checker("hallo + {welt} <= 10", false);
	internal_equation_checker("hallo 𝙥𝙡𝙪𝙨 welt <= 10", false);

	console.log(`Ran ${test_counter} tests (${failed} failed)`);
}

function toggle_disabled_status_of_remove_parameters_depending_on_if_there_are_more_than_one() {
	var nr_params = $(".parameterName").length;

	if (nr_params <= 1) {
		$(".remove_parameter").prop("disabled", true);
	} else {
		$(".remove_parameter").prop("disabled", false);
	}
}

function toggleHiddenConfigTableIfError() {
	let table = $("#hidden_config_table");

	if (!table.is(":visible") && table.find(".error_element").filter(function() {
		var this_display = $(this).css("display");
		return this_display !== "none" && $(this).text() !== "";
	}).length > 0) {
		table.toggle();
	}
}

function show_warning_for_model_when_custom_generation_strategy_is_set() {
	smoothShow($("#model_error").html("Custom generation strategy is set, so --model is ignored."));
}

function hide_warning_when_custom_custom_generation_strategy_isnt_set() {
	smoothHide($("#model_error").html(""));
}

function toggle_model_warning_for_custom_generation_strategy() {
	if($("#generation_strategy").val() == "") {
		hide_warning_when_custom_custom_generation_strategy_isnt_set();
	} else {
		show_warning_for_model_when_custom_generation_strategy_is_set();
	}
}

function add_equation_spaces(expression) {
	const operators = {
		'>=': '__GE__',
		'<=': '__LE__',
		'==': '__EQ__',
		'!=': '__NE__',
		'=>': '__AR__',
	};

	for (const [op, placeholder] of Object.entries(operators)) {
		expression = expression.replaceAll(op, placeholder);
	}

	expression = expression.replace(/([+\-*/()=<>])/g, ' $1 ');

	for (const [op, placeholder] of Object.entries(operators)) {
		expression = expression.replaceAll(placeholder, ` ${op} `);
	}

	return expression.replace(/\s+/g, ' ').trim();
}
