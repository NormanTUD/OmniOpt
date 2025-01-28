var initialized = false;
var shown_operation_insecure_without_server = false;

var l = typeof log === "function" ? log : console.log;

var tableData = [
	{ label: "Partition", id: "partition", type: "select", value: "", options: [], "required": true, "help": "The Partition your job will run on. This choice may restrict the amount of workers, GPUs, maximum time limits and a few more options." },
	{ label: "Experiment name", id: "experiment_name", type: "text", value: "", placeholder: "Name of your experiment (only letters and numbers)", "required": true, "regex": "^[a-zA-Z0-9_]+$", "help": "Name of your experiment. Will be used for example for the foldername it's results will be saved in." },
	{ label: "Reservation", id: "reservation", type: "text", value: "", placeholder: "Name of your reservation (optional)", "required": false, "regex": "^[a-zA-Z0-9_]*$", "help": "If you have a reservation, use it here. It makes jobs start faster, but is not necessary technically." },
	{ label: "Account", id: "account", type: "text", value: "", placeholder: "Account the job should run on", "help": "Depending on which groups you are on, this determines to which account group on the Slurm-system that job should be linked. If left empty, it will solely be determined by your login-account." },
	{ label: "Memory (in GB)", id: "mem_gb", type: "number", value: 1, placeholder: "Memory in GB per worker", min: 1, max: 1000 },
	{ label: "Timeout for the main program", id: "time", type: "number", value: 60, placeholder: "Timeout for the whole program", min: 1, "help": "This is the maximum amount of time that your main job will run, spawn jobs and collect results." },
	{ label: "Timeout for a single worker", id: "worker_timeout", type: "number", value: 60, placeholder: "Timeout for a single worker", min: 1, "help": "This is the maximum amount of time a single worker may run." },
	{ label: "Maximal number of evaluations", id: "max_eval", type: "number", value: 500, placeholder: "Maximum number of evaluations", min: 1, "max": 100000000, "help": "This number determines how many successful workers in total are needed to end the job properly." },
	{ label: "Max. number of Workers", id: "num_parallel_jobs", type: "number", value: 20, placeholder: "Maximum number of workers", "min": 1, "max": 100000000, "help": "The number maximum of workers that can run in parallel. While running, the number may be below this some times." },
	{ label: "GPUs per Worker", id: "gpus", type: "number", value: 0, placeholder: "Number of GPUs per worker", min: 0, max: 10, "help": "How many GPUs each worker should have." },
	{ label: "Number of random steps", id: "num_random_steps", type: "number", value: 20, placeholder: "Number of random steps", min: 1, "help": "At the beginning, some random jobs are started. By default, it is 20. This is needed to 'calibrate' the surrogate model." },
	{ label: "Follow", id: "follow", type: "checkbox", value: 1, "help": "tail -f the .out-file automatically, so you can see the output as soon as it appears. This does not change the results of OmniOpt2, but only the user-experience. This way, you see results as soon as they are available without needing to manually look for the outfile. Due to it using tail -f internally, you can simply CTRL-c out of it without cancelling the job." },
	{ label: "Live-Share", id: "live_share", type: "checkbox", value: 0, "help": "Automatically uploads the results to our servers for 30 days, so you can trace the output live in the browser, without needing SSH. By using this, you agree to have your username published online." },
	{ label: "Send anonymized usage statistics?", id: "send_anonymized_usage_stats", type: "checkbox", value: 1, "help": "This contains the time the job was started and ended, it's exit code, and runtime-uuid to count the number of unique runs and a 'user-id', which is a hashed output of the aes256 encrypted username/groups combination and some other values, but cannot be traced back to any specific user." },
	{ label: "Automatically checkout to latest checked version", id: "checkout_to_latest_tested_version", type: "checkbox", value: 1, "help": "For every commit, the CI pipeline checks all the tests and if they succeed, create a new version tag. If this is activated, you get the latest version that was tested properly and where all tests succeeded. If disabled, you may get the newest version, but it may has preventable bugs." },
	//{ label: "Show graphics at end?", id: "show_sixel_graphics", type: "checkbox", value: 0, "info": "May not be supported on all terminals.", "help": "This will use the module sixel to try to print your the results to the command line. If this doesn't work for you, please disable it. It has no effect on the results of OmniOpt2." },
	{ label: "Run program", id: "run_program", type: "textarea", value: "", placeholder: "Your program with parameters", "required": true, "info": "Use Variable names like this: <br><code class=\"highlight_me dark_code_bg invert_in_dark_mode\">bash /absolute/path/to/run.sh --lr=%(learning_rate) --epochs=%(epochs)</code>. See <a target=\"_blank\" href=\"tutorials.php?tutorial=run_sh\">this tutorial</a> to learn about the <code>run.sh</code>-file", "help": "This is the program that will be optimized. Use placeholder names for places where your hyperparameters should be, like '%(epochs)'. The GUI will warn you about missing parameter definitions, that need to be there in the parameter selection menu, and will not allow you to run OmniOpt2 unless all parameters are filled." }
];

var hiddenTableData = [
	{ label: "Result-Names", id: "result_names", type: "text", value: "RESULT", placeholder: "Name of the value that should be searched for, like RESULT", "required": true, "regex": /^(((([a-zA-Z][a-zA-Z0-9_]*)(=(min|max))?)(\s|$)?)+)$/, "help": "A space-seperated list of strings to search for in the STDOUT of your program like, for example, the loss. Default is RESULT=min.", "info": "This is used for the regex to search through the STDOUT of your program to find RESULT-values. You can define multiple result values like this: <tt>RESULT1 RESULT2 RESULT3</tt>. Can also be defined with min and max: <tt>LOSS=min PERFORMANCE=max ...</tt>" },
	{ label: "CPUs per Task", id: "cpus_per_task", type: "number", value: 1, placeholder: "CPUs per Task", min: 1, max: 10, "help": "How many CPUs should be assigned to each task (for workers)" },
	{ label: "Number of nodes", id: "nodes_per_job", type: "number", value: 1, placeholder: "tasks", min: 1, "help": "How many nodes (for each worker)" },
	{ label: "Seed", id: "seed", type: "number", value: "", placeholder: "Seed for reproducibility", "info": "When set, this will make OmniOpt2 runs reproducible, given your program also acts deterministically.", required: false },
	{ label: "Verbose", id: "verbose", type: "checkbox", value: 0, "help": "This enables more output to be shown. Useful for debugging. Does not change the outcome of your Optimization." },
	{ label: "Debug", id: "debug", type: "checkbox", value: 0, "help": "This enables more output to be shown. Useful for debugging. Does not change the outcome of your Optimization." },
	//{ label: "Maximize?", id: "maximize", type: "checkbox", value: 0, "help": "When set, the job will be maximized instead of minimized. This option may not work with all plots currently (TODO).", 'info': 'Currently, this is in alpha and may give wrong results!' },
	{ label: "Grid search?", id: "gridsearch", type: "checkbox", value: 0, info: "Switches range parameters to choice with <tt>max_eval</tt> number of steps. Converted to int when parameter is int. Only use together with the <i>FACTORIAL</i>-model.", "help": "This internally converts range parameters to choice parameters by laying them out seperated by the max eval number through the search space with intervals. Use FACTORIAL model to make it work properly. Still beta, though! (TOOD)" },
	{ label: "Model", id: "model", type: "select", value: "",
		options: [
			{ "text": "BOTORCH_MODULAR", "value": "BOTORCH_MODULAR" },
			{ "text": "SOBOL", "value": "SOBOL" },
			{ "text": "GPEI", "value": "GPEI" },
			{ "text": "FACTORIAL", "value": "FACTORIAL" },
			{ "text": "SAASBO", "value": "SAASBO" },
			{ "text": "FULLYBAYESIAN", "value": "FULLYBAYESIAN" },
			//{ "text": "LEGACY_BOTORCH", "value": "LEGACY_BOTORCH" },
			{ "text": "UNIFORM", "value": "UNIFORM" },
			{ "text": "BO_MIXED", "value": "BO_MIXED" }
		], "required": true,
		info: `
			<ul>
			    <li>BOTORCH_MODULAR: <a href='https://web.archive.org/web/20240715080430/https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf' target='_blank'>Default model</a></li>
			    <li><a target="_blank" href="https://en.wikipedia.org/wiki/Sobol_sequence">SOBOL</a>: Random search</li>
			    <li><i><a href='https://arxiv.org/pdf/1807.02811'>GPEI</a></i>: Uses Expected Improvement based on a Gaussian Process model to choose the next evaluation point.</li>
			    <li>FACTORIAL: <a target='_blank' href='https://ax.dev/tutorials/factorial.html'>All possible combinations</a></li>
			    <li>SAASBO: <i><a target='_blank' href='https://arxiv.org/pdf/2103.00349'>Sparse Axis-Aligned Subspace Bayesian Optimization</a></i> for high-dimensional Bayesian Optimization, recommended for hundreds of dimensions</li>
			    <li>FULLYBAYESIAN: Considers the full uncertainty of the Bayesian model in the optimization process</li>
			    <!--<li>LEGACY_BOTORCH: ???</li>-->
			    <li>UNIFORM: Random (uniformly distributed)</li>
			    <li>BO_MIXED: '<i><a href='https://ax.dev/api/_modules/ax/modelbridge/dispatch_utils.html'>BO_MIXED</a></i>' optimizes all range parameters once for each combination of choice parameters, then takes the optimum of those optima. The cost associated with this method grows with the number of combinations, and so it is only used when the number of enumerated discrete combinations is below some maximum value.</li>
			</ul>
`,
		"help": "The model chosen here tries to make an informed choice (except SOBOL, which means random search) about where to look for new hyperparameters. Different models are useful for different optimization problems, though which is best for what is something that I still need to search exactly (TODO!)"
	},

	{ label: "Installation-Method", id: "installation_method", type: "select", value: "",
		options: [
			{ "text": "Use git clone to clone OmniOpt2", "value": "clone" },
			{ "text": "Use pip and install OmniOpt2 from pypi (may not be the latest version)", "value": "pip" }
		], "required": true,
		"info": "Changes the way OmniOpt2 is installed.",
		"help": "If you want to install OmniOpt2 via pip, chose it here. It may not always have the latest version.",
		"use_in_curl_bash": true
	},

	{ label: "Run-Mode", id: "run_mode", type: "select", value: "",
		options: [
			{ "text": "Locally or on a HPC system", "value": "local" },
			{ "text": "Docker", "value": "docker" }
		], "required": true,
		"info": "Changes the curl-command and how omniopt is installed and executed.",
		"help": "If set to docker, it will run in a local docker container."
	},

	{
		label: "Constraints",
		id: "constraints",
		type: "text",
		value: "",
		placeholder: "Constraints in the form of 'a + b >= 10', seperated by Semicolon (;)", info: "Use simple constraints in the form of <code>a + b >= 10</code>, where <code>a</code> and <code>b</code> are parameter names. Possible comparisons: <code>>=</code>, <code><=</code>", "help": "The contraints allow you to limit values of the hyperparameter space that are allowed. For example, you can set that the sum of all or some parameters must be below a certain number. This may be useful for simulations, or complex functions that have certain limitations depending on the hyperparameters."
	},
];



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
			<input type='number' min=-1 max=159 class="invert_in_dark_mode time_picker_hours" value='${_hours}' onchange='update_original_time_element("${input_id}", this)'></input> Hours,
			<input type='number' min=-1 step=31 class="invert_in_dark_mode time_picker_minutes" value='${_minutes}' onchange='update_original_time_element("${input_id}", this)'></input> Minutes
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
			errors.push(this_error);
			$("#" + item.id + "_error").html(this_error).show();
		} else {
			$("#" + item.id + "_error").html("").hide();
		}
	}

	if (item.type === "checkbox") {
		value = $("#" + item.id).is(":checked") ? "1" : "0";
		if (value === "1") {
			command += " --" + item.id;
		}
	} else if ((item.type === "textarea" || item.type === "text") && value === "") {
		if(item.required) {
			var this_error = "Field '" + item.label + "' is required.";
			$("#" + item.id + "_error").html(this_error).show();
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
			$("#time_error").html(string_or_array_to_list(new_errors)).show();
			errors.push(...new_errors);
		} else {
			$("#time_error").html("").hide();
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
			$("#worker_timeout_error").html(string_or_array_to_list(new_errors)).show();
			errors.push(...new_errors);
		} else {
			$("#worker_timeout_error").html("").hide();
			command += " --" + item.id + "=" + value;
		}
	} else if (item.type === "number") {
		var numValue = parseFloat(value);

		if (isNaN(numValue) && ((Object.keys(item).includes("required") && item.required) || !Object.keys(item).includes("required"))) {
			errors.push("Invalid value for '" + item.label + "'. Must be a number.");
		} else if (item.min && item.max && (numValue < item.min || numValue > item.max)) {
			errors.push("Value for '" + item.label + "' must be between " + item.min + " and " + item.max + ".");
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
				var err_msg = `<code>%(${test_this_var_name})</code> not in existing defined parameters.`;
				new_errors.push(err_msg);
			}
		}

		for (var k = 0; k < existing_parameter_names.length; k++) {
			var test_this_var_name = existing_parameter_names[k];

			if(!variables_in_run_program.includes(test_this_var_name)) {
				var err_msg = `<code>%(${test_this_var_name})</code> is defined but not used.`;
				new_errors.push(err_msg);
			}
		}

		if(new_errors.length) {
			$("#run_program_error").html(string_or_array_to_list(new_errors)).show();
			errors.push(...new_errors);
		} else {
			$("#run_program_error").html("").hide();
		}

		value = btoa(value);

		//var base_64_encoder = value; //.replaceAll(/"/g, '\\"');
		//log("base_64_encoder:", base_64_encoder);

		//value = `$(${base_64_encoder} | base64 -w 0)`;

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
				$("#" + item.id + "_error").html("").hide();
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
	return false;
}

function update_command() {
	set_min_max();

	var errors = [];
	var warnings = [];
	var command = "./omniopt";

	var curl_options = "";

	if ($("#run_mode").val() == "docker") {
		command = "bash omniopt_docker omniopt";
	}

	tableData.forEach(function(item) {
		if(!Object.keys(item).includes("use_in_curl_bash") || item["use_in_curl_bash"] === false) {
			var cew = update_table_row(item, errors, warnings, command);
			command = cew[0];
			errors = cew[1];
			warnings = cew[2];
		} else {
			if(item["type"] == "select") {
				var val = $(`#${item["id"]}`).val();
				curl_options = ` --${item["id"]}=${val} `;
			} else if(item["type"] == "checkbox" && $(`#${item["id"]}`).is(":checked")) {
				curl_options = ` --${item["id"]} `;
			} else {
				error("use_in_curl_bash currently only supports select and checkbox");
			}
		}
	});

	hiddenTableData.forEach(function(item) {
		if(!Object.keys(item).includes("use_in_curl_bash") || item["use_in_curl_bash"] === false) {
			var cew = update_table_row(item, errors, warnings, command);
			command = cew[0];
			errors = cew[1];
			warnings = cew[2];
		} else {
			if(item["type"] == "select") {
				var val = $(`#${item["id"]}`).val();
				curl_options = ` --${item["id"]}=${val} `;
			} else if(item["type"] == "checkbox" && $(`#${item["id"]}`).is(":checked")) {
				curl_options = ` --${item["id"]} `;
			} else {
				error("use_in_curl_bash currently only supports select and checkbox");
			}
		}
	});

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
		} else if(parameterName.match(/^[a-zA-Z_]+$/)) {
			if (option === "range") {
				var $this = $(this);
				//log("$this.find('.minValue').val():", $this.find(".minValue").val());
				var minValue = parseFloat($this.find(".minValue").val());
				var maxValue = parseFloat($this.find(".maxValue").val());

				var numberType = $($(".parameterRow")[i]).find(".numberTypeSelect").val();

				if (minValue === maxValue) {
					warn_msg.push("Warning: The minimum and maximum values for parameter " + parameterName + " are equal.");
				}

				var is_ok = true;

				//log("minValue:", minValue);

				if(isNaN(minValue)) {
					warn_msg.push("<i>minValue</i> for parameter <i>" + parameterName + "</i> is not a number.");
					is_ok = false;
				}

				if(isNaN(maxValue)) {
					warn_msg.push("<i>maxValue</i> for parameter <i>" + parameterName + "</i> is not a number.");
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
					warn_msg.push("<i>Value</i> is missing.");
				} else if(!fixedValue.match(/./)) {
					warn_msg.push("<i>Value</i> is missing.");
				} else if(!fixedValue.match(/^[a-zA-Z0-9,_]+$/)) {
					warn_msg.push("Invalid values. Must match Regex /[a-zA-Z0-9,_]/.");

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
		} else if(parameterName && !parameterName.match(/^[a-zA-Z_]+$/)) {
			warn_msg.push("Name contains invalid characters. Must be all-letters.");
		} else if(is_invalid_parameter_name(parameterName)) {
			warn_msg.push("Name is or contains a reserved keyword.");
		} else {
			warn_msg.push("<i>Name</i> is missing.");
		}

		if(warn_msg.length) {
			$($(".parameterError")[i]).html(string_or_array_to_list(warn_msg)).show();
			//set_row_background_color_red_color($($(".parameterRow")[i]));
		} else {
			$($(".parameterError")[i]).html("").hide();
		}

		i++;
	});

	if (parameters.length > 0) {
		command += " --parameter " + parameters.join(" --parameter ");
	}

	if ($("#constraints").val()) {
		var _constraints = $("#constraints").val().split(";");
		_constraints = _constraints.filter(function(entry) { return entry.trim() != ""; }).map(function (el) {
			return el.trim();
		});
		for (var r = 0; r < _constraints.length; r++) {
			command += " --experiment_constraints '" + _constraints[r] + "'";
		}
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

		var ui_url = btoa(window.location.toString());
		command += " --ui_url " + ui_url;

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
		base_url = base_url.replace(/\/index.php/, "");
		base_url = base_url.replace(/\/gui.php/, "");

		var curl_command = "";

		if(curl_or_cat == "curl") {
			curl_command = `${curl_or_cat} ${base_url}install_omniax.sh 2>/dev/null | bash -s -- "${base_64_string}"${curl_options}`;
		} else {
			curl_command = `${curl_or_cat} ${base_url}install_omniax.sh | bash -s -- "${base_64_string}"${curl_options}`;
		}

		$("#command_element_highlighted").html(highlight_bash(command)).show().parent().show().parent().show();
		$("#curl_command_highlighted").html(highlight_bash(curl_command)).show().parent().show().parent().show();

		$("#command_element").text(command);
		$("#curl_command").text(curl_command);
	} else {
		$("#command_element_highlighted").html("").hide().parent().hide().parent().hide();
		$("#curl_command_highlighted").html("").hide().parent().hide().parent().hide();

		$("#command_element").text("");
		$("#curl_command").text("");
	}

	update_url();
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
			<table>
			    <tr>
				<td>Name:</td>
				<td><input onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='invert_in_dark_mode parameterName'></td>
			    </tr>
			    <tr>
				<td>Min:</td>
				<td><input onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='number' class='invert_in_dark_mode minValue'></td>
			    </tr>
			    <tr>
				<td>Max:</td>
				<td><input onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='number' class='invert_in_dark_mode maxValue'></td>
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
			<table>
			    <tr>
				<td>Name:</td>
				<td><input onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='invert_in_dark_mode parameterName'></td>
			    </tr>
			    <tr>
				<td>Values (comma separated):</td>
				<td><input onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='text' class='invert_in_dark_mode choiceValues'></td>
			    </tr>
			</table>
		    `;
	} else if (selectedOption === "fixed") {
		valueCell.innerHTML = `
			<table>
			    <tr>
				<td>Name:</td>
				<td><input onchange="update_command()" onkeyup="update_command()" onclick="update_command()" value="${paramName}" type='text' class='invert_in_dark_mode parameterName'></td>
			    </tr>
			    <tr>
				<td>Value:</td>
				<td><input onchange="update_command()" onkeyup="update_command()" onclick="update_command()" type='text' class='invert_in_dark_mode fixedValue'></td>
			    </tr>
			</table>
		    `;
	}

	valueCell.innerHTML += "<div style='display: none' class='error_element parameterError invert_in_dark_mode'></div>";

	update_command();

	apply_theme_based_on_system_preferences();
}

function addRow(button) {
	var table = document.getElementById("config_table");
	var rowIndex = button.parentNode.parentNode.rowIndex;
	var numberOfParams = $(".parameterRow").length;
	var newRow = table.insertRow(rowIndex + numberOfParams + 1);

	var optionCell = newRow.insertCell(0);
	var valueCell = newRow.insertCell(1);
	var buttonCell = newRow.insertCell(2);

	optionCell.innerHTML = "<select onchange='updateOptions(this)' class='optionSelect'><option value='range'>Range</option><option value='choice'>Choice</option><option value='fixed'>Fixed</option></select>";
	valueCell.innerHTML = "";

	buttonCell.innerHTML = "<button class='remove_parameter invert_in_dark_mode' onclick='removeRow(this)'>Remove</button>";

	updateOptions(optionCell.firstChild);

	newRow.classList.add("parameterRow");
	optionCell.firstChild.classList.add("optionSelect");
	//valueCell.firstChild.classList.add('valueInput');

	update_command();
}

function removeRow(button) {
	var table = document.getElementById("config_table");
	var rowIndex = button.parentNode.parentNode.rowIndex;
	var rowCount = table.rows.length;
	if (rowCount > 2) {
		table.deleteRow(rowIndex);
		update_command();
	}
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

		$(input).addClass("invert_in_dark_mode");

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
	} else {
		var input = $("<input>").attr({ id: item.id, type: item.type, value: item.value, placeholder: item.placeholder, min: item.min, max: item.max }).css("width", "100%");

		$(input).addClass("invert_in_dark_mode");

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

	tbody.append("<tr><td><button onclick='addRow(this)' class='add_parameter invert_in_dark_mode' id='main_add_row_button'>Add variable</button></td><td colspan='2'></td></tr>");

	var hidden_table = $("#hidden_config_table");
	var hidden_tbody = hidden_table.find("tbody");

	hiddenTableData.forEach(function(item) {
		create_table_row(hidden_table, hidden_tbody, item);
	});

	highlight_all_bash();

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
	tableData.forEach(function(item) {
		params.push(item.id + "=" + encodeURIComponent($("#" + item.id).val()));
	});

	hiddenTableData.forEach(function(item) {
		params.push(item.id + "=" + encodeURIComponent($("#" + item.id).val()));
	});

	var parameterIndex = 0;
	$(".parameterRow").each(function() {
		var option = $(this).find(".optionSelect").val();
		var parameterName = $(this).find(".parameterName").val();

		if(parameterName && !parameterName.match(/^\w+$/)) {
			//error(`Parameter name "${parameterName}" does have invalid characters. Must be all letters.`)
		} else if (parameterName) {
			if (option === "range") {
				var minValue = $(this).find(".minValue").val();
				var maxValue = $(this).find(".maxValue").val();
				var numberType = $(this).find(".numberTypeSelect").val();
				var log_scale = $(this).find(".log_scale").is(":checked") ? "true" : "false";

				params.push("parameter_" + parameterIndex + "_name=" + encodeURIComponent(parameterName));
				params.push("parameter_" + parameterIndex + "_type=" + encodeURIComponent(option));
				params.push("parameter_" + parameterIndex + "_min=" + encodeURIComponent(minValue));
				params.push("parameter_" + parameterIndex + "_max=" + encodeURIComponent(maxValue));
				params.push("parameter_" + parameterIndex + "_number_type=" + encodeURIComponent(numberType));
				params.push("parameter_" + parameterIndex + "_log_scale=" + encodeURIComponent(log_scale));
			} else if (option === "choice") {
				var choiceValues = $(this).find(".choiceValues").val();

				params.push("parameter_" + parameterIndex + "_name=" + encodeURIComponent(parameterName));
				params.push("parameter_" + parameterIndex + "_type=" + encodeURIComponent(option));
				params.push("parameter_" + parameterIndex + "_values=" + encodeURIComponent(choiceValues));
			} else if (option === "fixed") {
				var fixedValue = $(this).find(".fixedValue").val();

				params.push("parameter_" + parameterIndex + "_name=" + encodeURIComponent(parameterName));
				params.push("parameter_" + parameterIndex + "_type=" + encodeURIComponent(option));
				params.push("parameter_" + parameterIndex + "_value=" + encodeURIComponent(fixedValue));
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

function copy_to_clipboard(text) {
	var dummy = document.createElement("textarea");
	document.body.appendChild(dummy);
	dummy.value = text;
	dummy.select();
	document.execCommand("copy");
	document.body.removeChild(dummy);
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

async function start_gremlins () {
	(function() {
		function callback() {
			var horde = gremlins.createHorde({
				species: [
					gremlins.species.clicker(),
					gremlins.species.toucher(),
					gremlins.species.formFiller(),
					gremlins.species.scroller(),
					gremlins.species.typer()
				],
				mogwais: [
					gremlins.mogwais.alert(),
					gremlins.mogwais.gizmo()
				],
				strategies: [
					gremlins.strategies.distribution()
				]
			});

			horde.unleash();
		}
		var s = document.createElement("script");
		s.src = "https://unpkg.com/gremlins.js";
		if (s.addEventListener) {
			s.addEventListener("load", callback, false);
		} else if (s.readyState) {
			s.onreadystatechange = callback;
		}
		document.body.appendChild(s);
	})();
}
