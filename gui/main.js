var formdata = [];

function log (...args) {
	for (var i = 0; i < args.length; i++) {
		console.log(args[i]);
	}
}

var valid_grid_search = ['hp.randint', 'hp.quniform', 'hp.qloguniform', 'hp.qnormal', 'hp.qlognormal', 'hp.categorical', 'hp.choice', 'hp.choiceint', 'hp.choicestep'];
var number_of_parameters = 0;
var missing_values = 0;

function get_url_param (name) {
	var value = "";
	if(urlParams.has(name)) {
		value = urlParams.get(name);
	}

	var encodedStr = value.replace(/[\u00A0-\u9999<>\&]/gim, function(i) {
		return '&#' + i.charCodeAt(0) + ';';
	});
	return encodedStr;
}

function onlyUnique(value, index, self) {
	return self.indexOf(value) === index;
}

function get_number_of_parameters () {
	return parseInt($("#number_of_parameters").val());
}

function show_error_message (divid, msg) {
	$(divid).show().html('').html(get_error_string(msg));
}

function get_error_string (msg) {
	return '<div class="ui-state-error ui-corner-all" style="padding: 0 .7em;"><p><span class="ui-icon ui-icon-alert" style="float: left;"></span><strong>Warning:</strong> ' + msg + '</p></div>';
}

function no_value_str (i, this_parameter_name_string, value_type) {
	missing_values++;
	return "# !!! Parameter " + this_parameter_name_string + " has no" + (value_type ? " " + value_type : "") + " value\n";
}

function copy_to_clipboard(text) {
	var dummy = document.createElement("textarea");
	document.body.appendChild(dummy);
	dummy.value = text;
	dummy.select();
	document.execCommand("copy");
	document.body.removeChild(dummy);
}

function copy_bashcommand_to_clipboard () {
	var serialized = $("#bashcommand").text();
	copy_to_clipboard(serialized);

	$('#copied').show();
	setTimeout(function() { 
		$('#copied').fadeOut(); 
	}, 5000);
}

function show_time_warning () {
	var partition = $("#partition").val();
	var runtime = $("#computing_time").val();
	for (var this_partition in partition_data) {
		if (partition == this_partition) {
			var max_runtime = partition_data[partition]["computation_time"];
			if(runtime > max_runtime) {
				$("#computing_time").val(max_runtime);
				$("#timewarning").html("<center><i style='color: red'>On " + partition + ", jobs can only run for " + max_runtime + " hours</i></center>");
			} else {
				$("#timewarning").html("");
			}
		}
	}
}

function show_worker_warning () {
	var max_evals = parseInt($("#max_evals").val());
	var partition = $("#partition").val();
	var number_of_workers = parseInt($("#number_of_workers").val());

	if(!isNaN(number_of_workers)) {
		$("#emptyworkerwarning").hide();
		$("#emptyworkerwarning").html("");
		if(max_evals < number_of_workers) {
			$("#workerevalswarning").html("<center><i style='color: orange'>It is probably not useful to have more workers than max evals.</i></center>");
			missing_values++;
		} else {
			$("#workerevalswarning").html("");
		}
	} else {
		show_error_message("#emptyworkerwarning", 'Number of workers is empty or not a valid number.');
		missing_values++;
	}
}

function show_max_worker_warning () {
	var partition = $("#partition").val();
	var number_of_workers = parseInt($("#number_of_workers").val());

	for (var this_partition in partition_data) {
		if (partition == this_partition) {
			var this_max_number_of_workers = partition_data[this_partition]["number_of_workers"];
			if (number_of_workers > this_max_number_of_workers) {
				$("#maxworkerwarning").html("<center><i style='color: orange'>From tests it has been found that no more than " + this_max_number_of_workers + " workers can be launched on the " + this_partition + " partition. Expect this command to fail or lower the number of workers.</i></center>");
			} else {
				$("#maxworkerwarning").html("");
			}
		}
	}
}

function show_link_if_available () {
	var partition = $("#partition").val();
	if(partition_data[partition]["link"] != "") {
		$("#link").html("<a target='_blank' href='" + partition_data[partition]["link"] + "'>Link to the HPC Compendium about this hardware</a>");
	} else {
		$("#link").html("");
	}
	update_config();
}

function show_warning_if_available () {
	var partition = $("#partition").val();
	if(partition_data[partition]["warning"] != "") {
		show_error_message("#partitionwarning", partition_data[partition]["warning"]);
	} else {
		$("#partitionwarning").html("");
	}
	update_config();
}

function use_max_memory_of_partition () {
	var partition = $("#partition").val();
	if(partition_data[partition]["max_mem_per_core"] != "") {
		$("#mem_per_cpu").val(partition_data[partition]["max_mem_per_core"]);
	} else {
		$("#mem_per_cpu").val("2000");
	}
	update_config();
}

function disable_gpu_when_none_available () {
	var partition = $("#partition").val();
	if(partition_data[partition]["max_number_of_gpus"] == 0) {
		$("#enable_gpus").prop("checked", false)
		$("#enable_gpus").attr("disabled", true);
		show_error_message("#gputext", 'No GPUs on ' + partition + '.');
	} else {
		$("#enable_gpus").prop("checked", true)
		$("#enable_gpus").removeAttr("disabled");
		$("#gputext").html("");
	}
	update_config();
}

function max_memory_per_worker () {
	var partition = $("#partition").val();
	var mem_per_cpu = $("#mem_per_cpu").val();

	for (var this_partition in partition_data) {
		if (partition == this_partition) {
			var this_max_mem_per_cpu = partition_data[this_partition]["mem_per_cpu"];
			if(mem_per_cpu > this_max_mem_per_cpu) {
				$("#mem_per_cpu").val(this_max_mem_per_cpu);
				$("#maxmemperworkertext").html("<center><i style='color: red'>On " + partition + ", there can be a maximum of " + this_max_mem_per_cpu + " MB of RAM per Worker</i></center>");
			} else {
				$("#maxmemperworkertext").html("");
			}
		}
	}
}

function save_form_data () {
	$('#all').find(':input').each(function(){
		formdata[$(this).attr("id")] = $(this).val();
	});
}

function restore_form_data () {
	for (var i in formdata) {
		$("#" + i).val(formdata[i]);
		$("#" + i).change();
	}

	formdata = [];
}

function isNumericList(str) {
	if(/^(\d+(\.\d+)?,?)+$/.test(str)) {
		return 1;
	}
	return 0;
}

function isNumeric(str) {
	if(str === 0) {
		return true;
	}

	if(typeof str == "number") {
		return str;
	}
	if (typeof str != "string") return false;
	return !isNaN(str) && !isNaN(parseFloat(str));
}

function show_missing_values_error () {
	if(missing_values == 0) {
		$("#errors").html("");
		$("#hidewhenmissingdata").show();
	} else {
		show_error_message("#errors", 'Missing data for ' + missing_values + ' value(s)! Check the comments in the config file to find what is missing.');
		$("#hidewhenmissingdata").hide();
		$("#hidewhendone").show();
	}
}

function replace_varnames_with_numbers (objective_program) {
	var param_name_items = $('input[id^=parameter_][id$=_name]');

	for (var i = 0; i < param_name_items.length; i++) {
		var name = $(param_name_items[i]).val();

		var search = "($" + name + ")";
		var replace_with = "($x_" + i + ")";

		objective_program = objective_program.replaceAll(search, replace_with);
	}

	
	return objective_program;
}

function update_config () {
	missing_values = 0;
	var config_string = "[DATA]\n";

	var seed = $("#seed").val();
	if(seed) {
		config_string += "seed = " + seed + "\n";
	}

	var precision = $("#precision").val();
	var partition = $("#partition").val();
	var account = $("#account").val();
	var reservation = $("#reservation").val();
	var max_evals = $("#max_evals").val();
	var objective_program = $("#objective_program").val() ;
	var projectname = $("#projectname").val();
	var number_of_workers = $("#number_of_workers").val();
	var enable_gpus = $("#enable_gpus").is(":checked") ? 1 : 0;
	var mem_per_cpu = $("#mem_per_cpu").val();
	var computing_time = $("#computing_time").val();
	var num_gpus_per_worker = $("#number_of_gpus").val();

	if(isNumeric(number_of_workers)) {
		config_string += "number_of_workers = " + number_of_workers + "\n";
	}

	if(isNumeric(num_gpus_per_worker)) {
		config_string += "num_gpus_per_worker = " + num_gpus_per_worker + "\n";
	}

	if(isNumeric(precision)) {
		config_string += "precision = " + precision + "\n";
	} else {
		config_string += "# !!! Precision-parameter is empty or not a valid number\n";
		missing_values++;
	}

	if(account) {
		config_string += "account = " + account + "\n";
	}

	if(reservation) {
		config_string += "reservation = " + reservation + "\n";
	}

	if(partition) {
		config_string += "partition = " + partition + "\n";
	}

	if(projectname) {
		config_string += "projectname = " + projectname + "\n";
	}

	config_string += "enable_gpus = " + (enable_gpus ? 1 : 0) + "\n";
	config_string += "mem_per_cpu = " + mem_per_cpu + "\n";
	config_string += "computing_time = " + computing_time + "\n";

	if(!projectname.match(projectname_regex)) {
		projectname = "";
		missing_values++;
	} else if(!projectname.length >= 1) {
		missing_values++;
		show_error_message("#noprojectnameerror", 'Missing or invalid project name.');
	} else {
		$("#noprojectnameerror").html('');
		$("#invalidprojectnameerror").html('');
	}


	if(isNumeric(max_evals)) {
		config_string += "max_evals = " + max_evals + "\n";
		$("#maxevalserror").html('');
		$("#maxevalserror").hide();
	} else {
		config_string += "# !!! Max evals is empty or not a valid number\n";
		show_error_message("#maxevalserror", 'max evals is empty or not a valid number.');
		missing_values++;
	}

	config_string += "algo_name = " + $("#algo_name").val() + "\n";

	config_string += "range_generator_name = hp.randint\n";

	var num_of_newlines_in_objective_program =  objective_program.split(/\r\n|\r|\n/).length;

	check_param_names();

	if(objective_program.length >= 1 && num_of_newlines_in_objective_program <= 1) {
		$("#objectiveprogramerror").hide();
		$("#objectiveprogramerror").html('');
		config_string += "objective_program = " + replace_varnames_with_numbers(objective_program) + "\n";
		for (var i = 0; i < number_of_parameters; i++) {
			var param_name = $("#parameter_" + i + "_name").val()
			var regex = new RegExp("\\(\\$(?:x_" + i + "|" + param_name + ")\\)");
			if(!objective_program.match(regex)) {
				var parameter_name = "($x_" + i + ")";
				if($("#parameter_" + i + "_name").val()) {
					parameter_name = "($" + param_name + ")";
				}
				config_string += "# !!! Missing parameter " + parameter_name + " in Objective program string" + "\n";
				//missing_values++;
			}
		}
	} else if (objective_program.length && num_of_newlines_in_objective_program != 0) {
		config_string += "# !!! Objective Program string contains newlines\n";
		missing_values++;
		show_error_message("#objectiveprogramerror", 'Objective program string cannot contain newlines.');
	} else {
		config_string += "# !!! Objective Program string is empty\n";
		missing_values++;
		show_error_message("#objectiveprogramerror", 'Objective program string cannot be empty.');
	}

	if(number_of_parameters >= 1) {
		$("#toofewparameterserror").html('');
		config_string += "\n[DIMENSIONS]\n";

		config_string += "dimensions = " + number_of_parameters + "\n\n";

		for (var i = 0; i < number_of_parameters; i++) {
			var this_type = $("#type_" + i).val();
			var this_parameter_name = $("#parameter_" + i + "_name").val() ;

			var this_parameter_name_string = "($x_" + i + ")";
			if(this_parameter_name) {
				this_parameter_name_string = "($" + this_parameter_name + ")";
			}

			if(this_parameter_name) {
				config_string += "dim_" + i + "_name = " + this_parameter_name + "\n";
			} else {
				config_string += "# !!! No parameter name set for parameter number " + this_parameter_name_string + "\n";
				missing_values++;
			}

			var range_generator_name_type = this_type;
			if(this_type == "hp.choiceint") {
				range_generator_name_type = "hp.choice";
			}

			config_string += "range_generator_" + i + " = " + range_generator_name_type + "\n";

			if(this_type == "hp.choice") {
				var this_val = $("#hp_choice_" + i).val();

				if(typeof(this_val) !== "undefined" && this_val.length >= 1) {
					if(isNumericList(this_val) || 1) {
						var re = new RegExp(',{2,}', 'g');
						var respace = new RegExp('\\s*,\\s*', 'g');
						this_val = this_val.replace(re, ',');
						this_val = this_val.replace(respace, ',');
						this_val = this_val.replace(/,+$/, '');

						config_string += "options_" + i + " = " + this_val + "\n";
					} else {
						config_string += "# Parameter " + i + " must consist of numbers only";
						missing_values++;
					}
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "");
				}
			} else if(this_type == "hp.pchoice") {
				var this_val = $("#hp_pchoice_" + i).val();

				if(typeof(this_val) !== "undefined" && this_val.length >= 1) {
					var re = new RegExp(',{2,}', 'g');
					var respace = new RegExp('\\s*,\\s*', 'g');
					this_val = this_val.replace(re, ',');
					this_val = this_val.replace(respace, ',');
					this_val = this_val.replace(/,+$/, '');

					config_string += "options_" + i + " = " + this_val + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "");
				}

			} else if(this_type == "hp.choicestep") {
				var this_min = parseInt($("#choiceint_" + i + "_min").val());
				var this_max = parseInt($("#choiceint_" + i + "_max").val());
				var this_step = parseInt($("#choiceint_" + i + "_step").val());

				if(parseInt(this_min) > parseInt(this_max)) {
					var tmp = this_min;
					this_min = this_max;
					this_max = tmp;
				}

				if(isNumeric(this_max) && isNumeric(this_min) && isNumeric(this_step)) {
					if(parseInt(this_min) <= parseInt(this_max)) {
						var numberArray = [];
						for (var j = parseInt(this_min); j <= parseInt(this_max); j++) {
							    numberArray.push(j);
						}

						config_string += "min_dim_" + i + " = " + this_min + "\n";
						config_string += "max_dim_" + i + " = " + this_max + "\n";
						config_string += "step_dim_" + i + " = " + this_step + "\n";
					} else {
						config_string += "# !!! Parameter " + this_parameter_name_string + " min-value is not smaller or equal to max value\n";
						missing_values++;
					}
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "");
				}
			} else if(this_type == "hp.choiceint") {
				var this_min = parseFloat($("#choiceint_" + i + "_min").val());
				var this_max = parseFloat($("#choiceint_" + i + "_max").val());

				if(parseInt(this_min) > parseInt(this_max)) {
					var tmp = this_min;
					this_min = this_max;
					this_max = tmp;
				}

				log("min/max:", this_min, this_max);

				if(isNumeric(this_max) && isNumeric(this_min)) {
					if(parseInt(this_min) <= parseInt(this_max)) {
						var numberArray = [];
						for (var j = parseInt(this_min); j <= parseInt(this_max); j++) {
							    numberArray.push(j);
						}
						var pseudostring = numberArray.join(",");

						config_string += "options_" + i + " = " + pseudostring + "\n";
					} else {
						config_string += "# !!! Parameter " + this_parameter_name_string + " min-value is not smaller or equal to max value\n";
						missing_values++;
					}
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "");
				}
			} else if(this_type == "hp.randint") {
				var this_max =  $("#randint_" + i + "_max").val();
				if(isNumeric(this_max) && this_max >= 0) {
					if(parseInt(this_max) == this_max) {
						config_string += "max_dim_" + i + " = " + this_max + "\n";
					} else {
						config_string += no_value_str(i, this_parameter_name_string, "integer");
					}
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "");
				}
			} else if(this_type == "hp.quniform") {
				var this_min = $("#min_" + i).val();
				var this_max = $("#max_" + i).val();
				var this_q = $("#q_" + i).val() ;

				if(parseInt(this_min) > parseInt(this_max)) {
					var tmp = this_min;
					this_min = this_max;
					this_max = tmp;
				}

				if(isNumeric(this_min)) {
					config_string += "min_dim_" + i + " = " + this_min + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "low");
				}

				if(isNumeric(this_max)) {
					config_string += "max_dim_" + i + " = " + this_max + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "high");
				}

				if(isNumeric(this_q)) {
					config_string += "q_" + i + " = " + this_q + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "q");
				}
			} else if(this_type == "hp.loguniform" || this_type == "hp.uniform" || this_type == "hp.uniformint") {
				var this_min = $("#min_" + i).val();
				var this_max = $("#max_" + i).val();

				if(parseInt(this_min) > parseInt(this_max)) {
					var tmp = this_min;
					this_min = this_max;
					this_max = tmp;
				}

				if(isNumeric(this_min)) {
					config_string += "min_dim_" + i + " = " + this_min + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "low");
				}

				if(isNumeric(this_max)) {
					config_string += "max_dim_" + i + " = " + this_max + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "high");
				}
			} else if(this_type == "hp.normal") {
				var mu = $("#mu_" + i).val();
				var sigma = $("#sigma_" + i).val();

				if(isNumeric(mu)) {
					config_string += "mu_" + i + " = " + mu + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "Mean &mu;");
				}

				if(isNumeric(sigma)) {
					config_string += "sigma_" + i + " = " + sigma + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "standard deviation &sigma;");
				}
			} else if(this_type == "hp.qloguniform") {
				var this_min = $("#min_" + i).val();
				var this_max = $("#max_" + i).val();
				var this_q = $("#q_" + i).val();

				if(parseInt(this_min) > parseInt(this_max)) {
					var tmp = this_min;
					this_min = this_max;
					this_max = tmp;
				}

				if(isNumeric(this_min)) {
					config_string += "min_dim_" + i + " = " + this_min + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "low");
				}

				if(isNumeric(this_max)) {
					config_string += "max_dim_" + i + " = " + this_max + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "high");
				}

				if(isNumeric(this_q)) {
					config_string += "q_" + i + " = " + this_q + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "q");
				}
			} else if(this_type == "hp.qnormal") {
				var mu = $("#mu_" + i).val();
				var sigma = $("#sigma_" + i).val();
				var q = $("#q_" + i).val();

				if(isNumeric(mu)) {
					config_string += "mu_" + i + " = " + mu + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "Mean &mu;");
				}

				if(isNumeric(sigma)) {
					config_string += "sigma_" + i + " = " + sigma + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "Standard deviation &sigma;");
				}

				if(isNumeric(q)) {
					config_string += "q_" + i + " = " + q + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "q");
				}
			} else if(this_type == "hp.qlognormal") {
				var mu = $("#mu_" + i).val();
				var sigma = $("#sigma_" + i).val();
				var q = $("#q_" + i).val();

				if(isNumeric(mu)) {
					config_string += "mu_" + i + " = " + mu + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "Mean &mu;");
				}

				if(isNumeric(sigma)) {
					config_string += "sigma_" + i + " = " + sigma + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "Standard deviation &sigma;");
				}

				if(isNumeric(q)) {
					config_string += "q_" + i + " = " + q + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "q");
				}
			} else if(this_type == "hp.lognormal") {
				var mu = $("#mu_" + i).val();
				var sigma = $("#sigma_" + i).val();

				if(isNumeric(mu)) {
					config_string += "mu_" + i + " = " + mu + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "Mean &mu;");
				}

				if(isNumeric(sigma)) {
					config_string += "sigma_" + i + " = " + sigma + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "Standard deviation &sigma");
				}
			}

			if(i != number_of_parameters - 1) {
				config_string += "\n";
			}
		}
	} else {
		missing_values++;
		show_error_message("#toofewparameterserror", 'You need at least one CLI parameter.');
	}

	config_string += "\n";

	var enable_debug = $("#enable_debug").is(":checked") ? 1 : 0;
	var show_live_output = $("#show_live_output").is(":checked") ? 1 : 0;
	var sbatch_or_srun = $("#sbatch_or_srun").val();
	var debug_sbatch_srun = $("#debug_sbatch_srun").is(":checked") ? 1 : 0;

	config_string += "[DEBUG]\n";
	config_string += "debug_xtreme = " + enable_debug + "\n";
	config_string += "debug = " + enable_debug + "\n";
	config_string += "info = " + enable_debug + "\n";
	config_string += "warning = " + enable_debug + "\n";
	config_string += "success = " + enable_debug + "\n";
	config_string += "stack = " + enable_debug + "\n";
	config_string += "show_live_output = " + show_live_output + "\n";
	config_string += "sbatch_or_srun = " + sbatch_or_srun + "\n";
	config_string += "debug_sbatch_srun = " + debug_sbatch_srun + "\n";

	config_string += "\n";
	config_string += "[MONGODB]\n";

	var worker_last_job_timeout = $("#worker_last_job_timeout").val();
	if(!isNumeric(worker_last_job_timeout)) {
		config_string += "# worker_last_job_timeout not set properly!\n";
		missing_values++;
	} else {
		config_string += "worker_last_job_timeout = " + worker_last_job_timeout + "\n";
	}

	var poll_interval = $("#poll_interval").val();
	if(!isNumeric(worker_last_job_timeout)) {
		config_string += "# poll_interval not set properly!\n";
		missing_values++;
	} else {
		config_string += "poll_interval = " + poll_interval + "\n";
	}


	var kill_after_n_no_results = $("#kill_after_n_no_results").val();
	if(!isNumeric(worker_last_job_timeout)) {
		config_string += "# kill_after_n_no_results not set properly!\n";
		missing_values++;
	} else {
		config_string += "kill_after_n_no_results = " + kill_after_n_no_results + "\n";
	}

	$("#config").html('<code class="language-ini">' + config_string + '</code>');
	Prism.highlightAll();

	show_missing_values_error();

	var projectname = $("#projectname").val();
	var projectname_regex = new RegExp("^[a-zA-Z0-9_]+$");

	var number_of_workers = $("#number_of_workers").val();
	var computing_time = $("#computing_time").val();
	var mem_per_cpu = $("#mem_per_cpu").val();
	var partition = $("#partition").val();

	if(partition == "ml") {
		show_error_message("#seed_ml_error", 'Warning: Setting the seed does not work on ML.');
	} else {
		$("#seed_ml_error").html("").hide();
	}

	if(mem_per_cpu > 0) {
		$("#memworkererror").hide();
		$("#memworkererror").html("");
	} else {
		show_error_message("#memworkererror", 'Memory per worker is empty or not a valid number.');
	}

	var singlelogsfolder_set = 0;
	if(projectname.length >= 1) {
		if(projectname.match(projectname_regex)) {
			$("#singlelogsfolder").html("cd projects/" + projectname + "/singlelogs");
			singlelogsfolder_set = 1;
		}
	}

	if(!singlelogsfolder_set) {
		$("#singlelogsfolder").html("cd projects/$PROJECTNAME/singlelogs");
	}

	if(computing_time > 0 || computing_time.match(/^\d+(?::\d+)*$/)) {
		$("#computingtimeerror").hide();
		$("#computingtimeerror").html("");
		if(mem_per_cpu > 0) {
			$("#memworkererror").hide();
			$("#memworkererror").html("");
			if(objective_program) {
				if(number_of_workers > 0) {
					if(projectname.length >= 1) {
						if(projectname.match(projectname_regex)) {
							$("#singlelogsfolder").html("cd projects/" + projectname + "/singlelogs");
							var config_string_echo = config_string;
							config_string_echo = config_string_echo.replace(/\$/g, "\\$");

							var debug_sbatch_srun = $("#debug_sbatch_srun").is(":checked") ? 1 : 0;
							var trace_omniopt = $("#trace_omniopt").is(":checked") ? 1 : 0;
							var sbatch_or_srun = $("#sbatch_or_srun").val();
							var output_path_var = "";

							if($("#enable_slurmlog_out").is(":checked")) {
								output_path_var = ` -o projects/${projectname}/slurmlogs/%a.out`;
							}
							var sbatch_string = sbatch_or_srun + output_path_var + (debug_sbatch_srun ? ' -vvvvvvvvvvv ' : '') + " -J "  + projectname + " \\\n";
							var enable_gpus = $("#enable_gpus").is(":checked") ? 1 : 0;

							var number_of_gpus = $("#number_of_gpus").val();

							var reservation_string = " --reservation=" + $("#reservation").val() + " \\\n";;
							var add_reservation_string_to_sbatch = 0;
							var add_account_string_to_sbatch = 0;
							var cpus_per_task = parseInt($("#cpus_per_task").val());

							if(enable_gpus == 1) {
								if($("#reservation").val()) {
									if(number_of_gpus <= 1) {
										sbatch_string += reservation_string;
									} else {
										add_reservation_string_to_sbatch = 1;
									}
								}
							}

							if($("#account").val()) {
								if(number_of_gpus == 1) {
									sbatch_string += " -A " + $("#account").val() + " \\\n";
								} else {
									add_account_string_to_sbatch = 1;
								}
							}

							var max_number_of_gpus = partition_data[partition]["max_number_of_gpus"];
							var max_mem_per_core = partition_data[partition]["max_mem_per_core"];

							var number_of_allocated_gpus = number_of_workers;
							var number_of_cpus_per_worker = $("#number_of_cpus_per_worker").val();

							if(number_of_allocated_gpus > max_number_of_gpus) {
								number_of_allocated_gpus = max_number_of_gpus;
							}

							var has_cpus_per_task = false;

							if(enable_gpus == 1) {
								if(number_of_gpus == 1) {
									sbatch_string += " --cpus-per-task=" + number_of_cpus_per_worker + " \\\n"
									sbatch_string += " --gres=gpu:" + number_of_allocated_gpus + " \\\n --gpus-per-task=1 \\\n";
									has_cpus_per_task = true;
								}
							}

							var fmin_partition = partition;

							if(enable_gpus == 1 && number_of_gpus >= 2) {
								fmin_partition = "alpha";
							}

							sbatch_string += " --ntasks=" + number_of_workers + " \\\n";
							if(!computing_time.includes(":")) {
								sbatch_string += " --time=" + computing_time + ":00:00 \\\n";
							} else {
								sbatch_string += " --time=" + computing_time + " \\\n";
							}
							sbatch_string += " --mem-per-cpu=" + mem_per_cpu + " \\\n";
							sbatch_string += " --partition=" + fmin_partition + " \\\n";


							var sbatch_pl_params = "";

							if($("#number_of_cpus_per_worker").val()) {
								if(parseInt($("#number_of_gpus").val()) >= 1 && $("#enable_gpus").is(":checked")) {
									if(has_cpus_per_task) {
										sbatch_pl_params += "\\\n --srun_cpus_per_task=" + $("#number_of_cpus_per_worker").val();
									}
								} else {
									sbatch_pl_params += "\\\n --srun_cpus_per_task=" + $("#number_of_cpus_per_worker").val();
								}
							} else if(cpus_per_task) {
								sbatch_pl_params += " --srun_cpus_per_task=" + cpus_per_task + " \\\n";
							}

							if($("#overcommit").is(":checked")) {
								sbatch_pl_params += " --overcommit \\\n";
							}

							if($("#overlap").is(":checked")) {
								sbatch_pl_params += " --overlap \\\n";
							}

							if(partition == "barnard") {
								sbatch_string += " run.sh --project=" + projectname + " \\\n";
							} else {
								sbatch_string += " sbatch.pl --project=" + projectname + " \\\n";
							}

							if(enable_debug == 1) {
								sbatch_string += " --debug";
							}

							if(trace_omniopt) {
								sbatch_string += " --trace_omniopt";
							}

							if(add_reservation_string_to_sbatch) {
								sbatch_string += reservation_string;
							}

							if(add_account_string_to_sbatch) {
								sbatch_string += " --account=" + $("#account").val() + " \\\n"
							}

							sbatch_string += sbatch_pl_params;

							sbatch_string += " --partition=" + partition;

							if(enable_gpus == 1) {
								sbatch_string += "\\\n" + " --num_gpus_per_worker=" + $("#number_of_gpus").val() + " ";
								if(!computing_time.includes(":")) {
									sbatch_string += "\\\n" + " --max_time_per_worker=" + computing_time + ":00:00";
								} else {
									sbatch_string += "\\\n" + " --max_time_per_worker=" + computing_time;
								}
							} else {
								sbatch_string += "\\\n" + " --num_gpus_per_worker=0 ";
							}

							sbatch_string = sbatch_string.replace(/ {2,}/g, ' ');
							var sbatch_string_no_newlines = sbatch_string.replace(/\\+\n/g, ' ');

							//var base_url = document.URL.substr(0,document.URL.lastIndexOf('/'));
							var base_url = "https://imageseg.scads.ai/omnioptgui";
							var bash_command = "curl " + base_url + "/omniopt_script.sh 2>/dev/null | bash -s -- --projectname=" + projectname + " --config_file=" + btoa(config_string) + " --sbatch_command=" + btoa(sbatch_string_no_newlines);

							var workdir = $("#workdir").val();
							if(typeof workdir !== "undefined" && workdir != "") {
								bash_command = bash_command + " --workdir=" + workdir + " ";
							}

							if ($("#enable_curl_debug").is(":checked")) {
								bash_command = bash_command + " --debug ";
							}

							if ($("#dont_ask_to_start").is(":checked")) {
								bash_command = bash_command + " --dont_start_job ";
							}

							if (!$("#dont_add_to_shell_history").is(":checked")) {
								bash_command = bash_command + '; if [[ "$SHELL" == "/bin/bash" ]]; then history -r 2>&1 >/dev/null; elif [[ "$SHELL" == "/bin/zsh" ]]; then fc -R; fi';
							} else {
								bash_command = bash_command + " --dont_add_to_shell_history ";
							}

							$("#sbatch").html('<code class="language-bash">' + sbatch_string + '</code>');
							if(missing_values == 0) {
								$("#bashcommand").html('<code class="language-bash">' + bash_command + '</code>');
								$("#copytoclipboard").show();
								if($("#autohide_config_and_sbatch").is(":checked")) {
									$('#hidewhendone').hide();
								}
							}
						} else {
							$("#downloadbashlink").html('');
							$("#sbatch").html('<code class="language-bash"># Invalid project name given (only a-z, A-Z, 0-9 and _)</code>');
							$("#copytoclipboard").hide();
						}
					} else {
						$("#downloadbashlink").html('');
						$("#sbatch").html('<code class="language-bash"># No valid project name given</code>');
						$("#copytoclipboard").hide();
					}
				} else {
					$("#downloadbashlink").html('');
					$("#sbatch").html('<code class="language-bash"># You need at least 1 worker or more</code>');
					$("#copytoclipboard").hide();
				}
			} else {
				$("#downloadbashlink").html('');
				$("#sbatch").html('<code class="language-bash"># An objective_program string needs to be set</code>');
				$("#copytoclipboard").hide();
			}
		}
	} else {
		missing_values++;
		$("#downloadbashlink").html('');
		$("#sbatch").html('<code class="language-bash"># Computing time and memory per CPU must be higher than 0</code>');
		$("#copytoclipboard").hide();

		show_error_message("#computingtimeerror", 'Computing time is empty or not a valid number.');
	}

	$("#downloadlink").html('<a download="config.ini" href="data:application/octet-stream,' + encodeURIComponent(config_string) + '">Download this <tt>config.ini</tt>.</a>');

	Prism.highlightAll();

	show_multigpu();
}

function change_parameter_and_url(i, t) {
	if(typeof(t) == "string") {
		update_url_param("param_" + i + "_type", t);
		change_parameter_settings(i);
	} else {
		update_url_param("param_" + i + "_type", t.value);
		change_parameter_settings(i);
	}
}

function get_select_type (i) {
	var spaces = "&nbsp;&nbsp;&nbsp;&nbsp;";
	return "<input class='parameter_input' type='text' onkeyup='update_url_param(\"param_" + i + "_name\", this.value)' placeholder='Name of this parameter' id='parameter_" + i + "_name' value='" + get_url_param("param_" + i + "_name") + "' /><br /><span id='param_" + i + "_duplicate_name_error' class='name_error' style='display: none'></span><span id='param_" + i + "_invalid_name_error' class='name_error' style='display: none'></span>" +
		"<select class='parameter_input' id='type_" + i + "' onchange='change_parameter_and_url(" + i + ", this)'>" +
		"\t<option value='hp.randint' " + (get_url_param("param_" + i + "_type") == "hp.randint" ? ' selected="true" ' : '') + ">randint: Integer between 0 and a given maximal number</option>" +
		"\t<option value='' disabled='disabled'>&mdash;</option>\n" +
		"\t<option value='hp.choice' " + (get_url_param("param_" + i + "_type") == "hp.choice" ? ' selected="true" ' : '') + ">choice: One of the given options, separated by comma</option>" +
		"\t<option value='hp.pchoice' " + (get_url_param("param_" + i + "_type") == "hp.pchoice" ? ' selected="true" ' : '') + ">pchoice: One of the given options with a given probability, separated by comma</option>" +
		"\t<option value='hp.choiceint' " + (get_url_param("param_" + i + "_type") == "hp.choiceint" ? ' selected="true" ' : '') + ">choiceint: A pseudo-generator for hp.choice, for using integers between a and b</option>" +
		"\t<option value='hp.choicestep' " + (get_url_param("param_" + i + "_type") == "hp.choicestep" ? ' selected="true" ' : '') + ">choicestep: A pseudo-generator for hp.choice, for using integers between a and b with a certain step size</option>" +
		"\t<option value='' disabled='disabled'>&mdash;</option>\n" +
		"\t<option value='hp.uniform' " + (get_url_param("param_" + i + "_type") == "hp.uniform" ? ' selected="true" ' : '') + ">uniform: Uniformly distributed value between two values</option>" +
		"\t<option value='hp.uniformint' " + (get_url_param("param_" + i + "_type") == "hp.uniformint" ? ' selected="true" ' : '') + ">uniformint: Uniformly distributed integer value between two values</option>" +
		"\t<option value='hp.quniform' " + (get_url_param("param_" + i + "_type") == "hp.quniform" ? ' selected="true" ' : '') + "'>quniform: Values like round(uniform(min, max)/q)&#8901;q</option>" +
		"\t<option value='hp.loguniform' " + (get_url_param("param_" + i + "_type") == "hp.loguniform" ? ' selected="true" ' : '') + "'>loguniform: Values so that the log of the value is uniformly distributed</option>" +
		"\t<option value='hp.qloguniform' " + (get_url_param("param_" + i + "_type") == "hp.qloguniform" ? ' selected="true" ' : '') + "'>qloguniform: Values like round(exp(uniform(min, max))/q) &#8901;q</option>" +
		"\t<option value='' disabled='disabled'>&mdash;</option>\n" +
		"\t<option value='hp.normal' " + (get_url_param("param_" + i + "_type") == "hp.normal" ? ' selected="true" ' : '') + "'>normal: Real values that are normally distributed with Mean &mu; and Standard deviation &sigma;</option>" +
		"\t<option value='hp.qnormal' " + (get_url_param("param_" + i + "_type") == "hp.qnormal" ? ' selected="true" ' : '') + "'>qnormal: Values like round(normal(&mu;, &sigma;)/q)&#8901;q</option>" +
		"\t<option value='hp.lognormal' " + (get_url_param("param_" + i + "_type") == "hp.lognormal" ? ' selected="true" ' : '') + ">lognormal: Values so that the logarithm of normal(&mu;, &sigma;) is normally distributed</option>" +
		"\t<option value='hp.qlognormal' " + (get_url_param("param_" + i + "_type") == "hp.qlognormal" ? ' selected="true" ' : '') + ">qlognormal: Values like round(exp(lognormal(&mu;, &sigma;))/q)&#8901;q</option>" +
		"</select>" +
		"<div id='type_" + i + "_settings'></div>";
}

function change_parameter_settings(i) {
	var chosen_type = $("#type_" + i).val();
	var set_html = '<div style="background-color: red">Unknown parameter &raquo;' + chosen_type + "&laquo;</div>";

	if (chosen_type == "hp.choice") {
		set_html = "<input class='parameter_input' id='hp_choice_" + i + "' type='text' placeholder='Comma separated list of possible values' value='" + get_url_param("param_" + i + "_values") + "' onkeyup='update_url_param(\"param_" + i + "_values\", this.value)' a/>";
	} else if (chosen_type == "hp.randint") {
		set_html = "<input class='parameter_input' id='randint_" + i + "_max' type='number' placeholder='Maximal number' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' max='9223372036854775807' />";
	} else if (chosen_type == "hp.pchoice") {
		set_html = "<p>Example: <tt>a=20,b=70,c=10</tt>: <tt>a</tt> gets chosen with 20% probability, <tt>b</tt> with 70% and <tt>c</tt> with 10%. All percentages should add up to 100%.</p>"
		set_html += "<input class='parameter_input' id='hp_pchoice_" + i + "' type='text' placeholder='Comma separated list of possible values, with VALUE=PROBABILITY (probability between 0 and 100)' value='" + get_url_param("param_" + i + "_values") + "' onkeyup='update_url_param(\"param_" + i + "_values\", this.value)' a/>";
	} else if (chosen_type == "hp.choicestep") {
		set_html = "<input class='parameter_input' id='choiceint_" + i + "_min' type='number' placeholder='Minimal number' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='choiceint_" + i + "_max' type='number' placeholder='Maximal number' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='choiceint_" + i + "_step' type='number' placeholder='Step' value='" + get_url_param("param_" + i + "_value_step") + "' onkeyup='update_url_param(\"param_" + i + "_value_step\", this.value)' min='0' />";
	} else if (chosen_type == "hp.choiceint") {
		set_html = "<input class='parameter_input' id='choiceint_" + i + "_min' type='number' placeholder='Minimal number' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='choiceint_" + i + "_max' type='number' placeholder='Maximal number' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
		set_html += "<span style='color: orange;'>This is a pseudo-generator that HyperOpt does not really support. Remember this: in the DB, the index of the value will be saved, NOT the actual value listed here. When outputting anything via OmniOpts scripts, you will get results as expected. But if you change the <tt>config.ini</tt>, the results outputted by OmniOpts scripts will change too, even after it already ran!</span>";
	} else if (chosen_type == "hp.uniformint") {
		set_html = "<p>Return an integer uniformly distributed between the min and max value.</p>";
		set_html += "<input class='parameter_input' id='min_" + i + "' type='number' step='any' placeholder='min' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='max_" + i + "' type='number' step='any' placeholder='max' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
	} else if (chosen_type == "hp.uniform") {
		set_html = "<p>When optimizing, this variable is constrained to a two-sided interval.</p>";
		set_html += "<input class='parameter_input' id='min_" + i + "' type='number' step='any' placeholder='min' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='max_" + i + "' type='number' step='any' placeholder='max' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
	} else if (chosen_type == "hp.loguniform") {
		set_html = "<p>When optimizing, this variable is constrained to the interval <math>[<msup><mi>e</mi><mn mathvariant='normal'>min</mn></msup>,<mspace width='.1em' /><msup><mi>e</mi><mn mathvariant='normal'>max</mn></msup>]</math>.</p>";
		set_html += "<input class='parameter_input' id='min_" + i + "' type='number' step='any' placeholder='min' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='max_" + i + "' type='number' step='any' placeholder='max' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
	} else if (chosen_type == "hp.quniform") {
		set_html = '<p>Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below.</p>';
		set_html += "<input class='parameter_input' id='min_" + i + "' type='number' step='any' placeholder='min' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='max_" + i + "' type='number' step='any' placeholder='max' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='q_" + i + "' type='number' step='any' placeholder='q' value='" + get_url_param("param_" + i + "_value_q") + "' onkeyup='update_url_param(\"param_" + i + "_value_q\", this.value)' min='0' />";
	} else if (chosen_type == "hp.qloguniform") {
		set_html = '<p>Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.</p>';
		set_html += "<input class='parameter_input' id='min_" + i + "' type='number' step='any' placeholder='min' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='max_" + i + "' type='number' step='any' placeholder='max' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='q_" + i + "' type='number' step='any' placeholder='q' value='" + get_url_param("param_" + i + "_value_q") + "' onkeyup='update_url_param(\"param_" + i + "_value_q\", this.value)' min='0' />";
	} else if (chosen_type == "hp.normal") {
		set_html = "<input class='parameter_input' id='mu_" + i + "' type='number' step='any' placeholder='Mean &mu;' value='" + get_url_param("param_" + i + "_mu") + "' onkeyup='update_url_param(\"param_" + i + "_mu\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='sigma_" + i + "' type='number' step='any' placeholder='Standard deviation &sigma;' value='" + get_url_param("param_" + i + "_sigma") + "' onkeyup='update_url_param(\"param_" + i + "_sigma\", this.value)' min='0' />";
	} else if (chosen_type == "hp.qnormal") {
		set_html = "<p>Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.</p>";
		set_html += "<input class='parameter_input' id='mu_" + i + "' type='number' step='any' placeholder='Mean &mu;' value='" + get_url_param("mu_" + i) + "' min='0' onkeyup='update_url_param(\"mu_" + i + "\", this.value)' />";
		set_html += "<input class='parameter_input' id='sigma_" + i + "' type='number' step='any' placeholder='Standard deviation &sigma;' value='" + get_url_param("param_" + i + "_sigma") + "' onkeyup='update_url_param(\"param_" + i + "_sigma\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='q_" + i + "' type='number' step='any' placeholder='q' value='" + get_url_param("param_" + i + "_value_q") + "' min='0' onkeyup='update_url_param(\"param_" + i + "_value_q\", this.value)' />";
	} else if (chosen_type == "hp.lognormal") {
		set_html = "<input class='parameter_input' id='mu_" + i + "' type='number' step='any' placeholder='Mean &mu;' value='" + get_url_param("param_" + i + "_mu") + "' min='0' onkeyup='update_url_param(\"param_" + i + "_mu\", this.value)' />";
		set_html += "<input class='parameter_input' id='sigma_" + i + "' type='number' step='any' placeholder='Standard deviation &sigma;' value='" + get_url_param("param_" + i + "_sigma") + "' min='0' onkeyup='update_url_param(\"param_" + i + "_sigma\", this.value)' />";
	} else if (chosen_type == "hp.qlognormal") {
		set_html = "<p>Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.</p>";
		set_html += "<input class='parameter_input' id='mu_" + i + "' type='number' step='any' placeholder='Mean &mu;' value='" + get_url_param("param_" + i + "_mu") + "' min='0' onkeyup='update_url_param(\"param_" + i + "_mu\", this.value)' />";
		set_html += "<input class='parameter_input' id='sigma_" + i + "' type='number' step='any' placeholder='Standard deviation &sigma;' value='" + get_url_param("param_" + i + "_sigma") + "' min='0' onkeyup='update_url_param(\"param_" + i + "_sigma\", this.value)' />";
		set_html += "<input class='parameter_input' id='q_" + i + "' type='number' step='any' placeholder='q' value='" + get_url_param("param_" + i + "_value_q") + "' min='0' onkeyup='update_url_param(\"param_" + i + "_value_q\", this.value)' />";
	}

	set_html += get_gridsearch_warning(chosen_type);

	$("#type_" + i + "_settings").html(set_html);
	show_missing_values_error();
}

function get_gridsearch_warning (chosen_type) {
	if($("#algo_name").val() != "gridsearch") {
		return "";
	}

	if(valid_grid_search.includes(chosen_type)) {
		return "";
	}

	return '<div class="ui-state-error ui-corner-all" style="padding: 0 .7em;"><span class="ui-icon ui-icon-alert" style="float: left;"></span>' + chosen_type + ' is invalid for grid search. Valid types are: ' + valid_grid_search.join(", ") + "</div>";
}

function get_parameter_config (i) {
	return "<div class='parameter' id='parameter_" + i + "'><div class='errors' id='parameter_" + i + "_errors'></div>" + get_select_type(i) + "</div>";
}

function change_number_of_parameters() {
	save_form_data();
	var new_number_of_parameters = parseInt($("#number_of_parameters").val());
	if(!new_number_of_parameters) {
		new_number_of_parameters = 0;
	}

	if(new_number_of_parameters < 0) {
		alert("Number of parameters cannot be negative!");
		$(item).val(0);
	} else {
		if(new_number_of_parameters > number_of_parameters) {
			for (var i = number_of_parameters; i < new_number_of_parameters; i++) {
				$("#parameters").html($("#parameters").html() + get_parameter_config(i));
			}
		} else if(number_of_parameters - new_number_of_parameters != 0) {
			for (var i = number_of_parameters; i >= new_number_of_parameters; i--) {
				$("#parameter_" + i).remove();
			}
		}

	}

	number_of_parameters = new_number_of_parameters;

	for (var i = 0; i < number_of_parameters; i++) {
		change_parameter_settings(i);
	}

	show_missing_values_error();

	restore_form_data();

	update_config();
}

function add_listener(id, func) {
	if(typeof(id) == "string") {
		document.getElementById(id).addEventListener(
			"change",
			func,
			false
		);
		document.getElementById(id).addEventListener(
			"input",
			func,
			false
		);
	} else {
		for (this_id in id) {
			add_listener(id[this_id], func);
		}
	}
}

function parseINIString(data){
	var regex = {
		section: /^\s*\[\s*([^\]]*)\s*\]\s*$/,
	param: /^\s*([^=]+?)\s*=\s*(.*?)\s*$/,
	comment: /^\s*;.*$/
	};
	var value = {};
	var lines = data.split(/[\r\n]+/);
	var section = null;
	lines.forEach(function(line){
		if(regex.comment.test(line)) {
			return;
		} else if(regex.param.test(line)) {
			var match = line.match(regex.param);
			if(section) {
				value[section][match[1]] = match[2];
			} else {
				value[match[1]] = match[2];
			}
		} else if (regex.section.test(line)) {
			var match = line.match(regex.section);
			value[match[1]] = {};
			section = match[1];
		} else if (line.length == 0 && section) {
			section = null;
		};
	});
	return value;
}

function set_if_exists(data, category, name, id) {
	if(category in data) {
		if(name in data[category]) {
			var value = data[category][name];
			var jqid = "#" + id;
			if($(jqid).is(":checkbox")) {
				if(value == 1) {
					$(jqid).prop("checked", true);
				} else {
					$(jqid).prop("checked", false);
				}
			} else {
				$(jqid).val(value);
			}
		} else {
			console.log("name " + name + " cannot be found in data[" + category + "]");
		}
	} else {
		console.log("category " + category + " cannot be found in data");
	}
}

function show_multigpu () {
	var enable_gpus = $("#enable_gpus").is(":checked") ? 1 : 0;
	if(enable_gpus) {
		$("#multigpu").show();
		if($("#number_of_gpus").val() >= 2) {
			show_error_message("#multigputext", "Using more than 1 GPU works, but may take more time to allocate sub-jobs.");
		} else {
			$("#multigputext").html("");
		}
	} else {
		$("#multigpu").hide();
	}
}

function parse_from_config_ini () {
	var data = parseINIString($("#configini").val());
	

	var structure = {
		"DATA": [
			"projectname",
			"partition",
			"account",
			"reservation",
			"precision",
			"max_evals",
			"algo_name",
			"objective_program",
			"enable_gpus",
			"mem_per_cpu",
			"computing_time",
		],
		"DEBUG": [
			"show_live_output",
			"sbatch_or_srun",
			"debug_sbatch_srun",
			"trace_omniopt",
		],
		"MONGODB": [
			"worker_last_job_timeout",
			"poll_interval",
			"kill_after_n_no_results"
		]
	};

	var structure_keys = Object.keys(structure);

	for (var i = 0; i < structure_keys.length; i++) {
		var key = structure_keys[i];

		for (var j = 0; j < structure[key].length; j++) {
			var value = structure[key][j];
			set_if_exists(data, key, value, value);
		}
	}

	set_if_exists(data, "DATA", "num_gpus_per_worker", "number_of_gpus");

	set_if_exists(data, "DEBUG", "debug_xtreme", "enable_curl_debug");
	set_if_exists(data, "DEBUG", "debug", "enable_debug");

	set_if_exists(data, "DIMENSIONS", "dimensions", "number_of_parameters");
	change_number_of_parameters();
	
	if("DIMENSIONS" in data) {
		if("dimensions" in data["DIMENSIONS"]) {
			var number_of_parameters = data["DIMENSIONS"]["dimensions"];
			for (var i = 0; i < number_of_parameters; i++) {
				set_if_exists(data, "DIMENSIONS", "dim_" + i + "_name", "parameter_" + i + "_name");
				set_if_exists(data, "DIMENSIONS", "range_generator_" + i, "type_" + i);

				change_parameter_and_url(i, $("#parameter_" + i + "_name").val());
				change_parameter_and_url(i, $("#type_" + i).val());

				if("DIMENSIONS" in data) {
					var this_range_generator = "range_generator_" + i;
					if(this_range_generator in data["DIMENSIONS"]) {
						var this_range_generator = data["DIMENSIONS"][this_range_generator];
						if(this_range_generator == 'hp.randint') {
							set_if_exists(data, "DIMENSIONS", "max_dim_" + i, "randint_" + i + "_max");
						} else if(this_range_generator == 'hp.choicestep') {
							set_if_exists(data, "DIMENSIONS", "min_dim_" + i, "min_dim_" + i);
							set_if_exists(data, "DIMENSIONS", "max_dim_" + i, "max_dim_" + i);
							set_if_exists(data, "DIMENSIONS", "step_dim_" + i, "step_dim_" + i);
						} else if(this_range_generator == 'hp.choice') {
							set_if_exists(data, "DIMENSIONS", "options_" + i, "hp_choice_" + i);
						} else if(this_range_generator == 'hp.pchoice') {
							set_if_exists(data, "DIMENSIONS", "options_" + i, "hp_pchoice_" + i);
						} else if(this_range_generator == 'hp.qnormal' || this_range_generator == "hp.qlognormal") {
							set_if_exists(data, "DIMENSIONS", "mean_" + i, "mean_" + i);
							set_if_exists(data, "DIMENSIONS", "sigma_" + i, "sigma_" + i);
							set_if_exists(data, "DIMENSIONS", "q_" + i, "q_" + i);
						} else if(this_range_generator == 'hp.normal' || this_range_generator == 'hp.lognormal') {
							set_if_exists(data, "DIMENSIONS", "mu_" + i, "mu_" + i);
							set_if_exists(data, "DIMENSIONS", "sigma_" + i, "sigma_" + i);
						} else if(this_range_generator == 'hp.uniform' || this_range_generator == "hp.loguniform" || this_range_generator == "hp.uniformint") {
							set_if_exists(data, "DIMENSIONS", "min_dim_" + i, "min_" + i);
							set_if_exists(data, "DIMENSIONS", "max_dim_" + i, "max_" + i);
						} else if(this_range_generator == 'hp.quniform' || this_range_generator == "hp.qloguniform") {
							set_if_exists(data, "DIMENSIONS", "min_dim_" + i, "min_" + i);
							set_if_exists(data, "DIMENSIONS", "max_dim_" + i, "max_" + i);
							set_if_exists(data, "DIMENSIONS", "q_" + i, "q_" + i);
						} else {
							alert("Unhandled range generator: " + this_range_generator);
						}
					} else {
						console.log(this_range_generator + " cannot be found in data['DIMENSIONS']");
					}
				}
			}
		} else {
			console.log("dimensions not found in data['DIMENSIONS']");
		}
	} else {
		console.log("DIMENSIONS not found in data");
	}

	update_config();

	$("input").trigger("keyup")
}

function fill_test_data() {
	var num_params = $("#number_of_parameters").val();
	var example_program = "perl -e 'print qq#RESULT: #.(";
	var x = Array();
	for (var i = 0; i < num_params; i++) {
		x.push("($x_" + i + ")");
	}
	example_program += x.join(" + ");
	example_program += ").qq#\\n#'";

	$("#objective_program").val(example_program);

	$("#projectname").val("test_project");

	$('#partition').val("alpha");

	disable_gpu_when_none_available();
	show_warning_if_available();
	show_link_if_available();
	show_time_warning();

	$("#max_evals").val(100);

	update_config();
}

$(document).ready(function() {
	change_number_of_parameters();
	disable_gpu_when_none_available();
	show_warning_if_available();
	show_link_if_available();
	max_memory_per_worker();
	show_max_worker_warning();
	show_worker_warning();
	//use_max_memory_of_partition()

	add_listener("max_evals", show_worker_warning);

	add_listener("number_of_workers", show_max_worker_warning);
	add_listener("number_of_workers", show_worker_warning);

	add_listener("partition", show_max_worker_warning);
	add_listener("partition", max_memory_per_worker);
	add_listener("partition", show_time_warning);
	add_listener("partition", disable_gpu_when_none_available);
	add_listener("partition", show_warning_if_available);
	add_listener("partition", show_link_if_available);
	//add_listener("partition", use_max_memory_of_partition);

	add_listener("mem_per_cpu", max_memory_per_worker);

	add_listener("computing_time", show_time_warning);

	add_listener(["max_evals", "number_of_workers", "partition", "mem_per_cpu", "computing_time", "number_of_cpus_per_worker", "enable_gpus", "number_of_gpus", "seed"], update_config);

	document.getElementById("copytoclipboardbutton").addEventListener(
		"click",
		copy_bashcommand_to_clipboard,
		false
	)

	change_number_of_parameters();

	$("tr:even").css("background-color", "#ffffff");
	$("tr:odd").css("background-color", "#dadada");

	$( ".helpicon" ).each(function( index ) {
		var current_text = $(this).html();
		$(this).html(current_text + " <span title=\"" + $(this).data("help") + "\">&#x2753;</span>");
	});

	document.addEventListener('keydown', function(event) {
		if (event.ctrlKey && event.key === '*') {
			if (confirm('You pressed ctrl + *. Fill with auto-generated test-data? This may overwrite some of your input, so be careful.')) {
				fill_test_data();
			}
		}
	});
});

function get_i_for_param_name (pname) {
	var param_name_items = $('input[id^=parameter_][id$=_name]');

	var names = [];

	for (var i = 0; i < param_name_items.length; i++) {
		var t = param_name_items[i];

		var name = $(t).val()

		if(name == pname) {
			return i;
		}
	}

	return -1;
}


function check_param_names () {
	var param_name_items = $('input[id^=parameter_][id$=_name]');

	var names = [];

	for (var i = 0; i < param_name_items.length; i++) {
		var t = param_name_items[i];

		var name = $(t).val()

		var had_error = 0;

		if(names.includes(name)) {
			show_error_message("#param_" + i + "_duplicate_name_error", "The name of this layer already exists for a previous parameter!");
			had_error++;
		}

		if(!name.match(/^[a-zA-Z0-9_]*$/)) {
			show_error_message("#param_" + i + "_invalid_name_error", "Names for parameters can only consist of alphanumericals and underscores");
			had_error++;
		}

		if(had_error) {
			missing_values++;
		} else {
			$("#param_" + i + "_duplicate_name_error").html("").hide();
			$("#param_" + i + "_invalid_name_error").html("").hide();
		}

		names.push(name);
	}
}

