var formdata = [];

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

function get_number_of_parameters_from_objective_program () {
	const re = /\(\$x_\d+\)/g;
	return ($('#objective_program').val().match(re) || []).filter(onlyUnique).length;
}

function get_number_of_parameters () {
	return parseInt($("#number_of_parameters").val());
}

function show_error_message (divid, msg) {
	$(divid).show();
	$(divid).html('');
	$(divid).html(get_error_string(msg));
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

function update_config () {
	missing_values = 0;
	var config_string = "[DATA]\n";

	var precision = $("#precision").val();
	var max_evals = $("#max_evals").val();
	var objective_program = $("#objective_program").val() ;
	var projectname = $("#projectname").val();

	if(isNumeric(precision)) {
		config_string += "precision = " + precision + " \n";
	} else {
		config_string += "# !!! Precision-parameter is empty or not a valid number\n";
		missing_values++;
	}

	if(projectname == "cpu_test" || projectname == "cpu_test2" || projectname == "DONOTDELETE_testcase" || projectname == "gpu_test" || projectname == "test" || projectname == "testlowmem") {
		show_error_message("#invalidprojectnameerror", 'The project name <b>' + projectname + '</b> is already in use by one of the default projects. Choose another one.');
		projectname = "";
		missing_values++;
	} else if(!projectname.match(projectname_regex)) {
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
		config_string += "max_evals = " + max_evals + " \n";
		$("#maxevalserror").html('');
		$("#maxevalserror").hide();
	} else {
		config_string += "# !!! Max evals is empty or not a valid number\n";
		show_error_message("#maxevalserror", 'max evals is empty or not a valid number.');
		missing_values++;
	}

	config_string += "algo_name = " + $("#algo_name").val() + " \n";

	config_string += "range_generator_name = hp.randint\n";

	if(objective_program.length >= 1) {
		$("#objectiveprogramerror").hide();
		$("#objectiveprogramerror").html('');
		config_string += "objective_program = " + objective_program + " \n";
		for (var i = 0; i < number_of_parameters; i++) {
			var regex = new RegExp("\\(\\$x_" + i + "\\)");
			if(!objective_program.match(regex)) {
				var parameter_name = "($x_" + i + ")";
				if($("#parameter_" + i + "_name").val()) {
					parameter_name = $("#parameter_" + i + "_name").val() + " ($x_" + i + ")";
				}
				config_string += "# !!! Missing parameter " + parameter_name + " in Objective program string" + "\n";
				missing_values++;
			}
		}
		
		//if(get_number_of_parameters_from_objective_program() > get_number_of_parameters()) {
		//	$("#number_of_parameters").val(get_number_of_parameters_from_objective_program());
		//	number_of_parameters = $("#number_of_parameters").val();
		//	change_number_of_parameters();
		//}
	} else {
		config_string += "# !!! Objective Program string is empty\n";
		missing_values++;
		show_error_message("#objectiveprogramerror", 'Objective program string cannot be empty.');
	}

	if(number_of_parameters >= 1) {
		$("#toofewparameterserror").html('');
		config_string += "[DIMENSIONS]\n";

		config_string += "dimensions = " + number_of_parameters + "\n\n";

		for (var i = 0; i < number_of_parameters; i++) {
			var this_type = $("#type_" + i).val();
			var this_parameter_name = $("#parameter_" + i + "_name").val() ;

			var this_parameter_name_string = "($x_" + i + ")";
			if(this_parameter_name) {
				this_parameter_name_string = this_parameter_name + " ($x_" + i + ")";
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
					if(1 || isNumericList(this_val)) {
						var re = new RegExp(',{2,}', 'g');
						var respace = new RegExp('\\s*,\\s*', 'g');
						this_val = this_val.replace(re, ',');
						this_val = this_val.replace(respace, ',');
						this_val = this_val.replace(/,+$/, '');

						config_string += "options_" + i + " = " + this_val + "\n";
					} else {
						config_string += "# Parameter " + i + " must consist of numbers only";
					}
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "");
				}
			} else if(this_type == "hp.choiceint") {
				var this_min_dim = $("#choiceint_" + i + "_min").val();
				var this_max_dim = $("#choiceint_" + i + "_max").val();
				if(isNumeric(this_max_dim) && isNumeric(this_min_dim)) {
					if(parseInt(this_min_dim) <= parseInt(this_max_dim)) {
						var numberArray = [];
						for (var j = parseInt(this_min_dim); j <= parseInt(this_max_dim); j++) {
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
				var this_max_dim = $("#randint_" + i + "_max").val();
				if(isNumeric(this_max_dim) && this_max_dim >= 0) {
					config_string += "max_dim_" + i + " = " + this_max_dim + "\n";
				} else {
					config_string += no_value_str(i, this_parameter_name_string, "");
				}
			} else if(this_type == "hp.quniform") {
				var this_min = $("#min_" + i).val();
				var this_max = $("#max_" + i).val();
				var this_q = $("#q_" + i).val() ;

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
			} else if(this_type == "hp.loguniform" || this_type == "hp.uniform") {
				var this_min = $("#min_" + i).val();
				var this_max = $("#max_" + i).val();

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

	config_string += "[DEBUG]\n";
	config_string += "debug_xtreme = " + enable_debug + "\n";
	config_string += "debug = " + enable_debug + "\n";
	config_string += "info = " + enable_debug + "\n";
	config_string += "warning = " + enable_debug + "\n";
	config_string += "success = " + enable_debug + "\n";
	config_string += "stack = " + enable_debug + "\n";
	config_string += "show_live_output = " + show_live_output + "\n";

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

	if(computing_time > 0) {
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

							var sbatch_string = "sbatch -J " + projectname + " \\\n";
							var enable_gpus = $("#enable_gpus").is(":checked") ? 1 : 0;

							if($("#reservation").val()) {
								sbatch_string += " --reservation=" + $("#reservation").val() + " \\\n";
							}

							if($("#account").val()) {
								sbatch_string += " -A " + $("#account").val() + " \\\n";
							}

							var max_number_of_gpus = partition_data[partition]["max_number_of_gpus"];
							var max_mem_per_core = partition_data[partition]["max_mem_per_core"];

							var number_of_allocated_gpus = number_of_workers;
							var number_of_cpus_per_worker = $("#number_of_cpus_per_worker").val();

							if(number_of_allocated_gpus > max_number_of_gpus) {
								number_of_allocated_gpus = max_number_of_gpus;
							}

							if(enable_gpus == 1) {
								sbatch_string += " --cpus-per-task=" + number_of_cpus_per_worker + " --gres=gpu:" + number_of_allocated_gpus + " \\\n --gpus-per-task=1 \\\n";
							}

							sbatch_string += " --ntasks=" + number_of_workers + " \\\n";
							sbatch_string += " --time=" + computing_time + ":00:00 \\\n";
							sbatch_string += " --mem-per-cpu=" + mem_per_cpu + " \\\n";
							sbatch_string += " --partition=" + partition + " \\\n";

							sbatch_string += " sbatch.pl --project=" + projectname + " ";
							if(enable_debug == 1) {
								sbatch_string += " --debug";
							}

							sbatch_string = sbatch_string.replace(/ {2,}/g, ' ');
							var sbatch_string_no_newlines = sbatch_string.replace(/\\+\n/g, ' ');

							//var base_url = document.URL.substr(0,document.URL.lastIndexOf('/'));
							var base_url = "https://imageseg.scads.de/omnioptgui";
							var bash_command = "curl " + base_url + "/omniopt_script.sh 2>/dev/null | bash -s -- --projectname=" + projectname + " --config_file=" + btoa(config_string) + " --sbatch_command=" + btoa(sbatch_string_no_newlines);
							if ($("#enable_curl_debug").is(":checked")) {
								bash_command = bash_command + " --debug ";
							}

							if ($("#dont_ask_to_start").is(":checked")) {
								bash_command = bash_command + " --dont_start_job ";
							}

							if (!$("#dont_add_to_shell_history").is(":checked")) {
								bash_command = bash_command + '; if [[ "$SHELL" == "/bin/bash" ]]; then history -r; elif [[ "$SHELL" == "/bin/zsh" ]]; then fc -R; fi';
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
}

function change_parameter_and_url(i, t) {
	update_url_param("param_" + i + "_type", t.value);
	change_parameter_settings(i);
}

function get_select_type (i) {
	var spaces = "&nbsp;&nbsp;&nbsp;&nbsp;";
	return "<input class='parameter_input' type='text' onkeyup='update_url_param(\"param_" + i + "_name\", this.value)' placeholder='Name of this parameter' id='parameter_" + i + "_name' value='" + get_url_param("param_" + i + "_name") + "' /><br />" +
		"<select class='parameter_input' id='type_" + i + "' onchange='change_parameter_and_url(" + i + ", this)'>" +
		"\t<option value='hp.randint' " + (get_url_param("param_" + i + "_type") == "hp.randint" ? ' selected="true" ' : '') + ">hp.randint: Integer between 0 and a given maximal number</option>" +
		"\t<option value='' disabled='disabled'>────────────</option>\n" +
		"\t<option value='hp.choice' " + (get_url_param("param_" + i + "_type") == "hp.choice" ? ' selected="true" ' : '') + ">hp.choice: One of the given options, seperated by comma</option>" +
		"\t<option value='hp.choiceint' " + (get_url_param("param_" + i + "_type") == "hp.choiceint" ? ' selected="true" ' : '') + ">hp.choiceint: A pseudo-generator for hp.choice, for using integers between a and b</option>" +
		"\t<option value='' disabled='disabled'>────────────</option>\n" +
		"\t<option value='hp.uniform' " + (get_url_param("param_" + i + "_type") == "hp.uniform" ? ' selected="true" ' : '') + ">hp.uniform: Uniformly distributed value between two values</option>" +
		"\t<option value='hp.quniform' " + (get_url_param("param_" + i + "_type") == "hp.quniform" ? ' selected="true" ' : '') + "'>hp.quniform: Values like round(uniform(min, max)/q)&#8901;q</option>" +
		"\t<option value='hp.loguniform' " + (get_url_param("param_" + i + "_type") == "hp.loguniform" ? ' selected="true" ' : '') + "'>hp.loguniform: Values so that the log of the value is uniformly distributed</option>" +
		"\t<option value='hp.qloguniform' " + (get_url_param("param_" + i + "_type") == "hp.qloguniform" ? ' selected="true" ' : '') + "'>hp.qloguniform: Values like round(exp(uniform(min, max))/q) &#8901;q</option>" +
		"\t<option value='' disabled='disabled'>────────────</option>\n" +
		"\t<option value='hp.normal' " + (get_url_param("param_" + i + "_type") == "hp.normal" ? ' selected="true" ' : '') + "'>hp.normal: Real values that are normally distributed with Mean &mu; and Standard deviation &sigma;</option>" +
		"\t<option value='hp.qnormal' " + (get_url_param("param_" + i + "_type") == "hp.qnormal" ? ' selected="true" ' : '') + "'>hp.qnormal: Values like round(normal(&mu;, &sigma;)/q)&#8901;q</option>" +
		"\t<option value='hp.lognormal' " + (get_url_param("param_" + i + "_type") == "hp.lognormal" ? ' selected="true" ' : '') + ">hp.lognormal: Values so that the logarithm of normal(&mu;, &sigma;) is normally distributed</option>" +
		"\t<option value='hp.qlognormal' " + (get_url_param("param_" + i + "_type") == "hp.qlognormal" ? ' selected="true" ' : '') + ">hp.qlognormal: Values like round(exp(lognormal(&mu;, &sigma;))/q)&#8901;q</option>" +
		"</select>" +
		"<div id='type_" + i + "_settings'></div>";
}

function change_parameter_settings(i) {
	var chosen_type = $("#type_" + i).val();
	var set_html = '<div style="background-color: red">Unknown parameter &raquo;' + chosen_type + "&laquo;</div>";

	if (chosen_type == "hp.choice") {
		set_html = "<input class='parameter_input' id='hp_choice_" + i + "' type='text' placeholder='Comma seperated list of possible values' value='" + get_url_param("param_" + i + "_values") + "' onkeyup='update_url_param(\"param_" + i + "_values\", this.value)' a/>";
	} else if (chosen_type == "hp.randint") {
		set_html = "<input class='parameter_input' id='randint_" + i + "_max' type='number' placeholder='Maximal number' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
	} else if (chosen_type == "hp.choiceint") {
		set_html = "<input class='parameter_input' id='choiceint_" + i + "_min' type='number' placeholder='Minimal number' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='choiceint_" + i + "_max' type='number' placeholder='Maximal number' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
		set_html += "<span style='color: orange;'>This is a pseudo-generator that HyperOpt does not really support. Remember this: in the DB, the index of the value will be saved, NOT the actual value listed here. When outputting anything via OmniOpts scripts, you will get results as expected. But if you change the <tt>config.ini</tt>, the results outputted by OmniOpts scripts will change too, even after it already ran!</span>";
	} else if (chosen_type == "hp.uniform") {
		set_html = "<p>When optimizing, this variable is constrained to a two-sided interval.</p>";
		set_html += "<input class='parameter_input' id='min_" + i + "' type='number' step='any' placeholder='min' value='" + get_url_param("param_" + i + "value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='max_" + i + "' type='number' step='any' placeholder='max' value='" + get_url_param("param_" + i + "value_max") + "' onkeyup='update_url_param(\"param_" + i + "_value_max\", this.value)' min='0' />";
	} else if (chosen_type == "hp.loguniform") {
		set_html = "<p>When optimizing, this variable is constrained to the interval <math>[<msup><mi>e</mi><mn mathvariant='normal'>min</mn></msup>,<mspace width='.1em' /><msup><mi>e</mi><mn mathvariant='normal'>max</mn></msup>]</math>.</p>";
		set_html += "<input class='parameter_input' id='min_" + i + "' type='number' step='any' placeholder='min' value='" + get_url_param("param_" + i + "_value_min") + "' onkeyup='update_url_param(\"param_" + i + "_value_min\", this.value)' min='0' />";
		set_html += "<input class='parameter_input' id='max_" + i + "' type='number' step='any' placeholder='max' value='" + get_url_param("param_" + i + "_value_max") + "' onkeyup='update_url_param(\"param_i" + i + "_value_max\", this.value)' min='0' />";
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
		set_html += "<input class='parameter_input' id='mu_" + i + "' type='number' step='any' placeholder='Mean &mu;' value='" + get_url_param("param_" + i + "_mu") + "' min='0' />";
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

	$("#type_" + i + "_settings").html(set_html);
	show_missing_values_error();
}

function get_parameter_config (i) {
	return "<div class='parameter' id='parameter_" + i + "'><h3>Parameter <tt>($x_" + i + ")</tt></h3><div class='errors' id='parameter_" + i + "_errors'></div>" + get_select_type(i) + "</div>";
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
			var r = confirm("This will delete the last " + (number_of_parameters - new_number_of_parameters) + " parameter(s) and all entered data for that parameter! Are you sure about this?");
			if (r == true) {
				for (var i = number_of_parameters; i >= new_number_of_parameters; i--) {

					$("#parameter_" + i).remove();
				}
			} else {
				$("#number_of_parameters").blur();
				$("#number_of_parameters").val(number_of_parameters);
				new_number_of_parameters = number_of_parameters;
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

$(document).ready(function() {
	change_number_of_parameters();
	disable_gpu_when_none_available();
	show_warning_if_available();
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
	//add_listener("partition", use_max_memory_of_partition);

	add_listener("mem_per_cpu", max_memory_per_worker);

	add_listener("computing_time", show_time_warning);

	add_listener(["max_evals", "number_of_workers", "partition", "mem_per_cpu", "computing_time", "number_of_cpus_per_worker"], update_config);

	document.getElementById("copytoclipboardbutton").addEventListener(
		"click",
		copy_bashcommand_to_clipboard,
		false
	)

	change_number_of_parameters();

	/*
	window.addEventListener("beforeunload", function (e) {
		var confirmationMessage = 'It looks like you have been editing something. '
			+ 'If you leave before saving, your changes will be lost.';

		(e || window.event).returnValue = confirmationMessage; //Gecko + IE
		return confirmationMessage; //Gecko + Webkit, Safari, Chrome etc.
	});
	*/

	$("tr:even").css("background-color", "#ffffff");
	$("tr:odd").css("background-color", "#dadada");

	$( ".helpicon" ).each(function( index ) {
		var current_text = $(this).html();
		$(this).html(current_text + " <span title=\"" + $(this).data("help") + "\">&#x2753;</span>");
	});
});
