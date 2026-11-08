var invalid_names = ["generation_node", "start_time", "end_time", "hostname", "signal", "exit_code", "run_time", "program_string", "arm_name", "trial_index", "generation_method", "trial_status", "idxs", "submit_time", "queue_time", "worker_generator_uuid"];
var document_is_ready = false;
var fadeTime = 0;
var fadeTimeAfterLoading = 300;

function is_visible(e) {
	if (!e || e.nodeType !== 1) return false;
	if (!document.body.contains(e)) return false;

	let style = e.ownerDocument.defaultView.getComputedStyle(e);
	if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;

	let parent = e.parentElement;
	while (parent && parent !== document.body) {
		let ps = parent.ownerDocument.defaultView.getComputedStyle(parent);
		if (ps.display === "none" || ps.visibility === "hidden" || ps.opacity === "0") return false;
		parent = parent.parentElement;
	}

	return true;
}

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

function time_picker_max_hours (max_minutes) {
	var max = parseInt(max_minutes, 10);

	if (isNaN(max) || max <= 0) {
		return 99999;
	}

	return Math.ceil(max / 60);
}

function time_picker_parts_from_minutes (total_minutes, max_minutes) {
	var minutes = parseInt(total_minutes, 10);
	var max = parseInt(max_minutes, 10);

	if (isNaN(minutes) || minutes < 0) {
		minutes = 0;
	}

	if (!isNaN(max) && max > 0 && minutes > max) {
		minutes = max;
	}

	return {
		hours: Math.floor(minutes / 60),
		minutes: minutes % 60,
		totalMinutes: minutes
	};
}

function time_picker_parts_from_hours_and_minutes (hours, minutes, max_minutes) {
	var normalizedHours = parseInt(hours, 10);
	var normalizedMinutes = parseInt(minutes, 10);
	var max = parseInt(max_minutes, 10);

	if (isNaN(normalizedHours) || normalizedHours < 0) {
		normalizedHours = 0;
	}

	if (isNaN(normalizedMinutes)) {
		normalizedMinutes = 0;
	}

	if (normalizedMinutes < 0) {
		if (normalizedHours > 0) {
			normalizedHours -= 1;
			normalizedMinutes += 60;
		} else {
			normalizedMinutes = 0;
		}
	}

	if (normalizedMinutes >= 60) {
		normalizedHours += Math.floor(normalizedMinutes / 60);
		normalizedMinutes = normalizedMinutes % 60;
	}

	var totalMinutes = normalizedHours * 60 + normalizedMinutes;

	if (!isNaN(max) && max > 0 && totalMinutes > max) {
		totalMinutes = max;
	}

	return {
		hours: Math.floor(totalMinutes / 60),
		minutes: totalMinutes % 60,
		totalMinutes: totalMinutes
	};
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

	var maxMinutes = $input.attr("max");
	var maxHours = time_picker_max_hours(maxMinutes);
	var parts = time_picker_parts_from_minutes($input.val(), maxMinutes);

	var $div = $(`
		    <div class='time_picker_container'>
			<input type='number' min=0 max=${maxHours} class="time_picker_hours" value='${parts.hours}' onchange='update_original_time_element("${input_id}", this)'></input> Hours,
			<input type='number' min=0 max=59 step=1 class="time_picker_minutes" value='${parts.minutes}' onchange='update_original_time_element("${input_id}", this)'></input> Minutes
		    </div>
		`);

	$parent.prepend($div);

	$input.hide();
}

function update_original_time_element (original_element_id, new_element) {
	var $parent = $(new_element).parent();
	var $time_picker_minutes = $parent.find(".time_picker_minutes");
	var $time_picker_hours = $parent.find(".time_picker_hours");
	var $original = $("#" + original_element_id);
	var parts = time_picker_parts_from_hours_and_minutes(
		$time_picker_hours.val(),
		$time_picker_minutes.val(),
		$original.attr("max")
	);

	$time_picker_hours.val(parts.hours);
	$time_picker_minutes.val(parts.minutes);
	$original.val(parts.totalMinutes).trigger("change");
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
	const pattern = /(?:\$|\%)?\([a-zA-Z_][a-zA-Z0-9_]*\)|(?:\$|%)[a-zA-Z_][a-zA-Z0-9_]*/g;
	const variableNames = [];

	let match;
	while ((match = pattern.exec(run_program_string)) !== null) {
		let previous = match.index > 0 ? run_program_string.charAt(match.index - 1) : "";
		if (previous === "\\" || (match[0].charAt(0) === "%" && previous === "%") || (match[0].charAt(0) === "$" && previous === "$")) {
			continue;
		}
		let varName = match[0];
		varName = varName.replace(/^(\$|%)/, "");
		varName = varName.replace(/^(\$|%)?\(|\)$/g, "");
		if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(varName)) {
			variableNames.push(varName);
		}
	}

	return variableNames;
}

var RUN_PROGRAM_KEYWORD_SUGGESTIONS = [
	{ key: "lr", preferred_name: "learning_rate", aliases: ["lr", "learning_rate", "learningrate", "learn_rate", "learnrate", "lrate", "eta", "learn", "learning", "learing_rate", "lerning_rate", "learning_rat", "lr_rate"], kind: "range", min: "0.00001", max: "0.001", value_type: "float", log_scale: true },
	{ key: "decay", aliases: ["decay", "decay0", "dec", "decay_rate", "decayrate", "dec_rate", "decai", "deacy"], kind: "range", min: "0", max: "0.01", value_type: "float", log_scale: false },
	{ key: "optimizer", aliases: ["optimizer", "optimiser", "opt", "optim", "optimizer_name", "optimiser_name", "optimization", "optimizer_type", "optimzer", "optmizer", "optimizor", "optimise", "optimize"], kind: "choice", values: "adam,adamax,adamw,sgd,rmsprop,adagrad,adadelta,lion,adafactor" },
	{ key: "epochs", aliases: ["epochs", "epoch", "num_epochs", "n_epochs", "max_epochs", "epochsz", "epoch_count", "num_epoch", "total_epochs", "epocs", "epoches", "epohcs", "num_epoc", "epch"], kind: "range", min: "1", max: "100", value_type: "int", log_scale: false },
	{ key: "batch_size", aliases: ["batch_size", "batchsize", "batch", "bs", "batchsz", "bsize", "size", "batch_sz", "bsz", "batch_dim", "batch_n", "nbatch", "batch_count", "bat", "batch_sze", "batch_zise", "batsh_size", "bath_size"], kind: "range", min: "8", max: "512", value_type: "int", log_scale: false },
	{ key: "dropout", aliases: ["dropout", "dropout_rate", "drop_out", "drop", "drop_rate", "droprate", "dropout_prob", "dropout_p", "p_dropout", "dropuot", "droupout", "droput"], kind: "range", min: "0", max: "0.5", value_type: "float", log_scale: false },
	{ key: "momentum", aliases: ["momentum", "mom", "momentum_rate", "momentum_coef", "momentum_factor", "momentom", "momemtum", "momntum"], kind: "range", min: "0", max: "0.99", value_type: "float", log_scale: false },
	{ key: "weight_decay", aliases: ["weight_decay", "wd", "l2_penalty", "weightdecay", "weight_dec", "l2", "l2_reg", "l2_regularization", "weight_decay_rate", "wieght_decay", "weight_deacy"], kind: "range", min: "0", max: "0.01", value_type: "float", log_scale: false },
	{ key: "warmup", aliases: ["warmup", "warm_up", "warmup_steps", "warmup_step", "warmup_iters", "warmup_epochs", "wu", "warm_up_steps", "warmup_iterations"], kind: "range", min: "0", max: "1000", value_type: "int", log_scale: false },
	{ key: "hidden_size", aliases: ["hidden_size", "hidden_dim", "hidden_units", "hidden", "hsize", "hdim", "hidden_sz", "hidden_layer_size", "hidden_layers_size", "n_hidden", "num_hidden", "hidden_width", "hid", "hid_size", "hidden_siz", "hidde_size"], kind: "range", min: "16", max: "1024", value_type: "int", log_scale: false },
	{ key: "num_layers", aliases: ["num_layers", "n_layers", "layers", "num_hidden_layers", "layer_count", "nlayer", "nlayers", "numlayer", "numlayers", "num_layer", "layer_num", "num_layers_total", "num_hidden_layer", "hidden_layer_count"], kind: "range", min: "1", max: "12", value_type: "int", log_scale: false },
	{ key: "num_heads", aliases: ["num_heads", "n_heads", "heads", "head_count", "nhead", "nheads", "numhead", "numheads", "attention_heads", "num_attention_heads", "n_head"], kind: "range", min: "1", max: "16", value_type: "int", log_scale: false },
	{ key: "embedding_dim", aliases: ["embedding_dim", "emb_dim", "embedding_size", "emb_size", "embdim", "embeddingdim", "emb", "embedding", "dim_embedding", "embedding_dimension", "embed_dim", "embed_size", "embed_dims", "embedding_dims"], kind: "range", min: "8", max: "512", value_type: "int", log_scale: false },
	{ key: "kernel_size", aliases: ["kernel_size", "kernel", "ksize", "kernelsize", "kernel_sz", "k_size", "ksz", "filter_size", "filtersize", "filter_dim", "kern", "kernal_size", "kernal", "kerel_size"], kind: "range", min: "1", max: "15", value_type: "int", log_scale: false },
	{ key: "stride", aliases: ["stride", "strides", "stride_size", "strid"], kind: "range", min: "1", max: "4", value_type: "int", log_scale: false },
	{ key: "padding", aliases: ["padding", "pad", "padding_size", "padsize", "pad_size", "padding_mode", "padding_type"], kind: "choice", values: "same,valid,causal" },
	{ key: "activation", aliases: ["activation", "act", "activation_fn", "activation_function", "activation_func", "activation_name", "nonlinearity", "non_linearity", "activ", "actvation", "activaton", "activation_type"], kind: "choice", values: "relu,tanh,sigmoid,gelu,leaky_relu,elu,selu,swish,silu,mish,softplus" },
	{ key: "seed", aliases: ["seed", "random_seed", "rng_seed", "seed_value", "seedval", "randomseed", "rngseed", "rng", "rand_seed", "random_state", "rand_state", "manual_seed", "sed"], kind: "range", min: "0", max: "100000", value_type: "int", log_scale: false },
	{ key: "gradient_clip", aliases: ["gradient_clip", "grad_clip", "clip_grad", "gradient_clipping", "gradclip", "grad_clip_value", "clip_value", "clip_norm", "clipnorm", "max_grad_norm", "gradient_clip_norm", "gclip", "gradientclip", "grad_norm", "clip_grad_norm"], kind: "range", min: "0", max: "5", value_type: "float", log_scale: false },
	{ key: "beta1", param_name: "beta_one", aliases: ["beta1", "beta_1", "adam_beta1", "b1", "beta1_value", "beta_one", "b_1"], kind: "range", min: "0.8", max: "0.999", value_type: "float", log_scale: false },
	{ key: "beta2", param_name: "beta_two", aliases: ["beta2", "beta_2", "adam_beta2", "b2", "beta2_value", "beta_two", "b_2"], kind: "range", min: "0.9", max: "0.9999", value_type: "float", log_scale: false },
	{ key: "epsilon", aliases: ["epsilon", "eps", "adam_epsilon", "epsilon_value", "eps_val", "epsilonval", "epsilon0", "epsilion", "epsiolon"], kind: "range", min: "0.00000001", max: "0.001", value_type: "float", log_scale: true },
	{ key: "patience", aliases: ["patience", "early_stopping_patience", "earlystop_patience", "pat", "patience_epochs", "es_patience", "pacience", "early_stopping_rounds"], kind: "range", min: "1", max: "50", value_type: "int", log_scale: false },
	{ key: "factor", aliases: ["factor", "lr_factor", "reduce_factor", "factor_lr", "lr_reduce_factor", "reduce_lr_factor", "decay_factor", "factor_value", "fact"], kind: "range", min: "0.1", max: "0.9", value_type: "float", log_scale: false },
	{ key: "gamma", aliases: ["gamma", "lr_gamma", "gamma_lr", "gamma_value", "gamma_decay", "gamme"], kind: "range", min: "0.1", max: "0.9", value_type: "float", log_scale: false },
	{ key: "min_lr", aliases: ["min_lr", "min_learning_rate", "min_lr_rate", "minimum_lr", "min_learningrate", "lr_min", "min_lrate", "min_learn_rate", "minlr"], kind: "range", min: "0.0000001", max: "0.00001", value_type: "float", log_scale: true },
	{ key: "max_lr", aliases: ["max_lr", "max_learning_rate", "max_lr_rate", "maximum_lr", "max_learningrate", "lr_max", "max_lrate", "max_learn_rate", "maxlr"], kind: "range", min: "0.001", max: "0.1", value_type: "float", log_scale: true },
	{ key: "scheduler", aliases: ["scheduler", "lr_scheduler", "sched", "lr_sched", "schedule", "lr_schedule", "scheduler_type", "scheduler_name", "schedueler", "schedular", "sheduler"], kind: "choice", values: "none,step,cosine,linear,plateau,exponential,polynomial,onecycle,constant" },
	{ key: "loss", aliases: ["loss", "loss_function", "criterion", "loss_fn", "loss_type", "objective", "cost", "cost_function", "error_function", "objective_function", "losses"], kind: "choice", values: "mse,mae,rmse,huber,logcosh,binary_crossentropy,categorical_crossentropy,sparse_categorical_crossentropy,focal_loss,hinge" },
	{ key: "num_classes", aliases: ["num_classes", "n_classes", "classes", "class_count", "nclass", "nclasses", "numclass", "numclasses", "ncls", "num_labels", "n_labels", "labels", "num_output_classes", "num_class", "n_class", "output_classes"], kind: "range", min: "2", max: "1000", value_type: "int", log_scale: false },
	{ key: "input_dim", aliases: ["input_dim", "input_size", "input_features", "input_feature", "inputdim", "input_dimension", "in_dim", "indim", "insize", "input_shape", "n_inputs", "num_inputs", "input_dimension_size", "num_features", "n_features", "n_feature", "feature_dim", "input", "inputs"], kind: "range", min: "1", max: "4096", value_type: "int", log_scale: false },
	{ key: "output_dim", aliases: ["output_dim", "output_size", "outputdim", "output_dimension", "out_dim", "outdim", "outsize", "output_shape", "n_outputs", "num_outputs", "output_dimension_size", "num_targets", "n_targets", "target_dim", "output_dims", "output", "outputs"], kind: "range", min: "1", max: "4096", value_type: "int", log_scale: false },
	{ key: "n_estimators", aliases: ["n_estimators", "num_estimators", "estimators", "n_est", "nest", "nestimators", "num_trees", "n_trees", "trees", "tree_count", "num_rounds", "n_rounds", "num_iterations", "n_iter", "num_estimator", "n_estimator"], kind: "range", min: "10", max: "1000", value_type: "int", log_scale: false },
	{ key: "max_depth", aliases: ["max_depth", "tree_depth", "maxdepth", "max_depth_tree", "depth", "tree_max_depth", "mdepth", "max_tree_depth", "tree_maxdepth"], kind: "range", min: "1", max: "50", value_type: "int", log_scale: false },
	{ key: "min_samples_split", aliases: ["min_samples_split", "minsamplesplit", "min_sample_split", "min_split", "min_samples", "samples_split", "min_samples_to_split", "minsplit"], kind: "range", min: "2", max: "20", value_type: "int", log_scale: false },
	{ key: "min_samples_leaf", aliases: ["min_samples_leaf", "minsamplesleaf", "min_sample_leaf", "min_leaf", "min_samples_per_leaf", "samples_leaf", "minleaf"], kind: "range", min: "1", max: "20", value_type: "int", log_scale: false },
	{ key: "learning_rate_init", aliases: ["learning_rate_init", "init_lr", "init_learning_rate", "lr_init", "learningrate_init", "initial_lr", "initial_learning_rate", "lr_start", "learning_rate_start", "learn_rate_init", "init_learningrate"], kind: "range", min: "0.00001", max: "0.001", value_type: "float", log_scale: true },
	{ key: "alpha", aliases: ["alpha", "reg_alpha", "alpha_value", "regularization_alpha", "alpha_reg", "l1_alpha", "alpha_l1"], kind: "range", min: "0.000001", max: "1", value_type: "float", log_scale: true },
	{ key: "regularization", aliases: ["regularization", "reg", "reg_strength", "l2_strength", "regularization_strength", "reg_strength_l2", "l2_lambda", "lambda", "lambda_reg", "reg_lambda", "weight_regularization", "regularization_rate", "regularisation", "regularizaton", "regul", "regularization_factor", "l2_alpha"], kind: "range", min: "0", max: "0.1", value_type: "float", log_scale: false },
	{ key: "max_features", aliases: ["max_features", "maxfeatures", "feature_fraction", "max_feature"], kind: "choice", values: "sqrt,log2,auto,none" },
	{ key: "max_leaf_nodes", aliases: ["max_leaf_nodes", "maxleafnodes", "max_leaves", "num_leaves", "n_leaves"], kind: "range", min: "2", max: "10000", value_type: "int", log_scale: false },
	{ key: "min_weight_fraction_leaf", aliases: ["min_weight_fraction_leaf", "minweightfractionleaf", "min_weight_leaf"], kind: "range", min: "0", max: "0.5", value_type: "float", log_scale: false },
	{ key: "min_impurity_decrease", aliases: ["min_impurity_decrease", "minimpuritydecrease", "min_impurity"], kind: "range", min: "0", max: "0.1", value_type: "float", log_scale: false },
	{ key: "bootstrap", aliases: ["bootstrap", "bootstrapping", "boot"], kind: "choice", values: "true,false" },
	{ key: "oob_score", aliases: ["oob_score", "oobscore", "out_of_bag_score"], kind: "choice", values: "true,false" },
	{ key: "ccp_alpha", aliases: ["ccp_alpha", "ccpalpha", "pruning_alpha"], kind: "range", min: "0", max: "0.1", value_type: "float", log_scale: false },
	{ key: "max_samples", aliases: ["max_samples", "maxsamples", "sample_fraction", "sample_ratio"], kind: "range", min: "0.1", max: "1", value_type: "float", log_scale: false },
	{ key: "subsample", aliases: ["subsample", "sub_sample", "sample_rate"], kind: "range", min: "0.5", max: "1", value_type: "float", log_scale: false },
	{ key: "colsample_bytree", aliases: ["colsample_bytree", "colsample", "col_sample", "feature_subsample"], kind: "range", min: "0.1", max: "1", value_type: "float", log_scale: false },
	{ key: "min_child_weight", aliases: ["min_child_weight", "minchildweight", "min_child", "child_weight"], kind: "range", min: "0", max: "20", value_type: "float", log_scale: false },
	{ key: "scale_pos_weight", aliases: ["scale_pos_weight", "scaleposweight", "pos_weight", "positive_weight"], kind: "range", min: "0.1", max: "10", value_type: "float", log_scale: true },
	{ key: "C", aliases: ["C", "c", "c_param", "regularization_c"], kind: "range", min: "0.001", max: "100", value_type: "float", log_scale: true },
	{ key: "tol", aliases: ["tol", "tolerance", "rtol", "atol", "convergence_tolerance"], kind: "range", min: "0.000001", max: "0.01", value_type: "float", log_scale: true },
	{ key: "solver", aliases: ["solver", "solver_type", "linear_solver"], kind: "choice", values: "lbfgs,adam,sgd,newton-cg,saga,cg,gmres,bicgstab,direct" },
	{ key: "penalty", aliases: ["penalty", "penalty_type", "regularization_type"], kind: "choice", values: "l2,l1,elasticnet,none" },
	{ key: "n_neighbors", aliases: ["n_neighbors", "nneighbors", "neighbors", "k_neighbors", "knn"], kind: "range", min: "1", max: "100", value_type: "int", log_scale: false },
	{ key: "weights", aliases: ["weights", "weighting", "neighbor_weights"], kind: "choice", values: "uniform,distance" },
	{ key: "leaf_size", aliases: ["leaf_size", "leafsize", "tree_leaf_size"], kind: "range", min: "10", max: "100", value_type: "int", log_scale: false },
	{ key: "n_components", aliases: ["n_components", "ncomponents", "num_components", "components", "latent_dim"], kind: "range", min: "2", max: "512", value_type: "int", log_scale: false },
	{ key: "perplexity", aliases: ["perplexity", "perp"], kind: "range", min: "5", max: "100", value_type: "float", log_scale: false },
	{ key: "max_iter", aliases: ["max_iter", "maxiter", "max_iters", "max_iterations", "iterations", "n_iterations", "num_iters", "iter", "iters", "iteration"], kind: "range", min: "10", max: "10000", value_type: "int", log_scale: false },
	{ key: "min_child_samples", aliases: ["min_child_samples", "minchildsamples", "min_data_in_leaf", "min_data_leaf"], kind: "range", min: "1", max: "100", value_type: "int", log_scale: false },
	{ key: "population_size", aliases: ["population_size", "pop_size", "popsize", "n_population", "population", "num_population"], kind: "range", min: "10", max: "1000", value_type: "int", log_scale: false },
	{ key: "generations", aliases: ["generations", "n_generations", "num_generations", "max_generations", "ngen"], kind: "range", min: "1", max: "1000", value_type: "int", log_scale: false },
	{ key: "mutation_rate", aliases: ["mutation_rate", "mutation_prob", "mutation_probability", "p_mutation", "mut_rate", "mutrate"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "crossover_rate", aliases: ["crossover_rate", "crossover_prob", "crossover_probability", "p_crossover", "cx_rate", "cxrate"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "elitism_rate", aliases: ["elitism_rate", "elite_size", "elitism", "n_elite", "num_elite", "elite_fraction"], kind: "range", min: "0", max: "0.5", value_type: "float", log_scale: false },
	{ key: "tournament_size", aliases: ["tournament_size", "tournament", "tour_size", "selection_pressure"], kind: "range", min: "2", max: "20", value_type: "int", log_scale: false },
	{ key: "swarm_size", aliases: ["swarm_size", "n_particles", "num_particles", "particle_count", "particles", "swarm"], kind: "range", min: "10", max: "500", value_type: "int", log_scale: false },
	{ key: "inertia_weight", aliases: ["inertia_weight", "inertia", "w_inertia", "omega_inertia"], kind: "range", min: "0", max: "1.2", value_type: "float", log_scale: false },
	{ key: "cognitive_coeff", aliases: ["cognitive_coeff", "c1", "cognitive", "cognitive_weight"], kind: "range", min: "0", max: "4", value_type: "float", log_scale: false },
	{ key: "social_coeff", aliases: ["social_coeff", "c2", "social", "social_weight"], kind: "range", min: "0", max: "4", value_type: "float", log_scale: false },
	{ key: "cooling_rate", aliases: ["cooling_rate", "cooling", "cooling_factor", "cool_rate"], kind: "range", min: "0.8", max: "0.999", value_type: "float", log_scale: false },
	{ key: "initial_temperature", aliases: ["initial_temperature", "initial_temp", "temp_init", "t_start", "t0"], kind: "range", min: "0.1", max: "1000", value_type: "float", log_scale: true },
	{ key: "final_temperature", aliases: ["final_temperature", "final_temp", "temp_final", "t_end_temp", "temp_min"], kind: "range", min: "0.000001", max: "1", value_type: "float", log_scale: true },
	{ key: "n_restarts", aliases: ["n_restarts", "num_restarts", "restarts", "restart_count"], kind: "range", min: "1", max: "20", value_type: "int", log_scale: false },
	{ key: "n_trials", aliases: ["n_trials", "num_trials", "trials", "trial_count"], kind: "range", min: "10", max: "1000", value_type: "int", log_scale: false },
	{ key: "n_initial_points", aliases: ["n_initial_points", "initial_points", "n_startup_trials", "startup_trials"], kind: "range", min: "5", max: "100", value_type: "int", log_scale: false },
	{ key: "acquisition_function", aliases: ["acquisition_function", "acquisition", "acq_func", "acq"], kind: "choice", values: "ei,ucb,poi,gp_hedge" },
	{ key: "kernel_type", aliases: ["kernel_type", "cov_kernel", "covariance", "covariance_function", "gp_kernel"], kind: "choice", values: "rbf,matern,linear,poly" },
	{ key: "length_scale", aliases: ["length_scale", "lengthscale", "ls", "gp_length_scale"], kind: "range", min: "0.001", max: "1000", value_type: "float", log_scale: true },
	{ key: "noise_level", aliases: ["noise_level", "noise", "noise_std", "noise_sigma", "measurement_noise"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "exploration", aliases: ["exploration", "exploration_rate", "explore_fraction"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "penalty_coeff", aliases: ["penalty_coeff", "constraint_penalty", "penalty_weight", "penalty_factor"], kind: "range", min: "0", max: "100", value_type: "float", log_scale: false },
	{ key: "time_step", aliases: ["time_step", "timestep", "dt", "delta_t", "delta_time"], kind: "range", min: "0.000001", max: "0.1", value_type: "float", log_scale: true },
	{ key: "end_time", aliases: ["end_time", "simulation_time", "duration", "t_end", "total_time"], kind: "range", min: "0.1", max: "1000", value_type: "float", log_scale: true },
	{ key: "mesh_size", aliases: ["mesh_size", "meshsize", "cell_size", "element_size", "dx"], kind: "range", min: "0.0001", max: "1", value_type: "float", log_scale: true },
	{ key: "grid_size", aliases: ["grid_size", "gridsize", "grid_dim", "grid_points", "n_grid"], kind: "range", min: "8", max: "1024", value_type: "int", log_scale: false },
	{ key: "resolution", aliases: ["resolution", "res", "image_size", "img_size"], kind: "range", min: "8", max: "4096", value_type: "int", log_scale: false },
	{ key: "cfl", aliases: ["cfl", "cfl_number", "courant", "courant_number"], kind: "range", min: "0.1", max: "1", value_type: "float", log_scale: false },
	{ key: "viscosity", aliases: ["viscosity", "kinematic_viscosity", "dynamic_viscosity", "visc"], kind: "range", min: "0.000001", max: "1", value_type: "float", log_scale: true },
	{ key: "density", aliases: ["density", "rho", "mass_density"], kind: "range", min: "0.1", max: "10000", value_type: "float", log_scale: true },
	{ key: "diffusivity", aliases: ["diffusivity", "diffusion_coeff", "diffusion_coefficient", "diff_coef"], kind: "range", min: "0.00000001", max: "1", value_type: "float", log_scale: true },
	{ key: "thermal_conductivity", aliases: ["thermal_conductivity", "conductivity", "heat_conductivity", "k_thermal"], kind: "range", min: "0.01", max: "500", value_type: "float", log_scale: true },
	{ key: "specific_heat", aliases: ["specific_heat", "heat_capacity", "cp", "specific_heat_capacity"], kind: "range", min: "100", max: "5000", value_type: "float", log_scale: false },
	{ key: "youngs_modulus", aliases: ["youngs_modulus", "young_modulus", "elastic_modulus", "e_modulus"], kind: "range", min: "1000000", max: "1000000000000", value_type: "float", log_scale: true },
	{ key: "poisson_ratio", aliases: ["poisson_ratio", "poissons_ratio", "nu_poisson"], kind: "range", min: "0", max: "0.5", value_type: "float", log_scale: false },
	{ key: "yield_strength", aliases: ["yield_strength", "yield_stress", "sigma_y"], kind: "range", min: "100000", max: "10000000000", value_type: "float", log_scale: true },
	{ key: "friction_coefficient", aliases: ["friction_coefficient", "friction", "mu_friction"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "restitution", aliases: ["restitution", "restitution_coefficient", "bounce"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "damping", aliases: ["damping", "damping_coefficient", "damping_ratio"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "gravity", aliases: ["gravity", "g", "gravitational_acceleration"], kind: "range", min: "0", max: "20", value_type: "float", log_scale: false },
	{ key: "relaxation_factor", aliases: ["relaxation_factor", "relaxation", "relax_factor", "omega_relax"], kind: "range", min: "0.1", max: "1.5", value_type: "float", log_scale: false },
	{ key: "preconditioner", aliases: ["preconditioner", "precond", "preconditioner_type"], kind: "choice", values: "jacobi,ilu,amg,none" },
	{ key: "scheme", aliases: ["scheme", "discretization", "time_scheme", "integrator"], kind: "choice", values: "euler,rk4,crank_nicolson,upwind" },
	{ key: "n_steps", aliases: ["n_steps", "num_steps", "steps", "step_count", "timesteps"], kind: "range", min: "1", max: "100000", value_type: "int", log_scale: false },
	{ key: "n_cells", aliases: ["n_cells", "num_cells", "cells", "cell_count"], kind: "range", min: "10", max: "1000000", value_type: "int", log_scale: false },
	{ key: "n_samples", aliases: ["n_samples", "num_samples", "samples", "sample_size", "sample_count"], kind: "range", min: "10", max: "100000", value_type: "int", log_scale: false },
	{ key: "n_jobs", aliases: ["n_jobs", "num_jobs", "jobs", "workers", "num_workers", "n_workers", "n_threads", "num_threads", "threads", "nthread", "nthreads", "n_thread"], kind: "range", min: "-1", max: "128", value_type: "int", log_scale: false },
	{ key: "num_gpus", aliases: ["num_gpus", "n_gpus", "gpus", "gpu_count"], kind: "range", min: "0", max: "8", value_type: "int", log_scale: false },
	{ key: "max_tokens", aliases: ["max_tokens", "max_new_tokens", "max_length", "max_seq_len", "max_seq_length", "seq_len", "sequence_length", "context_length", "context_window", "ctx_len", "context", "seq", "ctx", "max_seq", "max_context", "max_len"], kind: "range", min: "16", max: "8192", value_type: "int", log_scale: false },
	{ key: "temperature", aliases: ["temperature", "temp", "sampling_temperature"], kind: "range", min: "0", max: "2", value_type: "float", log_scale: false },
	{ key: "top_p", aliases: ["top_p", "nucleus", "nucleus_p", "sampling_p"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "top_k", aliases: ["top_k", "topk", "k_sampling"], kind: "range", min: "1", max: "100", value_type: "int", log_scale: false },
	{ key: "repetition_penalty", aliases: ["repetition_penalty", "rep_penalty", "repetition"], kind: "range", min: "1", max: "2", value_type: "float", log_scale: false },
	{ key: "frequency_penalty", aliases: ["frequency_penalty", "freq_penalty", "frequency_pen"], kind: "range", min: "-2", max: "2", value_type: "float", log_scale: false },
	{ key: "presence_penalty", aliases: ["presence_penalty", "pres_penalty", "presence"], kind: "range", min: "-2", max: "2", value_type: "float", log_scale: false },
	{ key: "num_beams", aliases: ["num_beams", "beams", "beam_size", "beam_width", "n_beams"], kind: "range", min: "1", max: "10", value_type: "int", log_scale: false },
	{ key: "early_stopping", aliases: ["early_stopping", "earlystop", "early_stop"], kind: "choice", values: "true,false" },
	{ key: "length_penalty", aliases: ["length_penalty", "len_penalty"], kind: "range", min: "0.1", max: "2", value_type: "float", log_scale: false },
	{ key: "no_repeat_ngram_size", aliases: ["no_repeat_ngram_size", "no_repeat_ngram", "ngram_size", "n_ngram"], kind: "range", min: "1", max: "5", value_type: "int", log_scale: false },
	{ key: "do_sample", aliases: ["do_sample", "sample", "sampling"], kind: "choice", values: "true,false" },
	{ key: "lora_rank", aliases: ["lora_rank", "lora_r", "peft_rank", "adapter_rank", "rank_lora"], kind: "range", min: "1", max: "128", value_type: "int", log_scale: false },
	{ key: "lora_alpha", aliases: ["lora_alpha", "lora_alpha_scale", "peft_alpha", "adapter_alpha"], kind: "range", min: "1", max: "128", value_type: "int", log_scale: false },
	{ key: "lora_dropout", aliases: ["lora_dropout", "lora_drop", "peft_dropout", "adapter_dropout"], kind: "range", min: "0", max: "0.5", value_type: "float", log_scale: false },
	{ key: "quantization_bits", aliases: ["quantization_bits", "quant_bits", "bits", "quantization", "precision_bits"], kind: "choice", values: "4,8,16,32" },
	{ key: "gradient_accumulation_steps", aliases: ["gradient_accumulation_steps", "grad_accum", "gradient_accumulation", "accum_steps", "grad_accum_steps"], kind: "range", min: "1", max: "64", value_type: "int", log_scale: false },
	{ key: "micro_batch_size", aliases: ["micro_batch_size", "microbatch_size", "micro_bs", "microbs"], kind: "range", min: "1", max: "64", value_type: "int", log_scale: false },
	{ key: "global_batch_size", aliases: ["global_batch_size", "train_batch_size", "effective_batch_size", "global_bs"], kind: "range", min: "8", max: "4096", value_type: "int", log_scale: false },
	{ key: "eval_batch_size", aliases: ["eval_batch_size", "validation_batch_size", "eval_bs", "val_batch_size"], kind: "range", min: "8", max: "512", value_type: "int", log_scale: false },
	{ key: "warmup_ratio", aliases: ["warmup_ratio", "warmup_fraction", "warmup_frac", "warmup_pct"], kind: "range", min: "0", max: "0.2", value_type: "float", log_scale: false },
	{ key: "min_lr_ratio", aliases: ["min_lr_ratio", "min_lr_scale", "min_lr_fraction"], kind: "range", min: "0", max: "0.1", value_type: "float", log_scale: false },
	{ key: "max_steps", aliases: ["max_steps", "train_steps", "num_train_steps", "total_steps", "n_train_steps"], kind: "range", min: "100", max: "100000", value_type: "int", log_scale: false },
	{ key: "save_steps", aliases: ["save_steps", "checkpoint_steps", "save_interval"], kind: "range", min: "50", max: "5000", value_type: "int", log_scale: false },
	{ key: "eval_steps", aliases: ["eval_steps", "evaluation_steps", "eval_interval"], kind: "range", min: "10", max: "5000", value_type: "int", log_scale: false },
	{ key: "logging_steps", aliases: ["logging_steps", "log_steps", "log_interval", "logging_interval"], kind: "range", min: "1", max: "1000", value_type: "int", log_scale: false },
	{ key: "attention_dropout", aliases: ["attention_dropout", "attn_dropout", "attention_dropout_prob", "attn_drop"], kind: "range", min: "0", max: "0.5", value_type: "float", log_scale: false },
	{ key: "hidden_dropout", aliases: ["hidden_dropout", "hidden_dropout_prob", "hidden_drop"], kind: "range", min: "0", max: "0.5", value_type: "float", log_scale: false },
	{ key: "layer_norm_eps", aliases: ["layer_norm_eps", "layer_norm_epsilon", "layernorm_eps", "ln_eps"], kind: "range", min: "0.00000001", max: "0.001", value_type: "float", log_scale: true },
	{ key: "rope_theta", aliases: ["rope_theta", "rotary_theta", "rope_base"], kind: "range", min: "1000", max: "100000", value_type: "float", log_scale: true },
	{ key: "intermediate_size", aliases: ["intermediate_size", "ffn_dim", "feed_forward_dim", "mlp_dim", "intermediate_dim"], kind: "range", min: "512", max: "16384", value_type: "int", log_scale: false },
	{ key: "vocab_size", aliases: ["vocab_size", "vocabsize", "vocabulary_size"], kind: "range", min: "1000", max: "200000", value_type: "int", log_scale: false },
	{ key: "max_position_embeddings", aliases: ["max_position_embeddings", "max_position", "max_positions", "position_size"], kind: "range", min: "128", max: "32768", value_type: "int", log_scale: false },
	{ key: "adapter_dim", aliases: ["adapter_dim", "bottleneck_dim", "adapter_size"], kind: "range", min: "1", max: "256", value_type: "int", log_scale: false },
	{ key: "prefix_length", aliases: ["prefix_length", "prompt_length", "prefix_len"], kind: "range", min: "1", max: "100", value_type: "int", log_scale: false },
	{ key: "use_lora", aliases: ["use_lora", "lora", "enable_lora"], kind: "choice", values: "true,false" },
	{ key: "freeze_layers", aliases: ["freeze_layers", "frozen_layers", "n_frozen_layers"], kind: "range", min: "0", max: "64", value_type: "int", log_scale: false },
	{ key: "unfreeze_layers", aliases: ["unfreeze_layers", "trainable_layers", "n_trainable_layers"], kind: "range", min: "0", max: "64", value_type: "int", log_scale: false },
	{ key: "num_filters", aliases: ["num_filters", "n_filters", "nfilters", "filters", "filter_count", "out_channels"], kind: "range", min: "8", max: "1024", value_type: "int", log_scale: false },
	{ key: "pool_size", aliases: ["pool_size", "poolsize", "pooling_size", "pool_dim"], kind: "range", min: "2", max: "4", value_type: "int", log_scale: false },
	{ key: "num_blocks", aliases: ["num_blocks", "n_blocks", "nblocks", "blocks", "num_res_blocks", "res_blocks", "block_count"], kind: "range", min: "1", max: "100", value_type: "int", log_scale: false },
	{ key: "normalization", aliases: ["normalization", "norm_type", "norm", "normalization_type", "norm_layer"], kind: "choice", values: "batch,layer,group,none" },
	{ key: "group_size", aliases: ["group_size", "groupsize", "num_groups", "n_groups"], kind: "range", min: "1", max: "64", value_type: "int", log_scale: false },
	{ key: "drop_path_rate", aliases: ["drop_path_rate", "stochastic_depth", "stochastic_depth_rate", "drop_path"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "label_smoothing", aliases: ["label_smoothing", "label_smooth", "smoothing"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "mixup_alpha", aliases: ["mixup_alpha", "mixup", "alpha_mixup"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "cutmix_alpha", aliases: ["cutmix_alpha", "cutmix"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "augment_prob", aliases: ["augment_prob", "augmentation_prob", "aug_prob", "augment_probability"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "patch_size", aliases: ["patch_size", "patchsize", "patch_dim"], kind: "range", min: "1", max: "64", value_type: "int", log_scale: false },
	{ key: "window_size", aliases: ["window_size", "windowsize", "window_dim"], kind: "range", min: "1", max: "64", value_type: "int", log_scale: false },
	{ key: "mlp_ratio", aliases: ["mlp_ratio", "mlp_expansion", "ffn_ratio"], kind: "range", min: "1", max: "8", value_type: "float", log_scale: false },
	{ key: "channels", aliases: ["channels", "num_channels", "in_channels", "n_channels", "channel_count"], kind: "range", min: "1", max: "1024", value_type: "int", log_scale: false },
	{ key: "num_patches", aliases: ["num_patches", "n_patches", "patches", "patch_count"], kind: "range", min: "4", max: "4096", value_type: "int", log_scale: false },
	{ key: "qkv_bias", aliases: ["qkv_bias", "use_qkv_bias", "qkv_bias_enabled"], kind: "choice", values: "true,false" },
	{ key: "layer_scale_init", aliases: ["layer_scale_init", "layer_scale"], kind: "range", min: "0", max: "0.01", value_type: "float", log_scale: false },
	{ key: "weight_init", aliases: ["weight_init", "init", "initializer", "initialization", "weight_initializer", "param_init"], kind: "choice", values: "xavier,he,normal,uniform,orthogonal,zeros" },
	{ key: "gradient_checkpointing", aliases: ["gradient_checkpointing", "grad_checkpointing", "use_gradient_checkpointing", "activation_checkpointing"], kind: "choice", values: "true,false" },
	{ key: "mixed_precision", aliases: ["mixed_precision", "amp", "autocast", "precision_mode"], kind: "choice", values: "no,fp16,bf16" },
	{ key: "device", aliases: ["device", "compute_device", "hardware"], kind: "choice", values: "cpu,cuda,mps" },
	{ key: "use_flash_attention", aliases: ["use_flash_attention", "flash_attention", "flash_attn", "attn_implementation", "attention_implementation"], kind: "choice", values: "auto,flash_attention_2,eager,sdpa" },
	{ key: "torch_dtype", aliases: ["torch_dtype", "dtype", "compute_dtype"], kind: "choice", values: "float32,float16,bfloat16,int8" },
	{ key: "load_in_4bit", param_name: "load_in_four_bit", aliases: ["load_in_4bit", "use_4bit", "q4bit", "four_bit"], kind: "choice", values: "true,false" },
	{ key: "load_in_8bit", param_name: "load_in_eight_bit", aliases: ["load_in_8bit", "use_8bit", "q8bit", "eight_bit"], kind: "choice", values: "true,false" },
	{ key: "zero_stage", aliases: ["zero_stage", "zero", "deepspeed_stage", "zero_optimization_stage"], kind: "choice", values: "0,1,2,3" },
	{ key: "gpu_memory_utilization", aliases: ["gpu_memory_utilization", "gpu_mem_util", "memory_utilization", "gpu_memory_fraction"], kind: "range", min: "0.1", max: "1", value_type: "float", log_scale: false },
	{ key: "max_model_len", aliases: ["max_model_len", "model_max_length", "model_context_length"], kind: "range", min: "128", max: "32768", value_type: "int", log_scale: false },
	{ key: "tensor_parallel_size", aliases: ["tensor_parallel_size", "tp_size", "tensor_parallel"], kind: "range", min: "1", max: "16", value_type: "int", log_scale: false },
	{ key: "pipeline_parallel_size", aliases: ["pipeline_parallel_size", "pp_size", "pipeline_parallel"], kind: "range", min: "1", max: "16", value_type: "int", log_scale: false },
	{ key: "data_parallel_size", aliases: ["data_parallel_size", "dp_size", "data_parallel"], kind: "range", min: "1", max: "64", value_type: "int", log_scale: false },
	{ key: "block_size", aliases: ["block_size", "blocksize", "kv_block_size"], kind: "range", min: "4", max: "128", value_type: "int", log_scale: false },
	{ key: "swap_space", aliases: ["swap_space", "swap_size"], kind: "range", min: "1", max: "128", value_type: "int", log_scale: false },
	{ key: "prefetch_factor", aliases: ["prefetch_factor", "prefetch"], kind: "range", min: "1", max: "16", value_type: "int", log_scale: false },
	{ key: "pin_memory", aliases: ["pin_memory", "pinned_memory"], kind: "choice", values: "true,false" },
	{ key: "dataloader_workers", aliases: ["dataloader_workers", "loader_workers", "num_dataloader_workers", "data_workers"], kind: "range", min: "0", max: "64", value_type: "int", log_scale: false },
	{ key: "packing", aliases: ["packing", "pack_sequences", "sequence_packing"], kind: "choice", values: "true,false" },
	{ key: "per_device_batch_size", aliases: ["per_device_batch_size", "batch_size_per_device", "device_batch_size"], kind: "range", min: "1", max: "64", value_type: "int", log_scale: false },
	{ key: "boosting", aliases: ["boosting", "boosting_type", "booster", "boost_type"], kind: "choice", values: "gbdt,dart,goss" },
	{ key: "tree_method", aliases: ["tree_method", "tree_construction", "tree_builder"], kind: "choice", values: "hist,exact,approx,gpu_hist" },
	{ key: "max_bin", aliases: ["max_bin", "maxbin", "n_bins", "num_bins"], kind: "range", min: "16", max: "1024", value_type: "int", log_scale: false },
	{ key: "bagging_fraction", aliases: ["bagging_fraction", "bagging_frac", "bag_frac"], kind: "range", min: "0.1", max: "1", value_type: "float", log_scale: false },
	{ key: "bagging_freq", aliases: ["bagging_freq", "bag_freq"], kind: "range", min: "0", max: "20", value_type: "int", log_scale: false },
	{ key: "min_gain_to_split", aliases: ["min_gain_to_split", "min_gain", "min_split_gain"], kind: "range", min: "0", max: "10", value_type: "float", log_scale: false },
	{ key: "lambda_l1", param_name: "lambda_l_one", aliases: ["lambda_l1", "l1_lambda", "reg_lambda_l1"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "lambda_l2", param_name: "lambda_l_two", aliases: ["lambda_l2", "l2_lambda2", "reg_lambda_l2"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "num_boost_round", aliases: ["num_boost_round", "n_boost_round", "boosting_rounds", "boost_rounds"], kind: "range", min: "10", max: "10000", value_type: "int", log_scale: false },
	{ key: "colsample_bylevel", aliases: ["colsample_bylevel", "col_sample_bylevel", "feature_fraction_bylevel"], kind: "range", min: "0.1", max: "1", value_type: "float", log_scale: false },
	{ key: "colsample_bynode", aliases: ["colsample_bynode", "col_sample_bynode", "feature_fraction_bynode"], kind: "range", min: "0.1", max: "1", value_type: "float", log_scale: false },
	{ key: "max_delta_step", aliases: ["max_delta_step", "maxdeltastep", "delta_step"], kind: "range", min: "0", max: "10", value_type: "float", log_scale: false },
	{ key: "base_score", aliases: ["base_score", "basescore", "initial_score"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "cat_smooth", aliases: ["cat_smooth", "catsmooth"], kind: "range", min: "0", max: "100", value_type: "float", log_scale: false },
	{ key: "cat_l2", param_name: "cat_l_two", aliases: ["cat_l2", "catl2"], kind: "range", min: "0", max: "100", value_type: "float", log_scale: false },
	{ key: "max_cat_threshold", aliases: ["max_cat_threshold", "maxcatthreshold", "cat_threshold"], kind: "range", min: "1", max: "100", value_type: "int", log_scale: false },
	{ key: "domain_size", aliases: ["domain_size", "domain_length", "domain_width", "domain_height", "domain_dim", "domain_extent"], kind: "range", min: "0.001", max: "1000", value_type: "float", log_scale: true },
	{ key: "pressure", aliases: ["pressure", "p_pressure", "static_pressure"], kind: "range", min: "0.000001", max: "1000000000", value_type: "float", log_scale: true },
	{ key: "flow_rate", aliases: ["flow_rate", "flowrate", "mass_flow", "volumetric_flow", "flow"], kind: "range", min: "0", max: "1000", value_type: "float", log_scale: false },
	{ key: "inlet_velocity", aliases: ["inlet_velocity", "inflow_velocity", "velocity", "flow_velocity"], kind: "range", min: "0", max: "1000", value_type: "float", log_scale: false },
	{ key: "reynolds_number", aliases: ["reynolds_number", "reynolds", "re_number", "reynolds_num"], kind: "range", min: "1", max: "100000000", value_type: "float", log_scale: true },
	{ key: "mach_number", aliases: ["mach_number", "mach", "mach_num"], kind: "range", min: "0.01", max: "10", value_type: "float", log_scale: true },
	{ key: "frequency", aliases: ["frequency", "freq", "freq_hz"], kind: "range", min: "0.001", max: "1000000", value_type: "float", log_scale: true },
	{ key: "amplitude", aliases: ["amplitude", "ampl", "amplitude_value"], kind: "range", min: "0.000001", max: "1000000", value_type: "float", log_scale: true },
	{ key: "wavelength", aliases: ["wavelength", "wave_length", "lambda_wave"], kind: "range", min: "0.000000001", max: "1000", value_type: "float", log_scale: true },
	{ key: "period", aliases: ["period", "time_period", "period_length"], kind: "range", min: "0.001", max: "1000", value_type: "float", log_scale: true },
	{ key: "mass", aliases: ["mass", "body_mass", "weight_mass"], kind: "range", min: "0.000001", max: "1000000", value_type: "float", log_scale: true },
	{ key: "stiffness", aliases: ["stiffness", "spring_constant", "spring_stiffness", "elastic_stiffness"], kind: "range", min: "0.001", max: "1000000000", value_type: "float", log_scale: true },
	{ key: "surface_tension", aliases: ["surface_tension", "sigma_surface", "tension"], kind: "range", min: "0.000001", max: "10", value_type: "float", log_scale: true },
	{ key: "contact_angle", aliases: ["contact_angle", "theta_contact", "wetting_angle"], kind: "range", min: "0", max: "180", value_type: "float", log_scale: false },
	{ key: "roughness", aliases: ["roughness", "surface_roughness", "roughness_height"], kind: "range", min: "0.000000001", max: "0.1", value_type: "float", log_scale: true },
	{ key: "porosity", aliases: ["porosity", "void_fraction", "porous_fraction"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "tortuosity", aliases: ["tortuosity", "tortuosity_factor"], kind: "range", min: "1", max: "10", value_type: "float", log_scale: false },
	{ key: "reaction_rate", aliases: ["reaction_rate", "rate_constant", "reaction_constant"], kind: "range", min: "0.000000000001", max: "1000", value_type: "float", log_scale: true },
	{ key: "activation_energy", aliases: ["activation_energy", "e_activation", "activation_energy_value"], kind: "range", min: "0.000001", max: "1000000", value_type: "float", log_scale: true },
	{ key: "thermal_expansion", aliases: ["thermal_expansion", "expansion_coefficient", "thermal_expansion_coeff"], kind: "range", min: "0.000000001", max: "0.001", value_type: "float", log_scale: true },
	{ key: "latent_heat", aliases: ["latent_heat", "heat_of_vaporization", "latent_heat_value"], kind: "range", min: "1000", max: "10000000", value_type: "float", log_scale: true },
	{ key: "melting_temperature", aliases: ["melting_temperature", "melting_point", "tm"], kind: "range", min: "1", max: "5000", value_type: "float", log_scale: false },
	{ key: "boiling_temperature", aliases: ["boiling_temperature", "boiling_point", "tb"], kind: "range", min: "1", max: "5000", value_type: "float", log_scale: false },
	{ key: "emissivity", aliases: ["emissivity", "epsilon_thermal", "emissive"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "absorptivity", aliases: ["absorptivity", "absorbance", "absorption"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "reflectivity", aliases: ["reflectivity", "reflectance", "reflection"], kind: "range", min: "0", max: "1", value_type: "float", log_scale: false },
	{ key: "refractive_index", aliases: ["refractive_index", "refraction_index", "index_of_refraction"], kind: "range", min: "1", max: "5", value_type: "float", log_scale: false },
	{ key: "permittivity", aliases: ["permittivity", "dielectric_constant", "relative_permittivity"], kind: "range", min: "1", max: "10000", value_type: "float", log_scale: false },
	{ key: "permeability", aliases: ["permeability", "magnetic_permeability", "relative_permeability"], kind: "range", min: "0.000000000001", max: "1", value_type: "float", log_scale: true },
	{ key: "heat_transfer_coeff", aliases: ["heat_transfer_coeff", "heat_transfer_coefficient", "htc", "convection_coeff"], kind: "range", min: "0.1", max: "10000", value_type: "float", log_scale: true },
	{ key: "mesh_order", aliases: ["mesh_order", "element_order", "polynomial_order", "fe_order"], kind: "range", min: "1", max: "5", value_type: "int", log_scale: false },
	{ key: "n_modes", aliases: ["n_modes", "num_modes", "modes", "mode_count"], kind: "range", min: "1", max: "1000", value_type: "int", log_scale: false }
];

var _run_program_keyword_suggestion_map = Object.create(null);
RUN_PROGRAM_KEYWORD_SUGGESTIONS.forEach(function (definition) {
	definition.aliases.forEach(function (alias) {
		_run_program_keyword_suggestion_map[alias.toLowerCase()] = definition;
	});
});

function get_run_program_placeholders(run_program_string) {
	var pattern = /(?:\$|%)\(([a-zA-Z_][a-zA-Z0-9_]*)\)|(?:\$|%)([a-zA-Z_][a-zA-Z0-9_]*)/g;
	var names = [];
	var match;
	while ((match = pattern.exec(run_program_string)) !== null) {
		var previous = match.index > 0 ? run_program_string.charAt(match.index - 1) : "";
		if (previous === "\\" || (match[0].charAt(0) === "%" && previous === "%") || (match[0].charAt(0) === "$" && previous === "$")) {
			continue;
		}
		var name = match[1] || match[2];
		if (name) {
			names.push(name);
		}
	}
	return names;
}

var _run_program_keyword_fuzzy_cache = Object.create(null);
var _run_program_keyword_fuzzy_by_length = Object.create(null);
var _run_program_keyword_fuzzy_by_first = Object.create(null);
var _run_program_fuzzy_deadline = 0;
var _run_program_keyword_parameter_name_cache = Object.create(null);

RUN_PROGRAM_KEYWORD_SUGGESTIONS.forEach(function (definition) {
	definition.aliases.forEach(function (alias) {
		var lower = alias.toLowerCase();
		if (lower.length < 6) return;
		var entry = { alias: lower, definition: definition, length: lower.length, first: lower.charAt(0) };
		var lengthBucket = _run_program_keyword_fuzzy_by_length[lower.length];
		if (!lengthBucket) {
			lengthBucket = _run_program_keyword_fuzzy_by_length[lower.length] = [];
		}
		lengthBucket.push(entry);
		var firstBucket = _run_program_keyword_fuzzy_by_first[lower.charAt(0)];
		if (!firstBucket) {
			firstBucket = _run_program_keyword_fuzzy_by_first[lower.charAt(0)] = Object.create(null);
		}
		var firstLengthBucket = firstBucket[lower.length];
		if (!firstLengthBucket) {
			firstLengthBucket = firstBucket[lower.length] = [];
		}
		firstLengthBucket.push(entry);
	});
});

var _run_program_keyword_distance_buffer_a = new Array(128);
var _run_program_keyword_distance_buffer_b = new Array(128);
var _run_program_keyword_distance_buffer_c = new Array(128);

function _run_program_keyword_edit_distance(a, b, maxDistance) {
	var aLength = a.length;
	var bLength = b.length;
	if (a === b) return 0;
	var limit = maxDistance === undefined ? Math.max(aLength, bLength) : maxDistance;
	if (Math.abs(aLength - bLength) > limit) return limit + 1;
	if (limit === 0) return 1;
	if (bLength >= _run_program_keyword_distance_buffer_a.length) return limit + 1;

	var infinity = limit + 1;
	var prevPrev = _run_program_keyword_distance_buffer_a;
	var prev = _run_program_keyword_distance_buffer_b;
	var curr = _run_program_keyword_distance_buffer_c;
	var prevPrevStart = 0;
	var prevPrevEnd = -1;
	var prevStart = 0;
	var prevEnd = Math.min(bLength, limit);
	for (var j = 0; j <= prevEnd; j++) {
		prev[j] = j;
	}

	for (var ai = 1; ai <= aLength; ai++) {
		var start = Math.max(1, ai - limit);
		var end = Math.min(bLength, ai + limit);
		curr[0] = ai <= limit ? ai : infinity;
		if (start > 1) {
			curr[start - 1] = infinity;
		}
		var rowMinimum = curr[0];
		for (var bi = start; bi <= end; bi++) {
			var cost = a.charCodeAt(ai - 1) === b.charCodeAt(bi - 1) ? 0 : 1;
			var deletion = (bi >= prevStart && bi <= prevEnd ? prev[bi] : infinity) + 1;
			var insertion = curr[bi - 1] + 1;
			var substitution = (bi - 1 >= prevStart && bi - 1 <= prevEnd ? prev[bi - 1] : infinity) + cost;
			var value = deletion < insertion ? deletion : insertion;
			if (substitution < value) value = substitution;
			if (ai > 1 && bi > 1 && a.charCodeAt(ai - 1) === b.charCodeAt(bi - 2) && a.charCodeAt(ai - 2) === b.charCodeAt(bi - 1)) {
				var transposition = (bi - 2 >= prevPrevStart && bi - 2 <= prevPrevEnd ? prevPrev[bi - 2] : infinity) + cost;
				if (transposition < value) value = transposition;
			}
			curr[bi] = value;
			if (value < rowMinimum) rowMinimum = value;
		}
		if (rowMinimum > limit) return infinity;

		var spare = prevPrev;
		prevPrev = prev;
		prev = curr;
		curr = spare;
		prevPrevStart = 0;
		prevPrevEnd = prevEnd;
		prevStart = 0;
		prevEnd = end;
	}

	return bLength >= prevStart && bLength <= prevEnd ? prev[bLength] : infinity;
}

function _find_fuzzy_run_program_keyword_match(name, deadline) {
	var token = String(name || "").toLowerCase();
	if (token.length < 4 || token.length > 40 || /[0-9]/.test(token)) return null;
	if (Object.prototype.hasOwnProperty.call(_run_program_keyword_fuzzy_cache, token)) {
		return _run_program_keyword_fuzzy_cache[token];
	}
	if (deadline && Date.now() > deadline) return null;

	var maxDistance = token.length >= 8 ? 2 : 1;
	var bestDistance = maxDistance + 1;
	var bestEntry = null;
	var ambiguous = false;
	var aborted = false;
	var first = token.charAt(0);

	function consider(entry) {
		var lengthDifference = entry.length - token.length;
		if (lengthDifference < -maxDistance || lengthDifference > maxDistance) return;
		var distance = _run_program_keyword_edit_distance(token, entry.alias, maxDistance);
		if (distance > maxDistance) return;
		if (distance < bestDistance) {
			bestDistance = distance;
			bestEntry = entry;
			ambiguous = false;
		} else if (distance === bestDistance && bestEntry && entry.definition !== bestEntry.definition) {
			ambiguous = true;
		}
	}

	function expired() {
		return deadline && Date.now() > deadline;
	}

	var firstBuckets = _run_program_keyword_fuzzy_by_first[first];
	if (firstBuckets) {
		for (var length = token.length - maxDistance; length <= token.length + maxDistance; length++) {
			var bucket = firstBuckets[length];
			if (!bucket) continue;
			for (var i = 0; i < bucket.length; i++) {
				if ((i & 15) === 0 && expired()) {
					aborted = true;
					break;
				}
				consider(bucket[i]);
			}
			if (aborted) break;
		}
	}

	if (!aborted && (!bestEntry || bestDistance > 1)) {
		for (var fallbackLength = token.length - maxDistance; fallbackLength <= token.length + maxDistance; fallbackLength++) {
			var fallbackBucket = _run_program_keyword_fuzzy_by_length[fallbackLength];
			if (!fallbackBucket) continue;
			for (var j = 0; j < fallbackBucket.length; j++) {
				if ((j & 15) === 0 && expired()) {
					aborted = true;
					break;
				}
				if (fallbackBucket[j].first !== first) {
					consider(fallbackBucket[j]);
				}
			}
			if (aborted) break;
		}
	}

	if (aborted) return null;

	var result = ambiguous ? null : (bestEntry ? { definition: bestEntry.definition, alias: bestEntry.alias } : null);
	_run_program_keyword_fuzzy_cache[token] = result;
	return result;
}

function find_run_program_keyword_suggestion_match(name, allow_fuzzy, deadline) {
	var lower = String(name || "").toLowerCase();
	var exact = _run_program_keyword_suggestion_map[lower];
	if (exact) {
		return { definition: exact, alias: lower };
	}
	if (allow_fuzzy === false) return null;
	return _find_fuzzy_run_program_keyword_match(name, deadline);
}

function find_run_program_keyword_suggestion(name) {
	var match = find_run_program_keyword_suggestion_match(name);
	return match ? match.definition : null;
}

function _escape_run_program_suggestion_attr(value) {
	return String(value).replace(/&/g, "&amp;").replace(/'/g, "&#39;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function _run_program_suggestions_signature(suggestions) {
	return suggestions.map(function (suggestion) {
		var definition = suggestion.definition || {};
		return [
			suggestion.name,
			suggestion.parameter_name || suggestion.name,
			definition.key,
			definition.kind,
			definition.min,
			definition.max,
			definition.value_type,
			definition.log_scale ? "1" : "0",
			definition.values,
			definition.value,
			suggestion.target_exists ? "1" : "0"
		].join(":");
	}).join("|");
}

function _render_run_program_suggestions(suggestions) {
	var $box = $("#run_program_suggestions");
	if (!$box.length) return;

	suggestions = suggestions || [];
	var signature = _run_program_suggestions_signature(suggestions);
	if ($box.attr("data-suggestion-signature") === signature) return;
	$box.attr("data-suggestion-signature", signature);

	if (!suggestions.length) {
		$box.hide().empty();
		return;
	}
	var html = "<div class='run_program_suggestions_title'>Suggested parameters for detected placeholders</div>";
	html += "<div class='run_program_suggestion_list'>";
	suggestions.forEach(function (suggestion) {
		var definition = suggestion.definition;
		var parameter_name = suggestion.parameter_name || suggestion.name;
		var display_name = parameter_name === suggestion.name ? suggestion.name : suggestion.name + " → " + parameter_name;
		var add_title = suggestion.target_exists ? "Use existing parameter and fix placeholder" : "Add parameter";
		html += "<div class='run_program_suggestion_chip' data-param-name='" + _escape_run_program_suggestion_attr(suggestion.name) + "'>";
		html += "<div class='run_program_suggestion_head'>";
		html += "<span class='run_program_suggestion_name'>+ " + _escape_run_program_suggestion_attr(display_name) + "</span>";
		html += "<button type='button' class='run_program_suggestion_add' title='" + add_title + "' aria-label='" + add_title + "' data-param-name='" + _escape_run_program_suggestion_attr(suggestion.name) + "' onmousedown='_apply_run_program_suggestion(this); return false;' onclick='_apply_run_program_suggestion(this)'>&#10003;</button>";
		if (suggestion.target_exists) {
			html += "<button type='button' class='run_program_suggestion_add_new' title='Add as new parameter' aria-label='Add as new parameter' data-param-name='" + _escape_run_program_suggestion_attr(suggestion.name) + "' onmousedown='_apply_run_program_suggestion_as_new(this); return false;' onclick='_apply_run_program_suggestion_as_new(this)'>+ new</button>";
		}
		html += "</div>";
		html += "<div class='run_program_suggestion_fields'>";
		if (definition.kind === "choice") {
			html += "<label><span>values</span><input type='text' data-suggestion-field='values' value='" + _escape_run_program_suggestion_attr(definition.values) + "'></label>";
		} else if (definition.kind === "fixed") {
			html += "<label><span>value</span><input type='text' data-suggestion-field='value' value='" + _escape_run_program_suggestion_attr(definition.value) + "'></label>";
		} else {
			html += "<label><span>min</span><input type='number' step='any' data-suggestion-field='min' value='" + _escape_run_program_suggestion_attr(definition.min) + "'></label>";
			html += "<label><span>max</span><input type='number' step='any' data-suggestion-field='max' value='" + _escape_run_program_suggestion_attr(definition.max) + "'></label>";
			html += "<label><span>type</span><select data-suggestion-field='value_type'>";
			html += "<option value='float'" + (definition.value_type === "float" ? " selected" : "") + ">float</option>";
			html += "<option value='int'" + (definition.value_type === "int" ? " selected" : "") + ">int</option>";
			html += "</select></label>";
			html += "<label class='run_program_suggestion_log'><input type='checkbox' data-suggestion-field='log_scale'" + (definition.log_scale ? " checked" : "") + "> log</label>";
		}
		html += "</div>";
		html += "</div>";
	});
	html += "</div>";
	$box.html(html).show();
}

function _run_program_suggestion_edited_definition(button, definition) {
	var edited = {};
	Object.keys(definition).forEach(function (key) {
		edited[key] = definition[key];
	});
	var $card = $(button).closest(".run_program_suggestion_chip");
	if (!$card.length) return edited;

	if (definition.kind === "choice") {
		var values = $card.find("[data-suggestion-field='values']").val();
		if (values !== undefined && String(values).trim() !== "") edited.values = values;
	} else if (definition.kind === "fixed") {
		var fixed_value = $card.find("[data-suggestion-field='value']").val();
		if (fixed_value !== undefined && String(fixed_value).trim() !== "") edited.value = fixed_value;
	} else {
		var min = $card.find("[data-suggestion-field='min']").val();
		var max = $card.find("[data-suggestion-field='max']").val();
		var valueType = $card.find("[data-suggestion-field='value_type']").val();
		if (min !== undefined && String(min).trim() !== "") edited.min = min;
		if (max !== undefined && String(max).trim() !== "") edited.max = max;
		if (valueType) edited.value_type = valueType;
		edited.log_scale = $card.find("[data-suggestion-field='log_scale']").is(":checked");
	}
	return edited;
}

function _run_program_suggestion_parameter_name(name, definition, matched_alias) {
	var token = String(name || "").toLowerCase();
	var cache_key = token + "|" + (definition ? definition.key : "");
	if (Object.prototype.hasOwnProperty.call(_run_program_keyword_parameter_name_cache, cache_key)) {
		return _run_program_keyword_parameter_name_cache[cache_key];
	}

	var alias = String(matched_alias || name || "").toLowerCase();
	var preferred = definition ? String(definition.preferred_name || definition.param_name || definition.key || "").toLowerCase() : "";
	var token_is_valid = /^[a-zA-Z_]+$/.test(name);
	var preferred_is_valid = /^[a-zA-Z_]+$/.test(preferred);
	var result = null;

	if (!token_is_valid) {
		if (preferred_is_valid) {
			result = preferred;
		} else if (alias && /^[a-zA-Z_]+$/.test(alias)) {
			result = alias;
		}
	} else {
		var maxDistance = token.length >= 8 ? 2 : 1;
		if (preferred_is_valid && preferred !== token && Math.abs(preferred.length - token.length) <= maxDistance && _run_program_keyword_edit_distance(token, preferred, maxDistance) <= maxDistance) {
			result = preferred;
		} else if (alias && /^[a-zA-Z_]+$/.test(alias) && alias !== token && Math.abs(alias.length - token.length) <= maxDistance && _run_program_keyword_edit_distance(token, alias, maxDistance) <= maxDistance) {
			result = alias;
		} else {
			result = token;
		}
	}

	_run_program_keyword_parameter_name_cache[cache_key] = result;
	return result;
}

function _run_program_definition_with_current_values(definition, parameter_info) {
	var merged = {};
	Object.keys(definition).forEach(function (key) {
		merged[key] = definition[key];
	});
	if (!parameter_info) return merged;

	if (parameter_info.kind === "range") {
		merged.kind = "range";
		if (parameter_info.min !== "" && parameter_info.min != null) merged.min = parameter_info.min;
		if (parameter_info.max !== "" && parameter_info.max != null) merged.max = parameter_info.max;
		if (parameter_info.type) merged.value_type = parameter_info.type;
		merged.log_scale = !!parameter_info.log_scale;
	} else if (parameter_info.kind === "choice") {
		merged.kind = "choice";
		if (parameter_info.values !== "" && parameter_info.values != null) merged.values = parameter_info.values;
	} else if (parameter_info.kind === "fixed") {
		merged.kind = "fixed";
		if (parameter_info.value != "" && parameter_info.value != null) merged.value = parameter_info.value;
	}
	return merged;
}

function _run_program_sanitize_parameter_name(name) {
	var digit_names = { "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine" };
	var sanitized = String(name || "").replace(/[^a-zA-Z0-9_]/g, "_");
	sanitized = sanitized.replace(/[0-9]/g, function (digit) {
		return "_" + digit_names[digit] + "_";
	});
	sanitized = sanitized.replace(/_+/g, "_").replace(/^_+|_+$/g, "");
	if (!/^[a-zA-Z_]/.test(sanitized)) {
		sanitized = "param_" + sanitized;
	}
	return sanitized;
}

function _replace_run_program_placeholder(value, name, parameter_name) {
	var escaped_name = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	var boundary = "(?![A-Za-z0-9_])";
	value = value.replace(new RegExp("%\\(" + escaped_name + "\\)", "g"), "%(" + parameter_name + ")");
	value = value.replace(new RegExp("\\$\\(" + escaped_name + "\\)", "g"), "$(" + parameter_name + ")");
	value = value.replace(new RegExp("%" + escaped_name + boundary, "g"), "%(" + parameter_name + ")");
	value = value.replace(new RegExp("\\$" + escaped_name + boundary, "g"), "$(" + parameter_name + ")");
	return value;
}

function _ensure_run_program_placeholder(name, parameter_name) {
	var $run_program = $("#run_program");
	if (!$run_program.length) return;

	var target_name = parameter_name || name;
	var value = $run_program.val() || "";
	var original_value = value;

	if (name !== target_name) {
		value = _replace_run_program_placeholder(value, name, target_name);
	}

	if (get_run_program_placeholders(value).indexOf(target_name) === -1) {
		if (value.trim() === "") {
			value = "--" + target_name + "=%(" + target_name + ")";
		} else {
			value = value.replace(/\s+$/, "") + " --" + target_name + "=%(" + target_name + ")";
		}
	}

	if (value !== original_value) {
		_run_program_history_flush();
		$run_program.val(value).trigger("input").trigger("change");
	}
}

function _run_program_parameter_exists(name) {
	var exists = false;
	$(".parameterName").each(function () {
		if ($(this).val() === name) {
			exists = true;
			return false;
		}
	});
	return exists;
}

function _add_run_program_parameter(name, definition) {
	if (_run_program_parameter_exists(name)) return $();

	var kind = definition ? definition.kind : "range";
	var $target = $();
	$(".parameterRow").each(function () {
		var value = $(this).find(".parameterName").val();
		if (!value || value.trim() === "") {
			$target = $(this);
			return false;
		}
	});
	if (!$target.length) {
		$("#main_add_row_button").trigger("click");
		$target = $(".parameterRow").last();
	}
	if (!$target.length) return $();

	$target.find(".parameterName").val(name);
	$target.find(".optionSelect").val(kind).trigger("change");

	if (kind === "range") {
		$target.find(".minValue").val(definition ? definition.min : "");
		$target.find(".maxValue").val(definition ? definition.max : "");
		$target.find(".numberTypeSelect").val(definition ? definition.value_type : "float");
		$target.find(".log_scale").prop("checked", definition ? !!definition.log_scale : false);
	} else if (kind === "choice") {
		$target.find(".choiceValues").val(definition ? definition.values : "");
	} else if (kind === "fixed") {
		$target.find(".fixedValue").val(definition ? definition.value : "");
	}

	return $target;
}

function _apply_run_program_suggestion(button) {
	var name = $(button).attr("data-param-name");
	var match = find_run_program_keyword_suggestion_match(name);
	if (!match) return;

	var parameter_name = _run_program_suggestion_parameter_name(name, match.definition, match.alias);
	if (!parameter_name) return;

	var definition = _run_program_suggestion_edited_definition(button, match.definition);
	_add_run_program_parameter(parameter_name, definition);
	_ensure_run_program_placeholder(name, parameter_name);
	update_command();
}

function _run_program_unique_parameter_name(name) {
	var candidate = String(name || "").toLowerCase();
	if (!/^[a-zA-Z_]+$/.test(candidate)) {
		candidate = _run_program_sanitize_parameter_name(candidate);
	}
	if (!candidate) return null;
	if (typeof is_invalid_parameter_name === "function" && is_invalid_parameter_name(candidate)) {
		candidate = "param_" + candidate;
	}
	if (!_run_program_parameter_exists(candidate)) return candidate;

	var base = candidate;
	candidate = base + "_new";
	var index = 2;
	while (_run_program_parameter_exists(candidate)) {
		candidate = base + "_new" + index;
		index++;
	}
	return candidate;
}

function _apply_run_program_suggestion_as_new(button) {
	var name = $(button).attr("data-param-name");
	var match = find_run_program_keyword_suggestion_match(name);
	if (!match) return;

	var parameter_name = _run_program_unique_parameter_name(name);
	if (!parameter_name) return;

	var definition = _run_program_suggestion_edited_definition(button, match.definition);
	_add_run_program_parameter(parameter_name, definition);
	_ensure_run_program_placeholder(name, parameter_name);
	update_command();
}

function _add_missing_run_program_parameter(button) {
	var name = $(button).attr("data-param-name");
	if (!name) return;

	var parameter_name = _run_program_suggestion_parameter_name(name, null) || _run_program_sanitize_parameter_name(name);
	if (!parameter_name) return;

	if (typeof is_invalid_parameter_name === "function" && is_invalid_parameter_name(parameter_name)) {
		parameter_name = "param_" + parameter_name;
	}

	var $row = _add_run_program_parameter(parameter_name, null);
	_ensure_run_program_placeholder(name, parameter_name);
	update_command();

	if ($row.length) {
		setTimeout(function () {
			$row.find(".minValue").trigger("focus");
		}, 0);
	}
}

var RUN_PROGRAM_HISTORY_LIMIT = 200;
var RUN_PROGRAM_HISTORY_DEBOUNCE_MS = 500;
var _run_program_history = { states: [], index: -1, timer: null, suppress: false, ready: false };

function _run_program_history_value() {
	return $("#run_program").val() || "";
}

function _run_program_history_clear_timer() {
	if (_run_program_history.timer) {
		clearTimeout(_run_program_history.timer);
		_run_program_history.timer = null;
	}
}

function _run_program_history_record(value) {
	if (!_run_program_history.ready) return;
	value = String(value == null ? "" : value);
	var states = _run_program_history.states;
	var index = _run_program_history.index;
	if (index >= 0 && states[index] === value) return;
	if (index >= 0 && index + 1 < states.length && states[index + 1] === value) {
		_run_program_history.index = index + 1;
		return;
	}
	states = states.slice(0, index + 1);
	states.push(value);
	if (states.length > RUN_PROGRAM_HISTORY_LIMIT) {
		states.shift();
	}
	_run_program_history.states = states;
	_run_program_history.index = states.length - 1;
}

function _run_program_history_schedule() {
	if (!_run_program_history.ready) return;
	_run_program_history_clear_timer();
	_run_program_history.timer = setTimeout(function () {
		_run_program_history.timer = null;
		_run_program_history_record(_run_program_history_value());
	}, RUN_PROGRAM_HISTORY_DEBOUNCE_MS);
}

function _run_program_history_flush() {
	if (!_run_program_history.ready) return;
	_run_program_history_clear_timer();
	_run_program_history_record(_run_program_history_value());
}

function _run_program_history_apply(value) {
	var $run_program = $("#run_program");
	if (!$run_program.length) return;
	_run_program_history.suppress = true;
	try {
		$run_program.val(value).trigger("input").trigger("change");
		var element = $run_program[0];
		if (element && typeof element.setSelectionRange === "function") {
			var length = String(value).length;
			element.setSelectionRange(length, length);
		}
	} finally {
		_run_program_history.suppress = false;
	}
}

function _run_program_history_undo() {
	if (!_run_program_history.ready) return;
	_run_program_history_clear_timer();
	var current = _run_program_history_value();
	if (_run_program_history.index < 0) {
		_run_program_history.states = [current];
		_run_program_history.index = 0;
	}
	if (_run_program_history.states[_run_program_history.index] !== current) {
		_run_program_history_record(current);
	}
	if (_run_program_history.index <= 0) return;
	_run_program_history.index--;
	_run_program_history_apply(_run_program_history.states[_run_program_history.index]);
}

function _run_program_history_redo() {
	if (!_run_program_history.ready) return;
	_run_program_history_clear_timer();
	var current = _run_program_history_value();
	if (_run_program_history.states[_run_program_history.index] !== current) {
		_run_program_history_record(current);
		return;
	}
	if (_run_program_history.index >= _run_program_history.states.length - 1) return;
	_run_program_history.index++;
	_run_program_history_apply(_run_program_history.states[_run_program_history.index]);
}

function _run_program_history_shortcut(event) {
	var key = "";
	if (event.key) {
		key = String(event.key).toLowerCase();
	} else if (event.keyCode === 90) {
		key = "z";
	} else if (event.keyCode === 89) {
		key = "y";
	}
	if (!(event.ctrlKey || event.metaKey) || event.altKey) return null;
	if (key === "z") return event.shiftKey ? "redo" : "undo";
	if (key === "y") return "redo";
	return null;
}

function _run_program_history_keydown(event) {
	var action = _run_program_history_shortcut(event);
	if (!action) return;
	event.preventDefault();
	if (action === "undo") {
		_run_program_history_undo();
	} else {
		_run_program_history_redo();
	}
}

function setup_run_program_history() {
	var $run_program = $("#run_program");
	if (!$run_program.length) return;

	_run_program_history_clear_timer();
	_run_program_history.states = [_run_program_history_value()];
	_run_program_history.index = 0;
	_run_program_history.suppress = false;
	_run_program_history.ready = true;

	$run_program.off(".run_program_history");
	$run_program.on("input.run_program_history", function () {
		if (_run_program_history.suppress) return;
		_run_program_history_schedule();
	});
	$run_program.on("change.run_program_history", function () {
		if (_run_program_history.suppress) return;
		_run_program_history_flush();
	});
	$run_program.on("blur.run_program_history", function () {
		if (_run_program_history.suppress) return;
		_run_program_history_flush();
	});
	$run_program.on("keydown.run_program_history", _run_program_history_keydown);
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
			$("#" + item.id + "_error").html(`<img src='i/warning.svg' style='height: 1em' /> ${this_error}`).show()
		} else {
			$("#" + item.id + "_error").html("").hide()
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
			$("#time_error").html("").hide()
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
			$("#worker_timeout_error").html("").hide()
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
	} else if (item.id == "modules" || item.id == "venv") {
		value = (value || "").trim();
		if (value !== "") {
			command += " --" + item.id + "='" + value.replace(/'/g, "'\\''") + "'";
		}
		$("#" + item.id + "_error").html("").hide();
	} else if (item.id == "external_generator") {
		command += " --" + item.id + "='" + encode_base64(value) + "'";
	} else if (item.id == "run_program_once") {
		command += " --" + item.id + "='" + encode_base64(value) + "'";
	} else if (item.id == "run_program") {
		// When a formula is active, the run_program is auto-generated from
		// it, so the placeholder-vs-parameter cross-check is misleading.
		// Skip it entirely in that case so the scientist can leave an
		// old run_program around without seeing red.
		var formulaIsActive = ($("#formula").val() || "").trim() !== "" || (value || "").includes("run_with_formula.py");
		if (!formulaIsActive) {
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

			var current_parameter_info = get_current_parameter_info();
			var new_errors = [];
			var suggestions = [];
			var suggested_names = Object.create(null);

			for (var k = 0; k < variables_in_run_program.length; k++) {
				var test_this_var_name = variables_in_run_program[k];

				if(!existing_parameter_names.includes(test_this_var_name)) {
					var suggestion_match = find_run_program_keyword_suggestion_match(test_this_var_name, true, _run_program_fuzzy_deadline);
					var suggestion_definition = suggestion_match ? suggestion_match.definition : null;
					var parameter_name = suggestion_match ? _run_program_suggestion_parameter_name(test_this_var_name, suggestion_match.definition, suggestion_match.alias) : null;
					if (suggestion_definition && parameter_name) {
						var target_exists = existing_parameter_names.includes(parameter_name);
						if (target_exists) {
							suggestion_definition = _run_program_definition_with_current_values(suggestion_definition, current_parameter_info[parameter_name]);
						}
						if (!suggested_names[test_this_var_name]) {
							suggested_names[test_this_var_name] = true;
							suggestions.push({ name: test_this_var_name, parameter_name: parameter_name, definition: suggestion_definition, target_exists: target_exists });
						}
					} else {
						var add_parameter_button = `<button type='button' class='run_program_error_add_parameter' data-param-name='${test_this_var_name}' onmousedown='_add_missing_run_program_parameter(this); return false;' onclick='_add_missing_run_program_parameter(this)'>Add to parameters</button>`;
						var err_msg = `<img src='i/warning.svg' style='height: 1em' /> <code>%(${test_this_var_name})</code> not in existing defined parameters. ${add_parameter_button}`;
						new_errors.push(err_msg);
					}
				}
			}

			for (var k = 0; k < existing_parameter_names.length; k++) {
				var test_this_var_name = existing_parameter_names[k];

				if(!variables_in_run_program.includes(test_this_var_name)) {
					var err_msg = `<img src='i/warning.svg' style='height: 1em' /> <code>%(${test_this_var_name})</code> is defined but not used.`;
					new_errors.push(err_msg);
				}
			}

			_render_run_program_suggestions(suggestions);

			if(new_errors.length) {
				$("#run_program_error").html(string_or_array_to_list(new_errors)).show()
				errors.push(...new_errors);
			} else {
				$("#run_program_error").html("").hide()
			}
		} else {
			$("#run_program_error").html("").hide()
			_render_run_program_suggestions([]);
		}

		value = encode_base64(value);

		command += " --" + item.id + "='" + value + "'";
		$("#" + item.id).css("background-color", "");
	} else {
		if(!errors.length) {
			if (item.id != "constraints") {
				if (item.id == "result_names") {
					value = value.replace(/\s+/g, ' ').trim()
					command += ` --${item.id} '${value}'`
				} else {
					command += " --" + item.id + "=" + value;
				}
				$("#" + item.id + "_error").html("").hide()
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
	_run_program_fuzzy_deadline = Date.now() + 5;
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

	var formulaHandledIds = new Set(["formula", "formula_mode"]);
	function processTableData(_tableData) {
		_tableData.forEach(function(item) {
			if (formulaHandledIds.has(item.id)) return;
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

	// Custom handling for the formula editor (lives outside tableData).
	(function handleFormula() {
		var formulaText = ($("#formula").val() || "").trim();
		var formulaMode = ($("#formula_mode").val() || "auto");
		if (formulaText) {
			try {
				var b64 = btoa(unescape(encodeURIComponent(formulaText)));
				command += " --formula='" + b64 + "'";
				if (formulaMode && formulaMode !== "auto") {
					command += " --formula_mode=" + formulaMode;
				}
			} catch (e) {
				console.error("Base64 encoding failed for formula:", e);
			}
		}
	})();

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
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>maxValue</i> for parameter <i>" + parameterName + "</i> is empty or not a number.");
					is_ok = false;
				}

				if(minValue == "") {
					warn_msg.push("<img src='i/warning.svg' style='height: 1em' /><i>minValue</i> for parameter <i>" + parameterName + "</i> is empty or not a number.");
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
			$($(".parameterError")[i]).html(string_or_array_to_list(warn_msg)).show()
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
			command += " --experiment_constraints '" + encode_base64(add_equation_spaces(_constraints[r])) + "'";
		}

		var constraints_string = $("#constraints").val();
		var errors_string = is_valid_constraints_string(constraints_string);

		if(isAnyLogScaleSet()) {
			errors_string += "Cannot set constraints if one or more log-scale parameters are there."
		}

		if (errors_string != "") {
			errors.push("Something was wrong in the constraints parameter");
			$("#constraints_error").html(errors_string).show()
		} else {
			$("#constraints_error").html("").hide()
		}
	} else {
		$("#constraints_error").html("");
	}

	var errors_visible = false;
	$(".parameterError").each(function (i, e) {
		if(is_visible($(e))) {
			errors_visible = true;
		}
	});

	// Cross-field: either Run program OR the formula must be filled.
	var rp = ($("#run_program").val() || "").trim();
	var fm = ($("#formula").val() || "").trim();
	// Highlight the offending field(s) in red so the user immediately
	// sees which one they need to fill.  The class is removed as soon
	// as the field has content, so the indicator is reactive.
	$("#run_program").toggleClass("field_missing", !rp && !fm);
	$("#formula").toggleClass("field_missing", !rp && !fm);
	$("#formula_pane_text").toggleClass("field_missing", !rp && !fm);
	if (!rp && !fm) {
		errors.push("<img src='i/warning.svg' style='height: 1em' /> Either <i>Run program</i> or the <i>Formula editor</i> must be filled.");
	}

	if (!errors.length && $(".optionSelect").length && !errors_visible) {
		var base_url = location.protocol + "//" + location.host + "/" + location.pathname + "/";

		base_url = base_url.replaceAll(/\/+/g, "/");

		base_url = base_url.replace(/^http:\//, "http://");
		base_url = base_url.replace(/^https:\//, "https://");
		base_url = base_url.replace(/^file:\//, "file://");

		var base_64_string = encode_base64(command);

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

		$("#curl_command_highlighted").text(curl_command).show();
		$("#command_element_highlighted").text(nicer_command).show();
		$("#install_and_run").show();
		$("#only_run").show();

		$("#curl_command").text(curl_command);
		$("#command_element").text(nicer_command);
	} else {
		$("#command_element_highlighted").hide();
		$("#curl_command_highlighted").hide();
		$("#install_and_run").hide();
		$("#only_run").hide();

		$("#curl_command").text("");
		$("#command_element").text("");
	}

	show_warnings_and_errors(warnings, errors);

	update_url();

	_schedule_formula_preview_refresh();
}

function encode_base64 (v) {
	return btoa(v);
}

function decode_base64 (input) {
	decoded = atob(input);

	return decoded;
}

function is_base64_like(s) {
	// Only base64-decode strings that consist entirely of base64
	// characters.  This protects against the recursive re-decode
	// path catching ``$(echo ...`` (the previous output of this
	// function) and trying to decode it.
	return typeof s === "string" && s.length >= 4 && /^[A-Za-z0-9+/]+={0,2}$/.test(s);
}

function addBase64DecodedVersions(cmdString) {
	return cmdString.replace(/(--[a-zA-Z0-9_]+)=('([^']+)'|"([^"]+)"|([^\s]+))/g, (match, key, _, singleQuoted, doubleQuoted, bare) => {
		const value = singleQuoted || doubleQuoted || bare;

		let decoded = null;
		try {
			if (
				(key === "--run_program" || key === "--run_program_once" || key === "--formula") &&
				is_base64_like(value)
			) {
				decoded = decode_base64(value);
			}
		} catch (e) {
			console.error(e);
		}

		if (decoded) {
			var safeDecoded = decoded.replace(/\x27/g, `'\\''`).trim();
			return ` ${key}=$(printf '%s' '${safeDecoded}' | base64 -w0)`;
		} else {
			return match;
		}
	});
}

// ---------------------------------------------------------------------------
// Formula card (left: tabs + textarea, right: live parameter suggestions)
// ---------------------------------------------------------------------------

function build_formula_card_html() {
	return (
		"<div style='display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px;'>" +
		"<div id='formula_card_left' style='flex: 1 1 380px; min-width: 320px;'>" +
		"<div id='formula_panel_text'>" +
		"<textarea id='formula_pane_text' placeholder=\"f(x, y) = \\frac{x}{y} + \\sin(a)   or   f(x) = 2*x + y\" style='width: 100%; min-height: 80px; font-family: monospace;'></textarea>" +
		"</div>" +
		"<div id='formula_preview' style='position: relative; margin-top: 8px; padding: 8px 8px 16px 8px; background: #fff; border: 1px dashed #c0c0d0; border-radius: 8px; min-height: 50px; font-size: 1.05em; overflow: visible;'></div>" +
		"<div id='formula_param_legend' style='margin-top: 4px; font-size: 0.82em; color: #444; min-height: 1.2em;'></div>" +
		"<div id='formula_error' style='margin-top: 4px; font-size: 0.85em; color: #b00020;'></div>" +
		"</div>" +
		"<div id='formula_card_right' style='flex: 0 0 380px; min-width: 320px; padding: 10px; background: #fafaff; border: 1px solid #d6d6e6; border-radius: 10px;'>" +
		"<h4 style='margin-top: 0; margin-bottom: 6px;'>Suggested parameters</h4>" +
		"<div id='formula_suggestions' style='font-size: 0.9em; color: #444;'><em>Type a formula to see suggestions…</em></div>" +
		"<div style='margin-top: 10px; display: flex; gap: 6px;'>" +
		"<button type='button' id='formula_apply_btn' class='formula_tab' style='background:#4a90d9;color:#fff;'>Apply → add as parameters</button>" +
		"<button type='button' id='formula_clear_btn' class='formula_tab'>Clear formula</button>" +
		"</div>" +
		"<div style='margin-top: 10px; font-size: 0.8em; color: #777;'>" +
		"<b>Parameters</b> are variables that appear on the left-hand side of the formula (e.g. <code>x</code> in <code>f(x) = …</code>).<br>" +
		"<b>Constants</b> are variables that appear only on the right-hand side — they keep their default value.<br>" +
		"Variables bound by <code>\\sum</code> or <code>\\prod</code> are excluded automatically.<br><br>" +
		"Click the numbers in the rendered formula (e.g. <code>[-5, 10]</code> under a parameter) to edit them." +
		"</div>" +
		"</div>" +
		"</div>"
	);
}


// ---------------------------------------------------------------------------
// Inline-editable parameter bounds in the rendered formula preview.
//
// MathJax CHTML renders each glyph as an (empty) ``<mjx-c class="mjx-cXXXX">``
// element where XXXX is the Unicode codepoint.  We decode the underbrace
// label back into text, locate the editable values of *each* annotated
// parameter in its own label, and lay a clickable (transparent) overlay on
// top of each value at its exact bounding box.  Clicking opens a small
// fixed-position popover that names the parameter and shows its current
// settings; committing writes the new value back into the parameter table,
// the command and the URL.  Parameters without an annotated underbrace slot
// (e.g. in exponent position, or hand-written formulas) get a small "body
// badge" over the parameter's own glyph instead.
// ---------------------------------------------------------------------------

function _normalize_math_text(str) {
	// MathJax renders variables in the Mathematical Alphanumeric Symbols
	// block (e.g. italic "a" = U+1D44E, a surrogate pair in UTF-16), so
	// decoded glyph text has to be mapped back to ASCII before comparing
	// against parameter names and label tokens.
	var out = "";
	var i = 0;
	while (i < str.length) {
		var cp = str.codePointAt(i);
		if (cp >= 0x1D400 && cp <= 0x1D419) { out += String.fromCharCode(0x41 + (cp - 0x1D400)); }
		else if (cp >= 0x1D41A && cp <= 0x1D433) { out += String.fromCharCode(0x61 + (cp - 0x1D41A)); }
		else if (cp >= 0x1D434 && cp <= 0x1D44D) { out += String.fromCharCode(0x41 + (cp - 0x1D434)); }
		else if (cp >= 0x1D44E && cp <= 0x1D467) { out += String.fromCharCode(0x61 + (cp - 0x1D44E)); }
		else if (cp >= 0x1D468 && cp <= 0x1D481) { out += String.fromCharCode(0x41 + (cp - 0x1D468)); }
		else if (cp >= 0x1D482 && cp <= 0x1D49B) { out += String.fromCharCode(0x61 + (cp - 0x1D482)); }
		else { out += String.fromCodePoint(cp); }
		i += (cp > 0xFFFF) ? 2 : 1;
	}
	return out;
}

function _decode_mjx_glyphs(rootEl) {
	// Collect (character, element) pairs in document order by decoding the
	// ``mjx-cXXXX`` class on every glyph element.  MathJax glyph codes are
	// Unicode codepoints; a few are mapped to ASCII so the decoded text can
	// be matched against the raw min/max strings from the parameter table.
	var glyphs = [];
	var els = rootEl.querySelectorAll("mjx-c, .mjx-c");
	for (var i = 0; i < els.length; i++) {
		var el = els[i];
		var m = null;
		var cls = el.className;
		if (typeof cls === "string") {
			m = /(?:^|\s)mjx-c([0-9A-Fa-f]{2,6})(?:\s|$)/.exec(cls);
		} else if (el.classList) {
			for (var ci = 0; ci < el.classList.length; ci++) {
				var cm = /^mjx-c([0-9A-Fa-f]{2,6})$/.exec(el.classList[ci]);
				if (cm) { m = cm; break; }
			}
		}
		if (!m) continue;
		var cp = parseInt(m[1], 16);
		if (!(cp >= 0x20) || (cp >= 0xD800 && cp <= 0xDFFF)) continue;
		var ch;
		if (cp === 0x2212 || cp === 0x2213) ch = "-";   // minus signs
		else if (cp === 0x2217) ch = "*";
		else if (cp === 0x2219) ch = ".";
		else {
			try {
				ch = String.fromCodePoint ? String.fromCodePoint(cp) : String.fromCharCode(cp);
			} catch (e) {
				continue;
			}
		}
		glyphs.push({ ch: ch, el: el });
	}
	return glyphs;
}

function _find_parameter_row(name) {
	var $row = null;
	$(".parameterRow").each(function () {
		if ($(this).find(".parameterName").val().trim() === name) {
			$row = $(this);
			return false;
		}
	});
	return $row;
}

// The table column that holds a value for a given kind/side.
function _cell_field_for(kind, side) {
	if (kind === "range") return (side === "min" || side === "max") ? (side === "min" ? ".minValue" : ".maxValue") : null;
	if (kind === "fixed") return ".fixedValue";
	if (kind === "choice") return ".choiceValues";
	return null;
}

// The list of editable (side, value) tokens we expect to find rendered in an
// underbrace *label*.  ``value`` is the text the decoder will produce for the
// glyphs (spaces/separators MathJax renders via CSS are intentionally not
// part of it).
function _label_tokens_for_param(param) {
	if (param.kind === "range") {
		var out = [];
		var minV = String(param.min == null ? "" : param.min).trim();
		var maxV = String(param.max == null ? "" : param.max).trim();
		if (minV !== "" && minV !== "?" && isFinite(parseFloat(minV))) out.push({ side: "min", value: minV });
		if (maxV !== "" && maxV !== "?" && isFinite(parseFloat(maxV))) out.push({ side: "max", value: maxV });
		return out;
	}
	if (param.kind === "fixed") {
		var v = String(param.value == null ? "" : param.value).trim();
		return (v !== "" && v !== "?") ? [{ side: "value", value: v }] : [];
	}
	if (param.kind === "choice") {
		var items = (param.values || "").split(",").map(function (x) { return x.trim(); }).filter(Boolean);
		return items.length ? [{ side: "values", value: items.join(",") }] : [];
	}
	return [];
}

// Resolve which parameter an underbrace ``<mjx-under>`` (label) belongs to.
// MathJax renders ``\underbrace{X}_{Y}`` as ``<mjx-munder>`` (base = X with
// the brace below it) followed by a *sibling* ``<mjx-under>`` holding Y, so
// the owning munder is the element that immediately precedes the label.  The
// parameter name is the first plain identifier inside that munder that does
// not sit inside another label.
function _resolve_under_label_param(underEl) {
	var mder = null;
	if (underEl.parentElement && underEl.parentElement.children) {
		var kids = underEl.parentElement.children;
		for (var ki = 0; ki < kids.length; ki++) {
			if (kids[ki] === underEl) {
				if (ki > 0) {
					var prev = kids[ki - 1];
					if (prev.tagName && prev.tagName.toLowerCase() === "mjx-munder" ||
						(typeof prev.className === "string" && prev.className.indexOf("mjx-munder") !== -1)) {
						mder = prev;
					}
				}
				break;
			}
		}
	}
	// Fallback: some MathJax versions nest the label inside the munder.
	if (!mder) {
		var walk = underEl;
		while (walk && walk !== document.body) {
			if (walk.tagName && walk.tagName.toLowerCase() === "mjx-munder" ||
				(typeof walk.className === "string" && walk.className.indexOf("mjx-munder") !== -1)) {
				mder = walk;
				break;
			}
			walk = walk.parentElement;
		}
	}
	if (!mder) return "";
	var mis = mder.querySelectorAll("mjx-mi, .mjx-mi");
	for (var i = 0; i < mis.length; i++) {
		var el = mis[i];
		var inside = el;
		var inLabel = false;
		while (inside && inside !== mder) {
			if (inside.tagName && (inside.tagName.toLowerCase() === "mjx-under" ||
				(typeof inside.className === "string" && inside.className.indexOf("mjx-under") !== -1))) {
				inLabel = true;
				break;
			}
			inside = inside.parentElement;
		}
		if (inLabel) continue;
		var g = _decode_mjx_glyphs(el);
		if (!g.length) continue;
		var t = "";
		for (var gi = 0; gi < g.length; gi++) t += g[gi].ch;
		var tn = _normalize_math_text(t).trim();
		if (tn) return tn;
	}
	return "";
}

// Whether an integer index points right after the closing match of a numeric
// token (protects "10" from matching inside "1000" or "−1" matching "1").
function _is_number_boundary(text, start, len) {
	var before = start > 0 ? text[start - 1] : "";
	var after = text[start + len] || "";
	return !/[0-9]/.test(before) && !/[0-9]/.test(after) && after !== ".";
}

// Locate the MathJax operator element that renders the ``=`` separating
// the function signature from the body, so we can keep LHS identifiers
// (the bare parameter list ``f(x, y, z)``) out of the body-badge search.
function _find_equals_operator(container) {
	var candidates = container.querySelectorAll("mjx-mo, .mjx-mo");
	for (var i = 0; i < candidates.length; i++) {
		var mo = candidates[i];
		var g = _decode_mjx_glyphs(mo);
		if (!g.length) continue;
		var t = "";
		for (var gi = 0; gi < g.length; gi++) t += g[gi].ch;
		if (t === "=") return mo;
	}
	return null;
}

// Find the first occurrence of ``name`` as a plain math identifier in the
// formula *body* (skipping anything inside an underbrace label, and
// optionally skipping anything before ``skipBefore`` so the LHS parameter
// declaration doesn't shadow the body's identifier).
function _find_body_identifier(container, name, skipBefore) {
	var candidates = container.querySelectorAll("mjx-mi, .mjx-mi");
	for (var i = 0; i < candidates.length; i++) {
		var el = candidates[i];
		if (skipBefore && el !== skipBefore) {
			try {
				var cmp = el.compareDocumentPosition(skipBefore);
				// DOCUMENT_POSITION_FOLLOWING (4) means ``skipBefore`` comes
				// after ``el`` in document order, i.e. ``el`` sits in the
				// LHS we want to ignore.
				if (cmp & 4) continue;
			} catch (e) { /* non-fatal */ }
		}
		var par = el;
		var inUnder = false;
		while (par && par !== container) {
			if (par.tagName.toLowerCase() === "mjx-under" ||
				String(par.className).indexOf("mjx-under") !== -1) {
				inUnder = true;
				break;
			}
			par = par.parentElement;
		}
		if (inUnder) continue;
		var g = _decode_mjx_glyphs(el);
		if (!g.length) continue;
		var t = "";
		for (var gi = 0; gi < g.length; gi++) t += g[gi].ch;
		if (_normalize_math_text(t) === name) return el;
	}
	return null;
}

function _attach_editable_preview_overlays(node) {
	// A stale popover (from a committed/re-rendered edit) must not linger.
	if ($(".omniopt_edit_pop").length) $(".omniopt_edit_pop").remove();

	var $prev = $(node);
	$prev.find(".omniopt_bound_overlay").remove();
	if (!node || !node.querySelector) return;

	var pi = (typeof get_current_parameter_info === "function") ? get_current_parameter_info() : {};
	if (!Object.keys(pi).length) return;

	var container = node.querySelector("mjx-container, .mjx-container");
	if (!container) return;

	// 1) Underbrace-label editing.  Each ``<mjx-under>`` (the brace bridge is
	//    empty and skipped) is resolved to its owning parameter via the
	//    enclosing ``<mjx-munder>``, then the min/max/value tokens expected
	//    for that parameter are located inside the decoded label.
	var labeled = {};
	var underEls = node.querySelectorAll("mjx-under, .mjx-under");
	for (var ui = 0; ui < underEls.length; ui++) {
		var underEl = underEls[ui];
		var glyphs = _decode_mjx_glyphs(underEl);
		if (!glyphs.length) continue;
		var text = "";
		for (var gi = 0; gi < glyphs.length; gi++) text += glyphs[gi].ch;
		text = _normalize_math_text(text);
		var name = _resolve_under_label_param(underEl);
		if (!name || !pi[name]) continue;
		var claimed = {};

		var tokens = _label_tokens_for_param(pi[name]);
		for (var ti = 0; ti < tokens.length; ti++) {
			var token = tokens[ti];
			var found = -1;
			var startFrom = 0;
			while ((found = text.indexOf(token.value, startFrom)) !== -1) {
				var isNum = isFinite(parseFloat(token.value)) && token.value.trim() !== "";
				if (isNum && !_is_number_boundary(text, found, token.value.length)) {
					startFrom = found + 1;
					continue;
				}
				var overlap = false;
				for (var ci = found; ci < found + token.value.length; ci++) {
					if (claimed[ci]) { overlap = true; break; }
				}
				if (!overlap) break;
				startFrom = found + 1;
			}
			if (found === -1) continue;
			for (var c2 = found; c2 < found + token.value.length; c2++) claimed[c2] = true;
			labeled[name] = true;
			_attach_editable_overlay($prev, glyphs, found, found + token.value.length,
				{ name: name, kind: pi[name].kind, side: token.side }, token.value);
		}
	}

	// 2) Body badges for params without an annotated label slot.
	var eqEl = _find_equals_operator(container);
	for (var nm in pi) {
		if (labeled[nm]) continue;
		var mi = _find_body_identifier(container, nm, eqEl);
		if (!mi) continue;
		var g = _decode_mjx_glyphs(mi);
		if (!g.length) continue;
		labeled[nm] = true;
		_attach_editable_overlay($prev, g, 0, g.length,
			{ name: nm, kind: pi[nm].kind, side: "body" }, nm);
	}
}

function _attach_editable_overlay($prev, glyphs, startIdx, endIdx, target, token) {
	var rect = null;
	for (var i = startIdx; i < endIdx && i < glyphs.length; i++) {
		var r = glyphs[i].el.getBoundingClientRect();
		if (!r || (r.width === 0 && r.height === 0)) continue;
		if (!rect) {
			rect = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
		} else {
			if (r.left < rect.left) rect.left = r.left;
			if (r.top < rect.top) rect.top = r.top;
			if (r.right > rect.right) rect.right = r.right;
			if (r.bottom > rect.bottom) rect.bottom = r.bottom;
		}
	}
	if (!rect) return;

	var host = $prev[0];
	var cRect = host.getBoundingClientRect();
	var bL = host.clientLeft || 0;
	var bT = host.clientTop || 0;
	var oW = Math.max((rect.right - rect.left) + 4, 26);
	var oH = Math.max((rect.bottom - rect.top) + 2, 16);
	var left = (rect.left - cRect.left - bL) - (oW - (rect.right - rect.left)) / 2;
	var top = (rect.top - cRect.top - bT) - (oH - (rect.bottom - rect.top)) / 2;

	var hint = "Edit " + target.name +
		(target.side === "body" ? " (" + target.kind + ")" : " (" + target.side + ")");
	var $ov = $("<span>", {
		"class": "omniopt_bound_overlay",
		"data-name": target.name,
		"data-kind": target.kind,
		"data-side": target.side,
		title: hint
	}).css({ left: left + "px", top: top + "px", width: oW + "px", height: oH + "px" });
	$prev.append($ov);

	$ov.on("mousedown", function (ev) { ev.preventDefault(); });
	$ov.on("click", function () {
		_open_inline_editor($ov, target);
	});
}

// Open the (fixed-position) popover editor that lets the user change
// ``target``; ``$handle`` is hidden while it is open and reappears when the
// user cancels (a successful commit re-renders the preview anyway).
function _open_inline_editor($handle, target) {
	// Capture the anchor rect *before* hiding it, otherwise a hidden element
	// reports a zero rect and the popover jumps to the top-left corner.
	var anchorRect = $handle[0].getBoundingClientRect();
	$handle.hide();
	var $pop = $("<div class='omniopt_edit_pop'></div>");
	document.body.appendChild($pop[0]);
	_build_edit_pop($pop, target, function (didCommit) {
		$pop.remove();
		if (didCommit) {
			$handle.remove();
			if (typeof update_command === "function") update_command();
			if (typeof _formula_preview_callback === "function") {
				try { _formula_preview_callback(); } catch (e) { /* non-fatal */ }
			}
		} else {
			$handle.show();
		}
	});
	_position_edit_pop($pop, anchorRect);
	var $inp = $pop.find("input").first();
	if ($inp.length) {
		$inp.focus();
		try { $inp[0].select(); } catch (e) { /* non-fatal */ }
	}
}

// The numbers under each underbrace are already wired up with hover
// feedback (a blue tint + pencil ``::after`` indicator) by the CSS in
// ``gui.php``, so no separate persistent pencil badge is needed here.

function _position_edit_pop($pop, rect) {
	var pad = 8;
	var w = $pop.outerWidth() || 170;
	var h = $pop.outerHeight() || 96;
	// Anchor to the right side of the editable region, vertically centered
	// on it, so the popover sits next to where the user clicked rather than
	// jumping to a corner.  Flip to the left when it would overflow, and
	// vertically clamp so it stays on screen.
	var left = rect.right + pad;
	if (left + w > window.innerWidth - pad) left = rect.left - pad - w;
	if (left < pad) left = pad;
	var top = rect.top + (rect.height / 2) - (h / 2);
	if (top + h > window.innerHeight - pad) top = window.innerHeight - pad - h;
	if (top < pad) top = pad;
	$pop.css({ left: left + "px", top: top + "px" });
}

function _build_edit_pop($pop, target, done) {
	var row = _find_parameter_row(target.name);
	var kind = target.kind;

	var head = $("<div class='omniopt_edit_head'></div>");
	head.append($("<code class='omniopt_edit_name'></code>").text(target.name));
	var chipTxt = kind;
	if (kind === "range" && target.side !== "body") chipTxt += " " + target.side;
	head.append($("<span class='omniopt_edit_chip'></span>").text(chipTxt));
	$pop.append(head);

	var labelText = "";
	var $inp = $("<input type='text' autocomplete='off' spellcheck='false'>");
	var metaText = "";

	if (kind === "range") {
		var min = row ? (row.find(".minValue").val() || "") : "";
		var max = row ? (row.find(".maxValue").val() || "") : "";
		if (target.side === "body") {
			labelText = "min, max";
			$inp.val((min !== "" ? min : "?" ) + ", " + (max !== "" ? max : "?"));
		} else {
			labelText = target.side;
			$inp.val(target.side === "min" ? min : max);
		}
		var typeTxt = row ? (row.find(".numberTypeSelect").val() || "float") : "float";
		metaText = "range [" + (min || "?") + ", " + (max || "?") + "]" +
			" \u00b7 " + typeTxt +
			(row && row.find(".log_scale").is(":checked") ? " \u00b7 log" : "");
	} else if (kind === "fixed") {
		labelText = "value";
		$inp.val(row ? (row.find(".fixedValue").val() || "") : "");
		metaText = "fixed constant \u00b7 passed to every run";
	} else if (kind === "choice") {
		labelText = "values";
		$inp.val(row ? (row.find(".choiceValues").val() || "") : "");
		metaText = "comma-separated choices";
	}

	var inputRow = $("<div class='omniopt_edit_inputrow'></div>");
	inputRow.append($("<label></label>").text(labelText));
	inputRow.append($inp);
	$pop.append(inputRow);
	if (metaText) $pop.append($("<div class='omniopt_edit_meta'></div>").text(metaText));
	$pop.append($("<div class='omniopt_edit_hint'></div>").text("Enter = save \u00b7 Esc = cancel"));

	var committed = false;
	function commit() {
		if (committed || !row) return;
		committed = true;
		var val = $inp.val().trim();
		var ok = false;
		if (kind === "range") {
			if (target.side === "min" || target.side === "max") {
				if (val !== "" && isFinite(parseFloat(val))) {
					row.find(target.side === "min" ? ".minValue" : ".maxValue")
						.val(String(parseFloat(val))).trigger("change");
					ok = true;
				}
			} else {
				var parts = val.split(",").map(function (x) { return x.trim(); });
				if (parts.length === 2 && parts[0] !== "" && isFinite(parseFloat(parts[0])) &&
					parts[1] !== "" && isFinite(parseFloat(parts[1]))) {
					row.find(".minValue").val(String(parseFloat(parts[0]))).trigger("change");
					row.find(".maxValue").val(String(parseFloat(parts[1]))).trigger("change");
					ok = true;
				}
			}
		} else if (kind === "fixed") {
			if (val !== "") {
				row.find(".fixedValue").val(val).trigger("change");
				ok = true;
			}
		} else if (kind === "choice") {
			var items = val.split(",").map(function (x) { return x.trim(); }).filter(Boolean);
			if (items.length) {
				row.find(".choiceValues").val(items.join(",")).trigger("change");
				ok = true;
			}
		}
		done(ok);
	}
	$inp.on("keydown", function (e) {
		if (e.key === "Enter") {
			e.preventDefault();
			commit();
		} else if (e.key === "Escape") {
			e.preventDefault();
			done(false);
		}
	});
	$inp.on("blur", commit);
}

// Keep inline-editable overlays glued to the rendered numbers when the page
// is resized or the sidebar changes the layout behind them.
if (typeof window !== "undefined" && window.addEventListener) {
	window.addEventListener("resize", function () {
		setTimeout(function () {
			var $prev = $("#formula_preview");
			if (!$prev.length) return;
			try { _attach_editable_preview_overlays($prev[0]); } catch (e) { /* swallow */ }
		}, 60);
	});
}


// Very small JS-side formula parser used to extract suggested parameter
// names.  It is intentionally permissive: it identifies every identifier-like
// token and removes ones that are obviously sympy built-ins or known math
// constants.  The Python side does the authoritative parse; this is just for
// quick client-side feedback.
function _strip_underbraces(text) {
	var result = "";
	var i = 0;
	var marker = "\\underbrace{";
	while (i < text.length) {
		var idx = text.indexOf(marker, i);
		if (idx === -1) {
			result += text.substring(i);
			break;
		}
		result += text.substring(i, idx);
		var pos = idx + marker.length;
		var depth = 1;
		while (pos < text.length && depth > 0) {
			if (text[pos] === "{") depth++;
			else if (text[pos] === "}") depth--;
			pos++;
		}
		var content = text.substring(idx + marker.length, pos - 1);
		result += content;
		if (pos < text.length && text[pos] === "_") pos++;
		if (pos < text.length && text[pos] === "{") {
			depth = 1;
			pos++;
			while (pos < text.length && depth > 0) {
				if (text[pos] === "{") depth++;
				else if (text[pos] === "}") depth--;
				pos++;
			}
		}
		i = pos;
	}
	return result;
}

var FORMULA_RESERVED = new Set([
	// sympy constants
	"pi", "E", "I", "oo", "inf", "infty", "nan", "NaN", "True", "False",
	// common mathematical functions
	"sin", "cos", "tan", "asin", "acos", "atan",
	"sinh", "cosh", "tanh",
	"exp", "log", "ln", "sqrt", "abs", "Min", "Max",
	"Sum", "Product", "Integral", "Derivative",
	"max", "min", "norm", "trace", "det", "rank",
	"ReLU", "relu", "softmax", "sigmoid", "tanh",
	"clip", "clamp", "ceil", "floor", "round",
	"argmax", "argmin", "argsort",
	// python built-ins that often leak in
	"math", "numpy", "np", "self", "def", "return", "import", "from",
	"if", "else", "elif", "for", "while", "in", "and", "or", "not",
	"params", "evaluate", "_raw",
	// LaTeX commands and environments
	"underbrace", "overbrace", "substack", "mathbb", "text", "mathrm",
	"mathbf", "mathcal", "operatorname", "left", "right", "displaystyle",
	"quad", "qquad", "cdot", "times", "limits", "hat", "bar",
	"vec", "dot", "tilde", "widehat", "overline", "underline",
	"int", "oint", "iint", "forall", "exists", "partial",
	"frac", "dfrac", "tfrac", "sqrt", "begin", "end",
	"cases", "sim", "propto", "approx", "neq", "leq", "geq",
	"to", "rightarrow", "mapsto", "sum", "prod",
]);

var FORMULA_CONSTANTS = {
	pi: 3.141592653589793,
	e: 2.718281828459045,
	E: 2.718281828459045,
	PI: 3.141592653589793,
};

// Strip formatting/structural macros that obscure identifiers.
function _strip_macros(text) {
	// \frac{num}{den} → (num)/(den)
	text = text.replace(/\\(?:frac|dfrac|tfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
	// \sqrt[n]{x} or \sqrt{x} → (x)
	text = text.replace(/\\sqrt\s*(?:\[[^[\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g, "($1)");
	// \text{...}, \mathbb{...}, etc. → keep content
	text = text.replace(/\\(?:text|textit|textbf|mathrm|operatorname|mathbf|mathcal|mathbb|mathfrak|mathsf|mathtt|mbox|boldsymbol)\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g, "$1");
	// \| → remove (norm delimiters)
	text = text.replace(/\\\|/g, "");
	// \max, \min, \argmax, \argmin → remove name, keep args
	text = text.replace(/\\(?:max|min|argmax|argmin|norm|trace|det)\b/g, "");
	// \begin{...}, \end{...} → remove
	text = text.replace(/\\(?:begin|end)\s*\{[^{}]*\}/g, " ");
	// \sim, \propto, \approx, \neq, \leq, \geq, \to, \left, \right → remove
	text = text.replace(/\\(?:sim|propto|approx|neq|leq|geq|to|rightarrow|mapsto|left|right|displaystyle|textstyle)\b[|(\[.]?/g, " ");
	// \sin, \cos, etc. → keep name (filtered by FORMULA_RESERVED)
	text = text.replace(/\\(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|ln|abs)\b/g, "$1");
	// Greek letters → ASCII names (these ARE parameter names)
	text = text.replace(/\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|omicron|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)\b/g, "$1");
	// \sum, \prod, \int, etc. → remove (handled by _strip_sumprod_bodies)
	text = text.replace(/\\(?:sum|prod|int|oint|iint)\b/g, "");
	// Any remaining \command → space
	text = text.replace(/\\[A-Za-z]+/g, " ");
	// Clean up: {single_var} → var
	text = text.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, "$1");
	return text;
}

// Collect identifiers from a piece of text, ignoring reserved ones and
// capturing duplicates (so we can detect overlap between left-hand and right-hand sides).
function _collect_identifiers(text) {
	var re = /[A-Za-z_][A-Za-z0-9_]*/g;
	var out = [];
	var seen = {};
	var m;
	while ((m = re.exec(text)) !== null) {
		var name = m[0];
		if (name === "_") continue;
		// Normalize indexed variables: x_1, x_2 → x
		var baseName = name.replace(/_\d+$/, "");
		if (baseName.length > 0 && baseName !== name) name = baseName;
		if (FORMULA_RESERVED.has(name)) continue;
		out.push(name);
		seen[name] = (seen[name] || 0) + 1;
	}
	return { list: out, set: Object.keys(seen) };
}

// Split a LaTeX/infix formula on `=` into {lhs, rhs}.  We use the FIRST `=`
// that is not inside a brace block and not a comparator (we only handle
// assignments here, so any `==`, `<=`, `>=` would also be picked up but the
// caller can deal with it).  Returns {lhs: "", rhs: text} if no `=` exists.
function _split_assignment(text) {
	// Walk through char-by-char, tracking brace depth.  Use the LAST
	// top-level ``=`` as the LHS / RHS separator so chained assignments
	// like ``a + b = c = c - d`` (mathematically nonsensical, but valid
	// Python) land as ``lhs = "a + b = c"``, ``rhs = "c - d"`` — the
	// outermost assignment is what controls the parameter list.
	var depth = 0;
	var lastIdx = -1;
	for (var i = 0; i < text.length; i++) {
		var ch = text[i];
		if (ch === "{") depth++;
		else if (ch === "}") depth--;
		else if (ch === "=" && depth === 0) {
			lastIdx = i;
		}
	}
	if (lastIdx >= 0) {
		return { lhs: text.substring(0, lastIdx), rhs: text.substring(lastIdx + 1) };
	}
	return { lhs: "", rhs: text };
}

// Strip the function name + parens from a left-hand side like `f(x, y)` or `f(g(x))`
// so we are left with the actual parameter list inside the outermost parens.
// Returns an empty list if the LHS isn't shaped like a function definition
// (no ``(`` present, or the LHS contains more than one identifier / operator).
function _lhs_parameter_names(lhs) {
	var open = lhs.indexOf("(");
	if (open < 0) return [];
	var close = lhs.lastIndexOf(")");
	if (close < open) return [];
	var name = lhs.substring(0, open).trim();
	// The LHS must look like ``f`` or ``f.f`` (a function name, not an
	// arithmetic expression).  This guards against ``a + b = c = c - d``
	// being interpreted as parameters.
	if (!/^[A-Za-z_][A-Za-z0-9_.]*$/.test(name)) return [];
	var inner = lhs.substring(open + 1, close);
	return _collect_identifiers(_strip_macros(inner)).list;
}

// Extract bound variable names from \sum_{...}, \prod_{...}, \int_{...},
// \mathbb{E}_{...}, \oint_{...}, \iint_{...}.
// Returns { text: text (unchanged), bound: {name: true, ...} }.
function _strip_sumprod_bodies(text) {
	var boundNames = {};
	// Match \sum_{...}, \prod_{...}, \int_{...}, \oint_{...}, \iint_{...}
	var re1 = /\\(?:sum|prod|int|oint|iint)\s*_\s*\{([^{}]+)\}/g;
	var m;
	while ((m = re1.exec(text)) !== null) {
		var sub = m[1];
		var name = sub.indexOf("=") >= 0 ? sub.split("=")[0].trim() : sub.trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) boundNames[name] = true;
	}
	// Match \sum_var, \int_var (single char subscript without braces)
	var re2 = /\\(?:sum|prod|int|oint|iint)\s*_([A-Za-z_][A-Za-z0-9_]*)/g;
	while ((m = re2.exec(text)) !== null) {
		var name = m[1];
		if (name.indexOf("=") >= 0) name = name.split("=")[0].trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) boundNames[name] = true;
	}
	// Match \mathbb{E}_{x\sim...} or E_{x~...}
	var re3 = /(?:\\mathbb\{E\}|E)\s*_\s*\{([^{}]+)\}/g;
	while ((m = re3.exec(text)) !== null) {
		var sub = m[1];
		var name = sub.split(/\\sim|~|=|,/)[0].trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) boundNames[name] = true;
	}
	return { text: text, bound: boundNames };
}


// Extract parameter names from a Python-mode formula by scanning for
// ``params['x']`` / ``params["x"]`` / ``params.get('x')`` patterns.
function client_extract_formula_params(text, mode) {
	// Guard rail: any thrown error must fall back to an empty suggestion
	// list so the rest of the GUI still works (no broken `apply` button).
	try {
		return client_extract_formula_params_impl(text, mode);
	} catch (e) {
		try {
			console.error("[formula extraction] failed:", e, "for text:", text);
		} catch (_) { /* swallow */ }
		return { parameters: [], constants: [], bound: [], error: (e && e.message) ? e.message : String(e) };
	}
}

function client_extract_formula_params_impl(text, mode) {
	if (!text || !text.trim()) return { parameters: [], constants: [], bound: [] };
	if (typeof text !== "string") return { parameters: [], constants: [], bound: [] };
	var raw = _strip_underbraces(text);

	var split;
	try { split = _split_assignment(raw); }
	catch (e) { return { parameters: [], constants: [], bound: [] }; }
	var lhsRaw = split.lhs;
	var rhsRaw = split.rhs;

	var boundInfo;
	try { boundInfo = _strip_sumprod_bodies(raw); }
	catch (e) { return { parameters: [], constants: [], bound: [] }; }
	var boundNames = (boundInfo && boundInfo.bound) || {};

	// Left-hand side: find the parameter names of the function (variables
	// that appear inside the outermost parentheses of the left side).
	// If there is no LHS, every free symbol in the RHS becomes a parameter
	// (otherwise bare expressions like ``a + sin(b)`` would lose all of
	// their variables to the ``constant`` bucket).
	var lhsIdents = [];
	if (lhsRaw.trim().length > 0) {
		try { lhsIdents = _lhs_parameter_names(lhsRaw) || []; }
		catch (e) { lhsIdents = []; }
	}
	var lhsSet = {};
	for (var i = 0; i < lhsIdents.length; i++) lhsSet[lhsIdents[i]] = true;

	// Right-hand side: strip the function macros, collect identifiers.
	var rhsClean = "";
	try { rhsClean = _strip_macros(boundInfo.text); } catch (e) {}
	// Re-locate `=` for the stripped right-hand side — the bound-stripping
	// may have shifted positions.
	var rhsOnly = rhsClean;
	try { rhsOnly = _split_assignment(rhsClean).rhs; } catch (e) {}
	var rhsInfo;
	try { rhsInfo = _collect_identifiers(rhsOnly); } catch (e) { rhsInfo = { list: [], set: [] }; }
	var rhsIdents = rhsInfo.list || [];

	// Build the parameter and constant lists.
	var parameters = [];
	var constants = [];
	var seenP = {};
	var seenC = {};
	var hasLhs = lhsIdents.length > 0;
	for (var j = 0; j < lhsIdents.length; j++) {
		var n = lhsIdents[j];
		if (!seenP[n]) {
			seenP[n] = true;
			parameters.push(n);
		}
	}
	for (var k = 0; k < rhsIdents.length; k++) {
		var rn = rhsIdents[k];
		if (!rn) continue;
		if (boundNames[rn]) continue;          // bound by sum/prod
		if (lhsSet[rn]) continue;              // already a parameter
		if (FORMULA_RESERVED && FORMULA_RESERVED.has && FORMULA_RESERVED.has(rn)) continue; // safety
		if (!seenP[rn] && !seenC[rn]) {
			seenP[rn] = true;
			seenC[rn] = true;
			// With an LHS, RHS-only identifiers are *constants* (so the user
			// sees the split between parameters and fixed values).
			// Without an LHS, every free symbol is a *parameter* (so a bare
			// expression like ``a + sin(b)`` ends up with ``a`` and ``b``
			// as optimisable ranges rather than mysterious constants).
			if (hasLhs) {
				constants.push(rn);
			} else {
				parameters.push(rn);
			}
		}
	}

	function asSuggestion(name, kind) {
		if (Object.prototype.hasOwnProperty.call(FORMULA_CONSTANTS, name)) {
			return {
				name: name,
				kind: "fixed",
				lower: FORMULA_CONSTANTS[name],
				upper: FORMULA_CONSTANTS[name],
				value_type: "float",
				log_scale: false,
				note: "well-known constant (default value preserved; switch to range to optimize)",
			};
		}
		if (name.endsWith("_int")) {
			return { name: name, kind: kind, lower: 0, upper: 10, value_type: "int", log_scale: false };
		}
		if (name.startsWith("lr_") || name.startsWith("log_") || name.endsWith("_log")) {
			return { name: name, kind: kind, lower: 1e-5, upper: 1e-1, value_type: "float", log_scale: true };
		}
		// Constants default to 1.0 (so multiplying by them has a clear effect).
		// Parameters default to a [-1, 1] range around 0.
		if (kind === "fixed") {
			return { name: name, kind: kind, lower: 1, upper: 1, value_type: "float", log_scale: false };
		}
		return { name: name, kind: kind, lower: -1, upper: 1, value_type: "float", log_scale: false };
	}

	return {
		parameters: parameters.map(function (n) { return asSuggestion(n, "range"); }),
		constants: constants.map(function (n) { return asSuggestion(n, "fixed"); }),
		bound: Object.keys(boundNames),
	};
}


function client_render_suggestions(result) {
	var $out = $("#formula_suggestions");
	if (!result || (!result.parameters.length && !result.constants.length)) {
		$out.html("<em>(no free symbols detected)</em>");
		return;
	}
	// Pull the current values from the parameter table so the suggestions
	// panel reflects what the user has actually configured, not just the
	// defaults we'd suggest for a brand-new parameter row.
	var current = {};
	$(".parameterRow").each(function () {
		var name = $(this).find(".parameterName").val().trim();
		if (!name) return;
		var kind = $(this).find(".optionSelect").val();
		var entry = { kind: kind };
		if (kind === "range") {
			entry.lower = $(this).find(".minValue").val();
			entry.upper = $(this).find(".maxValue").val();
		} else if (kind === "fixed") {
			entry.lower = $(this).find(".fixedValue").val();
			entry.upper = entry.lower;
		} else if (kind === "choice") {
			entry.lower = $(this).find(".choiceValues").val();
			entry.upper = entry.lower;
		}
		current[name] = entry;
	});
	function mergeWithCurrent(s) {
		var c = current[s.name];
		if (!c) return s;
		// The user's chosen kind (range/fixed/choice) wins over the
		// formula's suggested kind so the panel doesn't lie about what
		// the parameter table actually contains.
		s.kind = c.kind;
		if (c.kind === "fixed") {
			s.lower = c.lower != null ? c.lower : s.lower;
			s.upper = s.lower;
		} else if (c.kind === "range") {
			if (c.lower !== "" && c.lower != null) s.lower = c.lower;
			if (c.upper !== "" && c.upper != null) s.upper = c.upper;
		} else if (c.kind === "choice") {
			if (c.lower) s.lower = c.lower;
			if (c.upper) s.upper = c.upper;
		}
		return s;
	}
	var html = "<table style='width:100%; border-collapse: collapse;'>";
	html += "<tr><th align='left'>name</th><th align='left'>type</th><th align='left'>value</th><th align='left'>role</th></tr>";
	function row(s, role) {
		var val;
		if (s.kind === "fixed") val = s.lower;
		else if (s.kind === "choice") val = "{" + s.lower + "}";
		else val = "[" + s.lower + ", " + s.upper + "]";
		return "<tr><td><code>" + s.name + "</code></td><td>" + s.kind + "</td><td>" + val + "</td><td>" + role + "</td></tr>";
	}
	for (var i = 0; i < result.parameters.length; i++) {
		html += row(mergeWithCurrent(result.parameters[i]), "<span style='color:#1b6e1b'>parameter</span>");
	}
	for (var j = 0; j < result.constants.length; j++) {
		html += row(mergeWithCurrent(result.constants[j]), "<span style='color:#7a3e9e'>constant</span>");
	}
	if (result.bound && result.bound.length) {
		html += "<tr><td colspan='4' style='padding-top: 6px; color:#777; font-style: italic;'>" +
			"bound variables (excluded): " +
			result.bound.map(function (b) { return "<code>" + b + "</code>"; }).join(", ") +
			"</td></tr>";
	}
	html += "</table>";
	$out.html(html);
}


function client_apply_suggestions(result) {
	// For each parameter/constant suggestion, if a parameter row with the
	// same name already exists, leave it alone (user may have edited it).
	// Otherwise add a new row of the appropriate kind.
	if (!result) return;
	var all = [].concat(result.parameters || [], result.constants || []);
	if (!all.length) return;

	// Capture the row count BEFORE we add new ones so we can address them
	// by index even if other code paths mutate the DOM in between.
	var initialCount = $(".parameterRow").length;

	// Find the first existing empty row (the GUI always renders one
	// blank parameter row on load).  Filling it lets the first suggestion
	// land in row 0 instead of being pushed past an empty placeholder.
	var firstEmptyIdx = -1;
	for (var ei = 0; ei < initialCount; ei++) {
		var $probe = $(".parameterRow").eq(ei);
		var probeName = $probe.find(".parameterName").val();
		if (!probeName || probeName.trim() === "") {
			firstEmptyIdx = ei;
			break;
		}
	}

	for (var i = 0; i < all.length; i++) {
		(function (s, idx) {
			var existing = false;
			$(".parameterName").each(function () {
				if ($(this).val() === s.name) existing = true;
			});
			if (existing) return;

			var targetIndex;
			if (firstEmptyIdx >= 0) {
				// Reuse the blank row first.
				targetIndex = firstEmptyIdx;
				firstEmptyIdx = -1;
			} else {
				$("#main_add_row_button").click();
				targetIndex = $(".parameterRow").length - 1;
			}
			var $row = $(".parameterRow").eq(targetIndex);
			if ($row.length === 0) return;

			// Set the name BEFORE changing the kind so updateOptions picks
			// it up when re-rendering the value cell.
			$row.find(".parameterName").val(s.name);
			$row.find(".optionSelect").val(s.kind).trigger("change");

			// Now the value cell has been re-rendered for the new kind —
			// fill in the kind-specific fields synchronously.
			if (s.kind === "range") {
				$row.find(".minValue").val(s.lower);
				$row.find(".maxValue").val(s.upper);
				$row.find(".numberTypeSelect").val(s.value_type);
				$row.find(".log_scale").prop("checked", !!s.log_scale);
			} else if (s.kind === "fixed") {
				$row.find(".fixedValue").val(s.lower);
			}
		})(all[i], initialCount + i);
	}

	var suggestedNames = all.map(function (s) { return s.name; });
	$(".parameterRow").each(function () {
		var name = $(this).find(".parameterName").val().trim();
		if (name && suggestedNames.indexOf(name) === -1) {
			$(this).remove();
		}
	});
	var remaining = $(".parameterRow").length;
	if (remaining === 0) {
		$("#main_add_row_button").click();
	}

	update_command();
}

var _formula_preview_callback = null;
var _formulaPreviewTimer = null;

function _schedule_formula_preview_refresh() {
	if (_formulaPreviewTimer) clearTimeout(_formulaPreviewTimer);
	_formulaPreviewTimer = setTimeout(function () {
		_formulaPreviewTimer = null;
		if (typeof _formula_preview_callback === "function") {
			_formula_preview_callback();
		}
	}, 300);
}

function get_current_parameter_info() {
	var info = {};
	$(".parameterRow").each(function () {
		var name = $(this).find(".parameterName").val().trim();
		if (!name || !/^[a-zA-Z_]+$/.test(name)) return;
		var option = $(this).find(".optionSelect").val();
		if (option === "range") {
			info[name] = {
				kind: "range",
				min: $(this).find(".minValue").val() || "",
				max: $(this).find(".maxValue").val() || "",
				type: $(this).find(".numberTypeSelect").val() || "float",
				log_scale: $(this).find(".log_scale").is(":checked")
			};
		} else if (option === "fixed") {
			info[name] = {
				kind: "fixed",
				value: $(this).find(".fixedValue").val() || ""
			};
		} else if (option === "choice") {
			info[name] = {
				kind: "choice",
				values: $(this).find(".choiceValues").val() || ""
			};
		}
	});
	return info;
}

function _convert_infix_to_latex(text) {
	text = text.replace(/([a-zA-Z_)\d]|(\}))\s*\*\*\s*([a-zA-Z_]\w*|\d+(?:\.\d+)?)/g, "$1^{$3}");
	text = text.replace(/\*\*/g, "^");
	// Function names → LaTeX
	text = text.replace(/\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|ln)\b/g, "\\$1");
	text = text.replace(/\bsqrt\s*\(/g, "\\sqrt{");
	text = text.replace(/\babs\s*\(/g, "\\left|");
	text = text.replace(/\b(max|min)\s*\(/g, "\\$1(");
	text = text.replace(/\b(ReLU|relu|sigmoid|softmax|clip|clamp|ceil|floor)\s*\(/g, "\\text{$1}(");
	// abs( closing → |
	text = text.replace(/\)\s*(?=[+\-*/^,)\s]|$)/g, ")");
	return text;
}

function _format_param_label(info) {
	if (info.kind === "range") {
		var min = info.min !== "" ? info.min : "?";
		var max = info.max !== "" ? info.max : "?";
		var numberSet = (info.type === "int") ? "\\mathbb{Z}" : "\\mathbb{R}";
		var line1 = "[" + min + ",\\, " + max + "] \\in " + numberSet;
		var line2 = (info.type === "int") ? "discrete" : "continuous";
		if (info.log_scale) line2 += ", log";
		return "\\substack{" + line1 + " \\\\ \\text{" + line2 + "}}";
	} else if (info.kind === "fixed") {
		var val = info.value !== "" ? info.value : "?";
		return "\\substack{" + val + " \\\\ \\text{fixed}}";
	} else if (info.kind === "choice") {
		var vals = info.values ? info.values.split(",").map(function (v) { return "\\text{" + v.trim() + "}"; }).filter(Boolean).join(",\\, ") : "?";
		return "\\substack{\\{" + vals + "\\} \\\\ \\text{choice}}";
	}
	return "";
}

function _protect_subsup(text) {
	var parts = [];
	var result = "";
	var i = 0;
	while (i < text.length) {
		if (text[i] === "_" || text[i] === "^") {
			var j = i + 1;
			if (j < text.length && text[j] === "{") {
				var depth = 1;
				j++;
				while (j < text.length && depth > 0) {
					if (text[j] === "{") depth++;
					else if (text[j] === "}") depth--;
					j++;
				}
			} else if (j < text.length) {
				j++;
			}
			parts.push(text.substring(i, j));
			result += "\x00" + (parts.length - 1) + "\x00";
			i = j;
		} else {
			result += text[i];
			i++;
		}
	}
	return { text: result, parts: parts };
}

function _restore_subsup(text, parts) {
	return text.replace(/\x00(\d+)\x00/g, function (m, idx) {
		return parts[parseInt(idx)];
	});
}

function _add_explicit_grouping(latex) {
	var re = /(\\(?:int|oint|iint|iiint|sum|prod))(?:\s*_{\s*[^{}]*\s*}|\s*_\s*[A-Za-z][A-Za-z0-9_]*\s*)?(?:\s*\^\s*\{[^{}]*\}|\s*\^\s*[A-Za-z][A-Za-z0-9_]*\s*)?/g;
	var result = "";
	var lastEnd = 0;
	var m;
	while ((m = re.exec(latex)) !== null) {
		var cmdEnd = m.index + m[0].length;
		result += latex.slice(lastEnd, cmdEnd);
		var pos = cmdEnd;
		while (pos < latex.length && latex[pos] === " ") pos++;
		if (pos >= latex.length) { lastEnd = cmdEnd; continue; }
		var ch = latex[pos];
		if (ch === "(" || ch === "{") { lastEnd = cmdEnd; continue; }
		if (ch === "\\") { lastEnd = cmdEnd; continue; }
		// If the integral/sum is inside user-written parens (i.e. the char
		// before the command is `(`), skip adding \left( \right) — the
		// user's parens already define the scope.
		var beforeIdx = m.index - 1;
		while (beforeIdx >= 0 && latex[beforeIdx] === " ") beforeIdx--;
		var inUserParens = beforeIdx >= 0 && latex[beforeIdx] === "(";
		if (inUserParens) {
			lastEnd = cmdEnd;
			continue;
		}
		var termEnd = _read_term_end(latex, pos);
		if (termEnd > pos) {
			var body = latex.slice(pos, termEnd);
			result += "\\left(" + body + "\\right)";
			lastEnd = termEnd;
		} else {
			lastEnd = cmdEnd;
		}
	}
	result += latex.slice(lastEnd);
	return result;
}

function _read_term_end(text, start) {
	var i = start;
	var depth = 0;
	var sawAny = false;
	while (i < text.length) {
		var ch = text[i];
		if (ch === "(" || ch === "{" || ch === "[") { depth++; sawAny = true; i++; continue; }
		if (ch === ")" || ch === "}" || ch === "]") {
			if (depth === 0) break;
			depth--; sawAny = true; i++; continue;
		}
		if (depth === 0 && (ch === "+" || ch === "-") && sawAny) {
			var nxt = text[i + 1] || "";
			if (nxt !== ch) break;
		}
		if (depth === 0 && ch === "=" && sawAny) break;
		i++;
		sawAny = true;
	}
	return i;
}

function _add_parameter_underbraces(latex, paramInfo) {
	var names = Object.keys(paramInfo);
	if (!names.length) return latex;

	var eqIdx = -1;
	var depth = 0;
	for (var i = 0; i < latex.length; i++) {
		var ch = latex[i];
		if (ch === "{") depth++;
		else if (ch === "}") depth--;
		else if (ch === "=" && depth === 0) eqIdx = i;
	}

	var lhs, rhs;
	if (eqIdx >= 0) {
		lhs = latex.substring(0, eqIdx + 1);
		rhs = latex.substring(eqIdx + 1);
	} else {
		lhs = "";
		rhs = latex;
	}

	var prot = _protect_subsup(rhs);
	var work = prot.text;

	names.sort(function (a, b) { return b.length - a.length; });
	var escaped = names.map(function (n) { return n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); });
	var re = new RegExp("(?<![A-Za-z0-9_])(?:" + escaped.join("|") + ")(?![A-Za-z0-9_])", "g");

	work = work.replace(re, function (match, offset) {
		var info = paramInfo[match];
		var label = _format_param_label(info);
		var before = offset > 0 ? work[offset - 1] : "";
		var after = offset + match.length < work.length ? work[offset + match.length] : "";
		// Skip if in exponent position (preceded by ^) or base of an
		// exponent (followed by ^) — underbrace label collides with ^
		if (before === "^" || after === "^") return match;
		return "\\underbrace{" + match + "}_{" + label + "}";
	});

	// Subscript/superscript members (e.g. ``^z`` in ``\frac{x}{y}^z``)
	// were shielded by ``_protect_subsup`` so the regex above never sees
	// them.  Walk each placeholder and, if its inner content is a single
	// parameter name, wrap it in an underbrace INSIDE the ``^{}`` /
	// ``_{}`` so the user still gets an editable label slot.
	var newParts = prot.parts.slice();
	for (var pi = 0; pi < newParts.length; pi++) {
		var part = newParts[pi];
		var sm = part.match(/^([\^_])(?:\{([^{}]*)\}|([A-Za-z_][A-Za-z0-9_]*))$/);
		if (!sm) continue;
		var marker = sm[1];
		var content = sm[2] !== undefined ? sm[2] : sm[3];
		if (!paramInfo.hasOwnProperty(content)) continue;
		var info2 = paramInfo[content];
		var label2 = _format_param_label(info2);
		newParts[pi] = marker + "{\\underbrace{" + content + "}_{" + label2 + "}}";
	}

	return lhs + _restore_subsup(work, newParts);
}

// The "Run program" / "Formula" tab bar lives inside the run_program row of
// the table.  On load we move it above the run_program textarea so it reads as
// a proper tab bar, then handle tab switching: the two are mutually exclusive
// (exactly one of them is filled in).  Defined at the top level because
// ``restoreFormula`` (in a different scope) also switches tabs.
function _position_run_program_tabbar() {
	var $tabbar = $("#run_program_tabbar");
	var $wrapper = $("#run_program_wrapper");
	if ($tabbar.length && $wrapper.length) {
		var $next = $tabbar.next();
		if ($next.length === 0 || $next[0] !== $wrapper[0]) {
			$tabbar.insertBefore($wrapper);
		}
	}
}

function _set_tab(state) {
	_position_run_program_tabbar();
	if ($(".omniopt_edit_pop").length) $(".omniopt_edit_pop").remove();
	var $card = $("#formula_card");
	var $wrapper = $("#run_program_wrapper");
	if (state === "formula") {
		if (!$card.data("built")) {
			$card.html(build_formula_card_html());
			$card.data("built", true);
			setup_formula_card_inner();
		}
		$card.show();
		$wrapper.hide();
		$("#rp_tab_formula").addClass("rp_tab_active");
		$("#rp_tab_run").removeClass("rp_tab_active");
	} else {
		$card.hide();
		$wrapper.show();
		$("#rp_tab_run").addClass("rp_tab_active");
		$("#rp_tab_formula").removeClass("rp_tab_active");
	}
}

function setup_formula_editor() {

	$(document).on("click", "#rp_tab_run", function () {
		_set_tab("run");
	});

	$(document).on("click", "#rp_tab_formula", function () {
		_set_tab("formula");
		// Re-anchor the editable bound overlays to the now-visible layout.
		setTimeout(function () {
			var $prev = $("#formula_preview");
			if ($prev.length) {
				try { _attach_editable_preview_overlays($prev[0]); } catch (e) { /* swallow */ }
			}
		}, 10);
	});

	// If the user starts typing into run_program, switch to the Run program
	// tab so the (hidden, stale) formula card doesn't linger.
	$(document).on("input", "#run_program", function () {
		if ($(this).val().trim() !== "") {
			_set_tab("run");
		}
	});

	// If the user starts typing into the formula editor, switch to the
	// Formula tab.
	$(document).on("input", "#formula", function () {
		if ($(this).val().trim() !== "") {
			_set_tab("formula");
		}
	});

	// Persist the hidden #formula and #formula_mode on every change so the
	// URL always carries the latest values (a refresh restores them).
	$(document).on("change input", "#formula, #formula_mode", function () {
		if (typeof update_command === "function") update_command();
	});

	// Parameter row edits already flow through ``update_command()`` (every
	// input has ``onkeyup/onchange/onclick="update_command()"``) and that
	// function ends with ``_schedule_formula_preview_refresh()``, so the
	// suggestions panel re-renders automatically once the debounced timer
	// fires — no extra listener needed here.

	// Arrange the tab bar above the run_program textarea right away (on
	// initial load neither the Formula tab nor restoreFormula will have run).
	_position_run_program_tabbar();
	_set_tab($("#formula_card").length && $("#formula_card").is(":visible") ? "formula" : "run");
}

function setup_formula_card_inner() {
	function sync_to_main_textarea(text) {
		$("#formula").val(text).trigger("change");
		var $panes = $("#formula_card #formula_pane_text");
		if ($panes.length) {
			$panes.val(text);
		}
	}

	function auto_detect_mode(text) {
		if (!text) return "auto";
		if (/\\(sin|cos|tan|sum|prod|frac|sqrt|text|textit|begin|end)\b/.test(text)) {
			return "latex";
		}
		return "infix";
	}

	function current_mode() {
		var m = ($("#formula_mode").val() || "").trim();
		return (m === "latex" || m === "infix") ? m : "auto";
	}

	function client_render_formula_preview(text) {
		var $prev = $("#formula_card #formula_preview");
		var $err = $("#formula_card #formula_error");

		// Guard rail: if the card isn't in the DOM yet, do nothing.
		if (!$prev.length || !$err.length) {
			return;
		}

		try {
			client_render_formula_preview_inner(text, $prev, $err);
		} catch (e) {
			// Last-resort guard rail — anything that throws here must
			// not break the rest of the page (form, command, etc.).
			try {
				console.error("[formula preview] uncaught error:", e);
				$err.text("Preview error: " + (e && e.message ? e.message : String(e)));
			} catch (_) { /* swallow */ }
		}
	}

	function client_render_formula_preview_inner(text, $prev, $err) {
		var $legend = $("#formula_card #formula_param_legend");

		if (!text || !text.trim()) {
			$prev.empty();
			$err.empty();
			if ($legend.length) $legend.html("");
			return;
		}

		var mode;
		try {
			mode = current_mode();
		} catch (e) {
			mode = "auto";
		}

		if (mode === "infix" || mode === "text" || mode === "auto") {
			var detected = auto_detect_mode(text);
			if (detected === "infix") {
				text = _convert_infix_to_latex(text);
			}
		}
		text = text.replace(/\*/g, "\\cdot ");
		text = text.replace(/([\d)])([a-zA-Z_])/g, "$1\\cdot $2");
		text = _add_explicit_grouping(text);

		var _pi = get_current_parameter_info();
		var _hasUnderbraces = text.indexOf("\\underbrace") !== -1;
		if (!_hasUnderbraces && Object.keys(_pi).length) {
			text = _add_parameter_underbraces(text, _pi);
			_hasUnderbraces = true;
		}

		if ($legend.length) {
			if (Object.keys(_pi).length) {
				var legendParts = [];
				for (var _ln in _pi) {
					var _li = _pi[_ln];
					if (_li.kind === "range") {
						legendParts.push("<code>" + _ln + "</code> ∈ [" + (_li.min || "?") + ", " + (_li.max || "?") + "] (" + (_li.type || "float") + (_li.log_scale ? ", log" : "") + ")");
					} else if (_li.kind === "fixed") {
						legendParts.push("<code>" + _ln + "</code> = " + (_li.value || "?") + " (fixed)");
					} else if (_li.kind === "choice") {
						legendParts.push("<code>" + _ln + "</code> ∈ {" + (_li.values || "?") + "}");
					}
				}
				$legend.html(legendParts.join(" &nbsp;·&nbsp; "));
			} else {
				$legend.html("");
			}
		}

		// Sanity-check the input: refuse to render empty / whitespace /
		// extreme-length LaTeX so we never break MathJax.
		if (typeof text !== "string") {
			$err.text("Preview error: formula is not a string");
			return;
		}
		if (text.length > 5000) {
			$err.text("Formula is too long to preview (>5000 chars)");
			$prev.empty();
			return;
		}

		// Detect obviously broken LaTeX before handing it to MathJax so we
		// can show a useful error message instead of a giant red ``?``.
		var stripped = text
			.replace(/\\underbrace\{[^{}]*\}\{[^{}]*\}/g, "")
			.replace(/^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*=\s*/, "")
			.replace(/^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*/, "");
		var balanceErrors = [];
		if (/\\\s/.test(text) && !_hasUnderbraces) balanceErrors.push("stray space after a backslash");
		if (!_hasUnderbraces && /[{}]\s*[+\-*/^=]/.test(stripped)) balanceErrors.push("brace next to an operator (probably missing ``\\right``)");
		// Count opening and closing braces; an imbalance is almost
		// certainly a typo.
		var depth = 0, maxDepth = 0, bchar;
		for (var bi = 0; bi < text.length; bi++) {
			bchar = text[bi];
			if (bchar === "{") { depth++; if (depth > maxDepth) maxDepth = depth; }
			else if (bchar === "}") { depth--; if (depth < 0) break; }
		}
		if (depth !== 0) balanceErrors.push("unbalanced braces (depth=" + depth + ")");
		// Reject control characters / null bytes that MathJax can't digest.
		if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(text)) balanceErrors.push("control characters in formula");
		if (/\{[ \t]*\}/.test(text)) balanceErrors.push("empty LaTeX group ``{}`` (no-op)");
		var bareMacros = text.match(/\\(?:int|sum|prod|oint|iint)\b/g) || [];
		var bareChainRe = /\\(?:int|sum|prod|oint|iint)\b(?:\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9]+))?\s*\\(?:int|sum|prod|oint|iint)\b(?:\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9]+))?\s*\\(?:int|sum|prod|oint|iint)\b/;
		if (bareChainRe.test(text)) balanceErrors.push("three or more bare ``\\int``/``\\sum``/``\\prod`` in a row (no integrand/summand between them)");
		if (bareMacros.length >= 6 && text.length < 120) balanceErrors.push("many ``\\int``/``\\sum``/``\\prod`` macros (" + bareMacros.length + ") for a formula of length " + text.length);

		// HTML-escape the four characters that would otherwise let the
		// user's text break out of the preview div; keep ``\\`` intact so
		// LaTeX commands like ``\\sin`` survive.
		var safe;
		try {
			safe = text
				.replace(/&/g, "&amp;")
				.replace(/</g, "&lt;")
				.replace(/>/g, "&gt;")
				.replace(/[\x00-\x1f]/g, "");
		} catch (e) {
			$err.text("Preview error: failed to escape formula");
			return;
		}

		$err.empty();

		// Show a quick "Loading…" hint so the user sees feedback before
		// MathJax has finished typesetting (it's async, so there can be
		// a noticeable delay the first time around).
		try {
			$prev.empty().html("<span style='color:#777'>&#x2026;rendering preview&#x2026;</span>");
		} catch (e) {
			$err.text("Preview error: failed to clear preview area");
			return;
		}

		// Insert the display-math wrapper directly into the preview div
		// (a child wrapper would otherwise get lost when MathJax replaces
		// the element after typesetting).
		var renderNode;
		try {
			renderNode = document.getElementById("formula_preview");
			if (!renderNode) {
				throw new Error("preview node missing");
			}
			renderNode.innerHTML = "\\[" + safe + "\\]";
		} catch (e) {
			$err.text("Preview error: failed to insert formula into DOM");
			return;
		}

		if (balanceErrors.length) {
			$err.text("Possible issues: " + balanceErrors.join("; "));
		}

		// Ask MathJax to typeset.  Wait for its startup promise first so we
		// don't lose the first render when MathJax is still loading.
		client_mathjax_typeset(renderNode, $err);
	}

	function client_mathjax_typeset(node, $err) {
		// If MathJax isn't loaded at all, show a friendly fallback so
		// the user still sees *something* rather than a blank box.
		if (!window.MathJax) {
			if ($err) {
				$err.text($err.text() ? $err.text() + "; " : "" +
					"MathJax not loaded — showing raw formula.");
			}
			return;
		}
		// MathJax v3 exposes ``MathJax.typesetPromise``.  Earlier versions
		// used ``MathJax.Hub.Queue``; if neither is available we just
		// keep the rendered (but un-typeset) HTML.
		try {
			var run = function () {
				// Strip any previous MathJax output so the user always
				// sees the current formula, not a stale render.
				if (window.MathJax.typesetClear) {
					try { window.MathJax.typesetClear([node]); } catch (_) {}
				} else if (node) {
					var old = node.querySelectorAll("mjx-container, .mjx-container, [data-mathjax]");
					for (var oi = 0; oi < old.length; oi++) {
						if (old[oi].parentNode) old[oi].parentNode.removeChild(old[oi]);
					}
				}
				if (window.MathJax.typesetPromise) {
					window.MathJax.typesetPromise([node])
						.then(function () {
							// After the math is rendered, make auto-generated
							// parameter bounds ([min, max] under braces) clickable
							// for inline editing.
							try {
								_attach_editable_preview_overlays(node);
							} catch (e) {
								console.error("[formula preview] overlay attach failed:", e);
							}
						})
						.catch(function (err) {
							console.error("[formula preview] typeset failed:", err);
							if ($err) {
								$err.text("LaTeX render failed: " + (err && err.message ? err.message : "unknown"));
							}
						});
				} else if (window.MathJax.Hub && window.MathJax.Hub.Queue) {
					window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, node]);
				} else {
					console.warn("[formula preview] no typeset API on MathJax");
				}
			};
			if (window.MathJax.startup && window.MathJax.startup.promise) {
				// Race the startup promise against a hard timeout so the
				// preview never gets stuck in "Loading…" forever.
				var startupTimeout = setTimeout(function () {
					console.warn("[formula preview] MathJax startup timed out, rendering raw formula");
					if ($err) {
						$err.text(($err.text() ? $err.text() + "; " : "") +
							"MathJax startup timed out");
					}
				}, 10000);
				window.MathJax.startup.promise.then(function () {
					clearTimeout(startupTimeout);
					run();
				}, function (err) {
					clearTimeout(startupTimeout);
					console.error("[formula preview] MathJax startup failed:", err);
				});
			} else {
				run();
			}
		} catch (e) {
			console.error("[formula preview] typeset threw:", e);
		}
	}

	function refresh_suggestions() {
		var text = $("#formula").val() || "";
		var mode = current_mode();
		var result;
		try {
			result = client_extract_formula_params(text, mode);
		} catch (e) {
			$("#formula_card #formula_suggestions").html(
				"<span style='color:#b00020'>Broken formula: " + client_escape_html(String(e)) + "</span>"
			);
			return;
		}
		client_render_suggestions(result);
	}

	function update_everything() {
		refresh_suggestions();
		client_render_formula_preview($("#formula").val() || "");
	}

	function client_escape_html(s) {
		return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
			.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
	}

	$("#formula_card #formula_pane_text").on("input", function () {
		var text = $(this).val();
		sync_to_main_textarea(text);
		update_everything();
		if (typeof update_command === "function") update_command();
	});

	// Ctrl/Cmd + Enter on the formula pane applies the suggestions.
	$("#formula_card #formula_pane_text").on("keydown", function (ev) {
		if ((ev.ctrlKey || ev.metaKey) && (ev.key === "Enter" || ev.keyCode === 13)) {
			ev.preventDefault();
			$("#formula_card #formula_apply_btn").trigger("click");
		}
	});

	$("#formula_card #formula_apply_btn").on("click", function () {
		var text = $("#formula").val() || "";
		var mode = current_mode();
		var result = client_extract_formula_params(text, mode);
		client_apply_suggestions(result);
		update_command();
	});

	$("#formula_card #formula_clear_btn").on("click", function () {
		$("#formula_card #formula_pane_text").val("");
		sync_to_main_textarea("");
		update_everything();
		update_command();
	});

	// Initial population from the hidden #formula textarea.
	var initial = $("#formula").val() || "";
	if (initial) {
		$("#formula_card #formula_pane_text").val(initial);
	}
	update_everything();

	_formula_preview_callback = function () {
		refresh_suggestions();
		client_render_formula_preview($("#formula").val() || "");
	};
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
					<td>Log-Scale<a class='tooltip invert_in_dark_mode' title='Ensures parameters (e.g., learning rate, weight decay, epsilon) are optimized in log-space, so all magnitudes are handled evenly.'><img src='i/help.svg' /></a>:</td>
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

	$(newRow).show()

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

		left_side_content += `<a class='tooltip invert_in_dark_mode' title='${escapeQuotes(item.help)}'><img src='i/help.svg' /></a>`;
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

		var $container = $("<div>").attr("id", item.id + "_wrapper");
		$container.append(input);
		valueCell.append($container);

		if (Object.keys(item).includes("onchange")) {
			$(input).change(item.onchange);
		}

		if (Object.keys(item).includes("append_html")) {
			valueCell.append($(item.append_html));
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

	if (item.id === "run_program") {
		valueCell.append($("<div class='run_program_suggestions invert_in_dark_mode' id='run_program_suggestions' style='display: none;'></div>"));
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

	// Field IDs that are emitted by ``pushFormula`` below; skip them
	// here so we don't write the same key twice into the URL.
	var FORMULA_FIELDS = {
		formula: true,
		formula_mode: true,
	};

	function push_value(item) {
		if (FORMULA_FIELDS[item.id]) return;
		var element = $("#" + item.id);
		var value;

		if (element.is(":checkbox")) {
			value = element.is(":checked") ? 1 : 0;
		} else {
			value = element.val();

			// Base64-encode run_program, run_program_once and external_generator
			// for the URL so newlines and special characters survive a refresh
			// and we don't blow past the browser's URL-length limit.
			if (item.id === "run_program" || item.id === "run_program_once" || item.id === "external_generator") {
				if (value && value.trim() !== "") {
					try {
						// encodeURIComponent trick preserves multi-byte chars
						value = btoa(unescape(encodeURIComponent(value)));
					} catch (e) {
						console.error("Base64 encoding failed for " + item.id + ":", e);
					}
				}
			}

			value = encodeURIComponent(value);
		}

		// Skip empty / default values so the URL stays compact.
		if (value === "" || value === "0" || value === "false") {
			// Keep checkboxes (booleans default to 0 = off) but skip empty text fields.
			if (typeof value === "string" && value === "") {
				return;
			}
		}

		params.push(item.id + "=" + value);
	}

	tableData.forEach(function(item) {
		push_value(item);
	});

	hiddenTableData.forEach(function(item) {
		push_value(item);
	});

	// Custom URL params for the formula editor (lives outside tableData).
	(function pushFormula() {
		var formulaEl = $("#formula");
		var modeEl = $("#formula_mode");
		if (formulaEl.length === 0 || modeEl.length === 0) return;
		var formulaVal = formulaEl.val() || "";
		var hasFormula = formulaVal.trim() !== "";
		if (hasFormula) {
			try {
				var b64 = btoa(unescape(encodeURIComponent(formulaVal)));
				params.push("formula=" + encodeURIComponent(b64));
			} catch (e) {
				console.error("Base64 encoding failed for formula:", e);
			}
			params.push("formula_mode=" + encodeURIComponent(modeEl.val() || "auto"));
		}
	})();

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
			parameterIndex++;
		}
		// Rows without a parameter name are skipped but their index is
		// preserved so the URL stays compact (no gaps in numbering).
	});

	if (initialized) {
		var url = window.location.origin + window.location.pathname + "?" + params.join("&") + "&num_parameters=" + parameterIndex;

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

	setup_formula_editor();

	var urlParams = new URLSearchParams(window.location.search);

	// --- tableData loop ---
	tableData.forEach(function(item) {
		var paramValue = urlParams.get(item.id);

		if (paramValue !== null) {
			// Strip surrounding single quotes if present
			if (item.id === "result_names") {
				paramValue = paramValue.replace(/^'(.*)'$/, '$1');
			}

			// Base64-decode the long text fields.  We detect them by trying
			// to decode; if that fails we leave the value alone (so the
			// URL-encoded form still works for older / hand-crafted URLs).
			if (item.id === "run_program" || item.id === "run_program_once" || item.id === "external_generator") {
				if (paramValue !== "") {
					try {
						paramValue = decodeURIComponent(escape(atob(paramValue)));
					} catch (e) {
						// Not base64 — keep the raw URL-decoded value.
						try {
							paramValue = decodeURIComponent(paramValue);
						} catch (e2) { /* keep as-is */ }
					}
				}
			}

			var $element = $("#" + item.id);
			if ($element.is(":checkbox")) {
				var boolValue = /^(1|true)$/i.test(paramValue);
				$element.prop("checked", boolValue).trigger("change");
			} else {
				$element.val(paramValue).trigger("change");
			}
		}
	});

	// --- formula editor (lives outside tableData) ---
	(function restoreFormula() {
		var fm = urlParams.get("formula");
		var fmMode = urlParams.get("formula_mode") || "auto";
		var fmEmpty = (fm === null || fm === "");
		var rpEl = document.getElementById("run_program");
		var rpEmpty = !rpEl || !(rpEl.value || "").trim();
		// If the URL carries a formula, open the formula card immediately
		// and hide run_program so the scientist lands back in the same
		// workflow they left.
		if (!fmEmpty) {
			fm = fm.replace(/^'(.*)'$/, '$1');
			try {
				if (is_base64_like(fm)) {
				fm = decodeURIComponent(escape(atob(fm)));
			}
			} catch(e) {
				// If atob throws (not valid base64), keep the raw value.
			}
			// If not base64, it's already the raw formula (e.g. from a
			// share-page link) — use it as-is.
			// Validate the mode before assigning.
			var validModes = ["auto", "latex", "infix"];
			if (validModes.indexOf(fmMode) < 0) fmMode = "auto";
			$("#formula").val(fm);
			$("#formula_mode").val(fmMode);
			// Open the formula card and hide run_program to mirror the
			// scientist's prior state.
			var $card = $("#formula_card");
			if ($card.length && !$card.data("built")) {
				$card.html(build_formula_card_html());
				$card.data("built", true);
				setup_formula_card_inner();
			}
			_set_tab("formula");
		// Clear any leftover run_program text — when the formula
		// editor is active we generate the run_program automatically,
		// so the leftover is just confusing.
		$("#run_program").val("").trigger("change");
		} else if (rpEmpty && fmEmpty) {
			// Nothing yet: keep run_program visible and don't auto-open
			// the formula editor — let the scientist pick.
		}
		// Defer so the formula card's input handlers are wired up.
		setTimeout(function () {
			if (!fmEmpty) {
				// Push the formula into all three panes and trigger the
				// active pane so ``update_everything`` runs and MathJax
				// gets a chance to render the preview.
				$("#formula_card #formula_pane_text").val(fm);
				$("#formula_pane_text").trigger("input");
				// Belt-and-suspenders: also call the renderer directly so
				// the preview shows up even if the input handler is
				// somehow shadowed by a third-party script.
				if (typeof _formula_preview_callback === "function") {
					_formula_preview_callback();
				}
				// Auto-apply the suggestions so the parameter table is
				// populated when restoring from URL.
				setTimeout(function () {
					$("#formula_card #formula_apply_btn").trigger("click");
				}, 20);
			}
		}, 0);
	})();

	// --- hiddenTableData loop ---
	hiddenTableData.forEach(function(item) {
		var paramValue = urlParams.get(item.id);
		if (paramValue !== null) {
			// Base64-decode for run_program_once and external_generator
			if (item.id === "run_program_once" || item.id === "external_generator") {
				// Strip surrounding single quotes FIRST (they come from the URL encoding)
				paramValue = paramValue.replace(/^'(.*)'$/, '$1');

				try {
					paramValue = decodeURIComponent(escape(atob(paramValue)));
				} catch (e) {
					console.error("Base64 decoding failed for " + item.id + ":", e);
				}
			}

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

	setup_run_program_history();

	initialized = true;

	fadeTime = fadeTimeAfterLoading;

	setTimeout(update_command, 200);

	update_url();

	update_command();

	document_is_ready = true;
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
	const table = document.getElementById("hidden_config_table");
	if (table.style.display !== "none") return;

	const errors = table.querySelectorAll(".error_element");
	for (let el of errors) {
		if (el.offsetParent !== null && el.textContent.trim() !== "") {
			table.style.display = "";
			return;
		}
	}
}

function show_warning_for_model_when_custom_generation_strategy_is_set() {
	$("#model_error").html("Custom generation strategy is set, so --model is ignored.").show();
}

function hide_warning_when_custom_custom_generation_strategy_isnt_set() {
	$("#model_error").html("").hide()
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
