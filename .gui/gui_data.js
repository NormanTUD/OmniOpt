"use strict";

var model_data = [
	{
		id: 'BOTORCH_MODULAR',
		name: 'Default model',
		link: 'https://web.archive.org/web/20240715080430/https://proceedings.neurips.cc/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf'
	},
	{
		id: 'SOBOL',
		name: 'SOBOL: Quasi-random sequence',
		link: 'https://en.wikipedia.org/wiki/Sobol_sequence'
	},
	{
		id: 'PSEUDORANDOM',
		name: 'PSEUDORANDOM: Uniformly random sequence. PSEUDORANDOM runs cannot be continued (yet).',
		link: 'https://en.wikipedia.org/wiki/Pseudorandomness',
		hide_in_custom_generation_strategy: true
	},
	{
		id: 'FACTORIAL',
		name: 'All possible combinations',
		link: 'https://ax.dev/docs/tutorials/factorial/'
	},
	{
		id: 'SAASBO',
		name: 'Sparse Axis-Aligned Subspace Bayesian Optimization',
		link: 'https://arxiv.org/pdf/2103.00349',
		italic: true,
		extra: ' for high-dimensional Bayesian Optimization, recommended for hundreds of dimensions'
	},
	{
		id: 'UNIFORM',
		name: 'Random (uniformly distributed)'
	},
	{
		id: 'BO_MIXED',
		name: 'Optimizes all range parameters once for each combination of choice parameters, then takes the optimum of those optima.',
		extra: 'The cost associated with this method grows with the number of combinations, and so it is only used when the number of enumerated discrete combinations is below some maximum value.'
	},
	{
		id: 'RANDOMFOREST',
		name: 'Random Forest',
		link: 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html',
		extra: ' Tree-based ensemble method often used as a baseline or benchmark. It is robust to overfitting and works well with default settings. Hyperparameters like number of trees, depth, and max features can be optimized.',
		hide_in_custom_generation_strategy: true
	},
	{
		id: 'EXTERNAL_GENERATOR',
		name: 'External Generator',
		//link: 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html',
		extra: ' Use an external program to create new hyperparameter points from previous data. Requires --external_generator to be set to a program that fulfills certain requirements. <a href="tutorials?tutorial=external_generator" target="_blank">See this tutorial on how to use external generators</a>. EXTERNAL_GENERATOR-runs cannot be continued (yet).',
		hide_in_custom_generation_strategy: true
	}
];

var valid_models_with_descriptions = Object.fromEntries(
	model_data.map(model => [
		model.id, 
		model.link ? `<a target="_blank" href="${model.link}">${model.italic ? '<i>' + model.name + '</i>' : model.name}</a>${model.extra || ''}` : model.name + (model.extra || '')
	])
);

var model_list_html = `<ul class="make_markable">${model_data.map(model => `
	<li><strong>${model.id}</strong>: ${valid_models_with_descriptions[model.id]}</li>
`).join('')}</ul>`;

var model_list_html_custom_generation_strategy = `<ul class="make_markable">${model_data.map(model => Object.keys(model).includes("hide_in_custom_generation_strategy") ? "" : `
	<li><strong>${model.id}</strong>: ${valid_models_with_descriptions[model.id]}</li>
`).join('')}</ul>`;

var valid_models = model_data.map(model => model.id);

var valid_models_generation_strategy = model_data
    .filter(model => !model.hide_in_custom_generation_strategy)
    .map(model => model.id);

var regex_path = "^(/([a-zA-Z0-9_-]+/?)*)?$";

var tableData = [
	{
		label: "Partition",
		id: "partition",
		type: "select",
		value: "",
		options: [],
		required: true,
		help: "The Partition your job will run on. This choice may restrict the amount of workers, GPUs, maximum time limits and a few more options."
	},
	{
		label: "Experiment name",
		id: "experiment_name",
		type: "text",
		value: "",
		placeholder: "Name of your experiment (only letters and numbers)",
		required: true,
		regex: "^[a-zA-Z0-9_]+$",
		help: "Name of your experiment. Will be used for example for the foldername it's results will be saved in.",
		regex_does_not_match_text: "The experiment name must consist of alphanumeric characters and underscores only."
	},
	{
		label: "Reservation",
		id: "reservation",
		type: "text",
		value: "",
		placeholder: "Name of your reservation (optional)",
		required: false,
		help: "If you have a reservation, use it here. It makes jobs start faster, but is not necessary technically.",
		regex: "^[a-zA-Z0-9_]*$",
		regex_does_not_match_text: "The reservation name must consist of alphanumeric characters and underscores only."
	},
	{
		label: "Account",
		id: "account",
		type: "text",
		value: "",
		placeholder: "Account the job should run on",
		help: "Depending on which groups you are on, this determines to which account group on the Slurm-system that job should be linked. If left empty, it will solely be determined by your login-account.",
		regex: "^[a-zA-Z0-9_]*$",
		regex_does_not_match_text: "The account name must consist of alphanumeric characters and underscores only."
	},
	{
		label: "Memory (in GB)",
		id: "mem_gb",
		type: "number",
		value: 10,
		placeholder: "Memory in GB per worker",
		min: 1,
		max: 1000
	},
	{
		label: "Timeout for the main program",
		id: "time",
		type: "number",
		value: 60,
		placeholder: "Timeout for the whole program",
		min: 1,
		help: "This is the maximum amount of time that your main job will run, spawn jobs and collect results."
	},
	{
		label: "Timeout for a single worker",
		id: "worker_timeout",
		type: "number",
		value: 60,
		placeholder: "Timeout for a single worker",
		min: 1,
		help: "This is the maximum amount of time a single worker may run."
	},
	{
		label: "Maximal number of evaluations",
		id: "max_eval",
		type: "number",
		value: 500,
		placeholder: "Maximum number of evaluations",
		min: 1,
		max: 100000000,
		help: "This number determines how many successful workers in total are needed to end the job properly."
	},
	{
		label: "Max. number of Workers",
		id: "num_parallel_jobs",
		type: "number",
		value: 20,
		placeholder: "Maximum number of workers",
		min: 1,
		max: 100000000,
		help: "The number maximum of workers that can run in parallel. While running, the number may be below this some times."
	},
	{
		label: "GPUs per Worker",
		id: "gpus",
		type: "number",
		value: 0,
		placeholder: "Number of GPUs per worker",
		min: 0,
		max: 10,
		help: "How many GPUs each worker should have."
	},
	{
		label: "Number of random steps",
		id: "num_random_steps",
		type: "number",
		value: 20,
		placeholder: "Number of random steps",
		min: 1,
		help: "At the beginning, some random jobs are started. By default, it is 20. This is needed to 'calibrate' the surrogate model."
	},
	{
		label: "Follow",
		id: "follow",
		type: "checkbox",
		value: 1,
		help: "tail -f the .out-file automatically, so you can see the output as soon as it appears. This does not change the results of OmniOpt2, but only the user-experience. This way, you see results as soon as they are available without needing to manually look for the outfile. Due to it using tail -f internally, you can simply CTRL-c out of it without cancelling the job.",
		info: `With <i>Follow</i>, the job's .out file will automatically be <code style="white-space: pre" class="highlight_me dark_code_bg invert_in_dark_mode">tail -f</code>'d on Slurm-systems. Having this set or not does not change the output of your calculations.`
	},
	{
		label: "Live-Share",
		id: "live_share",
		type: "checkbox",
		value: 0,
		help: "Automatically uploads the results to our servers for 30 days, so you can trace the output live in the browser, without needing SSH.",
		info: "By using this, you agree to have your username published online."
	},
	{
		label: "Send anonymized usage statistics?",
		id: "send_anonymized_usage_stats",
		type: "checkbox",
		value: 1,
		help: "This contains the time the job was started and ended, it's exit code, and runtime-uuid to count the number of unique runs and a 'user-id', which is a hashed output of the aes256 encrypted username/groups combination and some other values, but cannot be traced back to any specific user."
	},
	{
		label: "Automatically checkout to latest checked version",
		id: "checkout_to_latest_tested_version",
		type: "checkbox",
		value: 1,
		help: "For every commit, the CI pipeline checks all the tests and if they succeed, create a new version tag. If this is activated, you get the latest version that was tested properly and where all tests succeeded. If disabled, you may get the newest version, but it may has preventable bugs."
	},
	{
		label: "Constraints",
		id: "constraints",
		type: "text",
		value: "",
		placeholder: "Constraints like 'a + b >= 10', seperated by Semicolon (;)",
		info: "Use linear constraints in the form of <code>a*x + b*y - cz <= d</code>, where <code>a</code>, <code>b</code>, <code>c</code>, and <code>d</code> are float constants, and <code>x</code>, <code>y</code>, <code>z</code> are parameter names.There should be no space in each term around the operator <code></code> while there should be a single space around each operator <code>+</code>, <code>-</code>, <code><=</code>, and <code>>=</code>.",
		help: "The contraints allow you to limit values of the hyperparameter space that are allowed. For example, you can set that the sum of all or some parameters must be below a certain number. This may be useful for simulations, or complex functions that have certain limitations depending on the hyperparameters."
	},
	{
		label: "Result-Names",
		id: "result_names",
		type: "text",
		value: "RESULT=min",
		placeholder: "Name of the value that should be searched for, like 'result'",
		required: true,
		regex: /^(((([a-zA-Z][a-zA-Z0-9_]*)(=(min|max)(\s\s*|$))?)(\s|$)?)+)$/,
		help: "A space-seperated list of strings to search for in the STDOUT of your program like, for example, the loss. Default is RESULT=min.",
		info: "This is used for the regex to search through the <tt>STDOUT</tt> of your program to find result-values. You can define multiple result values like this: <code class='highlight_me dark_code_bg invert_in_dark_mode'>result1 result2 result3</code>. Can also be defined with min and max: <code class='highlight_me dark_code_bg invert_in_dark_mode'>LOSS=min PERFORMANCE=max ...</code>. Default, if min or max is not specified, is minimizing. Adding values here is the same as doing Multi-Objective-Optimization.",
		regex_does_not_match_text: "The value must consist of one or more alphanumeric identifiers separated by spaces. Each identifier can optionally have an assignment, which can either be '=min' or '=max', which is optional though. Default is min. The assignment must end with a space or the end of the string. Each identifier must start with a letter and can contain letters, numbers, or underscores."

	},
	{
		label: "Run program",
		id: "run_program",
		type: "textarea",
		value: "",
		placeholder: "Your program with parameters",
		required: true,
		info: "Use Variable names like this: <br><code style=\"white-space: pre\" class=\"highlight_me dark_code_bg invert_in_dark_mode\">bash /absolute/path/to/run.sh --lr=%(lr) --epochs=%(epochs)</code>. See <a target=\"_blank\" href=\"tutorials.php?tutorial=run_sh\">this tutorial</a> to learn about the <code>run.sh</code>-file",
		help: "This is the program that will be optimized. Use placeholder names for places where your hyperparameters should be, like '%(epochs)'. The GUI will warn you about missing parameter definitions, that need to be there in the parameter selection menu, and will not allow you to run OmniOpt2 unless all parameters are filled."
	}
];

var hiddenTableData = [
	{
		label: "CPUs per Task",
		id: "cpus_per_task",
		type: "number",
		value: 1,
		placeholder: "CPUs per Task",
		min: 1,
		max: 10,
		help: "How many CPUs should be assigned to each task (for workers)"
	},
	{
		label: "Number of nodes",
		id: "nodes_per_job",
		type: "number",
		value: 1,
		placeholder: "tasks",
		min: 1,
		help: "How many nodes (for each worker)"
	},
	{
		label: "Seed",
		id: "seed",
		type: "number",
		value: "",
		placeholder: "Seed for reproducibility",
		info: "When set, this will make OmniOpt2 runs reproducible, given your program also acts deterministically. If on Sbatch-Systems, you need to have <code style='white-space: pre' class='highlight_me dark_code_bg invert_in_dark_mode'>--generate_all_jobs_at_once</code> enabled, otherwise, runs will not be deterministic!",
		required: false
	},
	{
		label: "Verbose",
		id: "verbose",
		type: "checkbox",
		value: 0,
		help: "This enables more output to be shown. Useful for debugging. Does not change the outcome of your Optimization."
	},
	{
		label: "Generate all jobs at once",
		id: "generate_all_jobs_at_once",
		type: "checkbox",
		value: 1,
		help: "This generates all hyperparameter sets for the set of workers at once."
	},
	{
		label: "Debug",
		id: "debug",
		type: "checkbox",
		value: 0,
		help: "This enables more output to be shown. Useful for debugging. Does not change the outcome of your Optimization."
	},
	{
		label: "Revert to random",
		id: "revert_to_random_when_seemingly_exhausted",
		type: "checkbox",
		value: 1,
		help: "Reverts to random model if the systematic model cannot generate new points."
	},
	{
		label: "Grid search?",
		id: "gridsearch",
		type: "checkbox",
		value: 0,
		info: "Switches range parameters to choice with <tt>max_eval</tt> number of steps. Converted to int when parameter is int. Only use together with the <i>FACTORIAL</i>-model.",
		help: "This internally converts range parameters to choice parameters by laying them out seperated by the max eval number through the search space with intervals. Use FACTORIAL model to make it work properly. Still beta, though! (TOOD)"
	},
	{
		label: "Model",
		id: "model",
		type: "select",
		value: "",
		options: valid_models.map(model => ({ text: model, value: model })),
		required: true,
		info: model_list_html,
		help: "The model chosen here tries to make an informed choice (except SOBOL, which means random search) about where to look for new hyperparameters. Different models are useful for different optimization problems, though which is best for what is something that I still need to search exactly (TODO!)"
	},
	{
		label: "External program for EXTERNAL_GENERATOR",
		id: "external_generator",
		type: "text",
		value: "",
		info: "A call to a program to calculate hyperparameters. <a href='tutorials?tutorial=external_generator' target='_blank'>See this tutorial on how to use external generators</a>. For this option to take effect, the '--model' must be set to 'EXTERNAL_GENERATOR'.",
		required: false
	},
	{
		label: "Number of estimators for RANDOMFOREST",
		id: "n_estimators_randomforest",
		type: "number",
		value: 100,
		info: "The number of trees in the forest for RANDOMFOREST (default: 100). This is ignored when you don't have the --model set to RANDOMFOREST",
		min: 1
	},
	{
		label: "Installation-Method",
		id: "installation_method",
		type: "select",
		value: "",
		options: [
			{
				text: "Use git clone to clone OmniOpt2",
				value: "clone"
			},
			{
				text: "Use pip and install OmniOpt2 from pypi (may not be the latest version)",
				value: "pip"
			}
		],
		required: true,
		info: "Changes the way OmniOpt2 is installed.",
		help: "If you want to install OmniOpt2 via pip, chose it here. It may not always have the latest version.",
		use_in_curl_bash: true
	},
	{
		label: "Run-Mode",
		id: "run_mode",
		type: "select",
		value: "",
		options: [
			{
				text: "Locally or on a HPC system",
				value: "local"
			},
			{
				text: "Docker",
				value: "docker"
			}
		],
		required: true,
		info: "Changes the curl-command and how omniopt is installed and executed.",
		help: "If set to docker, it will run in a local docker container."
	},
	{
		label: "Decimal places",
		id: "decimalrounding",
		type: "number",
		value: 4,
		placeholder: "Number of decimal places to be rounded to",
		min: 0,
		max: 32
	},
	{
		label: "Disable TQDM",
		id: "disable_tqdm",
		type: "checkbox",
		value: 0,
		info: "Disable TQDM."
	},
	{
		label: "Verbose TQDM?",
		id: "verbose_tqdm",
		type: "checkbox",
		value: 0,
		info: "Show more verbose TQDM output."
	},
	{
		label: "Force local execution?",
		id: "force_local_execution",
		type: "checkbox",
		value: 0,
		info: "Forces local execution, even when SLURM is installed."
	},
	{
		label: "Auto exclude defective hosts?",
		id: "auto_exclude_defective_hosts",
		type: "checkbox",
		value: 0,
		info: "If set, before each evaluation there is a tiny test run on the GPU (if available) to test if the GPU is working properly. If not, the host chosen will be added to the excluded hosts list automatically."
	},
	{
		label: "Show SIXEL general?",
		id: "show_sixel_general",
		type: "checkbox",
		value: 0,
		info: "Show sixel general."
	},
	{
		label: "Show SIXEL trial-index-to-result?",
		id: "show_sixel_trial_index_result",
		type: "checkbox",
		value: 0,
		info: "Show sixel trial-index-to-result."
	},
	{
		label: "Show SIXEL scatter?",
		id: "show_sixel_scatter",
		type: "checkbox",
		value: 0,
		info: "Show sixel scatter graphic."
	},
	{
		label: "Show worker percentage table?",
		id: "show_worker_percentage_table_at_end",
		type: "checkbox",
		value: 0,
		info: "Shows a table of the workers percentage at the end."
	},
	{
		label: "Optimization with combined criteria?",
		id: "occ",
		type: "checkbox",
		value: 0,
		info: "Use optimization with combined criteria (OCC)."
	},
	{
		label: "Optimization with combined criteria type",
		id: "occ_type",
		type: "select",
		value: "euclid",
		options: [
			{
				text: "Calculate the euclidean distance to the origo of the search space",
				value: "euclid"
			},
			{
				text: "Calculate the geometric distance to the origo of the search space",
				value: "geometric"
			},
			{
				text: "Calculate the signed harmonic distance to the origo of the search space",
				value: "signed_harmonic"
			},
		],
		required: true,
		info: "How to merge multiple results into one. Doesn't affect single result jobs or jobs with specified multiple result names.",
		help: "How to merge multiple results into one."
	},
	{
		label: "No sleep?",
		id: "no_sleep",
		type: "checkbox",
		value: 0,
		info: "Disables sleeping in certain parts of the code."
	},
	{
		label: "Use srun instead of sbatch?",
		id: "slurm_use_srun",
		type: "checkbox",
		value: 0,
		info: "Use srun instead of sbatch for starting slurm jobs"
	},
	{
		label: "Verbose break run search table?",
		id: "verbose_break_run_search_table",
		type: "checkbox",
		value: 0,
		info: "Show a verbose table when the break_run_search is run."
	},
	{
		label: "Abbreviate job names?",
		id: "abbreviate_job_names",
		type: "checkbox",
		value: 0,
		info: "Abbreviate job names (Running -> R, Pending -> P and so on)."
	},
	{
		label: "Main process GB",
		id: "main_process_gb",
		type: "number",
		value: 8,
		info: "How much RAM the main process should have. Default is 8GB.",
		min: 1
	},
	{
		label: "Max nr. of zero results",
		id: "max_nr_of_zero_results",
		type: "number",
		value: 50,
		info: "Max nr. of jobs where the fetch_next_trial is empty to be considered as search space exhausted",
		min: 1
	},
	{
		label: "Pareto-Front-confidence",
		id: "pareto_front_confidence",
		type: "number",
		value: 1,
		info: "How certain OmniOpt2 should be regarding the pareto-front to count values in",
		min: 0,
		max: 1,
		step: 0.01
	},
	{
		label: "Slurm-Signal-Delay",
		id: "slurm_signal_delay_s",
		type: "number",
		value: 0,
		info: "When the workers end, they get a signal so your program can react to it. Default is 0, but set it to any number of seconds you wish your program to be able to react to USR1.",
		min: 0
	},

	{
		label: "Max nr. failed jobs",
		id: "max_failed_jobs",
		type: "number",
		value: 0,
		info: "How many failed jobs should suffice to cancel the whole job. If set to 0 or not at all, it will be --max_eval.",
		min: 0
	},
	{
		label: "Exclude",
		id: "exclude",
		type: "text",
		value: "",
		placeholder: "A comma separated list of values of excluded nodes (taurusi8009,taurusi8010)",
		required: false,
		regex: "^([a-zA-Z0-9_]+,?)*([a-zA-Z0-9_]+)*$",
		info: "A comma separated list of values of excluded nodes (taurusi8009,taurusi8010)",
		regex_does_not_match_text: "The value must consist of one or more alphanumeric identifiers (letters, numbers, or underscores), separated by commas. No leading or trailing commas are allowed."
	},

	{
		label: "Username",
		id: "username",
		type: "text",
		value: "",
		placeholder: "The username that will be used for OmniOpt2 share, if used at all",
		required: false,
		regex: "^([a-zA-Z0-9_]+,?)*([a-zA-Z0-9_]+)*$",
		regex_does_not_match_text: "The value must consist of one or more alphanumeric identifiers (letters, numbers, or underscores), separated by commas. No leading or trailing commas are allowed."
	},
	{
		label: "Generation strategy",
		id: "generation_strategy",
		onblur: function() { 
			toggle_model_warning_for_custom_generation_strategy();
		},
		onfocus: function() { 
			toggle_model_warning_for_custom_generation_strategy();
		},
		onkeypress: function() { 
			toggle_model_warning_for_custom_generation_strategy();
		},
		onchange: function() { 
			toggle_model_warning_for_custom_generation_strategy();
		},
		type: "text",
		value: "",
		info: `A comma-seperated list of strings of the form 'MODELNAME=count', for example, <code style="white-space: pre" class="highlight_me dark_code_bg invert_in_dark_mode">SOBOL=10,BOTORCH_MODULAR=20,SOBOL=10</code>. This will override the number of random steps and the --model option. Valid models are: ${model_list_html_custom_generation_strategy}`,
		required: false,
		regex: `^((?:${valid_models_generation_strategy.join("|")})+=\\d+,?)*$`,
		help: "Specify a custom generation strategy",
		regex_does_not_match_text: `The value must consist of one or more strategies from the list: ${valid_models_generation_strategy.join(", ")}. Each strategy must be followed by '=<number>' and can be separated by commas. No trailing commas are allowed.`
	},
	{
		label: "Root venv dir",
		id: "root_venv_dir",
		type: "text",
		value: "",
		info: "An absolute path where the virtual env should be installed to",
		required: false,
		regex: regex_path,
		regex_does_not_match_text: "The value must be a valid directory path, which can include alphanumeric characters, underscores, hyphens, and forward slashes. Each segment of the path must start and end with a valid character, and no trailing slashes are allowed at the end.",
		help: "Path where the virtual env should be installed to",
	},
	{
		label: "Workdir",
		id: "workdir",
		type: "text",
		value: "",
		info: "An absolute path what the working directory of jobs should be",
		required: false,
		regex: regex_path,
		regex_does_not_match_text: "The value must be a valid directory path, which can include alphanumeric characters, underscores, hyphens, and forward slashes. Each segment of the path must start and end with a valid character, and no trailing slashes are allowed at the end.",
	},
	{
		label: "Do not JIT-Compile",
		id: "dont_jit_compile",
		type: "checkbox",
		value: 0,
		info: "JIT-Compiling makes the model fitting faster by Just-in-Time-Compiling the Python-Code creating new models"
	},
	{
		label: "Fit out of design",
		id: "fit_out_of_design",
		type: "checkbox",
		value: 0,
		info: "Ignore data points outside of the design to improve generation-of-points-speed"
	},
	{
		label: "Refit on Cross-Validation",
		id: "refit_on_cv",
		type: "checkbox",
		value: 0,
		info: "Enable warm-fitting for Cross-validation on model-fitting (may lead to better results, but is slower)"
	},
	{
		label: "Show a table at the end showing how much time was spent at the different iterations of the model to create new points",
		id: "show_generate_time_table",
		type: "checkbox",
		value: 0
	},
	{
		label: "Don't warm-start-refitting",
		id: "dont_warm_start_refitting",
		type: "checkbox",
		value: 0,
		info: "Disable warm-fitting for model-fitting"
	},
	{
		label: "Max. attempts for generating new points",
		id: "max_attempts_for_generation",
		type: "number",
		value: 20,
		placeholder: "Number of attempts for generating the sets of new hyperparameters",
		min: 1,
	},
	{
		label: "num_restarts Option for optimizer_options",
		id: "num_restarts",
		type: "number",
		value: 5,
		placeholder: "Number of restarts for the Optimizer",
		min: 1,
	},
	{
		label: "raw_samples Option for the optimizer_options",
		id: "raw_samples",
		type: "number",
		value: 128,
		placeholder: "Number of raw samples for the Optimizer",
		min: 1,
	},
	{
		label: "Disable input transformations",
		id: "no_transform_inputs",
		type: "checkbox",
		value: 0,
		info: "Skip transforming input parameters (e.g., scaling, encoding) before training. Speeds up generation but may hurt model quality."
	},
	{
		label: "Also fit jobs that were abandoned",
		id: "fit_abandoned",
		type: "checkbox",
		value: 0,
		info: "Use jobs that were abandoned while fitting a new model."
	},
	{
		label: "Force sequential acquisition function",
		id: "acquisition_sequential",
		type: "checkbox",
		value: 0,
		info: "Using a non-sequential acquisition function may speed up the process of OmniOpt2's point generation."
	},
	{
		label: "Disable target normalization",
		id: "no_normalize_y",
		type: "checkbox",
		value: 0,
		info: "Avoid normalizing the output (objective) values. Faster, but less stable when outputs vary in scale."
	}
];
