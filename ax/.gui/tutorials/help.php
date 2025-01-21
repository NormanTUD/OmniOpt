<h1><samp>--help</samp></h1>

<!-- The <samp>--help</samp> of OmniOpt2 -->

<div id="toc"></div>

<h2 id="available_parameters">Available parameters (<samp>--help</samp>)</h2>

	<table>
		<thead>
			<tr class="invert_in_dark_mode">
				<th>Parameter</th>
				<th>Description</th>
				<th>Default Value</th>
			</tr>
		</thead>
		<tbody>
			<tr class="section-header invert_in_dark_mode">
				<td colspan="3">Required Arguments</td>
			</tr>
			<tr>
				<td><samp>--num_random_steps NUM_RANDOM_STEPS</samp></td>
				<td>Number of random steps to start with.</td>
				<td><samp>20</samp></td>
			</tr>
			<tr>
				<td><samp>--max_eval MAX_EVAL</samp></td>
				<td>Maximum number of evaluations.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--run_program RUN_PROGRAM [RUN_PROGRAM ...]</samp></td>
				<td>A program that should be run. Use, for example, <samp>%(x)</samp> for the parameter named <i>x</i>.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--experiment_name EXPERIMENT_NAME</samp></td>
				<td>Name of the experiment.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--mem_gb MEM_GB</samp></td>
				<td>Amount of RAM for each worker in GB.</td>
				<td><samp>1</samp></td>
			</tr>
			<tr class="section-header invert_in_dark_mode">
				<td colspan="3">Required Arguments That Allow A Choice</td>
			</tr>
			<tr>
				<td><samp>--parameter PARAMETER [PARAMETER ...]</samp></td>
				<td>Experiment parameters in the formats: <br>
					- <samp>&lt;NAME&gt; range &lt;NAME&gt; &lt;LOWER BOUND&gt; &lt;UPPER BOUND&gt; (&lt;INT, FLOAT&gt;) (&lt;log_scale: true or false, default false&gt;)</samp><br>
					- <samp>&lt;NAME&gt; fixed &lt;NAME&gt; &lt;VALUE&gt;</samp><br>
					- <samp>&lt;NAME&gt; choice &lt;NAME&gt; &lt;Comma-separated list of values&gt;</samp>
				</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--continue_previous_job CONTINUE_PREVIOUS_JOB</samp></td>
				<td>Continue from a previous checkpoint, use run-dir as argument.</td>
				<td>-</td>
			</tr>
			<tr class="section-header invert_in_dark_mode">
				<td colspan="3">Optional</td>
			</tr>
			<tr>
				<td><samp>--result_names</samp></td>
				<td>Name of hyperparameters. Example --result_names result1=max result2=min result3. Default: result=min, or result=max when --maximize is set. Default is min.</td>
				<td><samp>result=<min, max></samp></td>
			</tr>
			<tr>
				<td><samp>--moo_type</samp></td>
				<td>Set the type of the multi-objective-parameter collection function. Options are geometric, signed harmonic, and euclid.</td>
				<td><samp>euclid</samp></td>
			</tr>
			<tr>
				<td><samp>--max_parallelism</samp></td>
				<td>Set how the ax max parallelism flag should be set. Possible options: <samp>None</samp>, <samp>max_eval</samp>, <samp>num_parallel_jobs</samp>, <samp>twice_max_eval</samp>, <samp>twice_num_parallel_jobs</samp>, <samp>max_eval_times_thousand_plus_thousand</samp>, or any positive integer.</td>
				<td><samp>max_eval_times_thousand_plus_thousand</samp></td>
			</tr>
			<tr>
				<td><samp>--should_deduplicate</samp></td>
				<td>Try to de-duplicate ARMs</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--workdir</samp></td>
				<td>Working directory (doesn't yet work! TODO)</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--disable_tqdm</samp></td>
				<td>Disable the TQDM progress bar</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--live_share</samp></td>
				<td>Automatically live-share the current optimization run automatically</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--checkout_to_latest_tested_version</samp></td>
				<td>Automatically checkout to latest version that was tested in the CI pipeline</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--exclude "taurusi8009,taurusi8010"</samp></td>
				<td>A comma seperated list of values of excluded nodes.</td>
				<td><samp>None</samp></td>
			</tr>
			<tr>
				<td><samp>--maximize</samp></td>
				<td>Maximize instead of minimize (which is default).</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--experiment_constraints EXPERIMENT_CONSTRAINTS [EXPERIMENT_CONSTRAINTS ...]</samp></td>
				<td>Constraints for parameters. Example: <samp>x + y &lt;= 2.0</samp>.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--stderr_to_stdout</samp></td>
				<td>Redirect stderr to stdout for subjobs.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--run_dir RUN_DIR</samp></td>
				<td>Directory in which runs should be saved.</td>
				<td><samp>runs</samp></td>
			</tr>
			<tr>
				<td><samp>--seed SEED</samp></td>
				<td>Seed for random number generator.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--enforce_sequential_optimization</samp></td>
				<td>Enforce sequential optimization.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--verbose_tqdm</samp></td>
				<td>Show verbose tqdm messages.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--load_previous_job_data LOAD_PREVIOUS_JOB_DATA [LOAD_PREVIOUS_JOB_DATA ...]</samp></td>
				<td>Paths of previous jobs to load from.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--hide_ascii_plots</samp></td>
				<td>Hide ASCII-plots.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--model MODEL</samp></td>
				<td>Use special models for nonrandom steps. Valid models are: SOBOL, GPEI, FACTORIAL, SAASBO, FULLYBAYESIAN, LEGACY_BOTORCH, BOTORCH_MODULAR, UNIFORM, BO_MIXED.</td>
				<td><samp>BOTORCH_MODULAR</samp></td>
			</tr>
			<tr>
				<td><samp>--gridsearch</samp></td>
				<td>Enable gridsearch.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--occ</samp></td>
				<td>Enable optimization with combined criteria.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--show_sixel_scatter</samp></td>
				<td>Show <a href="https://en.wikipedia.org/wiki/Sixel" target="_blank">sixel</a> graphics of scatter plot in the end.</td>
				<td><samp>False</samp></td>
			</tr>

			<tr>
				<td><samp>--show_sixel_general</samp></td>
				<td>Show <a href="https://en.wikipedia.org/wiki/Sixel" target="_blank">sixel</a> graphics of general plot in the end.</td>
				<td><samp>False</samp></td>
			</tr>

			<tr>
				<td><samp>--show_sixel_trial_index_result</samp></td>
				<td>Show <a href="https://en.wikipedia.org/wiki/Sixel" target="_blank">sixel</a> graphics of trial_index_result plot in the end.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--follow</samp></td>
				<td>Automatically follow log file of sbatch.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--send_anonymized_usage_stats</samp></td>
				<td>Send anonymized usage stats.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--ui_url UI_URL</samp></td>
				<td>Site from which the OO-run was called.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--root_venv_dir ROOT_VENV_DIR</samp></td>
				<td>Where to install your modules to (<samp>$root_venv_dir/.omniax_...</samp>)</td>
				<td><samp>$HOME</samp></td>
			</tr>
			<tr>
				<td><samp>--main_process_gb (INT)</samp></td>
				<td>Amount of RAM the main process should have</td>
				<td><samp>4</samp></td>
			</tr>
			<tr>
				<td><samp>--max_nr_of_zero_results (INT)</samp></td>
				<td>Max. nr of successive zero results by ax_client.get_next_trials() before the search space is seen as exhausted.</td>
				<td><samp>20</samp></td>
			</tr>
			<tr>
				<td><samp>--abbreviate_job_names</samp></td>
				<td>Abbreviate pending job names (r = running, p = pending, u = unknown, c = cancelling)</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--disable_search_space_exhaustion_detection</samp></td>
				<td>Disables automatic search space reduction detection</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--orchestrator_file PATH/TO/orchestrator.YAML</samp></td>
				<td>An orchestrator file.</td>
				<td><samp>None</samp></td>
			</tr>
			<tr class="section-header invert_in_dark_mode">
				<td colspan="3">SLURM</td>
			</tr>
			<tr>
				<td><samp>--worker_timeout WORKER_TIMEOUT</samp></td>
				<td>Timeout for slurm jobs (i.e., for each single point to be optimized).</td>
				<td><samp>30</samp></td>
			</tr>
			<tr>
				<td><samp>--num_parallel_jobs NUM_PARALLEL_JOBS</samp></td>
				<td>Number of parallel slurm jobs (only used when SLURM is installed).</td>
				<td><samp>20</samp></td>
			</tr>
			<tr>
				<td><samp>--auto_exclude_defective_hosts</samp></td>
				<td>Run a Test if you can allocate a GPU on each node and if not, exclude it since the GPU driver seems to be broken somehow.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--slurm_use_srun</samp></td>
				<td>Using srun instead of sbatch. <a href="https://slurm.schedmd.com/srun.html" target="_blank">Learn more</a></td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--time TIME</samp></td>
				<td>Time for the main job.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--partition PARTITION</samp></td>
				<td>Partition to be run on.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--reservation RESERVATION</samp></td>
				<td>Reservation. <a href="https://slurm.schedmd.com/reservations.html" target="_blank">Learn more</a></td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--force_local_execution</samp></td>
				<td>Forces local execution even when SLURM is available.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--slurm_signal_delay_s SLURM_SIGNAL_DELAY_S</samp></td>
				<td>When the workers end, they get a signal so your program can react to it. Default is 0, but set it to any number of seconds you wish your program to be able to react to USR1.</td>
				<td><samp>0</samp></td>
			</tr>
			<tr>
				<td><samp>--nodes_per_job NODES_PER_JOB</samp></td>
				<td>Number of nodes per job due to the new alpha restriction.</td>
				<td><samp>1</samp></td>
			</tr>
			<tr>
				<td><samp>--cpus_per_task CPUS_PER_TASK</samp></td>
				<td>CPUs per task.</td>
				<td><samp>1</samp></td>
			</tr>
			<tr>
				<td><samp>--account ACCOUNT</samp></td>
				<td>Account to be used. <a href="https://slurm.schedmd.com/accounting.html" target="_blank">Learn more</a></td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--gpus GPUS</samp></td>
				<td>Number of GPUs.</td>
				<td><samp>0</samp></td>
			</tr>
			<tr class="section-header invert_in_dark_mode">
				<td colspan="3">Installing</td>
			</tr>
			<tr>
				<td><samp>--run_mode</samp></td>
				<td>Either <i>local</i> or <i>docker</i>.</td>
				<td><samp>local</samp></td>
			</tr>
			<tr class="section-header invert_in_dark_mode">
				<td colspan="3">Debug</td>
			</tr>
			<tr>
				<td><samp>--verbose</samp></td>
				<td>Verbose logging.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--debug</samp></td>
				<td>Enable debugging.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--no_sleep</samp></td>
				<td>Disables sleeping for fast job generation (not to be used on HPC).</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--tests</samp></td>
				<td>Run simple internal tests.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--raise_in_eval</samp></td>
				<td>Raise a signal in eval (only useful for debugging and testing)</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--run_tests_that_fail_on_taurus</samp></td>
				<td>Run tests on Taurus that usually fail</td>
				<td><samp>False</samp></td>
			</tr>
			<tr>
				<td><samp>--show_worker_percentage_table_at_end</samp></td>
				<td>Show a table of percentage of usage of max worker over time.</td>
				<td><samp>False</samp></td>
			</tr>
			<tr class="section-header invert_in_dark_mode">
				<td colspan="3">Config</td>
			</tr>
			<tr>
				<td><samp>--config_json</samp></td>
				<td>Allows a path to a json file containing all parameters for the run.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--config_toml</samp></td>
				<td>Allows a path to a toml file containing all parameters for the run.</td>
				<td>-</td>
			</tr>
			<tr>
				<td><samp>--config_yaml</samp></td>
				<td>Allows a path to a yaml file containing all parameters for the run.</td>
				<td>-</td>
			</tr>
		</tbody>
	</table>
