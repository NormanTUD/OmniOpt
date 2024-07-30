<?php
	include("../_header_base.php");
?>
<h1><tt>--help</tt></h1>

<div id="toc"></div>

<h2 id="available_parameters">Available parameters (<tt>--help</tt>)</h2>

	<table>
	<thead>
		<tr>
		<th>Parameter</th>
		<th>Description</th>
		<th>Default Value</th>
		</tr>
	</thead>
	<tbody>
		<tr class="section-header">
			<td colspan="3">Required Arguments</td>
		</tr>
		<tr>
			<td><tt>--num_parallel_jobs NUM_PARALLEL_JOBS</tt></tt>
			<td>Number of parallel slurm jobs (only used when Slurm is installed).</td>
			<td><tt>20</tt></td>
		</tr>
		<tr>
			<td><tt>--num_random_steps NUM_RANDOM_STEPS</tt></td>
			<td>Number of random steps to start with.</td>
			<td><tt>20</tt></td>
		</tr>
		<tr>
			<td><tt>--max_eval MAX_EVAL</tt></td>
			<td>Maximum number of evaluations.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--worker_timeout WORKER_TIMEOUT</tt></td>
			<td>Timeout for slurm jobs (i.e., for each single point to be optimized).</td>
			<td><tt>30</tt></td>
		</tr>
		<tr>
			<td><tt>--run_program RUN_PROGRAM [RUN_PROGRAM ...]</tt></td>
			<td>A program that should be run. Use, for example, <tt>%(x)</tt> for the parameter named <i>x</i>.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--experiment_name EXPERIMENT_NAME</tt></td>
			<td>Name of the experiment.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--mem_gb MEM_GB</tt></td>
			<td>Amount of RAM for each worker in GB.</td>
			<td><tt>1</tt></td>
		</tr>
		<tr class="section-header">
			<td colspan="3">Required Arguments That Allow A Choice</td>
		</tr>
		<tr>
			<td><tt>--parameter PARAMETER [PARAMETER ...]</tt></td>
			<td>Experiment parameters in the formats: <br>
				- <tt>&lt;NAME&gt; range &lt;NAME&gt; &lt;LOWER BOUND&gt; &lt;UPPER BOUND&gt; (&lt;INT, FLOAT&gt;)</tt><br>
				- <tt>&lt;NAME&gt; fixed &lt;NAME&gt; &lt;VALUE&gt;</tt><br>
				- <tt>&lt;NAME&gt; choice &lt;NAME&gt; &lt;Comma-separated list of values&gt;</tt>
			</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--continue_previous_job CONTINUE_PREVIOUS_JOB</tt></td>
			<td>Continue from a previous checkpoint, use run-dir as argument.</td>
			<td>-</td>
		</tr>
		<tr class="section-header">
			<td colspan="3">Optional</td>
		</tr>
		<tr>
			<td><tt>--exclude "taurusi8009,taurusi8010"</tt></td>
			<td>A comma seperated list of values of excluded nodes.</td>
			<td><tt>None</tt></td>
		</tr>
		<tr>
			<td><tt>--maximize</tt></td>
			<td>Maximize instead of minimize (which is default).</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--experiment_constraints EXPERIMENT_CONSTRAINTS [EXPERIMENT_CONSTRAINTS ...]</tt></td>
			<td>Constraints for parameters. Example: <tt>x + y <= 2.0</tt>.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--stderr_to_stdout</tt></td>
			<td>Redirect stderr to stdout for subjobs.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--run_dir RUN_DIR</tt></td>
			<td>Directory in which runs should be saved.</td>
			<td><tt>runs</tt></td>
		</tr>
		<tr>
			<td><tt>--seed SEED</tt></td>
			<td>Seed for random number generator.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--enforce_sequential_optimization</tt></td>
			<td>Enforce sequential optimization.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--verbose_tqdm</tt></td>
			<td>Show verbose tqdm messages.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--load_previous_job_data LOAD_PREVIOUS_JOB_DATA [LOAD_PREVIOUS_JOB_DATA ...]</tt></td>
			<td>Paths of previous jobs to load from.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--hide_ascii_plots</tt></td>
			<td>Hide ASCII-plots.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--model MODEL</tt></td>
			<td>Use special models for nonrandom steps. Valid models are: SOBOL, GPEI, FACTORIAL, SAASBO, FULLYBAYESIAN, LEGACY_BOTORCH, BOTORCH_MODULAR, UNIFORM, BO_MIXED.</td>
			<td><tt>BOTORCH_MODULAR</tt></td>
		</tr>
		<tr>
			<td><tt>--gridsearch</tt></td>
			<td>Enable gridsearch.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--show_sixel_scatter</tt></td>
			<td>Show <a href="https://en.wikipedia.org/wiki/Sixel" target="_blank">sixel</a> graphics of scatter plot in the end.</td>
			<td><tt>False</tt></td>
		</tr>

		<tr>
			<td><tt>--show_sixel_general</tt></td>
			<td>Show <a href="https://en.wikipedia.org/wiki/Sixel" target="_blank">sixel</a> graphics of general plot in the end.</td>
			<td><tt>False</tt></td>
		</tr>

		<tr>
			<td><tt>--show_sixel_trial_index_result</tt></td>
			<td>Show <a href="https://en.wikipedia.org/wiki/Sixel" target="_blank">sixel</a> graphics of trial_index_result plot in the end.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--follow</tt></td>
			<td>Automatically follow log file of sbatch.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--send_anonymized_usage_stats</tt></td>
			<td>Send anonymized usage stats.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--ui_url UI_URL</tt></td>
			<td>Site from which the OO-run was called.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--root_venv_dir ROOT_VENV_DIR</tt></td>
			<td>Where to install your modules to (<tt>$root_venv_dir/.omniax_...</tt>)</td>
			<td><tt>$HOME</tt></td>
		</tr>
		<tr>
			<td><tt>--main_process_gb (INT)</tt></td>
			<td>Amount of RAM the main process should have</td>
			<td><tt>4</tt></td>
		</tr>
		<tr>
			<td><tt>--max_nr_of_zero_results (INT)</tt></td>
			<td>Max. nr of successive zero results by ax_client.get_next_trials() before the search space is seen as exhausted.</td>
			<td><tt>20</tt></td>
		</tr>
		<tr>
			<td><tt>--abbreviate_job_names</tt></td>
			<td>Abbreviate pending job names (r = running, p = pending, u = unknown, c = cancelling)</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--disable_search_space_exhaustion_detection</tt></td>
			<td>Disables automatic search space reduction detection</td>
			<td><tt>False</tt></td>
		</tr>
		<tr class="section-header">
			<td colspan="3">Experimental</td>
			</tr>
		<tr>
			<td><tt>--experimental</tt></td>
			<td class="warning">Do some stuff not well tested yet.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--auto_execute_suggestions</tt></td>
			<td class="warning">Automatically run again with suggested parameters (NOT FOR SLURM YET!).</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--auto_execute_counter AUTO_EXECUTE_COUNTER</tt></td>
			<td class="warning">(Will automatically be set).</td>
			<td><tt>0</tt></td>
		</tr>
		<tr>
			<td><tt>--max_auto_execute MAX_AUTO_EXECUTE</tt></td>
			<td class="warning">How many nested jobs should be done.</td>
			<td><tt>3</tt></td>
		</tr>
		<tr>
			<td><tt>--show_parameter_suggestions</tt></td>
			<td class="warning">Show suggestions for possible promising parameter space changes.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--maximizer MAXIMIZER</tt></td>
			<td class="warning">Value to expand search space for suggestions. Calculation is point [+-] maximizer * abs(point).</td>
			<td><tt>2</tt></td>
		</tr>
			<tr class="section-header">
			<td colspan="3">Slurm</td>
		</tr>
		<tr>
			<td><tt>--auto_exclude_defective_hosts</tt></td>
			<td>Run a Test if you can allocate a GPU on each node and if not, exclude it since the GPU driver seems to be broken somehow.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--slurm_use_srun</tt></td>
			<td>Using srun instead of sbatch. <a href="https://slurm.schedmd.com/srun.html" target="_blank">Learn more</a></td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--time TIME</tt></td>
			<td>Time for the main job.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--partition PARTITION</tt></td>
			<td>Partition to be run on.</td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--reservation RESERVATION</tt></td>
			<td>Reservation. <a href="https://slurm.schedmd.com/reservations.html" target="_blank">Learn more</a></td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--force_local_execution</tt></td>
			<td>Forces local execution even when SLURM is available.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--slurm_signal_delay_s SLURM_SIGNAL_DELAY_S</tt></td>
			<td>When the workers end, they get a signal so your program can react to it. Default is 0, but set it to any number of seconds you wish your program to be able to react to USR1.</td>
			<td><tt>0</tt></td>
		</tr>
		<tr>
			<td><tt>--nodes_per_job NODES_PER_JOB</tt></td>
			<td>Number of nodes per job due to the new alpha restriction.</td>
			<td><tt>1</tt></td>
		</tr>
		<tr>
			<td><tt>--cpus_per_task CPUS_PER_TASK</tt></td>
			<td>CPUs per task.</td>
			<td><tt>1</tt></td>
		</tr>
		<tr>
			<td><tt>--account ACCOUNT</tt></td>
			<td>Account to be used. <a href="https://slurm.schedmd.com/accounting.html" target="_blank">Learn more</a></td>
			<td>-</td>
		</tr>
		<tr>
			<td><tt>--gpus GPUS</tt></td>
			<td>Number of GPUs.</td>
			<td><tt>0</tt></td>
		</tr>
		<tr>
			<td><tt>--tasks_per_node TASKS_PER_NODE</tt></td>
			<td>Number of tasks per node.</td>
			<td><tt>1</tt></td>
		</tr>
		<tr>
			<tr class="section-header">
			<td colspan="3">Installing</td>
		</tr>
		<tr>
			<td><tt>--run_mode</tt></td>
			<td>Either <i>local</i> or <i>docker</i>.</td>
			<td><tt>local</tt></td>
		</tr>
		<tr>
			<tr class="section-header">
			<td colspan="3">Debug</td>
		</tr>
		<tr>
			<td><tt>--verbose</tt></td>
			<td>Verbose logging.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--debug</tt></td>
			<td>Enable debugging.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--no_sleep</tt></td>
			<td>Disables sleeping for fast job generation (not to be used on HPC).</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--tests</tt></td>
			<td>Run simple internal tests.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--evaluate_to_random_value</tt></td>
			<td>Evaluate to random values.</td>
			<td><tt>False</tt></td>
		</tr>
		<tr>
			<td><tt>--show_worker_percentage_table_at_end</tt></td>
			<td>Show a table of percentage of usage of max worker over time.</td>
			<td><tt>False</tt></td>
		</tr>
	</tbody>
	</table>
