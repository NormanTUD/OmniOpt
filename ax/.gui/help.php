<?php
	include("_header_base.php");
?>
	<link href="tutorial.css" rel="stylesheet" />
	<link href="jquery-ui.css" rel="stylesheet">
	<link href="prism.css" rel="stylesheet" />

	<h1>Basics</h1>
    
	<div id="toc"></div>

	<h2 id="available_parameters">Available parameters (<tt>--help</tt>)</h2>

    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            text-align: center;
        }
        .table-container {
            width: 100%;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .section-header {
            background-color: #4CAF50;
            color: white;
        }
        .warning {
            color: red;
        }
    </style>
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
                <td>--num_parallel_jobs NUM_PARALLEL_JOBS</td>
                <td>Number of parallel slurm jobs.</td>
                <td>20</td>
            </tr>
            <tr>
                <td>--num_random_steps NUM_RANDOM_STEPS</td>
                <td>Number of urandom steps to start with.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--max_eval MAX_EVAL</td>
                <td>Maximum number of evaluations.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--worker_timeout WORKER_TIMEOUT</td>
                <td>Timeout for slurm jobs (i.e., for each single point to be optimized).</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--run_program RUN_PROGRAM [RUN_PROGRAM ...]</td>
                <td>A program that should be run. Use, for example, <tt>%(x)</tt> for the parameter named <i>x</i>.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--experiment_name EXPERIMENT_NAME</td>
                <td>Name of the experiment.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--mem_gb MEM_GB</td>
                <td>Amount of RAM for each worker in GB.</td>
                <td>1GB</td>
            </tr>
            <tr class="section-header">
                <td colspan="3">Required Arguments That Allow A Choice</td>
            </tr>
            <tr>
                <td>--parameter PARAMETER [PARAMETER ...]</td>
                <td>Experiment parameters in the formats: <br>
                    - <NAME> range &lt;NAME&gt; &lt;LOWER BOUND&gt; &lt;UPPER BOUND&gt; (&lt;INT, FLOAT&gt;) <br>
                    - <NAME> fixed &lt;NAME&gt; &lt;VALUE&gt; <br>
                    - <NAME> choice &lt;NAME&gt; &lt;Comma-separated list of values&gt;
                </td>
                <td>-</td>
            </tr>
            <tr>
                <td>--continue_previous_job CONTINUE_PREVIOUS_JOB</td>
                <td>Continue from a previous checkpoint, use run-dir as argument.</td>
                <td>-</td>
            </tr>
            <tr class="section-header">
                <td colspan="3">Optional</td>
            </tr>
            <tr>
                <td>--maximize</td>
                <td>Maximize instead of minimize (which is default).</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--experiment_constraints EXPERIMENT_CONSTRAINTS [EXPERIMENT_CONSTRAINTS ...]</td>
                <td>Constraints for parameters. Example: x + y <= 2.0.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--stderr_to_stdout</td>
                <td>Redirect stderr to stdout for subjobs.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--run_dir RUN_DIR</td>
                <td>Directory in which runs should be saved.</td>
                <td>runs</td>
            </tr>
            <tr>
                <td>--seed SEED</td>
                <td>Seed for random number generator.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--enforce_sequential_optimization</td>
                <td>Enforce sequential optimization.</td>
                <td>false</td>
            </tr>
            <tr>
                <td>--verbose_tqdm</td>
                <td>Show verbose tqdm messages.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--load_previous_job_data LOAD_PREVIOUS_JOB_DATA [LOAD_PREVIOUS_JOB_DATA ...]</td>
                <td>Paths of previous jobs to load from.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--hide_ascii_plots</td>
                <td>Hide ASCII-plots.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--model MODEL</td>
                <td>Use special models for nonrandom steps. Valid models are: SOBOL, GPEI, FACTORIAL, SAASBO, FULLYBAYESIAN, LEGACY_BOTORCH, BOTORCH_MODULAR, UNIFORM, BO_MIXED.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--gridsearch</td>
                <td>Enable gridsearch.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--show_sixel_graphics</td>
                <td>Show sixel graphics in the end.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--follow</td>
                <td>Automatically follow log file of sbatch.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--send_anonymized_usage_stats</td>
                <td>Send anonymized usage stats.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--ui_url UI_URL</td>
                <td>Site from which the OO-run was called.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--root_venv_dir ROOT_VENV_DIR</td>
                <td>Where to install your modules to ($root_venv_dir/.omniax_...)</td>
                <td><tt>$HOME</tt></td>
            </tr>
            <tr class="section-header">
                <td colspan="3">Experimental</td>
            </tr>
            <tr>
                <td>--experimental</td>
                <td class="warning">Do some stuff not well tested yet.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--auto_execute_suggestions</td>
                <td class="warning">Automatically run again with suggested parameters (NOT FOR SLURM YET!).</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--auto_execute_counter AUTO_EXECUTE_COUNTER</td>
                <td class="warning">(Will automatically be set).</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--max_auto_execute MAX_AUTO_EXECUTE</td>
                <td class="warning">How many nested jobs should be done.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--show_parameter_suggestions</td>
                <td class="warning">Show suggestions for possible promising parameter space changes.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--maximizer MAXIMIZER</td>
                <td class="warning">Value to expand search space for suggestions. Calculation is point [+-] maximizer * abs(point).</td>
                <td>0.5</td>
            </tr>
            <tr class="section-header">
                <td colspan="3">Slurm</td>
            </tr>
            <tr>
                <td>--slurm_use_srun</td>
                <td>Using srun instead of sbatch. <a href="https://slurm.schedmd.com/srun.html" target="_blank">Learn more</a></td>
                <td>-</td>
            </tr>
            <tr>
                <td>--time TIME</td>
                <td>Time for the main job.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--partition PARTITION</td>
                <td>Partition to be run on.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--reservation RESERVATION</td>
                <td>Reservation. <a href="https://slurm.schedmd.com/reservations.html" target="_blank">Learn more</a></td>
                <td>-</td>
            </tr>
            <tr>
                <td>--force_local_execution</td>
                <td>Forces local execution even when SLURM is available.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--slurm_signal_delay_s SLURM_SIGNAL_DELAY_S</td>
                <td>When the workers end, they get a signal so your program can react to it. Default is 0, but set it to any number of seconds you wish your program to react to USR1.</td>
                <td>0</td>
            </tr>
            <tr>
                <td>--nodes_per_job NODES_PER_JOB</td>
                <td>Number of nodes per job due to the new alpha restriction.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--cpus_per_task CPUS_PER_TASK</td>
                <td>CPUs per task.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--account ACCOUNT</td>
                <td>Account to be used. <a href="https://slurm.schedmd.com/accounting.html" target="_blank">Learn more</a></td>
                <td>-</td>
            </tr>
            <tr>
                <td>--gpus GPUS</td>
                <td>Number of GPUs.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--tasks_per_node TASKS_PER_NODE</td>
                <td>Number of tasks per node.</td>
                <td>-</td>
            </tr>
            <tr class="section-header">
                <td colspan="3">Debug</td>
            </tr>
            <tr>
                <td>--verbose</td>
                <td>Verbose logging.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--debug</td>
                <td>Enable debugging.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--no_sleep</td>
                <td>Disables sleeping for fast job generation (not to be used on HPC).</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--tests</td>
                <td>Run simple internal tests.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--evaluate_to_random_value</td>
                <td>Evaluate to random values.</td>
                <td>-</td>
            </tr>
            <tr>
                <td>--show_worker_percentage_table_at_end</td>
                <td>Show a table of percentage of usage of max worker over time.</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

	<script src="prism.js"></script>
	<script src="footer.js"></script>
</body>
</html>
