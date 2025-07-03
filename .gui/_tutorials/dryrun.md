# <span class='tutorial_icon invert_in_dark_mode'>🧪</span> Dry-Runs

<!-- How to quickly test if your configuration works properly -->

<!-- Category: Preparations, Basics and Setup -->

<div id="toc"></div>

## What are dry-runs?

Dry-Runs allow you to test your OmniOpt2-configuration by running very quick jobs and skipping things like Slurm, just to see if things like environment variables are set properly, paths work, and so on.

## How to run dry-runs?

Simply attach `--dryrun` to your OmniOpt2-call, like this, or chose the dry-run-Option in the GUI:

```bash
./omniopt \
	--live_share \
	--send_anonymized_usage_stats \
	--partition alpha \
	--experiment_name xyz \
	--mem_gb=4 \
	--time 60 \
	--worker_timeout=5 \
	--max_eval 1 \
	--num_parallel_jobs 2 \
	--gpus 0 \
	--run_program Li8udGVzdHMvb3B0aW1pemF0aW9uX2V4YW1wbGUgIC0taW50X3BhcmFtPSclKGludF9wYXJhbSknIC0tZmxvYXRfcGFyYW09JyUoZmxvYXRfcGFyYW0pJyAtLWNob2ljZV9wYXJhbT0nJShjaG9pY2VfcGFyYW0pJyAtLWludF9wYXJhbV90d289JyUoaW50X3BhcmFtX3R3byknIC0tbnJfcmVzdWx0cz0x \
	--parameter int_param range -100 10 int \
	--parameter float_param range -100 10 float \
	--parameter choice_param choice 1,2,4,8,16,hallo \
	--parameter int_param_two range -100 10 int \
	--num_random_steps 1 \
	--model BOTORCH_MODULAR \
	--auto_exclude_defective_hosts \
	--generate_all_jobs_at_once \
	--follow \
	--show_generate_time_table \
	--dryrun

```

Dry-Runs allow you to chose the parameter of one job before it runs and run only that one job. It skips SLURM, so your job should run quickly with these parameters, or in debug-mode. You will be asked for parameters, but certain ones are suggested if named properly, i.e. `epochs` will be suggested to `1`. You will be able to change them before the run executes.
