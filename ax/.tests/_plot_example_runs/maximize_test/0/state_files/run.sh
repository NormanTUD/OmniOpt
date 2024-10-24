omniopt '--partition=alpha --experiment_name=maximize_test --mem_gb=1 --time=60 --worker_timeout=60 --max_eval=4 --num_parallel_jobs=1 --gpus=1 --num_random_steps=2 --follow --live_share --send_anonymized_usage_stats 
--run_program=ZWNobyAiUkVTVUxUOiAlKHJlcyki --cpus_per_task=1 --nodes_per_job=1 --model=BOTORCH_MODULAR --run_mode=local --parameter res range 0 10000 float false --maximize
