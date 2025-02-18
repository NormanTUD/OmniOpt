#!/bin/bash

for max_eval in $(seq 1 30); do
	for num_parallel_jobs in $(seq 1 30); do
		echo "$max_eval - $num_parallel_jobs"
		bash run_docker --num_random_steps=1 --max_eval=$max_eval --num_parallel_jobs=$num_parallel_jobs || {
			echo ">>>bash run_docker --num_random_steps=1 --max_eval=$max_eval --num_parallel_jobs=$num_parallel_jobs<<< failed with $?"
			exit 1
		}
	done
done
