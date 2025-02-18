#!/bin/bash

for max_eval in $(seq 1 30); do
	for num_parallel_jobs in $(seq 1 30); do
		for num_random_steps in $(seq 1 30); do
			echo "max_eval: $max_eval - num_parallel_jobs: $num_parallel_jobs - num_random_steps: $num_random_steps"
			bash run_docker --num_random_steps=$num_random_steps --max_eval=$max_eval --num_parallel_jobs=$num_parallel_jobs || {
				echo ">>>bash run_docker --num_random_steps=$num_random_steps --max_eval=$max_eval --num_parallel_jobs=$num_parallel_jobs<<< failed with $?"
				exit 1
			}
		done
	done
done
