#!/bin/bash

min=1
max=10
stepsize=2

for max_eval in $(seq $min $stepsize $max); do
	for num_parallel_jobs in $(seq $min $stepsize $max); do
		for num_random_steps in $(seq $min $stepsize $max); do
			echo "max_eval: $max_eval - num_parallel_jobs: $num_parallel_jobs - num_random_steps: $num_random_steps"
			bash run_docker --num_random_steps=$num_random_steps --max_eval=$max_eval --num_parallel_jobs=$num_parallel_jobs || {
				echo ">>>bash run_docker --num_random_steps=$num_random_steps --max_eval=$max_eval --num_parallel_jobs=$num_parallel_jobs<<< failed with $?"
				exit 1
			}
		done
	done
done
