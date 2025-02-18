#!/bin/bash

min=1
max=10
stepsize=2
shuffle=false

display_help() {
	echo "Usage: $0 [options]"
	echo ""
	echo "Options:"
	echo "  --min VALUE       Set minimum value (default: 1)"
	echo "  --max VALUE       Set maximum value (default: 10)"
	echo "  --stepsize VALUE  Set step size (default: 2)"
	echo "  --shuffle         Shuffle commands before execution"
	echo "  -h, --help        Show this help message"
	exit 0
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--min)
			min="$2"
			shift 2
			;;
		--max)
			max="$2"
			shift 2
			;;
		--stepsize)
			stepsize="$2"
			shift 2
			;;
		--shuffle)
			shuffle=true
			shift
			;;
		-h|--help)
			display_help
			;;
		*)
			echo "Unknown option: $1" >&2
			display_help
			;;
	esac
done

commands=()
for max_eval in $(seq "$min" "$stepsize" "$max"); do
	for num_parallel_jobs in $(seq "$min" "$stepsize" "$max"); do
		for num_random_steps in $(seq "$min" "$stepsize" "$max"); do
			commands+=("bash run_docker --num_random_steps=$num_random_steps --max_eval=$max_eval --num_parallel_jobs=$num_parallel_jobs")
		done
	done
done

if [[ "$shuffle" == true ]]; then
	mapfile -t commands < <(printf "%s\n" "${commands[@]}" | shuf)
fi

for cmd in "${commands[@]}"; do
	echo "Running: $cmd"
	eval "$cmd" || {
	echo ">>> Command failed: $cmd <<<"
		exit 1
	}
done
