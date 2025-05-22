#!/bin/bash
set -euo pipefail

ALWAYS_THERE="--no_normalize_y --fit_out_of_design"

PARAMS=(
	#"--dont_warm_start_refitting"
	#"--no_transform_inputs"
	#"--refit_on_cv"
	#
	#"--no_normalize_y"
	#"--fit_out_of_design"

	#"--num_restarts=1"
	#"--num_restarts=5"
	#"--num_restarts=10"
	#"--raw_samples=1"
	#"--raw_samples=10"
	#"--raw_samples=100"
	
	"--fit_abandoned"
)

FIXED_ARGS="--num_random_steps=5 --max_eval=20 --num_parallel_jobs=5 --nr_nodes=5 --generate_all_jobs_at_once"

generate_combinations() {
	local n="${#PARAMS[@]}"
	for ((i = 1; i < (1 << n); i++)); do
		local combination=()
		for ((j = 0; j < n; j++)); do
			if (( (i >> j) & 1 )); then
				combination+=("${PARAMS[j]}")
			fi
		done
		printf "%s\n" "${combination[*]}"
	done
}

sanitize_filename() {
	local input="$1"
	echo "$input" | sed 's/--//g; s/ /:/g; s/_/-/g'
}

human_time() {
	local seconds=$1
	local hours=$((seconds / 3600))
	local minutes=$(( (seconds % 3600) / 60 ))
	local secs=$((seconds % 60))
	if ((hours > 24)); then
		local days=$((hours / 24))
		hours=$((hours % 24))
		printf "%d days, %d hours and %d minutes" "$days" "$hours" "$minutes"
	else
		printf "%d hours, %d minutes and %d seconds" "$hours" "$minutes" "$secs"
	fi
}

mapfile -t COMBOS < <(generate_combinations)
TOTAL=${#COMBOS[@]}
mkdir -p output

declare -a DURATIONS=()

for index in "${!COMBOS[@]}"; do
	combo="${COMBOS[$index]}"
	ADDITIONAL_ARGS="$combo"
	FILENAME_PART=$(sanitize_filename "$ADDITIONAL_ARGS")
	OUTPUT_FILE="output/${FILENAME_PART}.txt"

	CURRENT=$((index + 1))
	PROGRESS=$((CURRENT * 100 / TOTAL))

	BAR_WIDTH=40
	FILLED=$((PROGRESS * BAR_WIDTH / 100))
	EMPTY=$((BAR_WIDTH - FILLED))
	BAR=$(printf "%0.s#" $(seq 1 $FILLED))$(printf "%0.s-" $(seq 1 $EMPTY))

	printf "\r[%s] %3d%% (%d/%d) Running with: %s" "$BAR" "$PROGRESS" "$CURRENT" "$TOTAL" "$ADDITIONAL_ARGS"
	echo ""

	START_TIME=$(date +%s)
	bash run_docker $FIXED_ARGS --additional_parameter="$ALWAYS_THERE $ADDITIONAL_ARGS" 2>&1 | tee "$OUTPUT_FILE"
	END_TIME=$(date +%s)

	DURATION=$((END_TIME - START_TIME))
	DURATIONS+=("$DURATION")

	if (( CURRENT > 1 )); then
		sorted=($(printf "%s\n" "${DURATIONS[@]}" | sort -n))
		mid=$((CURRENT / 2))
		if (( CURRENT % 2 == 0 )); then
			median=$(((sorted[mid - 1] + sorted[mid]) / 2))
		else
			median=${sorted[mid]}
		fi

		remaining=$(( (TOTAL - CURRENT) * median ))
		human=$(human_time "$remaining")

		echo -e "\nEstimated remaining time: ~ $human"
	else
		echo ""
	fi
done

echo -e "\nAll jobs done!"
