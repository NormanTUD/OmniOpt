#!/bin/bash

set -euo pipefail

# Parameterliste
PARAMS=(
	"--dont_warm_start_refitting"
	"--refit_on_cv"
	"--fit_out_of_design"
	"--jit_compile"
	"--num_restarts"
	"--raw_samples"
	"--no_transform_inputs"
	"--no_normalize_y"
)

# Feste Parameter
FIXED_ARGS="--num_random_steps=20 --max_eval=50 --num_parallel_jobs=20 --nr_nodes=3"

# Funktion zum erzeugen aller Kombinationen
generate_combinations() {
	local n="${#PARAMS[@]}"
	for ((i=1; i < (1 << n); i++)); do
		local combination=()
		for ((j=0; j < n; j++)); do
			if (( (i >> j) & 1 )); then
				combination+=("${PARAMS[j]}")
			fi
		done
		echo "${combination[@]}"
	done
}

# Funktion zum erzeugen des Dateinamens
sanitize_filename() {
	local input="$1"
	echo "$input" | sed 's/--//g; s/ /:/g; s/_/-/g'
}

# Hauptschleife über alle Kombinationen
for combo in $(generate_combinations); do
	ADDITIONAL_ARGS="$combo"
	FILENAME_PART=$(sanitize_filename "$ADDITIONAL_ARGS")
	OUTPUT_FILE="output/${FILENAME_PART}.txt"

	# Stelle sicher, dass das Ausgabe-Verzeichnis existiert
	mkdir -p output

	echo "Running with: $ADDITIONAL_ARGS > $OUTPUT_FILE"
	#bash run_docker $FIXED_ARGS --additional_parameter="$ADDITIONAL_ARGS" > "$OUTPUT_FILE" 2>&1
done
