#!/bin/bash

#set -x
cd ..

PRINT_EMPTY_SPACE=0

function get_project_name_from_nnopt_outfile {
	cat $1 | egrep "project'.*=>" | sed -e "s/.*project.*=> '//" | sed -e "s/',//" | head -n1
}

function get_last_best_loss_from_nnopt_outfile {
	cat $1 | egrep "best loss" | tail -n1 | sed -e "s/\]//" | sed -e 's/.*best loss: //' | sed -e 's/srun.*//' | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" | sed -e 's/[^.0-9]*$//' | sed -e 's/[a-zA-Z].*//'
}

function get_max_runtime_output_from_nnopt_logfile {
	cat $1 | egrep "[0-9]*it \[" | tail -n1 | sed -e 's/.*\[//' | sed -e 's/,.*//'
}

function get_last_changed_time {
	stat -c %y $1 | sed -e 's/\..*//'
}

function get_number_of_results {
	cat $1 | grep "RESULT found as" | grep -v inf | wc -l
}

function short_summary_nnopt_output_file {
	if [[ -e $1 ]]; then
		PROJECTNAME=$(get_project_name_from_nnopt_outfile $1)
		LAST_LOSS=$(get_last_best_loss_from_nnopt_outfile $1)
		MAX_RUNTIME=$(get_max_runtime_output_from_nnopt_logfile $1)
		LAST_CHANGE=$(get_last_changed_time $1)
		NUMBER_OF_RESULTS=$(get_number_of_results $1)

		INDETERMINABLE_STRING="\e[91mIndeterminable\e[0m"

		if [[ -z $PROJECTNAME ]]; then
			PROJECTNAME=$INDETERMINABLE_STRING
		fi

		if [[ -z $LAST_LOSS ]]; then
			LAST_LOSS=$INDETERMINABLE_STRING
		fi

		if [[ -z $MAX_RUNTIME ]]; then
			MAX_RUNTIME=$INDETERMINABLE_STRING
		fi

		if [[ -z $LAST_CHANGE ]]; then
			LAST_CHANGE=$INDETERMINABLE_STRING
		fi

		if [[ -z $NUMBER_OF_RESULTS ]]; then
			NUMBER_OF_RESULTS=$INDETERMINABLE_STRING
		fi

		echo -e "Project name:		${PROJECTNAME}"
		echo -e "Filename:		$1"
		echo -e "Last loss:		${LAST_LOSS}"
		echo -e "Max runtime:		${MAX_RUNTIME}"
		echo -e "Last modified time:	${LAST_CHANGE}"
		echo -e "Number of results:	${NUMBER_OF_RESULTS}"
		if [[ "$PRINT_EMPTY_SPACE" == "1" ]]; then
			echo -e ""
		fi
	else
		echo -e "\e[91m$1 not found or not a file\e[0m"
	fi
}

if [[ -e $1 ]]; then
	short_summary_nnopt_output_file $1
else
	PRINT_EMPTY_SPACE=1
	for i in $(ls -tr1 *.out); do
		short_summary_nnopt_output_file $i;
	done
fi
