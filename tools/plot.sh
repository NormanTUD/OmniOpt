#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
hostname=$(hostname)

PROJECTNAME=
PROJECTDIR=projects/

for i in "$@"
do
	case $i in
		--project=*)
			PROJECTNAME="${i#*=}"
			;;
		--projectdir=*)
			PROJECTDIR="${i#*=}"
			;;
		*)
			# unknown option
			;;
	esac
done

#--project=Hist4D____VVVVVVVVVVVVVVVVVVVVVVVV --projectdir=/home/s3811141/projects/

source $DIR/debug.sh

if [[ -d "$PROJECTDIR/$PROJECTNAME" ]]; then
	if [[ -e "$PROJECTDIR/$PROJECTNAME/config.ini" ]]; then
		echo ${hostname} | grep --quiet "ml"

		if [ $? = 0 ]; then
		    module_load modenv/ml
		    module_load MongoDB/4.0.3
		    module_load TensorFlow/2.0.0-PythonAnaconda-3.7
		else
		    module_load modenv/scs5
		    module_load MongoDB/4.0.3
		    module_load Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4
		    module_load Python/3.7.4-GCCcore-8.3.0
		    module_load matplotlib/3.1.1-foss-2019b-Python-3.7.4
		fi

		(pip3 install --user psutil && python3 script/plot3.py $@ && python3 script/endmongodb.py $@) || (bash tools/repair_database.sh "$PROJECTDIR/$PROJECTNAME/mongodb" && (pip3 install --user psutil && python3 script/plot3.py $@ && python3 script/endmongodb.py $@))
	else
		echo "$PROJECTDIR/$PROJECTNAME/config.ini does not exist"
	fi
else
	echo "$PROJECTDIR/$PROJECTNAME does not exist"
fi
