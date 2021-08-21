#!/bin/bash -l

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}


function echoerr() {    echo "$@" 1>&2
}

function red_text {
        echoerr -e "\e[31m$1\e[0m"
}

set -e
set -o pipefail

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

function help () {
        echo "Possible options:"
        echo "  --projectdir=(DIREXISTS)=DIREXISTS"
        echo "  --project"
        echo "  --help                                             this help"
        echo "  --debug                                            Enables debug mode (set -x)"
        exit $1
}
export projectdir=projects
export project
for i in $@; do
        case $i in
                --projectdir=*)
                        projectdir="${i#*=}"
                        if [[ ! -d $projectdir ]]; then
                                red_text "error: directory $projectdir does not exist" >&2
                                help 1
                        fi
                        shift
                        ;;
                --project=*)
                        project="${i#*=}"
                        shift
                        ;;
                -h|--help)
                        help 0
                        ;;
                --debug)
                        set -x
                        ;;
                *)
                        red_text "Unknown parameter $i" >&2
                        help 1
                        ;;
        esac
done
if [[ -z "$projectdir" ]]; then red_text "Parameter --projectdir cannot be empty"; help 1; fi
if [[ -z "$project" ]]; then red_text "Parameter --project cannot be empty"; help 1; fi

ml purge
ml modenv/scs5
ml MongoDB/4.0.3
ml Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4
ml Python/3.7.4-GCCcore-8.3.0
ml matplotlib/3.1.1-foss-2019b-Python-3.7.4

python3 script/get_wallclock_time.py --projectdir=$projectdir --project=$project

python3 script/endmongodb.py --projectdir=$projectdir --project=$project 2>/dev/null
