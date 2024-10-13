#!/bin/bash -l

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib64

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}

function echoerr() {
    echo "$@" 1>&2
}

function p () {
        if [[ ! -z $2 ]]; then
            echoerr "PERCENTGAUGE: $1"
        fi

        if [[ ! -z $2 ]]; then
            echoerr "GAUGESTATUS: $2"
        fi
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
for i in "$@"; do
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

p "1" "Loading modules"
p "2" "Purging old modules"
ml --force purge
p "3" "Loading modenv/scs5"
ml release/23.04
p "5" "Loading MongoDB/4.0.3"
ml MongoDB/4.0.3
p "7" "Loading Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4"
ml GCC/11.3.0 OpenMPI/4.1.4 Hyperopt/0.2.7
p "9" "Loading Python/3.7.4-GCCcore-8.3.0"
ml Hyperopt/0.2.7 
p "9" "Loading matplotlib/3.1.1-foss-2019b-Python-3.7.4"
ml matplotlib/3.5.2

p "15" "Loaded all modules"

python3 script/get_wallclock_time.py --projectdir=$projectdir --project=$project

python3 script/endmongodb.py --projectdir=$projectdir --project=$project 2>/dev/null
