#!/bin/bash -l 

#SBATCH --time=10:00:00
#SBATCH --partition=haswell
#SBATCH --mem-per-cpu=20000
#SBATCH --N=1
#SBATCH --n=1
#SBATCH -J "videoexport"

function echoerr() {
        echo "$@" 1>&2
}

function red_text {
        echoerr -e "\e[31m$1\e[0m"
}

export DISPLAYGAUGE=0

set -e

function help () {
        echo "Create animation frames of an OmniOpt-run automatically"
        echo "Possible options:"
        echo "  --project=STR                                      Project name (needs to be set)"
        echo "  --todir=STR                                        Folder where the files should be written to (optional)"
        echo "  --frameno=INT                                      Number of frames (needs to be set)"
        echo "  --maxvalue=INT                                     Max value for graphs"
        echo "  --projectdir=STR                                   Maindir of projects (can be left empty)"
        echo "  --create_video                                     Auto-create video from single frames"
        echo "  --help                                             this help"
        echo "  --debug                                            Enables debug mode (set -x)"
        exit $1
}
export project=
export projectdir=./projects/
export NUMBEROFFRAMES=
export todir=
export maxvalue=9999999999999
export create_video=0
export MONGOTIMEOUT=1

for i in "$@"; do
        case $i in
                --create_video)
                        create_video=1
                        shift
                        ;;
                --maxvalue=*)
                        maxvalue="${i#*=}"
                        shift
                        ;;
                --todir=*)
                        todir="${i#*=}"
                        shift
                        ;;
                --projectdir=*)
                        projectdir="${i#*=}"
                        shift
                        ;;
                --frameno=*)
                        NUMBEROFFRAMES="${i#*=}"
                        export NUMBEROFFRAMES
                        re='^[+-]?[0-9]+$'
                        if ! [[ $NUMBEROFFRAMES =~ $re ]] ; then
                                red_text "error: Not a INT: $i" >&2
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

if [[ -z $project ]]; then
        red_text "--project= cannot be empty"
        exit 1
else
        if [[ ! -d "$projectdir/$project" ]]; then
                red_text "$projectdir/$project cannot be found"
                exit 2
        fi
fi

if [[ -z $NUMBEROFFRAMES ]]; then
        red_text "--frameno= cannot be empty"
        exit 1
fi

if [[ -z $todir ]]; then
        i=0
        MAINPATH=$HOME/${project}_graphanimation_${i}/
        while [[ -e $MAINPATH ]]; do
                MAINPATH=$HOME/${project}_graphanimation_${i}/
                i=$((i+1))
        done
        mkdir $MAINPATH
else
        MAINPATH=$todir
fi

function calltracer () {
        python3 script/endmongodb.py --projectdir=$projectdir --project=$project
        echo 'Last file/last line:'
        caller
}

trap 'calltracer' ERR

if [[ -z "$SLURM_JOB_ID" ]]; then
        red_text "If you use a large number of frames or a large database it is recommended to run this script inside an sbatch-job. You don't need GPUs for that, and you should allocate some x86_64 partition for this, e.g. haswell. Remember to use enough RAM!"
        red_text "Example-Call:"
        red_text "srun -N 1 -n 1 --partition=haswell --time=05:00:00 --mem-per-cpu=2400 --pty $SHELL"
fi

echo "Using $MAINPATH for animation path"


set -e

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
        eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
        eval `$LMOD_CMD sh "$@"`
}

ml release/23.04 2>&1 | grep -v load
ml MongoDB/4.0.3 2>&1 | grep -v load
ml GCC/11.3.0 2>&1 | grep -v load
ml OpenMPI/4.1.4 2>&1 | grep -v load
ml Hyperopt/0.2.7 2>&1 | grep -v load
ml matplotlib/3.5.2 2>&1 | grep -v load

IS_FIRST_LINE=1
export DONTCHECKMONGODBSTART=1
export EXPORT_FORMAT=png
export SVGEXPORTSIZE=1000
export DPISIZE=72

export PLOTPATH="$MAINPATH/${project}_%s.png"
perl tools/plot.pl --project=$project --projectdir=$projectdir --dontloadmodules --nopip --maxvalue=${maxvalue}

if [[ "$create_video" -eq "1" ]]; then
        cd $MAINPATH
        ml FFmpeg/4.2.2
        ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
        echo "Created video under $MAINPATH/out.mp4"
else
        red_text "No video will be created. Do that manually, e.g. with:"
        red_text "$MAINPATH"
        echo "ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4"
fi
