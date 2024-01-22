#!/bin/bash -l

PROJECTDIR=$1
PROJECT=$2

maxvalue=$(whiptail --inputbox "Max. result-value for $PROJECTDIR/$PROJECT?" 8 39 "10" --title "Max value" 3>&1 1>&2 2>&3)

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

ml () {
    eval $($LMOD_DIR/ml_cmd "$@")
}
module () {
    eval `$LMOD_CMD sh "$@"`
}
mml () {
        if ! ml is-loaded $1; then
            ml $1
        fi
}

mml release/23.04 2>&1 | grep -v load
mml MongoDB/4.0.3 2>&1 | grep -v load
mml GCC/11.3.0 2>&1 | grep -v load
mml OpenMPI/4.1.4 2>&1 | grep -v load
mml Hyperopt/0.2.7 2>&1 | grep -v load
mml matplotlib/3.5.2 2>&1 | grep -v load
mml GCCcore/11.3.0 2>&1 | grep -v load
mml gnuplot/5.4.4 2>&1 | grep -v load


RANDFILE=$RANDOM.txt
while [[ -e $RANDFILE ]]; do
    RANDFILE=$RANDOM.txt
done

GNUPLOTRANDFILE=$RANDOM.txt
while [[ -e $GNUPLOTRANDFILE ]]; do
    GNUPLOTRANDFILE=$RANDOM.txt
done

while true; do
    perl tools/run_mongodb_on_project.pl --project=$PROJECT --projectdir=$PROJECTDIR '--query=db.jobs.find({"result.status": { $eq: "ok" } }, { "result.loss": 1, "result.all_outputs.endtime": 1 } ).toArray()' --dontloadmodules | egrep '"(loss|endtime)"' | sed -e 's/^\s*//' | sed -e 's/.* : //' | paste -d " "  - - | sed -e 's/,//' > $RANDFILE
    if [[ "$?" -ne "0" ]]; then
        rm $RANDFILE
        exit $?
    fi

    if [[ ! -s $RANDFILE ]]; then
        rm $RANDFILE
        exit 1
    fi

    sleep 5
    clear
    
    echo "Max. value: $maxvalue"

    echo "reset
set term dumb size 180,40
set border lw 1
unset key
set xdata time
set timefmt '%s'
set format x '%H:%M:%S'
set yrange [0:$maxvalue]
plot '$RANDFILE' using 2:1 with points pt 'X'
" > $GNUPLOTRANDFILE

    gnuplot $GNUPLOTRANDFILE
    rm $GNUPLOTRANDFILE
    rm $RANDFILE
    sleep 1
done
