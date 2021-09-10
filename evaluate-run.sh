#!/bin/bash -l

# Find stdout files that have no "Resource exhausted: OOM" and no "RESULT: [0-9]", but a "RESULT: ", implying the script main-script finished, but not crashed internally
# for i in $(grep -L "Resource exhausted: OOM" *.stderr); do corresponding_outfile=$(basename $i ".stderr").stdout; if grep "RESULT:" $corresponding_outfile >/dev/null; then egrep -L "RESULT: [0-9]" $corresponding_outfile; fi; done

export BUBBLESIZEINPX=7
export NONZERODIGITS=2
export SHOWFAILEDJOBSINPLOT=0
export SVGEXPORTSIZE=2000
export SHOWALLGPUS=0
export HIDEMAXVALUESINPLOT=0
export ASKEDTOUPGRADE=0
export DISPLAYGAUGE=1
export DEBUG=0
export LOAD_MODULES=1
export UPGRADE=1

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
module () {
    eval `$LMOD_CMD sh "$@"`
}
ml () {
    eval $($LMOD_DIR/ml_cmd "$@")
}

#set -e
set -o pipefail

function calltracer () {
    echo 'Last file/last line:'
    caller
}
trap 'calltracer' ERR

numberre='^[0-9]+$'

PROJECTDIR=projects

echoerr() {
	echo "$@" 1>&2
}

function red_text {
	echoerr -e "\e[31m$1\e[0m"
}

function green_text {
	echoerr -e "\e[92m$1\e[0m"
}

function debug_code {
    if [[ "$DEBUG" -eq "1" ]]; then
        echoerr -e "\e[93m$1\e[0m"
    fi
}

function echo_red {
	echo -e "\e[31m$1\e[0m"
}

function echo_green {
	echo -e "\e[32m$1\e[0m"
}

function this_help {
	cat <<'EOF'
This helps graphically to use OmniOpt.

-p=/path/                                                   Path to available projects
--projectdir=/path/to/projects/

--nogauge                                                   Disables the gauges

-h                                                          This help
--help

--d                                                         Enable set -x debugging
--dont_load_modules                                         Don't load modules
--no_upgrade                                                Disables upgrades
--debug
EOF
}

function success_whiptail {
        OLD_NEWT_COLORS=$NEWT_COLORS
        export NEWT_COLORS='
root=white,blue
border=black,green
window=lightgray,green
shadow=darkgreen,black
title=black,green
button=black,cyan
actbutton=white,cyan
compactbutton=black,green
checkbox=black,green
actcheckbox=lightgray,cyan
entry=black,green
disentry=gray,green
label=black,green
listbox=black,green
actlistbox=black,cyan
sellistbox=lightgray,black
actsellistbox=lightgray,black
textbox=black,green
acttextbox=black,cyan
emptyscale=,gray
fullscale=,cyan
helpline=white,black
roottext=lightgrey,black
'
        whiptail "$@"
        export NEWT_COLORS="$OLD_NEWT_COLORS"
}

function error_whiptail {
        OLD_NEWT_COLORS=$NEWT_COLORS
        export NEWT_COLORS='
root=white,black
border=black,red
window=lightgray,red
shadow=black,darkred
title=black,red
button=black,cyan
actbutton=white,cyan
compactbutton=black,red
checkbox=black,red
actcheckbox=lightgray,cyan
entry=black,red
disentry=gray,red
label=black,red
listbox=black,red
actlistbox=black,cyan
sellistbox=lightgray,black
actsellistbox=lightgray,black
textbox=black,red
acttextbox=black,cyan
emptyscale=,gray
fullscale=,cyan
helpline=white,black
roottext=lightgrey,black
'
        whiptail "$@"
        export NEWT_COLORS="$OLD_NEWT_COLORS"
}

for i in "$@"; do
    case $i in
        -d|--debug)
            DEBUG=1
            set -x
            ;;

        --nogauge)
            export DISPLAYGAUGE=0
            ;;

        --dont_load_modules)
            export LOAD_MODULES=0
            ;;

        -p=*|--projectdir=*)
            PROJECTDIR="${i#*=}"
            ;;

        --no_upgrade)
            UPGRADE=0
            ;;

        -h|--help)
            this_help
            exit 0
            ;;

        *)
            echo_red "Unknown option $i"
            this_help
            exit 1
            ;;
    esac
done

function gaugecommand () {
    set +x
    PCT=0
    TITLE=$1
    TEXT=$2
    COMMAND=$3

    COMMANDFILE="$RANDOM.command"
    while [[ -e $COMMANDFILE ]]; do
            COMMANDFILE="$RANDOM.command"
    done

    echo $COMMAND > $COMMANDFILE 2>/dev/null

    if [[ "$DISPLAYGAUGE" -eq "1" ]]; then
            GAUGEFILE=$RANDOM.gauge
            while [[ -e $GAUGE ]]; do
                    GAUGEFILE=$RANDOM.gauge
            done

            (stdbuf -oL -eL bash $COMMANDFILE 2> $GAUGEFILE >/dev/null) &
            OLDTERM=$TERM
            export TERM=linux

            JOBPID=$(jobs -p | sed -e 's/ running.*//' | sed -e 's/.*\+ //')
            ( 
                    while ps -auxf | grep -v grep | grep $JOBPID >/dev/null; do
                            if [[ -e $GAUGEFILE ]]; then
                                echo "XXX"
                                CURRENT_PERCENTAGE=$(cat $GAUGEFILE | grep PERCENTGAUGE | sed -e 's/PERCENTGAUGE: //' | tail -n1)
                                echo $CURRENT_PERCENTAGE
                                HASMSG=$(cat $GAUGEFILE | grep GAUGESTATUS | tail -n1 | sed -e 's/GAUGESTATUS: //')
                                if [[ -z $HASMSG ]]; then
                                    echo "$TEXT"
                                else
                                    echo "$TEXT ($HASMSG)"
                                fi
                                echo "XXX"
                                if [[ "$CURRENT_PERCENTAGE" -eq "100" ]]; then
                                    exit
                                fi
                            fi
                            sleep 1
                    done
            ) | whiptail --title "$TITLE" --gauge "$TEXT" 8 120 0
            cat $GAUGEFILE
            rm $GAUGEFILE
            export TERM=$OLDTERM
    else
            bash $COMMANDFILE
    fi

    if [[ "$DEBUG" -eq "1" ]]; then
        set -x
    fi
    rm $COMMANDFILE
}


function info_message_large {
	MSG=$1
    eval `resize`
	echo_green "$MSG"
	whiptail --title "Info Message" --msgbox "$MSG" $LINES $COLUMNS
}

function info_message {
	MSG=$1
	echo_green "$MSG"
	whiptail --title "Info Message" --msgbox "$MSG" 8 78
}

function error_message {
	MSG=$1
	echo_red "$MSG"
	export NEWT_COLORS='
window=,red
border=white,red
textbox=white,red
button=black,white
'
	whiptail --title "Error Message" --msgbox "$MSG" 8 78
	export NEWT_COLORS=""
}

function inputbox {
	TITLE=$1
	MSG=$2
	DEFAULT=$3

	COLOR=$(whiptail --inputbox "$MSG" 8 39 "$DEFAULT" --title "$TITLE" 3>&1 1>&2 2>&3)
	exitstatus=$?
	if [[ $exitstatus == 0 ]]; then
		echo "$COLOR"
	else
		echo_red "You chose to cancel (1)"
		exit 1
	fi
}

function change_project_dir {
	PROJECTDIR=$(inputbox "Projectdir" "Enter a new Projectdir" "$PROJECTDIR")
}

mongodbtojson () {
    ip=$1 
    port=$2 
    dbname=$3 
    mongo --quiet mongodb://$ip:$port/$dbname --eval "db.jobs.find().pretty().toArray();"
}

function get_possible_options_for_gpu_plot {
    PROJECT=$1
    LOGDIR=$2

    POSSIBLE_OPTIONS=$(ls -t $LOGDIR/*/nvidia-*/gpu_usage.csv | sed -e 's/nvidia-.*//' | sed -e "s#$LOGDIR/*##" | sed -e 's#/$##' | sed -e 's/^//' | sed -e 's/$//' | perl -le 'while (<>) { chomp; s#(.*)#\1 \1#g; print qq#$_# } ' | uniq | tr '\n' ' ')

    echo $POSSIBLE_OPTIONS
}

function list_gpu_log_folders_for_plot_svg {
    PROJECT=$1

    LOGDIR=$PROJECTDIR/$PROJECT/logs/
    POSSIBLE_OPTIONS=$(get_possible_options_for_gpu_plot $PROJECT $LOGDIR)

    WHATTODO=$(whiptail --title "Available logfolders under ${LOGDIR} for plotting to svg" --menu "Chose any of the available logdirs" $LINES $COLUMNS $(( $LINES - 8 )) $POSSIBLE_OPTIONS "b)" "back" "q)" "quit" 3>&1 1>&2 2>&3)
    if [[ "$WHATTODO" =~ "b)" ]]; then
        list_option_for_job $PROJECT
    elif [[ "$WHATTODO" =~ "q)" ]]; then
        exit
    else
        export SVGFILE=$HOME/${PROJECT}.svg
        CNT=0
        while [[ -e $SVGFILE ]]; do
                CNT=$(($CNT+1))
                export SVGFILE=$HOME/${PROJECT}_${CNT}.svg
        done

        gaugecommand "Plotting GPU-Usage to $SVGFILE" "Please wait, this takes some time" "perl tools/plot_gpu.pl --project=$PROJECT --projectdir=$PROJECTDIR --logdate=$WHATTODO --filename=$SVGFILE"
        if [[ -e "$SVGFILE" ]]; then
                info_message "Wrote to file $SVGFILE"
        else
                error_message "Failed to write $SVGFILE"
        fi
        read -rsn1 -p"Press any key to continue";echo
        list_gpu_log_folders_for_plot_svg $PROJECT
    fi
}

function list_gpu_log_folders_for_plot {
    PROJECT=$1

    LOGDIR=$PROJECTDIR/$PROJECT/logs/
    POSSIBLE_OPTIONS=$(get_possible_options_for_gpu_plot $PROJECT $LOGDIR)

    WHATTODO=$(whiptail --title "Available logfolders under ${LOGDIR} for plotting to screen" --menu "Chose any of the available logdirs" $LINES $COLUMNS $(( $LINES - 8 )) $POSSIBLE_OPTIONS "b)" "back" "q)" "quit" 3>&1 1>&2 2>&3)
    if [[ "$WHATTODO" =~ "b)" ]]; then
        list_option_for_job $PROJECT
    elif [[ "$WHATTODO" =~ "q)" ]]; then
        exit
    else
        gaugecommand "Plotting GPU-Usage" "Please wait, this takes some time" "perl tools/plot_gpu.pl --project=$PROJECT --projectdir=$PROJECTDIR --logdate=$WHATTODO"
        list_gpu_log_folders_for_plot $PROJECT
    fi
}

function job_is_running {
        JOBNAME=$1

        if squeue -u "$USER" -o "%.100j" | grep "$JOBNAME"; then
            return 0
        else
            return 1
        fi
}

function list_option_for_job {
    PROJECT=$1
    eval `resize`

    THISPROJECTDIR="$PROJECTDIR/$PROJECT/"
    THISCONFIGINI="$THISPROJECTDIR/config.ini"
    THISMONGODIR="$THISPROJECTDIR/mongodb"
    THISSINGLELOGS="$THISPROJECTDIR/singlelogs"

    args=()

    if [[ -e $THISCONFIGINI ]]; then
        if [[ -d $THISMONGODIR ]]; then
            args+=("p)" "plot graph" "psvg)" "Plot graph to svg file" "P)" "plot graph with max value" "Psvg)" "Plot graph with max value to svg" "pa)" "Parallel plot" "v)" "Plot video"  "r)" "Repair Database" "c)" "get csv to stdout" "C)" "get csv to file" "wct)" "Get wallclock-time of all jobs (only useful for jobs that ran once)")
        fi

        args+=("co)" "Show run config")
    fi

    if [[ -d $THISSINGLELOGS ]]; then
        args+=("s)" "Auto-analyze jobs from singlelogs (may take very long)")
    fi

    if [[ -d $THISPROJECTDIR ]]; then
        args+=("e)" "Check this project for errors")
        args+=("d)" "Create debug-zip")
    fi

    if job_is_running $PROJECT; then
        args+=("n)" "Number of jobs with status OK (job must be running)" "N)" "Number of jobs with status OK every 10 seconds (job must be running)" "f)" "Number of jobs with status FAIL (job must be running)")
    fi
    
    if [[ -d "$PROJECTDIR/$PROJECT/logs/" ]]; then
        if [[ $(ls $PROJECTDIR/$PROJECT/logs/*/nvidia*/gpu_usage.csv | wc -l 2>/dev/null) -ne "0" ]]; then
            args+=("g)" "plot gpu usage" "G)" "plot gpu usage to svg file")
        fi
    fi

    WHATTODO=$(whiptail --title "Available options for >${PROJECT}<" --menu "Chose what to do with >$PROJECT<" $LINES $COLUMNS $(( $LINES - 8 )) "${args[@]}" "b)" "back" "q)" "quit" 3>&1 1>&2 2>&3)

    exitstatus=$?
    if [[ $exitstatus == 0 ]]; then
        if [[ "$WHATTODO" =~ "b)" ]]; then
                list_projects
        elif [[ "$WHATTODO" =~ "s)" ]]; then
            show_number_of_results $PROJECT
        elif [[ "$WHATTODO" =~ "q)" ]]; then
            exit
        elif [[ "$WHATTODO" =~ "co)" ]]; then
            if [[ -e $THISCONFIGINI ]]; then
                perl tools/showconfig.pl $PROJECT $THISCONFIGINI | less -R -c -S
            else
                whiptail --title "Error" --msgbox "'$THISCONFIGINI' cannot be found" 8 78
            fi
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "wct)" ]]; then
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                    bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            WCT_RESULT_FILE=${RANDOM}.txt
            while [[ -e $WCT_RESULT_FILE ]]; do
                    WCT_RESULT_FILE=${RANDOM}.txt
            done
            gaugecommand "Loading Wallclock-time" "This may take some time" "bash tools/get_wallclock_time.sh --project=$PROJECT --projectdir=$PROJECTDIR > $WCT_RESULT_FILE"
            WCT_RESULT=$(cat $WCT_RESULT_FILE)
            rm $WCT_RESULT_FILE

            whiptail --title "Wallclock-Time" --msgbox "$WCT_RESULT" 8 78

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "e)" ]]; then
            bash tools/error_analyze.sh --project=$PROJECT --projectdir=$PROJECTDIR

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "d)" ]]; then
            DEBUGFILE=debug.zip
            let i=0
            while [[ -e $DEBUGFILE ]]; do
                DEBUGFILE="debug_${i}.zip"
                let i++
            done

            error_analyze_file=error_analyze.log
            let i=0
            while [[ -e $error_analyze_file ]]; do
                error_analyze_file="error_analyze_${i}.log"
                let i++
            done

            bash tools/error_analyze.sh --project=$PROJECT --projectdir=$PROJECTDIR --nowhiptail 2>/dev/null > $error_analyze_file

            zip -r $DEBUGFILE $error_analyze_file *.out debuglogs/* $PROJECTDIR/$PROJECT/* -x $PROJECTDIR/$PROJECT/mongodb/\*

            rm $error_analyze_file

            if [[ -e $DEBUGFILE ]]; then
                if [[ "$USER" -eq "s3811141" ]]; then
                    info_message_large "Wrote $DEBUGFILE. Send this file to <norman.koch@tu-dresden.de> for debugging-help.\nscp_taurus $(pwd)/$DEBUGFILE .\n"
                else
                    info_message_large "Wrote $DEBUGFILE. Send this file to <norman.koch@tu-dresden.de> for debugging-help.\nscp $USER@taurus.hrsk.tu-dresden.de://$(pwd)/$DEBUGFILE .\n"
                fi
            else
                error_message "Could not write $DEBUFFILE"
            fi

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "c)" ]]; then
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                    bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            gaugecommand "CSV-Export" "Loading CSV-Export" "perl script/runningdbtocsv.pl --project=$PROJECT --projectdir=$PROJECTDIR"

            read -rsn1 -p"Press any key to continue";echo
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "pa)" ]]; then
            CSV_DIR=${PROJECTDIR}/${PROJECT}/csv/
            mkdir -p $CSV_DIR
            csv_filename=${CSV_DIR}/${PROJECT}.csv
            let i=1
            while [[ -e $csv_filename ]]; do
                csv_filename=${CSV_DIR}/${PROJECT}_${i}.csv
                let i++
            done

            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            gaugecommand "CSV-Export" "Loading CSV-Export, printing to $csv_filename" "perl script/runningdbtocsv.pl --project=$PROJECT --projectdir=$PROJECTDIR --filename=$csv_filename"
            if [[ -e "$csv_filename" ]]; then
                if [[ -s "$csv_filename" ]]; then
                    parallelplot_file=$PROJECTDIR/$PROJECT/parallel/plot.html
                    bash tools/parallel_plot.sh $csv_filename $parallelplot_file
                else
                    whiptail --title "File printed" --msgbox "The file was printed to $csv_filename but is empty" 8 78
                fi
            else
                whiptail --title "File not printed" --msgbox "The file was NOT printed to $csv_filename, this might be a Bug in OmniOpt. Contact <norman.koch@tu-dresden.de>." 8 78
            fi
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "C)" ]]; then
            CSV_DIR=${PROJECTDIR}/${PROJECT}/csv/
            mkdir -p $CSV_DIR
            csv_filename=${CSV_DIR}/${PROJECT}.csv
            let i=1
            while [[ -e $csv_filename ]]; do
                csv_filename=${CSV_DIR}/${PROJECT}_${i}.csv
                let i++
            done

            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            csv_filename=$(inputbox "Filename for the CSV file" "Path of the file for the CSV of $PROJECT" "$csv_filename")
            if [[ $? = 0 ]]; then
                echo "Filename: $csv_filename"

                gaugecommand "CSV-Export" "Loading CSV-Export, printing to $csv_filename" "perl script/runningdbtocsv.pl --project=$PROJECT --projectdir=$PROJECTDIR --filename=$csv_filename"
                if [[ -e "$csv_filename" ]]; then
                    if [[ -s "$csv_filename" ]]; then
                        whiptail --title "File printed" --msgbox "The file was printed to $csv_filename" 8 78
                    else
                        whiptail --title "File printed" --msgbox "The file was printed to $csv_filename but is empty" 8 78
                    fi
                else
                    whiptail --title "File not printed" --msgbox "The file was NOT printed to $csv_filename, this might be a Bug in OmniOpt. Contact <norman.koch@tu-dresden.de>." 8 78
                fi
            else
                echo_red "You cancelled the CSV creation"
            fi
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "p)" ]]; then
            echo_green "Plot"
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi
            if [[ -d "$PROJECTDIR/$PROJECT/singlelogs" ]]; then
                gaugecommand "Graph-Creation" "Please wait, this takes some time..." "perl tools/plot.pl --project=$PROJECT --projectdir=${PROJECTDIR}/"
            else
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." 8 78
            fi
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "psvg)" ]]; then
            export PLOTPATH=$HOME/${PROJECT}.svg
            export CNT=0
            while [[ -e $PLOTPATH ]]; do
                export CNT=$(($CNT+1))
                export PLOTPATH=$HOME/${PROJECT}_${CNT}.svg
            done

            echo_green "Plot"
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                    bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            if [[ -d "$PROJECTDIR/$PROJECT/singlelogs" ]]; then
                gaugecommand "Graph-Creation" "Please wait, this takes some time..." "perl tools/plot.pl --project=$PROJECT --projectdir=${PROJECTDIR}/"
            else
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." 8 78
            fi
            read -rsn1 -p"Press any key to continue";echo
            export PLOTPATH=
            list_option_for_job $PROJECT

        elif [[ "$WHATTODO" =~ "P)" ]]; then
            echo_green "Plot with max value"

            if [[ -d "$PROJECTDIR/$PROJECT/singlelogs" ]]; then
                maxvalue=$(inputbox "Max value for plot" "Enter a max value for plotting $PROJECT (float)" "0.1")
                echo_green "Got maxvalue = ${maxvalue}"
                if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                        bash tools/repair_database.sh $PROJECTDIR/$PROJECT
                fi
                gaugecommand "Graph-Creation" "Please wait, this takes some time..." "perl tools/plot.pl --project=$PROJECT --projectdir=${PROJECTDIR}/ --maxvalue=$maxvalue"
            else
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." 8 78
            fi



            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "v)" ]]; then
            length_in_seconds=$(inputbox "How many seconds should the video have?" "Number of seconds of the video" "30")
            number_of_frames=$((20 * $length_in_seconds))
            echo_green "Got number_of_frames= ${number_of_frames}"
            max_value=$(inputbox "Should there be a max value?" "Max value? (leave empty for none)" "")
            echo_green "Got max_value = ${max_value}"

            echo_green "Plotvideo"
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            if [[ -z $max_value ]]; then
                srun tools/animation.sh --project=$PROJECT --frameno=$number_of_frames --create_video
            else
                srun tools/animation.sh --project=$PROJECT --frameno=$number_of_frames --maxvalue=$max_value --create_video
            fi

            export PLOTPATH=

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "Psvg)" ]]; then
            maxvalue=$(inputbox "Max value for plot" "Enter a max value for plotting $PROJECT (float)" "0.1")
            echo_green "Got maxvalue = ${maxvalue}"

            export PLOTPATH=$HOME/${PROJECT}_limit_${maxvalue}.svg
            CNT=0
            while [[ -e $PLOTPATH ]]; do
                CNT=$(($CNT+1))
                export PLOTPATH=$HOME/${PROJECT}_${CNT}_limit_${maxvalue}_%s.svg
            done

            echo_green "Plot"
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi
            gaugecommand "Graph-Creation" "Please wait, this takes some time..." "perl tools/plot.pl --project=$PROJECT --projectdir=${PROJECTDIR}/ --maxvalue=$maxvalue"
            read -rsn1 -p"Press any key to continue";echo
            export PLOTPATH=

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "r)" ]]; then
            echo_green "Repair database"
            bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "g)" ]]; then
            echo_green "Plot GPU usage"
            list_gpu_log_folders_for_plot $PROJECT
        elif [[ "$WHATTODO" =~ "G)" ]]; then
            echo_green "Plot GPU usage to SVG"
            list_gpu_log_folders_for_plot_svg $PROJECT
        elif [[ "$WHATTODO" =~ "n)" ]]; then
            NUMBER_OF_OK_JOBS=$(perl tools/run_mongodb_on_project.pl --project=$PROJECT --projectdir=$PROJECTDIR --query='db.jobs.find({"result.status": { $eq: "ok" } }, { "result.loss": 1 } ).count()' 2>/dev/null)
            if [[ "$NUMBER_OF_OK_JOBS" == "" ]]; then
                    NUMBER_OF_OK_JOBS="*an unknown number of* (check if the job is running)"
            fi
            info_message "There are $NUMBER_OF_OK_JOBS jobs with status OK in the DB of $PROJECT"
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "N)" ]]; then
            NUMBER_OF_OK_JOBS=$(perl tools/run_mongodb_on_project.pl --project=$PROJECT --projectdir=$PROJECTDIR --query='db.jobs.find({"result.status": { $eq: "ok" } }, { "result.loss": 1 } ).count()' 2>/dev/null) 
            set +x
            while [ 1 ]; do
                {
                    for ((i = 0 ; i <= 100 ; i+=5)); do
                        if [[ i -eq 0 ]]; then
                            NUMBER_OF_OK_JOBS=$(perl tools/run_mongodb_on_project.pl --project=$PROJECT --projectdir=$PROJECTDIR --query='db.jobs.find({"result.status": { $eq: "ok" } }, { "result.loss": 1 } ).count()' 2>/dev/null) 
                        fi
                        sleep 0.05
                        echo $i
                        read -t 0.1 -N 1 input
                        if [[ $input == 'q' ]] || [[ $input == "Q" ]];then
                            exit 1
                        fi
                    done
                } | whiptail --gauge "Currently, there are $NUMBER_OF_OK_JOBS status OK Jobs in the DB for ${PROJECT}. Wait or keep pressing CTRL-c to exit..." 6 120 0
                echo "Press CTRL-C to cancel now"
                sleep 1
            done
            if [[ "$DEBUG" -eq "1" ]]; then
                set -x
            fi

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "f)" ]]; then
            NUMBER_OF_FAILED_JOBS=$(perl tools/run_mongodb_on_project.pl --project=$PROJECT --projectdir=$PROJECTDIR --query='db.jobs.find({"result.status": { $eq: "fail" } }, { "result.loss": 1 } ).count()' 2>/dev/null)
            if [[ "$NUMBER_OF_OK_JOBS" == "" ]]; then
                NUMBER_OF_FAILED_JOBS="*an unknown number of* (check if the job is running)"
            fi
            error_message "There are $NUMBER_OF_FAILED_JOBS jobs with status FAIL in the DB of $PROJECT"
            list_option_for_job $PROJECT
        else
            echo_red "ADA2"
        fi
    else
        echo_red "You chose to cancel (2)"
        exit 1
    fi
}

function get_squeue_from_format_string {
	if ! command -v squeue &> /dev/null; then
		red_text "squeue not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if [[ $FAILED == 0 ]]; then
		for line in $(squeue -u $USER --format "$1" | sed '1d' | sed -e "s/''/'/g"); do 
			echo "$line" | tr '\n' ' '; 
		done
	fi
}


function slurmlogpath {
	if command -v scontrol &> /dev/null; then
		if command -v grep &> /dev/null; then
			if command -v sed &> /dev/null; then
				debug_code "scontrol show job $1 | grep StdOut | sed -e 's/^\\s*StdOut=//'"
				scontrol show job $1 | grep StdOut | sed -e 's/^\s*StdOut=//'
			else
				red_text "sed not found"
			fi
		else
			red_text "grep not found"
		fi
	else
		red_text "scontrol not found"
	fi
}

function get_job_name {
	if command -v scontrol &> /dev/null; then
		scontrol show job $1 | grep JobName | sed -e 's/.*JobName=//'
	else
		red_text "Program scontrol does not exist, cannot run it"
	fi
}

function run_commands_in_parallel {
	args=$#
	if command -v screen &> /dev/null; then
		if command -v uuidgen &> /dev/null; then
			THISSCREENCONFIGFILE=/tmp/$(uuidgen).conf
			for (( i=1; i<=$args; i+=1 )); do
				thiscommand=${@:$i:1}
				echo "screen $thiscommand" >> $THISSCREENCONFIGFILE
				echo "title '$thiscommand'" >> $THISSCREENCONFIGFILE
				if [[ $i -ne $args ]]; then
					echo "split -v" >> $THISSCREENCONFIGFILE
					echo "focus right" >> $THISSCREENCONFIGFILE
				fi
			done
			if [[ -e $THISSCREENCONFIGFILE ]]; then
				echo $THISSCREENCONFIGFILE
				cat $THISSCREENCONFIGFILE
				screen -c $THISSCREENCONFIGFILE
				rm $THISSCREENCONFIGFILE
			else
				echo "$THISSCREENCONFIGFILE not found"
			fi
		else
			echo "Command uuidgen not found, cannot execute run_commands_in_parallel"
		fi
	else
		echo "Command screen not found, cannot execute run_commands_in_parallel"
	fi
}

function multiple_slurm_tails {
	if command -v screen &> /dev/null; then
		if command -v uuidgen &> /dev/null; then
			THISSCREENCONFIGFILE=/tmp/$(uuidgen).conf
			for slurmid in "$@"; do
				logfile=$(slurmlogpath $slurmid)
				if [[ -e $logfile ]]; then
					echo "screen tail -f $logfile" >> $THISSCREENCONFIGFILE
					echo "title '$(get_job_name $slurmid) ($slurmid)'" >> $THISSCREENCONFIGFILE
					if [[ ${*: -1:1} -ne $slurmid ]]; then
						echo "split -v" >> $THISSCREENCONFIGFILE
						echo "focus right" >> $THISSCREENCONFIGFILE
					fi
				fi
			done
			if [[ -e $THISSCREENCONFIGFILE ]]; then
				debug_code "Screen file:"
				cat $THISSCREENCONFIGFILE
				debug_code "Screen file end"
				screen -c $THISSCREENCONFIGFILE
				rm $THISSCREENCONFIGFILE
			else
				red_text "$THISSCREENCONFIGFILE not found"
			fi
		else
			red_text "Command uuidgen not found, cannot execute multiple tails"
		fi
	else
		red_text "Command screen not found, cannot execute multiple tails"
	fi
}



function kill_multiple_jobs_usrsignal {
	FAILED=0
	if ! command -v squeue &> /dev/null; then
		red_text "squeue not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v scancel &> /dev/null; then
		red_text "scancel not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v whiptail &> /dev/null; then
		red_text "whiptail not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if [[ $FAILED == 0 ]]; then
		TJOBS=$(get_squeue_from_format_string '"%A" "%j (%t, %M)" OFF')
		chosenjobs=$(eval "whiptail --title 'Which jobs to kill with USR1?' --checklist 'Which jobs to choose USR1?' $WIDTHHEIGHT $TJOBS" 3>&1 1>&2 2>&3)
		if [[ -z $chosenjobs ]]; then
			green_text "No jobs chosen to kill"
		else
			export NEWT_COLORS='
window=,red
border=white,red
textbox=white,red
button=black,white
'
			if (whiptail --title "Really kill multiple jobs ($chosenjobs)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill multiple jobs ($chosenjobs)?" 8 78); then
				debug_code "scancel --signal=USR1 --batch $chosenjobs"
				eval "scancel --signal=USR1 --batch $chosenjobs"
				NEWT_COLORS=''
				return 0
			fi
			NEWT_COLORS=''
		fi
	fi
	return 1
}

function kill_multiple_jobs {
	FAILED=0
	if ! command -v squeue &> /dev/null; then
		red_text "squeue not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v scancel &> /dev/null; then
		red_text "scancel not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v whiptail &> /dev/null; then
		red_text "whiptail not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if [[ $FAILED == 0 ]]; then
		TJOBS=$(get_squeue_from_format_string "'%A' '%j (%t, %M)' OFF")
		chosenjobs=$(eval "whiptail --title 'Which jobs to kill?' --checklist 'Which jobs to choose?' $WIDTHHEIGHT $TJOBS" 3>&1 1>&2 2>&3)
		if [[ -z $chosenjobs ]]; then
			green_text "No jobs chosen to kill"
		else
			export NEWT_COLORS='
window=,red
border=white,red
textbox=white,red
button=black,white
'
			if (whiptail --title "Really kill multiple jobs ($chosenjobs)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill multiple jobs ($chosenjobs)?" 8 78); then
				debug_code "scancel $chosenjobs"
				eval "scancel $chosenjobs"
				NEWT_COLORS=''
				return 0
			fi
			NEWT_COLORS=''
		fi
	fi
	return 1
}

function tail_multiple_jobs {
	FAILED=0
	AUTOON=OFF

	if [[ $1 == 'ON' ]]; then
		AUTOON=ON
	fi

	if ! command -v squeue &> /dev/null; then
		red_text "squeue not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v tail &> /dev/null; then
		red_text "tail not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v whiptail &> /dev/null; then
		red_text "whiptail not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v screen &> /dev/null; then
		red_text "screen not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if [[ $FAILED == 0 ]]; then
        TJOBS=$(get_squeue_from_format_string "'%A' '%j (%t, %M)' $AUTOON")
		chosenjobs=$(eval "whiptail --title 'Which jobs to tail?' --checklist 'Which jobs to choose for tail?' $WIDTHHEIGHT $TJOBS" 3>&1 1>&2 2>&3)
		if [[ -z $chosenjobs ]]; then
			green_text "No jobs chosen to tail"
		else
			#whiptail --title "Tail for multiple jobs with screen" --msgbox "To exit, press <CTRL> <a>, then <\\>" 8 78 3>&1 1>&2 2>&3
			eval "multiple_slurm_tails $chosenjobs"
		fi
	fi
}


function single_job_tasks {
	chosenjob=$1
	gobacktolist_running_slurm_jobs="$2"

    if [[ -z $chosenjob ]]; then
        exit "No job chosen!!!"
    fi

    if ! command -v scontrol &> /dev/null; then
		red_text "scontrol not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	if ! command -v tail &> /dev/null; then
		red_text "tail not found. Cannot execute list_running_slurm_jobs without it"
		FAILED=1
	fi

	WHYPENDINGSTRING=""
	if command -v whypending &> /dev/null; then
		WHYPENDINGSTRING="'w)' 'whypending'"
	fi

	SCANCELSTRING=""
	if command -v scancel &> /dev/null; then
		SCANCELSTRING="'k)' 'scancel' 'c)' 'scancel with signal USR1'"
	fi

	TAILSTRING=""
	if command -v tail &> /dev/null; then
		TAILSTRING="'t)' 'tail -f'"
	else
		red_text "Tail does not seem to be installed, not showing 'tail -f' option"
	fi

	jobname=$(get_job_name $chosenjob)
    whiptailoptions="'n)' 'nvidia-smi on every node every n seconds' 'o)' 'run command on every node once' 'O)' 'run command on every node every n seconds'  's)' 'Show log path' $WHYPENDINGSTRING $TAILSTRING $SCANCELSTRING 'm)' 'go to main menu' 'q)' 'quit'"
	whattodo=$(eval "whiptail --title 'list_running_slurm_jobs >$jobname< ($chosenjob)' --menu 'What to do with job >$jobname< ($chosenjob)' $WIDTHHEIGHT $whiptailoptions" 3>&2 2>&1 1>&3)
	case $whattodo in
		"q)")
			debug_code "quitting single_job_tasks"
            exit
			;;
		"s)")
			debug_code "slurmlogpath $chosenjob"
            logpath=$(slurmlogpath $chosenjob)
            echo $logpath
            whiptail --title "Logpath of $chosenjob" --msgbox "$logpath" 20 100
            single_job_tasks $chosenjob
			;;
		"t)")
			chosenjobpath=$(slurmlogpath $chosenjob)
			debug_code "tail -f $chosenjobpath"
			tail -f $chosenjobpath
            single_job_tasks $chosenjob
			;;
		"w)")
			debug_code "whypending $chosenjob"
            whypending_str=$(whypending $chosenjob)
            echo $whypending_str
            whiptail --title "whypending $chosenjob" --msgbox "$whypending_str" 20 100
            single_job_tasks $chosenjob
			;;
		"k)")
			export NEWT_COLORS='
window=,red
border=white,red
textbox=white,red
button=black,white
'
			if (whiptail --title "Really kill >$jobname< ($chosenjob)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill >$jobname< ($chosenjob)?" 8 78); then
				debug_code "scancel $chosenjob"
				scancel $chosenjob && green_text "$jobname ($chosenjob) killed" || red_text "Error killing $jobname ($chosenjob)"
			fi
			NEWT_COLORS=""
			;;
		"m)")
			list_running_slurm_jobs
			;;
		"o)")
			RUNCOMMAND=$(inputbox "Which command to run once on every node of job $chosenjob" "Enter a new command" "")
			perl tools/run_on_every_node.pl --jobid="$chosenjob" --command="${RUNCOMMAND}"
            single_job_tasks $chosenjob
			;;
		"O)")
			RUNCOMMAND=$(inputbox "Which command to run once on every node of job $chosenjob" "Enter a new command" "")
			NUMBEROFSECONDS=$(inputbox "How many seconds between executing the command on all nodes?" "Enter an integer" "")
			perl tools/run_on_every_node.pl --jobid="$chosenjob" --command="${RUNCOMMAND}" --repeat=${NUMBEROFSECONDS}
			;;
        "n)")
			RUNCOMMAND="nvidia-smi"
			NUMBEROFSECONDS=$(inputbox "How many seconds between executing the command on all nodes?" "Enter an integer" "")
			perl tools/run_on_every_node.pl --jobid="$chosenjob" --command="${RUNCOMMAND}" --repeat=${NUMBEROFSECONDS}
			;;
		"c)")
			export NEWT_COLORS='
window=,red
border=white,red
textbox=white,red
button=black,white
'
			if (whiptail --title "Really kill with USR1 >$jobname< ($chosenjob)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill >$jobname< ($chosenjob) with USR1?" 8 78); then
				debug_code "scancel --signal=USR1 --batch $chosenjob"
				scancel --signal=USR1 --batch $chosenjob && green_text "$chosenjob killed" || red_text "Error killing $chosenjob"
			fi
			NEWT_COLORS=""
			;;
		"q)")
			green_text "Ok, exiting"
			;;
	esac

	if [[ $gobacktolist_running_slurm_jobs -eq '1' ]]; then
		main
	else
		single_job_tasks $chosenjob
	fi
}

function list_running_slurm_jobs {
    FAILED=0

    NEWT_COLORS=""

    if ! command -v squeue &> /dev/null; then
        red_text "squeue not found. Cannot execute list_running_slurm_jobs without it"
        FAILED=1
    fi

    JOBS=$(get_squeue_from_format_string '"%A" "%j (%t, %M)"')

    if ! command -v whiptail &> /dev/null; then
        red_text "whiptail not found. Cannot execute list_running_slurm_jobs without it"
        FAILED=1
    fi

    if ! command -v scontrol &> /dev/null; then
        red_text "scontrol not found. Cannot execute list_running_slurm_jobs without it"
        FAILED=1
    fi

    SCANCELSTRING=""
    if command -v scancel &> /dev/null; then
        if [[ ! -z $JOBS ]]; then
            SCANCELSTRING="'k)' 'kill multiple jobs' 'n)' 'kill multiple jobs with USR1'"
        fi
    fi

    TAILSTRING=""
    if command -v tail &> /dev/null; then
        if command -v screen &> /dev/null; then
            if [[ ! -z $JOBS ]]; then
                    TAILSTRING="'t)' 'tail multiple jobs' 'e)' 'tail multiple jobs (all enabled by default)'"
            fi
        else
            red_text "Screen could not be found, not showing 'tail multiple jobs' option"
        fi
    else
        red_text "Tail does not seem to be installed, not showing 'tail multiple jobs'"
    fi

    FULLOPTIONSTRING="$JOBS $SCANCELSTRING $TAILSTRING"

    if [[ $FAILED == 0 ]]; then
        WIDTHHEIGHT="$LINES $COLUMNS $(( $LINES - 8 ))"

        chosenjob=$(
            eval "whiptail --title 'Slurm Manager' --menu 'Welcome to the Slurm-Manager' $WIDTHHEIGHT $FULLOPTIONSTRING 'r)' 'Reload Slurm-Manager' 'm)' 'Go back to main menu' 'q)' 'Quit Slurm-Manager'" 3>&2 2>&1 1>&3
        )

        chosenjob=$(echo $chosenjob | tr -d '\n');

        if [[ $chosenjob == 'q)' ]]; then
            green_text "Ok, exiting"
            exit
        elif [[ $chosenjob == 'r)' ]]; then
            list_running_slurm_jobs
        elif [[ $chosenjob == 'm)' ]]; then
            main	
        elif [[ $chosenjob == 't)' ]]; then
            tail_multiple_jobs
        elif [[ $chosenjob == 'e)' ]]; then
            tail_multiple_jobs ON
        elif [[ $chosenjob == 'n)' ]]; then
            kill_multiple_jobs_usrsignal || list_running_slurm_jobs
        elif [[ $chosenjob == 'k)' ]]; then
            kill_multiple_jobs || list_running_slurm_jobs
        elif [[ "$chosenjob" =~ $numberre ]]; then
            single_job_tasks $chosenjob 1
        else
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! UNKNOWN OPTION: "
            echo "$chosenjob"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            exit
        fi
    else
        red_text  "Missing requirements, cannot run list_running_slurm_jobs"
    fi
}

function show_number_of_results {
    PROJECTNAME=$1
    THISPROJECTDIR=$PROJECTDIR/$PROJECTNAME
    if [[ -d $THISPROJECTDIR ]]; then
        number_of_resultfiles_as_whole=$(ls $THISPROJECTDIR/singlelogs/*.stdout | wc -l)
        if [[ $number_of_resultfiles_as_whole == 0 ]]; then
            whiptail --title "ERROR" --msgbox "It seems like no job has ever run in $THISPROJECTDIR." 8 78
        else
            number_of_resultfiles_with_result=$(grep -rm1 'RESULT: ' $THISPROJECTDIR/singlelogs/*.stdout | wc -l)
            number_of_valid_results=$(grep -rm1 'RESULT: [0-9]' $THISPROJECTDIR/singlelogs/*.stdout | wc -l)
            number_of_invalid_results=$(echo "$number_of_resultfiles_with_result-$number_of_valid_results" | bc)
            number_of_oom=$(grep -m1 "Resource exhausted: OOM" $THISPROJECTDIR/singlelogs/*.stderr | wc -l)
            number_of_broken_pipes=$(grep -rm1 "Broken pipe" $THISPROJECTDIR/singlelogs/*.stderr | wc -l)

            number_of_unfinished_jobs=$(($number_of_resultfiles_as_whole-$number_of_resultfiles_with_result))
            number_of_invalid_results=$(($number_of_invalid_results+$number_of_unfinished_jobs))

            number_of_explained_crashes=$(($number_of_broken_pipes+$number_of_oom))
            number_of_explained_crashes=$(($number_of_explained_crashes+$number_of_unfinished_jobs))
            number_of_unexplained_crashes=$(($number_of_invalid_results-$number_of_explained_crashes))

            worst_result=$(egrep "RESULT: [0-9]" $THISPROJECTDIR/singlelogs/*.stdout | sed -e 's/.*RESULT: //' | sort -nr | head -n1)
            best_result=$(egrep "RESULT: [0-9]" $THISPROJECTDIR/singlelogs/*.stdout | sed -e 's/.*RESULT: //' | sort -nr | tail -n1)

            oom_msg=''
            if [[ "$number_of_oom" -ne "0" ]]; then
                oom_msg="-> $number_of_oom Out-of-Memory-errors were detected.\n"
            fi
            broken_pipe_msg=''
            if [[ "$number_of_broken_pipes" -ne "0" ]]; then
                broken_pipe_msg="-> $number_of_broken_pipes broken-pipe-errors were detected.\n"
            fi

            whiptail --title "Result-Analysis" --msgbox "There are $number_of_resultfiles_as_whole .stdout files.\n-> $number_of_resultfiles_with_result of them have a 'RESULT'-String.\n-> Of those $number_of_valid_results seem to be valid results\n\nThis leaves us with $number_of_invalid_results invalid evaluations.\n${oom_msg}${broken_pipe_msg}-> Number of unfinished jobs: $number_of_unfinished_jobs\nThis means, $number_of_explained_crashes invalid evaluations are explained and $number_of_unexplained_crashes remain unexplained.\n\nWorst result: $worst_result\nBest result: $best_result." 20 90
        fi
    else
        whiptail --title "ERROR" --msgbox "The directory '$THISPROJECTDIR' does not seem to exist." 8 78
    fi
    list_option_for_job $PROJECTNAME
}

function change_variables {
    MENU_CHOICE=$(whiptail --title "Change variables" --menu "Choose an option" 25 120 16 "NONZERODIGITS" "Max. number of non-zero decimal places in the graph plot (currently $NONZERODIGITS)" "SHOWFAILEDJOBSINPLOT" "Show failed runs in plots with really high values (currently $SHOWFAILEDJOBSINPLOT)" "BUBBLESIZEINPX" "Size of bubbles in the plot graph (currently $BUBBLESIZEINPX)" "SVGEXPORTSIZE" "Size of the exported SVG-Graphs of Plot and GPU-Plot (currently $SVGEXPORTSIZE)" "SHOWALLGPUS" "Show all GPUs in GPU-Plot (currently $SHOWALLGPUS)" "HIDEMAXVALUESINPLOT" "Hide max values in Plot (currently $HIDEMAXVALUESINPLOT)" "DISPLAYGAUGE" "Display gauge when possible (currently $DISPLAYGAUGE)" "PROJECTDIR" "The path where projects are (currently $PROJECTDIR)" "DEBUG" "Debug evaluate-run.sh" "m)" "Main menu" 3>&1 1>&2 2>&3)
    exitstatus=$?
    if [[ $exitstatus == 0 ]]; then
        if [[ "$MENU_CHOICE" =~ "m)" ]]; then
            main
        elif [[ "$MENU_CHOICE" =~ "SHOWALLGPUS" ]]; then
            chosenvar=$(whiptail --inputbox "Show all GPUs instead of the ones from the log file only in GPU-Plot?" 8 39 "$SHOWALLGPUS" --title "SHOWALLGPUS" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "BUBBLESIZEINPX" ]]; then
            chosenvar=$(whiptail --inputbox "Size of the plot bubbles in px?" 8 39 "$BUBBLESIZEINPX" --title "BUBBLESIZEINPX" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "NONZERODIGITS" ]]; then
            chosenvar=$(whiptail --inputbox "Max. number of non-zero decimal places in the graph plot?" 8 39 "$NONZERODIGITS" --title "NONZERODIGITS" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "SVGEXPORTSIZE" ]]; then
            chosenvar=$(whiptail --inputbox "Width of the SVG-Exports" 8 39 "$SVGEXPORTSIZE" --title "SVGEXPORTSIZE" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "SHOWFAILEDJOBSINPLOT" ]]; then
            chosenvar=$(whiptail --inputbox "Show invalid jobs with really high values in plot? (0 = no, 1 = yes)" 8 39 "$SHOWFAILEDJOBSINPLOT" --title "SHOWFAILEDJOBSINPLOT" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "HIDEMAXVALUESINPLOT" ]]; then
            chosenvar=$(whiptail --inputbox "Hide max values in Plot? (0 = no, 1 = yes)" 8 39 "$HIDEMAXVALUESINPLOT" --title "HIDEMAXVALUESINPLOT" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "DISPLAYGAUGE" ]]; then
            chosenvar=$(whiptail --inputbox "Display gauge when possible? (0 = no, 1 = yes)" 8 39 "$DISPLAYGAUGE" --title "DISPLAYGAUGE" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "PROJECTDIR" ]]; then
            chosenvar=$(whiptail --inputbox "Path of Projects" 8 39 "$PROJECTDIR" --title "PROJECTDIR" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "DEBUG" ]]; then
            chosenvar=$(whiptail --inputbox "Debug" 8 39 "$DEBUG" --title "DEBUG" 3>&1 1>&2 2>&3)
            if [[ "$chosenvar" -eq "1" ]]; then
                set -x
            else
                set +x
            fi
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        else
            whiptail --title "Invalid option" --msgbox "The option '$MENU_CHOICE' is not valid. Returning to the main menu" 8 78 3>&1 1>&2 2>&3
            change_variables
        fi
    else
        echo_red "You chose to cancel (3)"
        exit 1
    fi
}

function list_projects {
    AVAILABLE_PROJECTS=$(ls $PROJECTDIR/*/config.ini | sed -e "s#${PROJECTDIR}/##" | sed -e 's#/config.ini##' | perl -le 'while (<>) { chomp; chomp; print qq#$_ $_# }')
	eval `resize`

    WHATTODO=$(whiptail --title "Available projects under ${PROJECTDIR}" --menu "Chose any of the available projects" $LINES $COLUMNS $(( $LINES - 8 )) $AVAILABLE_PROJECTS "c)" "Change the project dir" "s)" "List running SLURM jobs" "v)" "Show/Change Variables" "t)" "Run OmniOpt-Tests (fast)" "T)" "Run OmniOpt-Tests (complete)" "q)" "quit" 3>&1 1>&2 2>&3)

	exitstatus=$?
	if [[ $exitstatus == 0 ]]; then
		if [[ "$WHATTODO" =~ "c)" ]]; then
			change_project_dir
			main
		elif [[ "$WHATTODO" =~ "s)" ]]; then
			list_running_slurm_jobs
			main
		elif [[ "$WHATTODO" =~ "v)" ]]; then
            change_variables
            main
		elif [[ "$WHATTODO" =~ "t)" ]]; then
			perl sbatch.pl --run_tests --debug && info_message "All tests ok." || error_message "At least one of the tests failed."
			main
		elif [[ "$WHATTODO" =~ "T)" ]]; then
			perl sbatch.pl --run_full_tests --debug && info_message "All tests ok." || error_message "$? tests failed."
			main
		elif [[ "$WHATTODO" =~ "q)" ]]; then
			debug_code "Exiting"
			exit
		else
			list_option_for_job "$WHATTODO"
		fi
	else
		echo_red "You chose to cancel (3)"
		exit 1
	fi
}

function main {
	if [[ -d $PROJECTDIR ]]; then
		debug_code "The folder '$PROJECTDIR' exists"

		list_projects
	else
		debug_code "The folder '$PROJECTDIR' does not exist."
		change_project_dir
		if [[ $? = 0 ]]; then
			main
		else
			exit $?
		fi
	fi
}

# https://stackoverflow.com/questions/2683279/how-to-detect-if-a-script-is-being-sourced
sourced=0
if [ -n "$ZSH_EVAL_CONTEXT" ]; then 
	case $ZSH_EVAL_CONTEXT in *:file) sourced=1;; esac
elif [ -n "$KSH_VERSION" ]; then
	[ "$(cd $(dirname -- $0) && pwd -P)/$(basename -- $0)" != "$(cd $(dirname -- ${.sh.file}) && pwd -P)/$(basename -- ${.sh.file})" ] && sourced=1
elif [ -n "$BASH_VERSION" ]; then
	(return 0 2>/dev/null) && sourced=1 
else # All other shells: examine $0 for known shell binary filenames
	# Detects `sh` and `dash`; add additional shell filenames as needed.
	case ${0##*/} in sh|dash) sourced=1;; esac
fi

if [[ $sourced -eq "0" ]]; then
    if uname -r | grep ppc64 2>/dev/null >/dev/null; then
        whiptail --title "Cannot run on PowerPC" --msgbox "The dostuff.sh cannot be run on a PowerPC-architecture. Please use the login-nodes like 'ssh -X $USER@taurus.hrsk.tu-dresden.de' and run this script again." 10 78
        exit 12
    fi

    if [[ -z $DISPLAY ]]; then
        if (whiptail --title "No X-Server detected" --yes-button "Continue without X-Server" --no-button "No, do not Continue without X-Server" --yesno "Without X-Server, some tools (like Graph-plotting with GUI) do not work, but some others (like plotting to SVG-files) still do. If you want to use the script fully, please use 'ssh -X $USER@taurus.hrsk.tu-dresden.de', then 'cd $(pwd)' and re-start this script" 10 120); then
            echo_green "Continue without X-Server"
        else
            echo_red "Don't continue without X-Server"
            exit 10
        fi
    fi

    if ! hostname | grep tauruslogin 2>/dev/null >/dev/null; then
        export THISHOSTNAME=$(hostname | sed -e 's/\..*//')
        if (whiptail --title "Not on login-node" --yes-button "Continue on $THISHOSTNAME" --no-button "No, do not continue on $THISHOSTNAME" --yesno "It is strongly recommended that this script only get's run at login-nodes and not on compute nodes and you seem to be on a compute-node ($THISHOSTNAME). Are you sure you want to continue?" 10 140); then
            echo_green "Continue on $THISHOSTNAME"
        else
            echo_red "Don't continue on $THISHOSTNAME"
            exit 11
        fi
    fi

    if [[ ! -e .dont_ask_upgrade ]] && [[ "$ASKEDTOUPGRADE" == 0 ]]; then
        if [[ "$UPGRADE" -eq "1" ]]; then
            ASKEDTOUPGRADE=1
            CURRENTHASH=$(git rev-parse HEAD)

            REMOTEURL=$(git config --get remote.origin.url)
            REMOTEHASH=$(git ls-remote $REMOTEURL HEAD | awk '{ print $1}')

            if [ "$CURRENTHASH" = "$REMOTEHASH" ]; then
                debug_code "Software seems up-to-date ($CURRENTHASH)"
            else
                if (whiptail --title "There is a new version of OmniOpt available" --yesno "Do you want to upgrade?" 8 78); then
                    git pull
                    bash evaluate-run.sh --dont_load_modules --no_upgrade $@
                    bash zsh/install.sh
                    exit
                else
                    if (whiptail --title "Ask again?" --yesno "You chose not to upgrade. Ask again at next start?" 8 78); then
                        echo "Asking again next time"
                    else
                        echo "OK, not asking again"
                        touch .dont_ask_upgrade
                    fi
                fi
            fi
        fi
    fi

    modules_to_load=(modenv/scs5 Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4 MongoDB/4.0.3)

    load_percent=0
    let stepsize=100/${#modules_to_load[*]}

    if [[ "$DISPLAYGAUGE" -eq "1" ]]; then
        set +x
        (
            for this_module in ${modules_to_load[*]}; do
                let load_percent=$load_percent+$stepsize
                echo "XXX"
                echo $load_percent
                echo "Loading modules... ($this_module...)"
                echo "XXX"
                ml $this_module 2>/dev/null
            done
        ) | whiptail --title "Loading Modules" --gauge "Loading modules..." 6 70 0

        if [[ "$DEBUG" -eq "1" ]]; then
            set -x
        fi
    else
        if [[ "$LOAD_MODULES" -eq "1" ]]; then
            for this_module in ${modules_to_load[*]}; do
                ml $this_module 2>/dev/null
            done
        fi
    fi


	main
fi
