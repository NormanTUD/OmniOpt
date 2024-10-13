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
export SCIENTIFICNOTATION=4
export DEBUG=0
export LOAD_MODULES=1
export UPGRADE=1
export SEPERATOR=";"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib64/

source tools/general.sh

mkdir -p debuglogs
bash_logname=debuglogs/evaluate_run.log
let i=1
while [[ -e $bash_logname ]]; do
    bash_logname=debuglogs/evaluate_run_${i}.log
    let i++
done
exec 1> >(tee -ia $bash_logname)
exec 2> >(tee -ia $bash_logname >& 2)

export BASH_XTRACEFD="$FD"

if [[ -e ".default_settings" ]]; then
    source ".default_settings"
fi

if [[ -e "$HOME/.oo_default_settings" ]]; then
    echo "Loading $HOME/.oo_default_settings"
    source "$HOME/.oo_default_settings"
fi

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

module () {
    eval `$LMOD_CMD sh "$@"`
}

ml () {
    eval $($LMOD_DIR/ml_cmd "$@")
}

#set -e
#set -o pipefail

function calltracer () {
    echo 'Last file/last line:'
    caller
}
trap 'calltracer' ERR

numberre='^[0-9]+$'

PROJECTDIR=projects



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

		eval `resize`

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
		) | whiptail --title "$TITLE" --gauge "$TEXT"  $LINES $COLUMNS $(( $LINES - 8 ))
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
	eval `resize`
	MSG=$1
	echo_green "$MSG"
	whiptail --title "Info Message" --msgbox "$MSG" $LINES $COLUMNS $(( $LINES - 8 ))
}


function inputbox {
	TITLE=$1
	MSG=$2
	DEFAULT=$3

	eval `resize`
	RESULT=$(whiptail --inputbox "$MSG" $LINES $COLUMNS "$DEFAULT" --title "$TITLE" 3>&1 1>&2 2>&3)
	exitstatus=$?
	if [[ $exitstatus == 0 ]]; then
		echo "$RESULT"
	else
		echo_red "You chose to cancel (1)"
		exit 1
	fi
}

function change_project_dir {
	if [[ -d "$PROJECTDIR" ]]; then
		PROJECTDIR=$(inputbox "Projectdir" "The project directory is currently '$PROJECTDIR'. Enter a new Projectdir (relative or absolute) or just press enter to continue using this one" "$PROJECTDIR")
	else
		PROJECTDIR=$(inputbox "Projectdir" "The directory '$PROJECTDIR' could not be found. Enter a new Projectdir (relative or absolute)" "$PROJECTDIR")
	fi
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
        SVGDIR=$PROJECTDIR/$PROJECT/gpu_plot/
        mkdir -p $SVGDIR
        export SVGFILE=$SVGDIR/${PROJECT}.svg
        CNT=0
        while [[ -e $SVGFILE ]]; do
                CNT=$(($CNT+1))
                export SVGFILE=$SVGDIR/${PROJECT}_${CNT}.svg
        done

        gaugecommand "Plotting GPU-Usage to $SVGFILE" "Please wait, this takes some time" "perl tools/plot_gpu.pl --project=$PROJECT --projectdir=$PROJECTDIR --logdate=$WHATTODO --filename=$SVGFILE"
        if [[ -e "$SVGFILE" ]]; then
                info_message "Wrote to file $SVGFILE"
        else
                error_message "Failed to write $SVGFILE"
        fi
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
            args+=("p)" "2d scatter-plots" "psvg)" "2d scatterplots to svg file" "P)" "2d scatterplots with max value" "Psvg)" "2d scatterplots with max value to svg" "pa)" "Parallel plot" "v)" "Plot video"  "r)" "Repair Database" "c)" "get csv to stdout" "C)" "get csv to file" "wct)" "Get wallclock-time of all jobs (only useful for jobs that ran once)" "i)" "Get general info for this job")
        fi

        args+=("co)" "Show run config")
    fi

    if [[ -d "$THISSINGLELOGS" ]]; then
        args+=("s)" "Auto-analyze jobs from singlelogs (may take very long)")
    fi

    if [[ -d "$THISPROJECTDIR" ]]; then
        args+=("e)" "Check this project for errors")
        args+=("d)" "Create debug-zip")
    fi

    if job_is_running "$PROJECT"; then
        args+=("n)" "Number of jobs with status OK (job must be running)" "N)" "Number of jobs with status OK every 10 seconds (job must be running)" "f)" "Number of jobs with status FAIL (job must be running)" "l)" "Live plot results (only vaguely)")
    fi
    
    if [[ -d "$PROJECTDIR/$PROJECT/logs/" ]]; then
        if [[ $(ls $PROJECTDIR/$PROJECT/logs/*/nvidia*/gpu_usage.csv | wc -l 2>/dev/null) -ne "0" ]]; then
            args+=("g)" "Plot GPU usage" "G)" "Plot GPU usage to svg file")
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
                clear
                perl tools/showconfig.pl $PROJECT $THISCONFIGINI | less -R -c -S
                read -p "Press enter to return to evaluate-run.sh"
            else
                eval `resize`
                whiptail --title "Error" --msgbox "'$THISCONFIGINI' cannot be found" $LINES $COLUMNS $(( $LINES - 8 ))
            fi
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "wct)" ]]; then
            if [[ -d "$PROJECTDIR/$PROJECT/singlelogs" ]]; then
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

                eval `resize`
                whiptail --title "Wallclock-Time" --msgbox "$WCT_RESULT" $LINES $COLUMNS $(( $LINES - 8 ))
            else
                eval `resize`
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot determine wallclock time from empty job." $LINES $COLUMNS $(( $LINES - 8 ))
            fi

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

            list_installed_modules=list_installed_modules.log
            let i=0
            while [[ -e $list_installed_modules ]]; do
                list_installed_modules="list_installed_modules_${i}.log"
                let i++
            done

            error_analyze_file=error_analyze.log
            let i=0
            while [[ -e $error_analyze_file ]]; do
                error_analyze_file="error_analyze_${i}.log"
                let i++
            done

            bash tools/error_analyze.sh --project=$PROJECT --projectdir=$PROJECTDIR --nowhiptail 2>/dev/null > $error_analyze_file

            pip3 list > $list_installed_modules

            zip -r $DEBUGFILE $list_installed_modules $error_analyze_file *.out debuglogs/* $PROJECTDIR/$PROJECT/* -x $PROJECTDIR/$PROJECT/mongodb/\*

            rm $error_analyze_file

            if [[ -e $DEBUGFILE ]]; then
                if (whiptail --title "Make this file available over a webserver?" --yesno "Do you want to make this file available over a webserver?" 8 78); then
                    spin_up_temporary_webserver . $DEBUGFILE
                else
                    if [[ "$USER" == "s3811141" ]]; then
                        info_message_large "Wrote $DEBUGFILE. Send this file to <norman.koch@tu-dresden.de> for debugging-help.\nscp_taurus $(pwd)/$DEBUGFILE .\n"
                    else
                        info_message_large "Wrote $DEBUGFILE. Send this file to <norman.koch@tu-dresden.de> for debugging-help.\nscp $USER@taurus.hrsk.tu-dresden.de://$(pwd)/$DEBUGFILE .\n"
                    fi
                fi
            else
                error_message "Could not write $DEBUFFILE"
            fi

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "c)" ]]; then
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                    bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            gaugecommand "CSV-Export" "Loading CSV-Export" "perl script/runningdbtocsv.pl --project=$PROJECT --projectdir=$PROJECTDIR --seperator='$SEPERATOR'"

            read -rsn1 -p"Press any key to continue";echo
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "i)" ]]; then
            if [[ -d "$PROJECTDIR/$PROJECT/singlelogs" ]]; then
                bash tools/show_info_whiptail.sh $PROJECT $PROJECTDIR
            else
                eval `resize`
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot determine general info from empty job." $LINES $COLUMNS $(( $LINES - 8 ))
            fi

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "pa)" ]]; then
            CSV_DIR=${PROJECTDIR}/${PROJECT}/csv/
            mkdir -p $CSV_DIR
            csv_filename=${CSV_DIR}/${PROJECT}.csv
            create_csv=1

            if [[ -e $csv_filename ]]; then
                timestamp=$(date -r $csv_filename)
                existing_files=("$(basename $csv_filename)" "use this one ($timestamp)")
                let i=1
                while [[ -e ${CSV_DIR}/${PROJECT}_${i}.csv ]]; do
                    timestamp=$(date -r ${CSV_DIR}/${PROJECT}_${i}.csv)
                    existing_files+=("${PROJECT}_${i}.csv" "use this one ($timestamp)")
                    let i++
                done

                option=$(whiptail --title "File already exists. What do you want to do?" --menu "File already exists. What do you want to do?" 25 78 16 \
                    "${existing_files[@]}" \
                    "new" "Create a new one" 3>&1 1>&2 2>&3)
                exitstatus=$?
                if [ $exitstatus = 0 ]; then
                    if [[ $option == "new" ]]; then
                        let i=1
                        while [[ -e $csv_filename ]]; do
                            csv_filename=${CSV_DIR}/${PROJECT}_${i}.csv
                            let i++
                        done
                    else
                        create_csv=0
                    fi
                else
                    list_option_for_job $PROJECT
                    return
                fi
            fi

            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            if [[ $create_csv == "1" ]]; then
                gaugecommand "CSV-Export" "Loading CSV-Export, printing to $csv_filename" "perl script/runningdbtocsv.pl --project=$PROJECT --projectdir=$PROJECTDIR --filename=$csv_filename"
            else
                echo "Not re-creating CSV. Using $csv_filename"
            fi

            if [[ -e "$csv_filename" ]]; then
                if [[ -s "$csv_filename" ]]; then
                    parallelplot_file=$PROJECTDIR/$PROJECT/parallel-plot/plot.html
                    bash tools/parallel_plot.sh $csv_filename $parallelplot_file
                else
                    eval `resize`
                    whiptail --title "File printed" --msgbox "The file was printed to $csv_filename but is empty" $LINES $COLUMNS $(( $LINES - 8 ))
                fi
            else
                eval `resize`
                whiptail --title "File not printed" --msgbox "The file was NOT printed to $csv_filename, this might be a Bug in OmniOpt. Contact <norman.koch@tu-dresden.de>." $LINES $COLUMNS $(( $LINES - 8 ))
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

                gaugecommand "CSV-Export" "Loading CSV-Export, printing to $csv_filename" "perl script/runningdbtocsv.pl --project=$PROJECT --projectdir=$PROJECTDIR --filename=$csv_filename --seperator='$SEPERATOR'"
                if [[ -e "$csv_filename" ]]; then
                    if [[ -s "$csv_filename" ]]; then
                        eval `resize`
                        whiptail --title "File printed" --msgbox "The file was printed to $csv_filename" $LINES $COLUMNS $(( $LINES - 8 ))
                    else
                        eval `resize`
                        whiptail --title "File printed" --msgbox "The file was printed to $csv_filename but is empty" $LINES $COLUMNS $(( $LINES - 8 ))
                    fi
                else
                    whiptail --title "File not printed" --msgbox "The file was NOT printed to $csv_filename, this might be a Bug in OmniOpt. Contact <norman.koch@tu-dresden.de>." 30 78
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
                eval `resize`
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." $LINES $COLUMNS $(( $LINES - 8 ))
            fi
            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "psvg)" ]]; then
            SVGDIR=$PROJECTDIR/$PROJECT/2d-scatterplots/
            mkdir -p $SVGDIR
            export PLOTPATH=$SVGDIR/${PROJECT}.svg
            export CNT=0
            while [[ -e $PLOTPATH ]]; do
                export CNT=$(($CNT+1))
                export PLOTPATH=$SVGDIR/${PROJECT}_${CNT}.svg
            done

            echo_green "Plot"
            if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                    bash tools/repair_database.sh $PROJECTDIR/$PROJECT
            fi

            if [[ -d "$PROJECTDIR/$PROJECT/singlelogs" ]]; then
                gaugecommand "Graph-Creation" "Please wait, this takes some time..." "perl tools/plot.pl --project=$PROJECT --projectdir=${PROJECTDIR}/"
                if [[ -e "$PLOTPATH" ]]; then
                    info_message "Wrote to file $PLOTPATH"
                else
                    error_message "Failed to write $PLOTPATH"
                fi
             else
                eval `resize`
                 whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." $LINES $COLUMNS $(( $LINES - 8 ))
            fi

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
                eval `resize`
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." $LINES $COLUMNS $(( $LINES - 8 ))
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
            if [[ -d "$PROJECTDIR/$PROJECT/singlelogs" ]]; then
                maxvalue=$(inputbox "Max value for plot" "Enter a max value for plotting $PROJECT (float)" "0.1")
                echo_green "Got maxvalue = ${maxvalue}"

                SVGDIR=$PROJECTDIR/$PROJECT/2d-scatterplots/
                mkdir -p $SVGDIR

                export PLOTPATH=$SVGDIR/${PROJECT}_limit_${maxvalue}.svg
                CNT=0
                while [[ -e $PLOTPATH ]]; do
                    CNT=$(($CNT+1))
                    export PLOTPATH=$SVGDIR/${PROJECT}_${CNT}_limit_${maxvalue}_%s.svg
                done

                echo_green "Plot"
                if [[ -e $PROJECTDIR/$PROJECT/mongodb/mongod.lock ]]; then
                    bash tools/repair_database.sh $PROJECTDIR/$PROJECT
                fi

                    gaugecommand "Graph-Creation" "Please wait, this takes some time..." "perl tools/plot.pl --project=$PROJECT --projectdir=${PROJECTDIR}/ --maxvalue=$maxvalue"


                if [[ -e "$PLOTPATH" ]]; then
                    info_message "Wrote to file $PLOTPATH"
                else
                    error_message "Failed to write $PLOTPATH"
                fi
            else
                eval `resize`
                whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$PROJECT/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." $LINES $COLUMNS $(( $LINES - 8 ))
            fi

            export PLOTPATH=

            list_option_for_job $PROJECT
        elif [[ "$WHATTODO" =~ "l)" ]]; then
            bash tools/live_plot.sh $PROJECTDIR $PROJECT
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
            if [[ "$DEBUG" == "1" ]]; then
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
        set +x
		chosenjobs=$(eval "whiptail --title 'Which jobs to kill with USR1?' --checklist 'Which jobs to choose USR1?' $WIDTHHEIGHT $TJOBS" 3>&1 1>&2 2>&3)

        if [[ $DEBUG -eq 1 ]]; then
            set -x
        fi
		if [[ -z $chosenjobs ]]; then
			green_text "No jobs chosen to kill"
		else
			export NEWT_COLORS='
window=,red
border=white,red
textbox=white,red
button=black,white
'
            eval `resize`
            if (whiptail --title "Really kill multiple jobs ($chosenjobs)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill multiple jobs ($chosenjobs)?" $LINES $COLUMNS $(( $LINES - 8 ))); then
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
            eval `resize`
            if (whiptail --title "Really kill multiple jobs ($chosenjobs)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill multiple jobs ($chosenjobs)?" $LINES $COLUMNS $(( $LINES - 8 ))); then
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
            eval `resize`
            #whiptail --title "Tail for multiple jobs with screen" --msgbox "To exit, press <CTRL> <a>, then <\\>" $LINES $COLUMNS $(( $LINES - 8 )) 3>&1 1>&2 2>&3
			multiple_slurm_tails $chosenjobs
		fi
	fi
}

function single_job_tasks {
	chosenjob=$1
	gobacktolist_running_slurm_jobs="$2"

    if [[ -z $chosenjob ]]; then
        echo "No job chosen!!!"
	exit 1
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
            eval `resize`
            if (whiptail --title "Really kill >$jobname< ($chosenjob)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill >$jobname< ($chosenjob)?" $LINES $COLUMNS $(( $LINES - 8 ))); then
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
            eval `resize`
            if (whiptail --title "Really kill with USR1 >$jobname< ($chosenjob)?" --yesno --defaultno --fullbuttons "Are you sure you want to kill >$jobname< ($chosenjob) with USR1?" $LINES $COLUMNS $(( $LINES - 8 ))); then
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

function plot_multiple_projects {
    PROJECTS=$(ls $PROJECTDIR/*/config.ini 2>/dev/null | sed -e "s#$PROJECTDIR/##" | sed -e 's#/config.ini##')

    if [[ -z "$PROJECTS" ]]; then
	error_message "No projects found (plot_multiple_projects)"
	return
    fi

    PROJECTS_STRING=""

    for p in $PROJECTS; do
        PROJECTS_STRING+=" $p $p ON "
    done

    RESULT=$(whiptail --title "Plot multiple projects" --checklist "Choose the projects that you want CSV files of" 20 78 4 \
        $PROJECTS_STRING \
            3>&1 1>&2 2>&3)
    exitstatus=$?

    if [ $exitstatus = 0 ]; then
        echo "User selected Ok and entered " $RESULT
    fi

    for chosen_project in $(echo $RESULT | sed -e 's/"//g'); do
        if [[ -e $PROJECTDIR/$chosen_project/mongodb/mongod.lock ]]; then
            bash tools/repair_database.sh $PROJECTDIR/$chosen_project
        fi
    done

    for chosen_project in $(echo $RESULT | sed -e 's/"//g'); do
        if [[ -d "$PROJECTDIR/$chosen_project/singlelogs" ]]; then
            perl tools/plot.pl --project=$chosen_project --projectdir=${PROJECTDIR}/ &
        else
            eval `resize`
            whiptail --title "No runs" --msgbox "The folder '$PROJECTDIR/$chosen_project/singlelogs' does not exist. This means the job has not yet ran. Cannot create graph from empty job." $LINES $COLUMNS $(( $LINES - 8 ))
        fi
    done

    echo "You will get back to the main screen once all the plot windows are closed."

    wait
}

function restart_old_jobs {
    PROJECTS_STRING=$(ls -1 sbatch_commands | sed -e 's/\(.*\)/\1 \1/' | tr '\n' ' ')

    eval `resize`
    PROJECT_TO_REDO=$(whiptail --title 'Menu example' --menu 'Choose an option' $LINES $COLUMNS $(( $LINES - 8 )) $PROJECTS_STRING '<- Back' 'Back to main menu' 3>&1 1>&2 2>&3)


    if [[ "$PROJECT_TO_REDO" =~ "<- Back" ]]; then
        return
    else
        NUMBER_OF_SBATCHES=$(ls sbatch_commands/$PROJECT_TO_REDO/*.sbatch | wc -l)
        if [[ $NUMBER_OF_SBATCHES -eq 1 ]]; then
            SOURCEME=$(ls sbatch_commands/$PROJECT_TO_REDO/*.sbatch)
            eval `resize`
            whiptail --title "Sbatch started" --msgbox "$(source $SOURCEME 2>&1)" $LINES $COLUMNS
        else
            eval `resize`
            SOURCEME=$(whiptail --title 'Menu example' --menu 'Choose an option' $LINES $COLUMNS $(( $LINES - 8 )) $(perl -e 'use Time::localtime; use File::stat; while ($file = <sbatch_commands/*/*.sbatch>) { $date = ctime( stat($file)->ctime ); $date =~ s#[^\w\d]#_#g; $date =~ s#_+#_#g; print qq#$file $date #; }') 3>&1 1>&2 2>&3)
            whiptail --title "Sbatch started" --msgbox "$(source $SOURCEME 2>&1)" $LINES $COLUMNS
        fi
    fi
}

function csv_multiple_projects {
    PROJECTS=$(ls $PROJECTDIR/*/config.ini 2>/dev/null | sed -e "s#$PROJECTDIR/##" | sed -e 's#/config.ini##')

    if [[ -z "$PROJECTS" ]]; then
	error_message "No projects found (csv_multiple_projects)"
	return
    fi

    PROJECTS_STRING=""

    for p in $PROJECTS; do
        PROJECTS_STRING+=" $p $p ON "
    done

    RESULT=$(whiptail --title "Plot multiple projects" --checklist "Choose the projects that you want to plot" 20 78 4 \
        $PROJECTS_STRING \
            3>&1 1>&2 2>&3)
    exitstatus=$?

    if [ $exitstatus = 0 ]; then
        echo "User selected Ok and entered " $RESULT
    fi

    for chosen_project in $(echo $RESULT | sed -e 's/"//g'); do
        CSV_DIR=${PROJECTDIR}/${chosen_project}/csv/
        mkdir -p $CSV_DIR
        csv_filename=${CSV_DIR}/${chosen_project}.csv
        let i=1

        create_new_csv_file=1
        if [[ -e $csv_filename ]]; then
            eval `resize`
            if (whiptail --title "New CSV-File?" --yesno "The file $csv_filename already exists. Do you want to create a new one?" $LINES $COLUMNS $(( $LINES - 8 ))); then
                create_new_csv_file=1
            else
                create_new_csv_file=0
            fi
        fi

        if [[ "$create_new_csv_file" -eq 1 ]]; then
            while [[ -e $csv_filename ]]; do
                csv_filename=${CSV_DIR}/${chosen_project}_${i}.csv
                let i++
            done

            if [[ -e $PROJECTDIR/$chosen_project/mongodb/mongod.lock ]]; then
                bash tools/repair_database.sh $PROJECTDIR/$chosen_project
            fi

            csv_filename=$(inputbox "Filename for the CSV file" "Path of the file for the CSV of $chosen_project" "$csv_filename")
            if [[ $? = 0 ]]; then
                echo "Filename: $csv_filename"

                gaugecommand "CSV-Export" "Loading CSV-Export, printing to $csv_filename" "perl script/runningdbtocsv.pl --project=$chosen_project --projectdir=$PROJECTDIR --filename=$csv_filename --seperator='$SEPERATOR'"
                if [[ -e "$csv_filename" ]]; then
                    if [[ -s "$csv_filename" ]]; then
                        echo "The file was printed to $csv_filename"
                        #eval `resize`
                        #whiptail --title "File printed" --msgbox "The file was printed to $csv_filename" $LINES $COLUMNS $(( $LINES - 8 ))
                    else
                        eval `resize`
                        whiptail --title "File printed" --msgbox "The file was printed to $csv_filename but is empty" $LINES $COLUMNS $(( $LINES - 8 ))
                    fi
                else
                    whiptail --title "File not printed" --msgbox "The file was NOT printed to $csv_filename, this might be a Bug in OmniOpt. Contact <norman.koch@tu-dresden.de>." 30 78
                fi
            else
                echo_red "You cancelled the CSV creation"
            fi
        fi
    done
}

function list_running_slurm_jobs {
    FAILED=0

    NEWT_COLORS=""

    if ! command -v squeue &> /dev/null; then
        red_text "squeue not found. Cannot execute list_running_slurm_jobs without it"
        FAILED=1
    fi


    if ! command -v whiptail &> /dev/null; then
        red_text "whiptail not found. Cannot execute list_running_slurm_jobs without it"
        FAILED=1
    fi

    if ! command -v scontrol &> /dev/null; then
        red_text "scontrol not found. Cannot execute list_running_slurm_jobs without it"
        FAILED=1
    fi

    JOBS=$(get_squeue_from_format_string "%A '%j (%t, %M)'")
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

    if [[ $FAILED == 0 ]]; then
        eval `resize`
        WIDTHHEIGHT="$LINES $COLUMNS $(( $LINES - 8 ))"

        TMPFILE=/tmp/$(uuidgen)

        RUNCOMMAND="whiptail --title 'Slurm Manager' --menu 'Welcome to the Slurm-Manager' $WIDTHHEIGHT $JOBS $SCANCELSTRING $TAILSTRING 'r)' 'Reload Slurm-Manager' 'm)' 'Go back to main menu' 'q)' 'Quit Slurm-Manager'"
        echo $RUNCOMMAND > $TMPFILE

        set +x

        chosenjob=$(source $TMPFILE 3>&2 2>&1 1>&3)

        if [[ $DEBUG -eq 1 ]]; then
            set -x
        fi
        
        echo "============> chosenjob:"
        echo $chosenjob

        rm $TMPFILE

        exit_code=$?

        if [[ $exit_code = 0 ]]; then
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
                error_message "Unknown option >>$chosenjob<< "
                main
            fi
        else
            error_message "The command\n$RUNCOMMAND\nreturn exit-code $exit_code. Returning to main menu."
            main
        fi
    else
        red_text  "Missing requirements, cannot run list_running_slurm_jobs"
    fi
}

function show_number_of_results {
    PROJECTNAME=$1
    THISPROJECTDIR=$PROJECTDIR/$PROJECTNAME
    set +x
    if [[ -d $THISPROJECTDIR ]]; then
        number_of_resultfiles_as_whole=$(ls $THISPROJECTDIR/singlelogs/*.stdout | wc -l)
        if [[ $number_of_resultfiles_as_whole == 0 ]]; then
            eval `resize`
            whiptail --title "ERROR" --msgbox "It seems like no job has ever run in $THISPROJECTDIR." $LINES $COLUMNS $(( $LINES - 8 ))
        else
            number_of_resultfiles_with_result=$(grep -irm1 'RESULT: ' $THISPROJECTDIR/singlelogs/*.stdout | wc -l)
            number_of_valid_results=$(grep -irm1 'RESULT: [0-9]' $THISPROJECTDIR/singlelogs/*.stdout | wc -l)
            number_of_invalid_results=$(echo "$number_of_resultfiles_with_result-$number_of_valid_results" | bc)
            number_of_oom=$(grep -m1 "Resource exhausted: OOM" $THISPROJECTDIR/singlelogs/*.stderr | wc -l)
            number_of_broken_pipes=$(grep -irm1 "Broken pipe" $THISPROJECTDIR/singlelogs/*.stderr | wc -l)
            number_of_permission_denied=$(grep -irm1 "Permission denied" $THISPROJECTDIR/singlelogs/*.stderr | wc -l)

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

            permission_error_msg=''
            if [[ "$number_of_permission_denied" -ne "0" ]]; then
                permission_error_msg="-> $number_of_permission_denied permission errors were detected.\n"
            fi


            whiptail --title "Result-Analysis" --msgbox "There are $number_of_resultfiles_as_whole .stdout files.\n-> $number_of_resultfiles_with_result of them have a 'RESULT'-String.\n-> Of those $number_of_valid_results seem to be valid results\n\nThis leaves us with $number_of_invalid_results invalid evaluations.\n${oom_msg}${broken_pipe_msg}${permission_error_msg}-> Number of unfinished jobs: $number_of_unfinished_jobs\nThis means, $number_of_explained_crashes invalid evaluations are explained and $number_of_unexplained_crashes remain unexplained.\n\nWorst result: $worst_result\nBest result: $best_result." 20 90


        fi
    else
        eval `resize`
        whiptail --title "ERROR" --msgbox "The directory '$THISPROJECTDIR' does not seem to exist." $LINES $COLUMNS $(( $LINES - 8 ))
    fi

    if [[ $DEBUG -eq 1 ]]; then
        set -x
    fi

    list_option_for_job $PROJECTNAME
}

function change_variables {
    eval `resize`
    MENU_CHOICE=$(whiptail --title "Change variables" --menu "Choose an option" $LINES $COLUMNS $(( $LINES - 8 )) "NONZERODIGITS" "Max. number of non-zero decimal places in the graph plot (currently $NONZERODIGITS)" "SHOWFAILEDJOBSINPLOT" "Show failed runs in plots with really high values (currently $SHOWFAILEDJOBSINPLOT)" "BUBBLESIZEINPX" "Size of bubbles in the plot graph (currently $BUBBLESIZEINPX)" "SVGEXPORTSIZE" "Size of the exported SVG-Graphs of Plot and GPU-Plot (currently $SVGEXPORTSIZE)" "SHOWALLGPUS" "Show all GPUs in GPU-Plot (currently $SHOWALLGPUS)" "HIDEMAXVALUESINPLOT" "Hide max values in Plot (currently $HIDEMAXVALUESINPLOT)" "DISPLAYGAUGE" "Display gauge when possible (currently $DISPLAYGAUGE)" "PROJECTDIR" "The path where projects are (currently $PROJECTDIR)" "DEBUG" "Debug evaluate-run.sh (currently $DEBUG)" "SCIENTIFICNOTATION" "Use scientific notation and with how many decimal places (currently $SCIENTIFICNOTATION)" "SEPERATOR" "Seperator for CSV files (currently $SEPERATOR)" "s)" "Save current settings as default for this OmniOpt-installation" "S)" "Save current settings as default for all OmniOpt-installations on your account" "m)" "Main menu" 3>&1 1>&2 2>&3)
    exitstatus=$?
    if [[ $exitstatus == 0 ]]; then
        if [[ "$MENU_CHOICE" =~ "m)" ]]; then
            main
        elif [[ "$MENU_CHOICE" =~ "SHOWALLGPUS" ]]; then
            DEFAULTNO=''
            if [[ "$SHOWALLGPUS" == "0" ]]; then
                DEFAULTNO=" --defaultno "
            fi

            eval `resize`
            if (whiptail --title "Show all GPUs in plot?" --yesno "Show all GPUs instead of the ones from the log file only in GPU-Plot?" $DEFAULTNO $LINES $COLUMNS $(( $LINES - 8 ))); then
                export SHOWALLGPUS=1
            else
                export SHOWALLGPUS=0
            fi

            change_variables
        elif [[ "$MENU_CHOICE" =~ "SEPERATOR" ]]; then
            chosenvar=$(whiptail --inputbox "Seperator for CSV Files?" 8 39 "$SEPERATOR" --title "SEPERATOR" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "BUBBLESIZEINPX" ]]; then
            chosenvar=$(whiptail --inputbox "Size of the plot bubbles in px?" 8 39 "$BUBBLESIZEINPX" --title "BUBBLESIZEINPX" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "NONZERODIGITS" ]]; then
            chosenvar=$(whiptail --inputbox "Max. number of non-zero decimal places in the graph plot?" 8 80 "$NONZERODIGITS" --title "NONZERODIGITS" 3>&1 1>&2 2>&3)
            while [[ ! $chosenvar =~ ^[0-9]+$ ]]; do
                chosenvar=$(whiptail --inputbox "The value you entered was not an integer. Max. number of non-zero decimal places in the graph plot?" 8 80 "$NONZERODIGITS" --title "NONZERODIGITS" 3>&1 1>&2 2>&3)
            done
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "SVGEXPORTSIZE" ]]; then
            chosenvar=$(whiptail --inputbox "Width of the SVG-Exports:" 8 50 "$SVGEXPORTSIZE" --title "SVGEXPORTSIZE" 3>&1 1>&2 2>&3)
            while [[ ! $chosenvar =~ ^[0-9]+$ ]]; do
                chosenvar=$(whiptail --inputbox "The value you entered was not an integer. Width of the SVG-Exports:" 8 50 "$SVGEXPORTSIZE" --title "SVGEXPORTSIZE" 3>&1 1>&2 2>&3)
            done
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "SHOWFAILEDJOBSINPLOT" ]]; then
            DEFAULTNO=''
            if [[ "$SHOWFAILEDJOBSINPLOT" == "0" ]]; then
                DEFAULTNO=" --defaultno "
            fi

            eval `resize`
            if (whiptail --title "Show invalid jobs in plot?" --yesno "Do you want to show invalid jobs with really high values in plot?" $DEFAULTNO $LINES $COLUMNS $(( $LINES - 8 ))); then
                export SHOWFAILEDJOBSINPLOT=1
            else
                export SHOWFAILEDJOBSINPLOT=0
            fi

            change_variables
        elif [[ "$MENU_CHOICE" =~ "HIDEMAXVALUESINPLOT" ]]; then
            DEFAULTNO=''
            if [[ "$HIDEMAXVALUESINPLOT" == "0" ]]; then
                DEFAULTNO=" --defaultno "
            fi

            eval `resize`
            if (whiptail --title "Hide max-values-string in plot?" --yesno "Do you want to hide the 'max value' string in plots?" $DEFAULTNO $LINES $COLUMNS $(( $LINES - 8 ))); then
                export HIDEMAXVALUESINPLOT=1
            else
                export HIDEMAXVALUESINPLOT=0
            fi

            change_variables
        elif [[ "$MENU_CHOICE" =~ "SCIENTIFICNOTATION" ]]; then
            chosenvar=$(whiptail --inputbox "Use scientific notation? 0 = no, 1 = yes with 1 decimal point, 2 = yes with 2 decimal points, ..." 8 80 "$SCIENTIFICNOTATION" --title "SCIENTIFICNOTATION" 3>&1 1>&2 2>&3)
            eval "export $MENU_CHOICE=$chosenvar"
            change_variables
        elif [[ "$MENU_CHOICE" =~ "DISPLAYGAUGE" ]]; then
            DEFAULTNO=''
            if [[ "$DISPLAYGAUGE" == "0" ]]; then
                DEFAULTNO=" --defaultno "
            fi

            eval `resize`
            if (whiptail --title "Enable gauge?" --yesno "Do you want to enable gauge whereever possible?" $DEFAULTNO $LINES $COLUMNS $(( $LINES - 8 ))); then
                export DISPLAYGAUGE=1
            else
                export DISPLAYGAUGE=0
            fi

            change_variables
        elif [[ "$MENU_CHOICE" =~ "PROJECTDIR" ]]; then
            chosenvar=$(whiptail --inputbox "Path of Projects" 8 80 "$PROJECTDIR" --title "PROJECTDIR" 3>&1 1>&2 2>&3)
            while [[ ! -d $chosenvar ]]; do
                chosenvar=$(whiptail --inputbox "The path you chose does not exist. Choose another project path:" 8 80 "$PROJECTDIR" --title "PROJECTDIR" 3>&1 1>&2 2>&3)
            done
            eval "export $MENU_CHOICE=$chosenvar"

            change_variables
        elif [[ "$MENU_CHOICE" =~ "DEBUG" ]]; then
            DEFAULTNO=''
            if [[ "$DEBUG" == "0" ]]; then
                DEFAULTNO=" --defaultno "
            fi

            eval `resize`
            if (whiptail --title "Enable debug?" --yesno "Do you want to enable debug?" $DEFAULTNO $LINES $COLUMNS $(( $LINES - 8 ))); then
                export DEBUG=1
                set -x
            else
                export DEBUG=0
                set +x
            fi

            change_variables
        elif [[ "$MENU_CHOICE" =~ "s)" ]]; then
            if [[ -e ".default_settings" ]]; then
                rm .default_settings
            fi

            echo "export NONZERODIGITS=$NONZERODIGITS" >> .default_settings
            echo "export SHOWFAILEDJOBSINPLOT=$SHOWFAILEDJOBSINPLOT" >> .default_settings
            echo "export BUBBLESIZEINPX=$BUBBLESIZEINPX" >> .default_settings
            echo "export SVGEXPORTSIZE=$SVGEXPORTSIZE" >> .default_settings
            echo "export SHOWALLGPUS=$SHOWALLGPUS" >> .default_settings
            echo "export HIDEMAXVALUESINPLOT=$HIDEMAXVALUESINPLOT" >> .default_settings
            echo "export DISPLAYGAUGE=$DISPLAYGAUGE" >> .default_settings
            echo "export PROJECTDIR=$PROJECTDIR" >> .default_settings
            echo "export DEBUG=$DEBUG" >> .default_settings
            echo "export SCIENTIFICNOTATION=$SCIENTIFICNOTATION" >> .default_settings
            echo "export SEPERATOR='$SEPERATOR'" >> .default_settings
        elif [[ "$MENU_CHOICE" =~ "S)" ]]; then
            if [[ -e "$HOME/.oo_default_settings" ]]; then
                rm ~/.oo_default_settings
            fi

            echo "export NONZERODIGITS=$NONZERODIGITS" >> ~/.oo_default_settings
            echo "export SHOWFAILEDJOBSINPLOT=$SHOWFAILEDJOBSINPLOT" >> ~/.oo_default_settings
            echo "export BUBBLESIZEINPX=$BUBBLESIZEINPX" >> ~/.oo_default_settings
            echo "export SVGEXPORTSIZE=$SVGEXPORTSIZE" >> ~/.oo_default_settings
            echo "export SHOWALLGPUS=$SHOWALLGPUS" >> ~/.oo_default_settings
            echo "export HIDEMAXVALUESINPLOT=$HIDEMAXVALUESINPLOT" >> ~/.oo_default_settings
            echo "export DISPLAYGAUGE=$DISPLAYGAUGE" >> ~/.oo_default_settings
            echo "export PROJECTDIR=$PROJECTDIR" >> ~/.oo_default_settings
            echo "export DEBUG=$DEBUG" >> ~/.oo_default_settings
            echo "export SCIENTIFICNOTATION=$SCIENTIFICNOTATION" >> ~/.oo_default_settings
            echo "export SEPERATOR='$SEPERATOR'" >> ~/.oo_default_settings
        else
            eval `resize`
            whiptail --title "Invalid option" --msgbox "The option '$MENU_CHOICE' is not valid. Returning to the main menu" $LINES $COLUMNS $(( $LINES - 8 )) 3>&1 1>&2 2>&3
            change_variables
        fi
    else
        echo_red "You chose to cancel (3)"
        exit 1
    fi
}

function list_projects {
	AVAILABLE_PROJECTS=$(ls $PROJECTDIR/*/config.ini 2>/dev/null | sed -e "s#${PROJECTDIR}/##" | sed -e 's#/config.ini##' | perl -le 'while (<>) { chomp; chomp; print qq#$_ $_# }')

	if [[ -z "$AVAILABLE_PROJECTS" ]]; then
		echo "No projects found (list_projects)"
	fi

	eval `resize`

	# REMOVED BECAUSE IT IS TOO BUGGY AND PROBABLY NOONE USES IT:
	# "s)" "List running SLURM jobs"
	WHATTODO=$(whiptail --title "Available projects under ${PROJECTDIR}" --menu "Chose any of the available projects or options:" $LINES $COLUMNS $(( $LINES - 8 )) $AVAILABLE_PROJECTS "S)" "Start http-server here" "p)" "Plot multiple projects" "R)" "Restart old jobs" "C)" "CSV from multiple projects" "c)" "Change the project dir" "v)" "Show/Change Variables" "t)" "Run OmniOpt-Tests (fast)" "T)" "Run OmniOpt-Tests (complete)" "q)" "quit" 3>&1 1>&2 2>&3)

	exitstatus=$?
	if [[ $exitstatus == 0 ]]; then
		if [[ "$WHATTODO" =~ "c)" ]]; then
			change_project_dir
			main
		elif [[ "$WHATTODO" =~ "p)" ]]; then
			plot_multiple_projects
			main
		elif [[ "$WHATTODO" =~ "C)" ]]; then
			csv_multiple_projects
			main
		elif [[ "$WHATTODO" =~ "R)" ]]; then
			restart_old_jobs
			main
		elif [[ "$WHATTODO" =~ "S)" ]]; then
			spin_up_temporary_webserver . ""
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
		echo_red "You chose to cancel (4)"
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
		main
	fi
}



#list_running_slurm_jobs
#exit




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

if [[ $sourced == "0" ]]; then
    if uname -r | grep ppc64 2>/dev/null >/dev/null; then
        whiptail --title "Cannot run on PowerPC" --msgbox "The evaluate-run.sh cannot be run on a PowerPC-architecture. Please use the login-nodes like 'ssh -X $USER@taurus.hrsk.tu-dresden.de' and run this script again." 10 78
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

    if [[ ! -e .dont_ask_upgrade ]] && [[ "$ASKEDTOUPGRADE" == 0 ]]; then
        if [[ "$UPGRADE" == "1" ]]; then
            ASKEDTOUPGRADE=1
            CURRENTHASH=$(git rev-parse HEAD)

            REMOTEURL=$(git config --get remote.origin.url)
            REMOTEHASH=$(git ls-remote $REMOTEURL HEAD | awk '{ print $1}')

            if [ "$CURRENTHASH" = "$REMOTEHASH" ]; then
                debug_code "Software seems up-to-date ($CURRENTHASH)"
            else
                eval `resize`
                if (whiptail --title "There is a new version of OmniOpt available" --yesno "Do you want to upgrade?" $LINES $COLUMNS $(( $LINES - 8 ))); then
                    git pull
                    bash evaluate-run.sh --dont_load_modules --no_upgrade "$@"
                    bash zsh/install.sh
                    exit
                else
                    eval `resize`
                    if (whiptail --title "Ask again?" --yesno "You chose not to upgrade. Ask again at next start?" $LINES $COLUMNS $(( $LINES - 8 ))); then
                        echo "Asking again next time"
                    else
                        echo "OK, not asking again"
                        touch .dont_ask_upgrade
                    fi
                fi
            fi
        fi
    fi

    modules_to_load=(release/23.04 GCC/11.3.0 OpenMPI/4.1.4 Hyperopt/0.2.7)

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
			if ! ml is-loaded $this_module; then
				ml $this_module 2>/dev/null
			fi
		done
        ) | whiptail --title "Loading Modules" --gauge "Loading modules..." 6 70 0

	if [[ "$DEBUG" -eq "1" ]]; then
		set -x
	fi
	else
		if [[ "$LOAD_MODULES" -eq "1" ]]; then
			for this_module in ${modules_to_load[*]}; do
				if ! ml is-loaded $this_module; then
					ml $this_module 2>/dev/null
				fi
			done
		fi
	fi

	main
fi
