#!/bin/bash

source tools/general.sh

CSVFILE=$1
OUTPUTPATH=$2
DONTSTARTFIREFOX=$3

if [ ! -f "$1" ]; then
    error_message "$1 not found"
    exit 1
fi

mkdir -p $(dirname $OUTPUTPATH)

if [[ -e tools/parallel_venv/bin/activate ]]; then
    source tools/parallel_venv/bin/activate
else
    python3 -m venv tools/parallel_venv
    source tools/parallel_venv/bin/activate
    pip3 install pandas 
    pip3 install hiplot
fi

csv_titles_on_or_off () {
        colname=$1
        droparray=(
                "time"
                "endtime"
                "hostname"
                "logfile_path"
                "loss"
                "starttime"
                "Beginning"
                "time.1"
                "ETA"
                "Set"
                "params"
                "GPU"
                "capability"
                "cc"
                "computeCapability"
                "coreClock"
                "coreCount"
                "device"
                "deviceMemoryBandwidth"
                "deviceMemorySize"
                "devices"
                "matrix"
                "pciBusID"
                "replica"
                "task"
        )
        if [[ "${droparray[*]}" =~ "${colname}" ]]; then
                echo "OFF"
        else
                echo "ON"
        fi
}

readarray -t titles < <(cat $CSVFILE | head -n1 | sed -e 's/;/\n/g' | egrep -v '^\s*$')

if [[ ${#titles[@]} -eq 1 ]]; then
    error_message "Only one column ($titles) found. Need at least two."
    exit 1
fi

options=()
for i in "${titles[@]}"; do
        options+=("$i" "" "$(csv_titles_on_or_off $i)")
done

eval `resize`
chosen_option=$(whiptail --title "Choose CSV-titles to plot" --checklist "Check/uncheck the titles you want to plot with the spacebar and press enter" $LINES $COLUMNS $(( $LINES - 8 )) "${options[@]}" 3>&1 1>&2 2>&3)
if [[ "$?" == 0 ]]; then
    export enabled_titles=$(echo $chosen_option | sed -e 's/" "/;/g' | sed -e 's/"//g')

    python3 tools/create_parallel_plot.py $CSVFILE $OUTPUTPATH
    if [[ "$?" -eq "0" ]]; then
        echo "Creating $OUTPUTPATH seemed to be succesful"
        if [[ -z $DONTSTARTFIREFOX ]]; then
            eval `resize`

            POSSIBLE_OPTIONS=(
                    "webserver" "Start Webserver"
                    "path" "Show the path of the file so you can copy it to your local machine"
            )

            if [[ $DISPLAY ]]; then
                POSSIBLE_OPTIONS+=("firefox" "Start firefox on taurus (might be slow)")
            fi

            CHOSEN=$(whiptail --title "What to do?" --menu "Choose an option" $LINES $COLUMNS $(( $LINES - 8 )) "${POSSIBLE_OPTIONS[@]}" 3>&1 1>&2 2>&3)

            if [[ "$?" == 0 ]]; then
                if [ $CHOSEN == "webserver" ]; then
                    spin_up_temporary_webserver $(dirname $OUTPUTPATH) "plot.html"
                elif [ $CHOSEN == "firefox" ]; then
                    firefox $OUTPUTPATH
                else
                    CUSTOM_TEXT="The file is available under\n$(pwd)/$OUTPUTPATH"
                    if [[ $DISPLAY ]]; then
                        zenity --info --width=800 --height=600 --text "$CUSTOM_TEXT"
                    else
                        whiptail --title "File path" --msgbox "$CUSTOM_TEXT" $LINES $COLUMNS $(( $LINES - 8 ))
                    fi
                fi
            else
                echo_green "OK, cancelling parallel plot"
                exit 0
            fi
        else
            echo "Path: $OUTPUTPATH"
        fi
    else
        echo "Creating $OUTPUTPATH seems to have failed"
    fi
else
    echo_green "Ok, cancelling parallel plot"
    exit 0
fi
