#!/bin/bash

PROJECT=$1
PROJECTDIR=$2

INFO=$(perl tools/get_info.pl --project=$PROJECT --projectdir=$PROJECTDIR)

eval `resize`
whiptail --title "Info for $PROJECT" --msgbox "$INFO" $LINES $COLUMNS
