#!/usr/bin/env bash

GENERAL_SCRIPT_DIR=$(cd -- "$( dirname -- ${BASH_SOURCE[0]} )" &> /dev/null && pwd)

source .colorfunctions.sh

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

function error_message {
	eval `resize`
	MSG=$1
	echo_red "$MSG"
	export NEWT_COLORS='
window=,red
border=white,red
textbox=white,red
button=black,white
'
    whiptail --title "Error Message" --scrolltext --msgbox "$MSG" $LINES $COLUMNS $(( $LINES - 8 ))
	export NEWT_COLORS=""
}

function spin_up_temporary_webserver {
	CDPATH=$1
	DOWNLOADPATH=$2

	cd $CDPATH

	lower_port=$(cat /proc/sys/net/ipv4/ip_local_port_range | sed -e 's/\s.*//')
	upper_port=$(cat /proc/sys/net/ipv4/ip_local_port_range | sed -e 's/.*\s//')

	free_port=$(comm -23 \
		<(seq "$lower_port" "$upper_port" | sort) \
		<(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
		| shuf | head -n1)

	python3 -u -m http.server $free_port 2>/dev/null >/dev/null

	#{ sleep 3; curl 127.0.0.1:$free_port/ 2>&1 >/dev/null  } &

	this_hostname=$(hostname | sed -e 's/\.taurus\././')

	PORT_FORWARDING_COMMAND="ssh -f -N -L $free_port:$this_hostname:$free_port $USER@$this_hostname; sensible-browser http://localhost:$free_port/$DOWNLOADPATH"

	CUSTOM_TEXT="The webserver has started. Run this command locally to forward the port $free_port to your local system:\n\n$PORT_FORWARDING_COMMAND\n\nWhen you close this window by clicking OK, the server will be shut down."

	if [[ -n "$DISPLAY" ]]; then
		if [[ $(arch) == "x86_64" ]] && [[ -e ./tools/xclip ]] ; then
			echo "$PORT_FORWARDING_COMMAND" | ./tools/xclip -sel clip
			whiptail --title "Webserver" --msgbox "$CUSTOM_TEXT\n\nThis command has already been copied to your clipboard. Paste it locally on your machine to view the file(s)." $LINES $COLUMNS $(( $LINES - 8 ))
		else
			zenity --info --width=800 --height=600 --text "$CUSTOM_TEXT"
		fi
	else
		whiptail --title "Webserver" --msgbox "$CUSTOM_TEXT" $LINES $COLUMNS $(( $LINES - 8 ))
	fi

	kill %1 # %2
}
