Green='\033[0;32m'
Color_Off='\033[0m'
Red='\033[0;31m'

function echoerr {
	echo "$@" 1>&2
}

function red_text {
	echo -ne "${Red}$1${Color_Off}"
}

function yellow_text {
	echoerr -e "\e\033[0;33m$1\e[0m"
}

function green_text_no_newline {
	echoerr -ne "\033[0;32m$1\e[0m"
}


function green_text {
	echoerr -e "\033[0;32m$1\e[0m"
}

function green {
	echo -ne "${Green}$1${Color_Off}"
}

function _tput {
	set +e
	CHAR=$1

	if ! command -v tput 2>/dev/null >/dev/null; then
		red_text "tput not installed" >&2
		set +e
		return 0
	fi

	if [[ -z $CHAR ]]; then
		red_text "No character given" >&2
		set +e
		return 0
	fi

	if ! tty 2>/dev/null >/dev/null; then
		echo ""
		set +e
		return 0
	fi

	if [[ "$CHAR" == "bel" ]] && [[ "$OO_MAIN_TESTS" -eq "1" ]]; then
		echo "Not print BEL-character for main-test-suite ($CHAR, $OO_MAIN_TESTS)"
	else
		tput "$CHAR"
	fi
	set +e
}

function green_reset_line {
	_tput cr
	_tput el
	green "$1"
}

function red_reset_line {
	_tput cr
	_tput el
	red_text "$1"
}

function _green_text_bold_underline {
	echoerr -e "\033[1;4;32m$1\033[0m"
}
