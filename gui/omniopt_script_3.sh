#!/bin/bash
#
set -e
set -o pipefail
set -u

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR



export LC_ALL=en_US.UTF-8

ALLPARAMS=$@

curl_commands=curl_commands
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DEBUG=0

function echo_green {
        echo -e "\e[42m\e[97m$1\e[0m"
}
function echo_yellow {
        echo -e "\e[43m\e[97m$1\e[0m"
}
function echo_red {
        echo -e "\e[41m\e[97m$1\e[0m"
}
function echo_headline {
        echo -e "\e[4m\e[96m$1\e[0m"
}

PROJECTNAME=
SBATCH_COMMAND=
CONFIG_FILE=
FORCE_DISABLE_TAURUS_CHECK=0
NO_CLONE=0
BRANCH=master
DONTSTARTJOB=0
AUTOSKIP=0
ADDSBATCHTOSHELLHISTORY=1
PARTITION=
WARNINGHOME=1
USEEXISTINGFOLDER=0
AUTOACCEPTNEWPROJECTNAME=0
AUTOCONTINUEJOB=0
INSTALL_ZSH_AUTOCOMP=1
WORKDIR="."
OOFOLDER=$WORKDIR/omniopt
ERRORANALYZE=1
INTERACTIVE=1
AUTOSTART=0
GITPULL=0

function help () {
        exitcode=$1
        echo_green "OmniOpt installer"
        echo "This script installs OmniOpt via the GUI"
        echo ""
        echo_yellow "Needed options"
        echo "--projectname=Projectname                   The name used for displaying a project name and for the config folder"
        echo "--config_file=[Base64]                      Base64-encoded config file"
        echo "--sbatch_command=[Base64]                   Base64-encoded sbatch-command"
        echo ""
        echo_yellow "Disable automatic checks:"
        echo "--no_taurus_check                           Disable the check if you're on Taurus or not"
        echo "--no_warning_home                           Disables warning for home folder"
        echo "--no_error_analyze                          Disables auto-error-analyzing"
        echo ""
        echo_yellow "Git options:"
        echo "--no_clone                                  Disable cloning (e.g. for debugging or if you're sure you have already cloned)"
        echo "--branch=branchname                         Default: master"
        echo ""
        echo_yellow "Taurus-options:"
        echo "--partition=parname                         Specifies the partition (normally parsed from --sbatch_command), only needed for installer tests"
        echo ""
        echo_yellow "Shell options:"
        echo "--dont_add_to_shell_history                 Don't add sbatch command to shell history"
        echo "--no_install_zsh_autocomp                   Do not install ZSH-autocompletions automatically"
        echo ""
        echo_yellow "Folder options:"
        echo "--omnioptfolder=folder                      Folder to install OmniOpt to (overrides --workdir)"
        echo "--use_existing_folder                       Use existing folder (if exists)"
        echo "--workdir=WORK/DIR                          Path where OmniOpt will be installed to, '.' if empty"
        echo ""
        echo_yellow "Auto-accept options:"
        echo "--auto_accept_projectname                   Auto accept new project name when already exists"
        echo "--auto_continue_job                         Automatically continue old job if it already exists"
        echo "--autoskip                                  Autoskip if OmniOpt-folder already exists"
        echo "--dont_start_job                            Don't start the job automatically and don't ask for it"
        echo ""
        echo_yellow "Scripting"
        echo "--noninteractive                            Disables whiptail, all options in this category require this to be set or else are ignored"
        echo "--autostart                                 Autostart job after cloning?"
        echo "--git_pull                                  Run git pull when the directory already exists and noninteractive is set"
        echo ""
        echo_yellow "Debug-Options"
        echo "--debug                                     Enables set -x"
        exit "$exitcode"
}

slurmlogpath () {
        if command -v scontrol &> /dev/null
        then
                if command -v grep &> /dev/null
                then
                        if command -v sed &> /dev/null
                        then
                                scontrol show job "$1" | grep --color=auto --exclude-dir={.bzr,CVS,.git,.hg,.svn,.idea,.tox} StdOut | sed -e 's/^\s*StdOut=//'
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

SET_OO_FOLDER=0

for i in "$@"; do
        case $i in
                --projectname=*)
                        PROJECTNAME="${i#*=}"
                        ;;

                --omnioptfolder=*)
                        OOFOLDER="${i#*=}"
                        SET_OO_FOLDER=1
                        ;;

                --config_file=*)
                        CONFIG_FILE=$(echo "${i#*=}" | base64 --decode)
                        ;;

                --branch=*)
                        BRANCH="${i#*=}"
                        ;;

                --dont_start_job*)
                        DONTSTARTJOB=1
                        ;;

                --sbatch_command=*)
                        SBATCH_COMMAND=$(echo "${i#*=}" | base64 --decode)
                        ;;

                --no_error_analyze*)
                        ERRORANALYZE=0
                        ;;

                --no_taurus_check*)
                        FORCE_DISABLE_TAURUS_CHECK=1
                        ;;

                --autoskip*)
                        AUTOSKIP=1
                        ;;

                --no_install_zsh_autocomp*)
                        INSTALL_ZSH_AUTOCOMP=0
                        ;;

                --workdir=*)
                        WORKDIR="${i#*=}"
                        ;;

                --auto_continue_job*)
                        AUTOCONTINUEJOB=1
                        ;;

                --auto_accept_projectname*)
                        AUTOACCEPTNEWPROJECTNAME=1
                        ;;

                --use_existing_folder*)
                        USEEXISTINGFOLDER=1
                        ;;

                --no_warning_home*)
                        WARNINGHOME=0
                        ;;

                --no_clone*)
                        NO_CLONE=1
                        ;;

                --dont_add_to_shell_history*)
                        ADDSBATCHTOSHELLHISTORY=0
                        ;;

                --partition*)
                        PARTITION="${i#*=}"
                        ;;

                --autostart*)
                        AUTOSTART=1
                        ;;

                --git_pull*)
                        GITPULL=1
                        ;;



                --noninteractive*)
                        INTERACTIVE=0
                        ;;

                --debug*)
                        DEBUG=1
                        set -x
                        ;;

                --help*)
                        help 0
                        ;;

                *)
                        echo_red "ERROR: Unknown parameter $i"
                        help 1
                        ;;
        esac
done

if [[ "$SET_OO_FOLDER" -eq "0" ]]; then
        OOFOLDER=$WORKDIR/omniopt
fi

if [[ -z "$PARTITION" ]]; then
        PARTITION=$(echo "$SBATCH_COMMAND" | sed -e 's/.*--partition=//' | sed -e 's/\s.*//')
fi

function error_analyze {
        if [[ "$ERRORANALYZE" -eq "1" ]]; then
                if [[ -e tools/error_analyze.sh ]]; then
                        bash tools/error_analyze.sh --project=$1 --project_has_not_run --nowhiptail --whiptail_only_on_error --no_ssh_key_errors
                else
                        echo_red "tools/error_analyze.sh not found.";
                fi
        fi
}

function start_job {
        if [[ $DONTSTARTJOB -eq "0" ]]; then
                if command -v sbatch; then
                        START_JOB=0
                        if [[ "$INTERACTIVE" == 1 ]]; then
                                if (whiptail --title "Run OmniOpt now?" --yesno "Do you want to start the Job $1 now?" 8 78); then
                                        START_JOB=1
                                fi
                        else
                                if [[ "$AUTOSTART" == 1 ]]; then
                                        START_JOB=1
                                fi
                        fi

                        if [[ "$START_JOB" == 1 ]]; then
                                echo_yellow "${2}"
                                SBATCH_OUTPUT=$($2)

                                SBATCH_EXIT_CODE=$?
                                if [[ "$SBATCH_EXIT_CODE" -eq "0" ]] ; then
                                        echo_green "The job seems to have been allocated succesfully"

                                        STARTEDSLURMID=$(echo "$SBATCH_OUTPUT" | sed -e 's/Submitted batch job //')
                                        SLURMLOGPATH=$(slurmlogpath $STARTEDSLURMID)

                                        echo_yellow "You end this job before it reaches any of the predefined limits with:"
                                        echo_yellow "scancel --signal=USR1 --batch $STARTEDSLURMID"
                                        echo ""
                                        echo_yellow "The output will be written to: $SLURMLOGPATH"
                                else
                                        echo_red "Allocating the job failed. Check the stdout for more details"
                                        exit 9
                                fi
                        else
                                echo_yellow 'You chose not to start the job right now. You can always run this job with'
                                echo_yellow "cd $(pwd); ${2}"
                        fi
                else
                        echo_red "sbatch cannot be found. Cannot continue."
                fi
        else
                echo_yellow "Because you added the parameter --dont_start_job, the job won't be started now. But you can always start it manually with:"
                echo_yellow "cd $(pwd); ${2}"
        fi
}

function continue_old_projects {
        OOFOLDER=$1

        PROJECTNAMEOLD=$2
        if [[ -d "$OOFOLDER/projects/$2" ]]; then
                if [[ "$AUTOCONTINUEJOB" -eq "1" ]]; then
                        CONTINUEJOB=1
                else
                        if [[ "$INTERACTIVE" == "1" ]]; then
                                if (whiptail --title "The project $2 already exists" --yesno --yes-button "Continue running old project" --no-button "Start new project" "The folder $OOFOLDER/projects/$2 already exists. Do you want to start a new project from scratch or continue running the old one where it left off?" 8 120); then
                                        CONTINUEJOB=1
                                else
                                        COUNTER=0
                                        while [[ -d "$OOFOLDER/projects/${2}_${COUNTER}" ]]; do
                                                COUNTER=$(($COUNTER+1))
                                        done
                                        PROJECTNAME="${2}_${COUNTER}"

                                        if [[ "$AUTOACCEPTNEWPROJECTNAME" -eq "0" ]]; then
                                                PROJECTNAME=$(whiptail --inputbox "The $PROJECTNAMEOLD already exists. Enter a new name" 8 39 "$2" --title "Project name" 3>&1 1>&2 2>&3)
                                                if [[ ! "$?" -eq "0" ]]; then
                                                        exit 3
                                                fi
                                        fi
                                fi
                        else
                                echo_red "The project $2 already exists in $OOFOLDER. Cannot continue in non-interactive mode"
                                exit 10
                        fi
                fi
        fi

        if [[ "$CONTINUEJOB" -eq "0" ]]; then
                SBATCH_COMMAND=$(echo "$SBATCH_COMMAND" | sed -e "s/\(-J \|--project=\)$PROJECTNAMEOLD/\1$2/g")
        fi
}

function write_sbatch_command_to_file {
        PROJECT=$1
        SBATCH="$2"

        mkdir -p sbatch_commands/$PROJECT

        sh_name=sbatch_commands/$PROJECT/0.sbatch
        prev_sh_name=$sh_name
        i=0

        while [ -e $sh_name ]; do
                prev_sh_name=$sh_name
                sh_name=sbatch_commands/${PROJECT}/${i}.sbatch
                i=$((i+1))
        done

        echo "$SBATCH" > $sh_name

        if [[ -e "$prev_sh_name" ]]; then
                if [[ -e "$sh_name" ]]; then
                        if [[ "$sh_name" == "$prev_sh_name" ]]; then
                                echo "\$sh_name and \$prev_sh_name are the same file ($prev_sh_name)"
                        else
                                if ! diff -q $prev_sh_name $sh_name &>/dev/null; then
                                        echo "Different. Keeping $prev_sh_name and $sh_name"
                                else
                                        echo "Same. Deleting $sh_name"
                                        rm $sh_name
                                fi
                        fi
                else
                        echo "$sh_name does not exist"
                fi
        else
                echo "$prev_sh_name does not exist"
        fi

}

function create_config_file {
        THISPROJECTDIR=projects/${1}/
        mkdir -p "$THISPROJECTDIR" && echo_green 'Project folder creation successful' || echo_red 'Failed to create project folder'

        TMP_CONFIG_INI=$THISPROJECTDIR/config_tmp
        echo "${2}" > "$TMP_CONFIG_INI"
	FILE_ENDING="ini"
        CONFIG_INI=$THISPROJECTDIR/config.${FILE_ENDING}


        if [[ -e $CONFIG_INI ]]; then
                if cmp --silent $TMP_CONFIG_INI $CONFIG_INI; then
                        mv "$TMP_CONFIG_INI" "$CONFIG_INI"
                else
                        ALLOW_SIMPLE_OVERWRITE=1
                        DIMENSIONS_OLD_RUN=$(cat "$CONFIG_INI" | grep "^dimensions =" | sed -e 's/^dimensions.*= //')
                        DIMENSIONS_NEW_RUN=$(cat "$TMP_CONFIG_INI" | grep "^dimensions =" | sed -e 's/^dimensions.*= //')

                        if [[ -d "$THISPROJECTDIR/mongodb" ]]; then
                                if [[ -e $CONFIG_INI ]]; then
                                        if [[ "$DIMENSIONS_OLD_RUN" -ne "$DIMENSIONS_NEW_RUN" ]]; then
                                                ALLOW_SIMPLE_OVERWRITE=0
                                        fi
                                fi
                        fi

                        if [[ "$ALLOW_SIMPLE_OVERWRITE" -eq "1" ]]; then
                                if [[ "$INTERACTIVE" == 1 ]]; then
                                        if (whiptail --title "$CONFIG_INI has changed" --yesno --yes-button "New one" --no-button "Old one" "Do you want to use the old one or the new one?" 8 78); then
                                                mv $TMP_CONFIG_INI $CONFIG_INI
                                        else
                                                mv $CONFIG_INI ${CONFIG_INI}_old
                                                mv $TMP_CONFIG_INI $CONFIG_INI
                                        fi
                                else
                                        echo_red "The files $CONFIG_FILE and $TMP_CONFIG_INI were different. Cannot resolve this in non-interactive mode (1)."
                                fi
                        else
                                if [[ "$INTERACTIVE" == 1 ]]; then
                                        OLDMONGODBFOLDER=$THISPROJECTDIR/mongodb
                                        COUNTER=0
                                        while [[ -d "${OLDMONGODBFOLDER}_${COUNTER}" ]]; do
                                                COUNTER=$(($COUNTER+1))
                                        done
                                        MONGODBFOLDER="${OLDMONGODBFOLDER}_${COUNTER}"

                                        if (whiptail --title "Cannot easily overwrite old config file" --yesno --yes-button "Move mongodb-folder to and use new config" --no-button "Use old config" "The number of parameters have changed from $DIMENSIONS_OLD_RUN to $DIMENSIONS_NEW_RUN and the job has already ran. Using old config file or move old MongoDB-folder to $MONGODBFOLDER and use new config file." 8 150); then
                                                mv "$OLDMONGODBFOLDER" "$MONGODBFOLDER"
                                                mv "$TMP_CONFIG_INI" "$CONFIG_INI"
                                        else
                                                echo_green "Using old MongoDB folder and old config file"
                                        fi
                                else
                                        echo_red "The files $CONFIG_FILE and $TMP_CONFIG_INI were different. Cannot resolve this in non-interactive mode (2)."
                                fi
                        fi
                fi
        else
                mv $TMP_CONFIG_INI $CONFIG_INI
        fi
}

function install_zsh_extensions {
        if [[ "$SHELL" =~ "zsh" ]]; then
                if [[ "$INSTALL_ZSH_AUTOCOMP" -eq "1" ]]; then
                        echo_green "Installing autocompletion for ZSH"
                        if [[ -e zsh/install.sh ]]; then
                                bash zsh/install.sh
                        else
                                echo_red "Cannot find zsh/install.sh"
                        fi
                fi
        fi
}

function add_to_shell_history {
        if [[ "$ADDSBATCHTOSHELLHISTORY" == "1" ]]; then
                if [[ "$SHELL" == "/bin/bash" ]]; then
                        echo "cd $(pwd); ${1}" >> ~/.bash_history
                        echo_yellow "Added the sbatch command to the history"
                elif [[ "$SHELL" == "/bin/zsh" ]]; then
                        echo ": $(date +%s):0;cd $(pwd); ${1}" >> ~/.zsh_history
                fi
        fi
}

echo_headline 'Welcome to the OmniOpt-Installer.'
echo_headline "This will automatically install OmniOpt and start the job with the specified parameters and config for the program ${PROJECTNAME}."

if [[ -z $PROJECTNAME ]]; then
        echo_red "No project name given. Exiting."
        exit 4
fi

if [[ -z $CONFIG_FILE ]]; then
        echo_red "No config file given or not in a valid base64 format. Exiting."
        exit 5
fi

if [[ -z $SBATCH_COMMAND ]]; then
        echo_red "No sbatch command given or not in a valid base64 format. Exiting."
        exit 6
fi

if (hostname | grep taurus 2>/dev/null >/dev/null); then
        echo_green 'OK, you seem to be on Taurus'
else
        if [[ "$FORCE_DISABLE_TAURUS_CHECK" -eq "0" ]]; then
                echo_red 'Not on Taurus. Not running this script on other computers than taurus'
                exit 1
        else
                echo_yellow 'Not on Taurus. Ignoring this because of --no_taurus_check';
        fi
fi

if echo "$PARTITION" | grep 'ml'; then
        if pwd | egrep "^/(lustre|scratch)(/|\$)"; then
                if [[ "$INTERACTIVE" == 1 ]]; then
                        if (whiptail --title "Problems with Scratch/Lustre and ML" --yesno --no-button "Don't continue" --yes-button "Continue"  "Since the Lustre-driver cannot be compiled for ppc64le, Lustre is only available with NFS. This causes a lot of problems, since reading is quite slow and might sometimes just stop working. Are you sure you want to continue? If not, create a new folder somewhere else and run this command again." 12 78); then
                                echo "User selected Yes, exit status was $?."
                        else
                                echo_yellow "Ok, I will not continue with $PARTITION in $(pwd)"
                                echo_yellow "Run this command somewhere else to continue:"
                                echo_yellow "curl https://imageseg.scads.de/omnioptgui/omniopt_script.sh 2>/dev/null | bash -s -- $ALLPARAMS"
                                exit 8
                        fi
                else
                        echo_yellow "Since the Lustre-driver cannot be compiled for ppc64le, Lustre is only available with NFS. This causes a lot of problems, since reading is quite slow and might sometimes just stop working."
                fi
        fi
fi

ASK_FOR_GIT_PULL=0

if [[ "$AUTOSKIP" -eq "0" ]]; then
        if [[ -d $OOFOLDER ]]; then
                if [[ "$USEEXISTINGFOLDER" -eq "1" ]]; then
                        NO_CLONE=1
                        ASK_FOR_GIT_PULL=1
                else
                        if [[ "$INTERACTIVE" == 1 ]]; then
                                if (whiptail --title "$OOFOLDER already exists" --yesno --no-button "Create new folder" --yes-button "Use existing folder"  "The folder '$OOFOLDER' already exists. You can create a new one or use the existing folder." 12 78); then
                                        NO_CLONE=1
                                        ASK_FOR_GIT_PULL=1
                                else
                                        echo_red "The folder '$OOFOLDER' already exists in this directory. This might lead to problems. Delete this folder or move into another folder before running this script again. Otherwise, you can resubmit jobs from this directory. But be warned if errors occur."
                                        COUNTER=0
                                        OOOLD=$OOFOLDER
                                        while [[ -d "${OOFOLDER}_${COUNTER}" ]]; do
                                                COUNTER=$(($COUNTER+1))
                                        done
                                        OOFOLDER="${OOFOLDER}_${COUNTER}"

                                        OOFOLDER=$(whiptail --inputbox "The $OOOLD already exists. Enter a new name" 8 39 "$OOFOLDER" --title "OmniOpt-Folder-name" 3>&1 1>&2 2>&3)
                                        if [[ ! "$?" -eq "0" ]]; then
                                                exit 3
                                        fi
                                fi
                        else
                                echo_red "The folder '$OOFOLDER' already exists in this directory. This might lead to problems. Delete this folder or move into another folder before running this script again. Otherwise, you can resubmit jobs from this directory. But be warned if errors occur."
                                COUNTER=0
                                OOOLD=$OOFOLDER
                                while [[ -d "${OOFOLDER}_${COUNTER}" ]]; do
                                        COUNTER=$(($COUNTER+1))
                                done
                                OOFOLDER="${OOFOLDER}_${COUNTER}"
                                echo_red "It will be cloned into the folder $OOFOLDER"
                        fi
                fi
        fi
fi

if [[ "$ASK_FOR_GIT_PULL" -eq "1" ]]; then
        cd "$OOFOLDER"

        CURRENTHASH=$(git rev-parse HEAD)     

        REMOTEURL=$(git config --get remote.origin.url)     
        if [[ -z $REMOTEHASH ]]; then
                echo_red "REMOTEHASH is undefined"
        else
                REMOTEHASH=$(git ls-remote "$REMOTEURL" HEAD | awk '{ print $1}')     

                if [ "$CURRENTHASH" = "$REMOTEHASH" ]; then      
                        echo_green "Software seems up-to-date ($CURRENTHASH)"      
                else                     
                        if [[ -d $OOFOLDER/.git ]]; then
                                GITPULLNOW=0
                                if [[ "$INTERACTIVE" == 1 ]]; then
                                        if (whiptail --title "There is a new version of OmniOpt available" --yesno --yes-button "Yes, upgrade" --no-button "No, don't upgrade" "Do you want to upgrade? (Strongly recommended)" 8 78); then      
                                                GITPULLNOW=1
                                        fi
                                else
                                        if [[ "$GITPULL" == 1 ]]; then
                                                GITPULLNOW=1
                                        fi
                                fi

                                if [[ "$GITPULLNOW" == 1 ]]; then
                                        git pull         
                                else                 
                                        echo "OK, not upgrading"
                                fi
                        else
                                echo_red "$OOFOLDER/.git is not a folder. Something went wrong"
                        fi
                fi
        fi
        cd -
fi

if [[ "$WARNINGHOME" -eq "1" ]]; then
        if (echo "$(realpath $OOFOLDER)" | grep $HOME 2>/dev/null >/dev/null); then
                if [[ "$INTERACTIVE" == "1" ]]; then
                        if (whiptail --title 'Home-Directory warning' --yesno --yes-button "Yes, I am sure" --no-button "No, don't continue" 'It is not recommended to run this script somewhere in your home directory, since the database might grow quite large. Are you sure about this?' 8 78); then
                                echo_red 'Ok, running in some sub folder of your home. I warned you. If something goes wrong or your home is full then it is your fault.'
                        else
                                echo_green 'OK, wise choice. cd into some other directory, preferrably a workspace, move this script there and run it again.'
                                exit 2
                        fi
                else
                        echo_yellow "It is not recommended to run this script somewhere in your home directory, since the database might grow quite large."
                fi
        fi
fi

if [[ "$NO_CLONE" -eq "0" ]]; then
        echo_green 'Cloning OmniOpt...'
        if [[ -d /projects/p_scads/nnopt/bare/ ]]; then
                total=0
                CLONECOMMAND="git clone --depth=1 file:///projects/p_scads/nnopt/bare/ $OOFOLDER"

                if [[ $DEBUG == 1 ]]; then
                        $CLONECOMMAND
                else
                        set +x
                        if [[ "$INTERACTIVE" == "1" ]]; then
                                $CLONECOMMAND 2>&1 | tr \\r \\n | {
                                        while read -r line ; do
                                                cur=`grep -oP '\d+(?=%)' <<< ${line}`
                                                total=$((total+cur))
                                                percent=$(bc <<< "scale=2;100*($total/100)")
                                                echo "$percent/1" | bc
                                        done
                                } | whiptail --title "Cloning" --gauge "Cloning OmniOpt for optimizing project '$PROJECTNAME'" 8 78 0 && echo_green 'Cloning successful' || echo_red 'Cloning failed'
                        else
                                $CLONECOMMAND
                        fi
                        if [[ "$DEBUG" -eq "1" ]]; then
                                set -x
                        fi
                fi
                if [[ ! "$BRANCH" == "master" ]]; then
                        cd "$OOFOLDER"
                        git pull --all
                        git checkout "$BRANCH"
                        cd -
                fi
        else
                echo_red "The folder /projects/p_scads/nnopt/bare/ does not seem to exist. Cannot continue."
                exit 7
        fi
else
        mkdir -p "$OOFOLDER"
fi

CONTINUEJOB=0

install_zsh_extensions

cd "$OOFOLDER"

mkdir -p $curl_commands

curl_sh_name=$curl_commands/$PROJECTNAME.sbatch
prev_curl_sh_name=$curl_sh_name
i=0

while [ -e $curl_sh_name ]; do
        prev_curl_sh_name=$curl_sh_name
        curl_sh_name=$curl_commands/${PROJECTNAME}_${i}.sbatch
        i=$((i+1))
done

mkdir -p $curl_commands

echo "#!/bin/bash" >> $curl_sh_name
echo "# Project: $PROJECTNAME" >> $curl_sh_name
echo "curl https://imageseg.scads.ai/omnioptgui/omniopt_script.sh 2>/dev/null | bash $*" >> $curl_sh_name

if [[ -e "$prev_curl_sh_name" ]]; then
        if [[ -e "$curl_sh_name" ]]; then
                if [[ "$curl_sh_name" == "$prev_curl_sh_name" ]]; then
			true
                        #echo "\$curl_sh_name and \$prev_curl_sh_name are the same file ($prev_curl_sh_name)"
                else
                        if ! diff -q $prev_curl_sh_name $curl_sh_name &>/dev/null; then
                                echo "Different. Keeping $prev_curl_sh_name and $curl_sh_name"
                        else
                                echo "Same. Deleting $curl_sh_name"
                                rm $curl_sh_name
                        fi
                fi
        else
                echo "$curl_sh_name does not exist"
        fi
else
        echo "$prev_curl_sh_name does not exist"
fi


echo $CONFIG_FILE
if echo "${CONFIG_FILE}" | grep "[DIMENSIONS]" 2>&1 >/dev/null; then
<<'END_COMMENT'
	echo "Got a valid json file as config.json. Parsing this for creating different runs for each seed."
	exit
	# do something with the JSON

	# Extract the seeds from the JSON
	seeds=$(echo "$CONFIG_FILE" | jq -r '.DATA.seed[]')

	if [[ "$seeds" == "" ]]; then
		seeds=("empty")
	fi

	# Loop through the seeds and create a config file for each
	for seed in $seeds; do
		# Create the config file name

		SEED_PROJECTNAME=${PROJECTNAME}_SEED_${seed}
		THISPROJECTDIR=projects/$SEED_PROJECTNAME
		if [[ "$seed" == "empty" ]]; then
			THISPROJECTDIR=projects/${PROJECTNAME}/
		fi
		mkdir -p "$THISPROJECTDIR" && echo_green 'Project folder creation successful' || echo_red 'Failed to create project folder'
		config_file="$THISPROJECTDIR/config.ini"

		# Extract the relevant data for this seed from the JSON
		data=$(echo "$CONFIG_FILE" | jq -r '.DATA')
		dimensions=$(echo "$CONFIG_FILE" | jq -r '.DIMENSIONS')
		debug=$(echo "$CONFIG_FILE" | jq -r '.DEBUG')
		mongodb=$(echo "$CONFIG_FILE" | jq -r '.MONGODB')

		# Write the data section to the config file
		echo "[DATA]" > "$config_file"
		echo "seed = $seed" >> "$config_file"
		echo "$data" | jq -r 'to_entries | .[] | "\(.key) = \(.value)"' | grep -v seed >> "$config_file"

		# Write the debug section to the config file
		echo "[DEBUG]" >> "$config_file"
		echo "$debug" | jq -r 'to_entries | .[] | "\(.key) = \(.value)"' >> "$config_file"

		# Write the mongodb section to the config file
		echo "[MONGODB]" >> "$config_file"
		echo "$mongodb" | jq -r 'to_entries | .[] | "\(.key) = \(.value)"' >> "$config_file"
		
		# Generate INI file content

		# Read JSON file and parse it using jq
		dimensions=$(echo "$CONFIG_FILE" | jq -r '.DIMENSIONS')

		# Generate INI file content
		ini="[DIMENSIONS]\n"
		loop=0
		for i in $(echo "${dimensions}" | jq -r '.[] | @base64'); do
			dim_name=$(echo ${i} | base64 --decode | jq -r '.name')
			for key in $(echo ${i} | base64 --decode | jq -r 'keys[]'); do
				if [[ "$key" == "name" ]]; then
					ini+="dim_${loop}_name = ${dim_name}\n"
				else
					value=$(echo ${i} | base64 --decode | jq -r ".$key")
					if [[ "$key" == "options" ]]; then
						value=$(echo ${value[@]} | jq -r 'join(",")')
					fi
					key=$(echo $key | tr '_' ' ' | sed 's/^./\L&/')
					if [[ "$key" == *"range generator"* ]]; then
						key=$(echo $key | sed 's/range generator/range_generator/')
						ini+="$(echo $key)_${loop} = ${value}\n"
					elif [[ "$key" == *"options"* ]]; then
						ini+="$(echo $key)_${loop} = ${value}\n"
					else
						ini+="$(echo $key)_dim_${loop} = ${value}\n"
					fi
				fi
			done
			loop=$((loop+1))
		done
		ini+="\ndimensions = ${loop}\n"

		# Write INI file
		echo -e $ini >> "$config_file"

		SEED_SBATCH_COMMAND=$(echo $SBATCH_COMMAND | sed -e "s/$PROJECTNAME/${SEED_PROJECTNAME}/g")

		continue_old_projects $OOFOLDER $SEED_PROJECTNAME
		error_analyze $SEED_PROJECTNAME
		write_sbatch_command_to_file $SEED_PROJECTNAME "$SEED_SBATCH_COMMAND"
		start_job $SEED_PROJECTNAME "$SEED_SBATCH_COMMAND"
		add_to_shell_history $SEED_SBATCH_COMMAND
	done
END_COMMENT

export CONFIG_FILE

yes yes | cpan -i JSON

#	" NEW "
perl -lne 'use strict;
use warnings;
use JSON;
use autodie;
use Data::Dumper;

my $config_file = $ENV{CONFIG_FILE};

# Extract the seeds from the JSON
my $json = JSON->new->decode($config_file);
my $seeds = $json->{DATA}->{seed};

if (!$seeds) {
        $seeds = ["empty"];
}

# Loop through the seeds and create a config file for each
foreach my $seed (@$seeds) {
        # Create the config file name
        my $seed_projectname = $json->{"DATA"}->{projectname} . "_SEED_" . $seed;
        my $thisprojectdir = "projects/" . $seed_projectname;

        if ($seed eq "empty") {
                $thisprojectdir = "projects/" . $json->{"DATA"}->{projectname} . "/";
        }

        if (!-d $thisprojectdir) {
                system("mkdir -p $thisprojectdir");
        }
        my $config_file_name = "$thisprojectdir/config.ini";

        open my $fh, ">", $config_file_name;

        # Extract the relevant data for this seed from the JSON
        my $data = $json->{DATA};
        my $dimensions = $json->{DIMENSIONS};
        my $debug = $json->{DEBUG};
        my $mongodb = $json->{MONGODB};

        my $c = "";

        # Write the data section to the config file
        $c .= "[DATA]\n";
        $c .= "seed = $seed\n";
        foreach my $key (keys %$data) {
                next if ($key eq "seed");
                my $value = $data->{$key};
                $c .= "$key = $value\n";
        }

        # Write the debug section to the config file
        $c .= "\n[DEBUG]\n";
        foreach my $key (keys %$debug) {
                my $value = $debug->{$key};
                $c .= "$key = $value\n";
        }

        # Write the mongodb section to the config file
        $c .= "\n[MONGODB]\n";
        foreach my $key (keys %$mongodb) {
                my $value = $mongodb->{$key};
                $c .= "$key = $value\n";
        }

        # Generate INI file content
        $c .= "\n[DIMENSIONS]\n";
        my $loop = 0;
        foreach my $dim (@$dimensions) {
                my $dim_name = $dim->{name};
                $c .= "dim_${loop}_name = $dim_name\n";
                foreach my $key (keys %$dim) {
                        next if ($key eq "name");
                        my $value = $dim->{$key};
                        if ($key eq "options") {
                                $value = join(",", @$value);
                        }
                        $key =~ s/_/ /g;
                        $key = lcfirst($key);
                        if ($key =~ /range generator/) {
                                $key =~ s/range generator/range_generator/;
                                $c .= "${key}_${loop} = $value\n";
                        } elsif ($key eq "options") {
                                $value = join(",", split /\s*,\s*/, $value);
                        }

                        # Modify the key to match the INI format
                        $key =~ tr/_/ /;
                        $key = lcfirst($key);
                        if ($key =~ /range generator/) {
                                $key =~ s/range generator/range_generator/;
                                $c .= "$key" . "_$loop = $value\n";
                        } elsif ($key eq "options") {
                                $c .= "$key" . "_$loop = $value\n";
                        } else {
                                $c .= "$key" . "_dim_$loop = $value\n";
                        }
                }
                $c .= "\n";
                $loop++;
        }
        $c .= "dimensions = $loop\n";

        print $fh $c;
        close($fh);
}'
#	" NEW "

	SEED_SBATCH_COMMAND=$(echo $SBATCH_COMMAND | sed -e "s/$PROJECTNAME/${SEED_PROJECTNAME}/g")

	continue_old_projects $OOFOLDER $SEED_PROJECTNAME
	error_analyze $SEED_PROJECTNAME
	write_sbatch_command_to_file $SEED_PROJECTNAME "$SEED_SBATCH_COMMAND"
	start_job $SEED_PROJECTNAME "$SEED_SBATCH_COMMAND"
	add_to_shell_history $SEED_SBATCH_COMMAND
else
	echo "Looks like ini"
	exit
	SEEDS=$(echo "$CONFIG_FILE" | grep "seed" | sed -e 's/.*:\s*//' | sed -e 's/.*=\s*//' | sed -e 's#,$##')
	if [ -z "$SEEDS" ]; then
		continue_old_projects $OOFOLDER $PROJECTNAME
		create_config_file $PROJECTNAME "$CONFIG_FILE"
		error_analyze $PROJECTNAME
		write_sbatch_command_to_file $PROJECTNAME "$SBATCH_COMMAND"
		start_job $PROJECTNAME "$SBATCH_COMMAND"
		add_to_shell_history $SBATCH_COMMAND
	else
		echo_green "Found seeds: $SEEDS, creating a project for each one."
		for SEED in $SEEDS; do
			SEED_PROJECTNAME=${PROJECTNAME}_SEED_${SEED}
			SEED_CONFIG_FILE=$(echo "$CONFIG_FILE" | sed -e "s/^seed\s*=.*/seed = $SEED/")

			SEED_SBATCH_COMMAND=$(echo $SBATCH_COMMAND | sed -e "s/$PROJECTNAME/${PROJECTNAME}_SEED_${SEED}/g")
			echo $SEED_SBATCH_COMMAND

			continue_old_projects $OOFOLDER $SEED_PROJECTNAME
			create_config_file $SEED_PROJECTNAME "$SEED_CONFIG_FILE"
			error_analyze $SEED_PROJECTNAME
			write_sbatch_command_to_file $SEED_PROJECTNAME "$SEED_SBATCH_COMMAND"
			start_job $SEED_PROJECTNAME "$SEED_SBATCH_COMMAND"
			add_to_shell_history $SEED_SBATCH_COMMAND
		done
	fi
fi

rm jq
