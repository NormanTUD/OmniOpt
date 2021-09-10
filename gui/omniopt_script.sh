#!/bin/bash

export LC_ALL=en_US.UTF-8

set -e

ALLPARAMS=$@

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
OOFOLDER=omniopt
PARTITION=
WARNINGHOME=1
USEEXISTINGFOLDER=0
AUTOACCEPTNEWPROJECTNAME=0
AUTOCONTINUEJOB=0
INSTALL_ZSH_AUTOCOMP=1

function help () {
	exitcode=$1
	echo_green "Valid Options:"
	echo "--projectname=Projectname	The name used for displaying a project name and for the config folder"
	echo "--config_file=[Base64]		Base64-decoded config file"
	echo "--sbatch_command=[Base64]	Base64 decoded sbatch-command"
	echo "--no_taurus_check		Disable the check if you're on Taurus or not"
	echo "--branch=branchname		Default: master"
	echo "--no_clone			Disable cloning (e.g. for debugging or if you're sure you have already cloned)"
	echo "--dont_start_job		Don't start the job automatically and don't ask for it"
	echo "--autoskip			Autoskip if OmniOpt-folder already exists"
	echo "--partition=parname		Specifies the partition (normally parsed from --sbatch_command), only needed for installer tests"
	echo "--dont_add_to_shell_history	Don't add sbatch command to shell history"
	echo "--omnioptfolder=folder		Folder to install OmniOpt to"
	echo "--no_warning_home		Disables warning for home folder"
	echo "--use_existing_folder		Use existing folder (if exists)"
	echo "--auto_accept_projectname	Auto accept new project name when already exists"
	echo "--auto_continue_job		Automatically continue old job if it already exists"
	echo "--no_install_zsh_autocomp         Do not install ZSH-autocompletions automatically"
	echo "--debug				Enables set -x"
	exit $exitcode
}

slurmlogpath () {
        if command -v scontrol &> /dev/null
        then
                if command -v grep &> /dev/null
                then
                        if command -v sed &> /dev/null
                        then
                                scontrol show job $1 | grep --color=auto --exclude-dir={.bzr,CVS,.git,.hg,.svn,.idea,.tox} StdOut | sed -e 's/^\s*StdOut=//'
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

for i in "$@"; do
    case $i in
        --projectname=*)
		PROJECTNAME="${i#*=}"
		;;
        
        --omnioptfolder=*)
		OOFOLDER="${i#*=}"
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
            
	--no_taurus_check*)
		FORCE_DISABLE_TAURUS_CHECK=1
		;;
            
	--autoskip*)
		AUTOSKIP=1
		;;

	--no_install_zsh_autocomp*)
		INSTALL_ZSH_AUTOCOMP=0
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
		exit
		;;
    esac
done

if [[ -z "$PARTITION" ]]; then
	PARTITION=$(echo "$SBATCH_COMMAND" | sed -e 's/.*--partition=//' | sed -e 's/\s.*//')
fi

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
		echo_red 'Not on Taurus. Ignoring this because of --no_taurus_check';
	fi
fi

if echo $PARTITION | grep 'ml'; then
	if pwd | egrep "^/(lustre|scratch)(/|\$)"; then
		if (whiptail --title "Problems with Scratch/Lustre and ML" --yesno --no-button "Don't continue" --yes-button "Continue"  "Since the Lustre-driver cannot be compiled for ppc64le, Lustre is only available with NFS. This causes a lot of problems, since reading is quite slow and might sometimes just stop working. Are you sure you want to continue? If not, create a new folder somewhere else and run this command again." 12 78); then
			echo "User selected Yes, exit status was $?."
		else
			echo_yellow "Ok, I will not continue with $PARTITION in $(pwd)"
			echo_yellow "Run this command somewhere else to continue:"
			echo_yellow "curl https://imageseg.scads.de/omnioptgui/omniopt_script.sh 2>/dev/null | bash -s -- $ALLPARAMS"
			exit 8
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
		fi
	fi
fi

if [[ "$ASK_FOR_GIT_PULL" -eq "1" ]]; then
	cd $OOFOLDER

	CURRENTHASH=$(git rev-parse HEAD)     

	REMOTEURL=$(git config --get remote.origin.url)     
	REMOTEHASH=$(git ls-remote $REMOTEURL HEAD | awk '{ print $1}')     

	if [ "$CURRENTHASH" = "$REMOTEHASH" ]; then      
		echo_green "Software seems up-to-date ($CURRENTHASH)"      
	else                     
		if (whiptail --title "There is a new version of OmniOpt available" --yesno --yes-button "Yes, upgrade" --no-button "No, don't upgrade" "Do you want to upgrade? (Strongly recommended)" 8 78); then      
			git pull         
		else                 
			echo "OK, not upgrading"
		fi
	fi
	cd -
fi

if [[ "$WARNINGHOME" -eq "1" ]]; then
	if (pwd | grep $HOME 2>/dev/null >/dev/null); then
		if (whiptail --title 'Home-Directory warning' --yesno --yes-button "Yes, I am sure" --no-button "No, don't continue" 'It is not recommended to run this script somewhere in your home directory, since the database might grow quite large. Are you sure about this?' 8 78); then
			echo_red 'Ok, running in some sub folder of your home. I warned you. If something goes wrong or your home is full then it is your fault.'
		else
			echo_green 'OK, wise choice. cd into some other directory, preferrably a workspace, move this script there and run it again.'
			exit 2
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
			$CLONECOMMAND 2>&1 | tr \\r \\n | {
				while read -r line ; do
					cur=`grep -oP '\d+(?=%)' <<< ${line}`
					total=$((total+cur))
					percent=$(bc <<< "scale=2;100*($total/100)")
					echo "$percent/1" | bc
				done
			} | whiptail --title "Cloning" --gauge "Cloning OmniOpt for optimizing project '$PROJECTNAME'" 8 78 0 && echo_green 'Cloning successful' || echo_red 'Cloning failed'
			if [[ "$DEBUG" -eq "1" ]]; then
				set -x
			fi
		fi
		if [[ ! "$BRANCH" == "master" ]]; then
			cd $OOFOLDER
			git pull --all
			git checkout $BRANCH
			cd -
		fi
	else
		echo_red "The folder /projects/p_scads/nnopt/bare/ does not seem to exist. Cannot continue."
		exit 7
	fi
else
	mkdir -p $OOFOLDER
fi

CONTINUEJOB=0
PROJECTNAMEOLD=$PROJECTNAME

if [[ -d "$OOFOLDER/projects/$PROJECTNAME" ]]; then
	if [[ "$AUTOCONTINUEJOB" -eq "1" ]]; then
		CONTINUEJOB=1
	else
		if (whiptail --title "The project $PROJECTNAME already exists" --yesno --yes-button "Continue running old project" --no-button "Start new project" "The folder $OOFOLDER/projects/$PROJECTNAME already exists. Do you want to start a new project from scratch or continue running the old one where it left off?" 8 120); then
			CONTINUEJOB=1
		else
			COUNTER=0
			while [[ -d "$OOFOLDER/projects/${PROJECTNAME}_${COUNTER}" ]]; do
				COUNTER=$(($COUNTER+1))
			done
			PROJECTNAME="${PROJECTNAME}_${COUNTER}"

			if [[ "$AUTOACCEPTNEWPROJECTNAME" -eq "0" ]]; then
				PROJECTNAME=$(whiptail --inputbox "The $PROJECTNAMEOLD already exists. Enter a new name" 8 39 "$PROJECTNAME" --title "Project name" 3>&1 1>&2 2>&3)
				if [[ ! "$?" -eq "0" ]]; then
					exit 3
				fi
			fi
		fi
	fi
fi

if [[ "$CONTINUEJOB" -eq "0" ]]; then
	SBATCH_COMMAND=$(echo $SBATCH_COMMAND | sed -e "s/\(-J \|--project=\)$PROJECTNAMEOLD/\1$PROJECTNAME/g")
fi

cd $OOFOLDER
THISPROJECTDIR=projects/${PROJECTNAME}/
mkdir -p $THISPROJECTDIR && echo_green 'Project folder creation successful' || echo_red 'Failed to create project folder'

CONFIG_INI=$THISPROJECTDIR/config.ini
TMP_CONFIG_INI=${CONFIG_INI}_tmp
echo "${CONFIG_FILE}" > $TMP_CONFIG_INI
if [[ -e $CONFIG_INI ]]; then
	if cmp --silent $TMP_CONFIG_INI $CONFIG_INI; then
		mv $TMP_CONFIG_INI $CONFIG_INI
	else
		ALLOW_SIMPLE_OVERWRITE=1
		DIMENSIONS_OLD_RUN=$(cat $CONFIG_INI | grep "^dimensions =" | sed -e 's/^dimensions.*= //')
		DIMENSIONS_NEW_RUN=$(cat $TMP_CONFIG_INI | grep "^dimensions =" | sed -e 's/^dimensions.*= //')

		if [[ -d "$THISPROJECTDIR/mongodb" ]]; then
			if [[ -e $CONFIG_INI ]]; then
				if [[ "$DIMENSIONS_OLD_RUN" -ne "$DIMENSIONS_NEW_RUN" ]]; then
					ALLOW_SIMPLE_OVERWRITE=0
				fi
			fi
		fi

		if [[ "$ALLOW_SIMPLE_OVERWRITE" -eq "1" ]]; then
			if (whiptail --title "$CONFIG_INI has changed" --yesno --yes-button "New one" --no-button "Old one" "Do you want to use the old one or the new one?" 8 78); then
				mv $TMP_CONFIG_INI $CONFIG_INI
			else
				mv $CONFIG_INI ${CONFIG_INI}_old
				mv $TMP_CONFIG_INI $CONFIG_INI
			fi
		else
			OLDMONGODBFOLDER=$THISPROJECTDIR/mongodb
			COUNTER=0
			while [[ -d "${OLDMONGODBFOLDER}_${COUNTER}" ]]; do
				COUNTER=$(($COUNTER+1))
			done
			MONGODBFOLDER="${OLDMONGODBFOLDER}_${COUNTER}"

			if (whiptail --title "Cannot easily overwrite old config file" --yesno --yes-button "Move mongodb-folder to and use new config" --no-button "Use old config" "The number of parameters have changed from $DIMENSIONS_OLD_RUN to $DIMENSIONS_NEW_RUN and the job has already ran. Using old config file or move old MongoDB-folder to $MONGODBFOLDER and use new config file." 8 150); then
				mv $OLDMONGODBFOLDER $MONGODBFOLDER
				mv $TMP_CONFIG_INI $CONFIG_INI
			else
				echo_green "Using old MongoDB folder and old config file"
			fi
		fi
	fi
else
	mv $TMP_CONFIG_INI $CONFIG_INI
fi

if [[ "$SHELL" =~ "zsh" ]]; then
	if [[ "$INSTALL_ZSH_AUTOCOMP" -eq "1" ]]; then
		echo_green "Installing autocompletion for ZSH"
		bash zsh/install.sh
	fi
fi

if [[ $DONTSTARTJOB -eq "0" ]]; then
	if (whiptail --title "Run OmniOpt now?" --yesno "Do you want to start the Job ${PROJECTNAME} now?" 8 78); then
		echo_yellow "${SBATCH_COMMAND}"
		SBATCH_OUTPUT=$($SBATCH_COMMAND)

		SBATCH_EXIT_CODE=$?
		if [[ "$SBATCH_EXIT_CODE" -eq "0" ]] ; then
			echo_green "The job seems to have been allocated succesfully"

			STARTEDSLURMID=$(echo $SBATCH_OUTPUT | sed -e 's/Submitted batch job //')
			SLURMLOGPATH=$(slurmlogpath $STARTEDSLURMID)

			echo_yellow "You end this job before it reaches any of the predefined limits with:"
			echo_yellow "scancel --signal=USR1 --batch $STARTEDSLURMID"
			echo ""
			echo_yellow "You can find it's output in the file $SLURMLOGPATH"
		else
			echo_red "Allocating the job failed. Check the stdout for more details"
			exit 1
		fi
	else
		echo_yellow 'You chose not to start the job right now. You can always run this job with'
		echo_yellow "cd $(pwd); ${SBATCH_COMMAND}"
	fi
else
	echo_yellow "Because you added the parameter --dont_start_job, the job won't be started now. But you can always start it manually with:"
	echo_yellow "cd $(pwd); ${SBATCH_COMMAND}"
fi

if [[ "$ADDSBATCHTOSHELLHISTORY" == "1" ]]; then
	if [[ "$SHELL" == "/bin/bash" ]]; then
		echo "cd $(pwd); ${SBATCH_COMMAND}" >> ~/.bash_history
		echo_yellow "Added the sbatch command to the history"
	elif [[ "$SHELL" == "/bin/zsh" ]]; then
		echo ": $(date +%s):0;cd $(pwd); ${SBATCH_COMMAND}" >> ~/.zsh_history
	fi
fi
