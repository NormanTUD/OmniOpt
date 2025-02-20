#!/bin/bash

DBNAME=$1
UNSUPERVISED=$2

LMOD_DIR=/usr/share/lmod/lmod/libexec/
LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod

module () {
	eval `$LMOD_CMD sh "$@"`
}

ml () {
	eval $($LMOD_DIR/ml_cmd "$@")
}

ml MongoDB/4.0.3 2>&1 | grep -v loaded

function echo_red {
	echo -e "\e[31m$1\e[0m"
}

function echo_green {
	echo -e "\e[32m$1\e[0m"
}

function get_random_open_port_localhost {
	lower_port=$(cat /proc/sys/net/ipv4/ip_local_port_range | cut -f1)
	upper_port=$(cat /proc/sys/net/ipv4/ip_local_port_range | cut -f2)
	comm -23 <(seq $lower_port $upper_port | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1
}

function main {
	mongodbpath=$1

	if [[ -z "$mongodbpath" ]]; then
		echo_red "Folder parameter not given"
	else
		if [[ -d $mongodbpath ]]; then
			echo_green "The folder $mongodbpath exists"
			if [[ ! -e "$mongodbpath/WiredTiger" ]]; then
				mongodbpath+="/mongodb"
			fi
			if [[ -e "$mongodbpath/WiredTiger" ]]; then
				i=0

				while [[ -d "$mongodbpath.backup.$i" ]]; do
					i=$((i+1))
				done
				backupfolder="$mongodbpath.backup.$i"

				if [[ -d "$mongodbpath/../ipfiles/" ]]; then
					echo_green "Found $mongodbpath/../ipfiles/ folder one folder upwards of the MongoDB-Folder. Checking if there's already an instance running."
					found_running_job=0
					for potential_slurm_id in $(ls $mongodbpath/../ipfiles/ | grep mongodbserverip | sed -e 's#.*-##'); do
						if [[ "$found_running_job" -eq "0" ]]; then
							if squeue -u $USER | grep $potential_slurm_id > /dev/null; then
								found_running_job=$potential_slurm_id
							fi
						fi
					done

					if [[ "$found_running_job" -ne "0" ]]; then
						echo_red "The Job is running right now. It's slurm ID is $found_running_job. Cannot run repair with a database that is running."
						exit
					fi
				else
					echo_red "Did not find $mongodbpath/../ipfiles/, cannot check for running instances of this DB"
				fi

				echo_green "The file $mongodbpath/WiredTiger exists, so it seems to be a MongoDB-folder"
				if [[ -e "$mongodbpath/mongod.lock" ]]; then
                    if [[ -z $UNSUPERVISED ]]; then
                        if (whiptail --title "The lock file already exists. Delete it?" --yesno "The file $mongodbpath/mongod.lock exists, but it needs to be deleted to continue. Are you sure a server is not running on this folder and you want to delete it?" 10 120); then
                            echo_green "Deleting $mongodbpath/mongod.lock" 
                            rm "$mongodbpath/mongod.lock" 
                        else
                            echo_red "Cancelled the repair-script because it needs to delete the $mongodbpath/mongod.lock file but you chose to cancel deletion";
                            exit
                        fi
                    else
                        echo_green "Deleting $mongodbpath/mongod.lock in unsupervised mode" 
                        rm "$mongodbpath/mongod.lock" 
                    fi
				fi

				echo_green "Backing up folder $mongodbpath to $backupfolder"
				cp -r $mongodbpath $backupfolder

				mongodbport=$(get_random_open_port_localhost)

				echo_green "Running mongod --repair --dbpath $mongodbpath --port $mongodbport"
				if mongod --repair --dbpath $mongodbpath --port $mongodbport; then
					echo_green "Repairing the DB folder seems to have worked"

					if [[ -e "$mongodbpath/mongod.lock" ]]; then
						echo_green "Deleting $mongodbpath/mongod.lock" 
						rm "$mongodbpath/mongod.lock" 
					fi
				else
					echo_red "Repairing the DB folder seems to have failed"
					mv "$mongodbpath" "$mongodbpath.repair_tried_but_failed"
					mv "$backupfolder" "$mongodbpath"
				fi
			else
				echo_red "$mongodbpath/WiredTiger does not exist. The folder does not seem to be a MongoDB-folder"
			fi
		else
			echo_red "The folder $mongodbpath does not exist"
		fi
	fi
}

main $DBNAME
