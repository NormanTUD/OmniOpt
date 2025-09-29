#!/bin/bash

# Check if file was provided
if [ $# -eq 0 ]; then
    echo "Please provide a file as an argument"
    exit 1
fi

# Check if file exists
if [ ! -f "$1" ]; then
    echo "File $1 does not exist"
    exit 1
fi

# Read each line from the file
while IFS= read -r line; do

    # Parse the experiment name
    experiment_name=$(echo "$line" | grep -oP '(?<=experiment_name=)[^&]*')
    IFS='_' read -r -a experiment_parts <<< "$experiment_name"

    # Check if experiment name has the correct format
    if [ ${#experiment_parts[@]} -ne 4 ]; then
        echo "Invalid experiment name format: $experiment_name"
        continue
    fi

    # Assign values to variables
    var1=${experiment_parts[0]}
    var2=${experiment_parts[1]}
    var3=${experiment_parts[2]}
    var4=${experiment_parts[3]}
    #http://172.26.63.64/share?user_id=s4122485&experiment_name=BNDM_Electricity_HoeffdingTreeClassifier_ACCURACY-RUNTIME&run_nr=0
    (curl "$line&sort=time_desc#tab_export" |  awk '/<!-- export.html -->/{p=!p; next} p' | perl -MHTML::Entities -pe 'decode_entities($_)') > "${var1}_${var2}_${var3}_${var4}.html"

done < "$1"
