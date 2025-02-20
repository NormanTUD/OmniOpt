#!/bin/bash

function _sbatch_pl {
    _describe 'command' "(
        '--help:This help'
        '--worker=10:Number of workers (usually automatically deducted from --ntasks)'
        '--mempercpu=1234:Defines how much memory every worker should get. Usually this is not needed, because this script gets it from the sbatch commands --mem-per-cpu'
        '--nosanitycheck:Disables the sanity checks (not recommended!)'
        '--project=projectname:Name of the project you want to execute'
        '--keep_db_up=1:Check if the DB is up every \$sleep_db_up seconds and restart it if its not (0 = no, 1 = yes)'
        '--sleep_db_up=30:Check if DB is up every n seconds and restart if its not'
        '--run_nvidia_smi=1:Run (1) or dont run (0) nvidia-smi periodically to get information about the GPU usage of the workers'
        '--sleep_nvidia_smi=10:Sleep n seconds after each try to get nvidia-smi-gpu-infos'
        '--debug_srun:Enables debug-output for srun'
        '--projectdir=/path/to/projects/:This allows you to change the project directory to anything outside of this script path (helpful, so that the git is kept small), only use absolute paths here!)'
        '--nomsgs:Disables messages'
        '--debug:Enables lots and lots of debug outputs'
        '--dryrun:Run this script without any side effects (i.e. not creating files, not starting workers etc.)'
        '--nowarnings:Disables the outputting of warnings (not recommended!)'
        '--nodryrunmsgs:Disables the outputting of --dryrun-messages'
        '--run_tests:Runs a bunch of tests and exits without doing anything else'
        '--run_full_tests:Run testsuite and also run a testjob (takes longer, but is more safe to ensure stability)'
    )"
}

function _evaluate_run {
    _describe 'command' "(
        '--projectdir=/path/to/projects/:Path to available projects'
        '--showtestprojects:Show test projects (default is not to show them)'
        '--nogauge:Disables the gauges'
        '--help:Help'
        '--dont_load_modules:Dont load modules'
        '--no_upgrade:Disables upgrade'
        '--debug:Debug mode'
    )"
}

function _get_wallclock_time_sh {
    _describe 'command' "(
        '--projectdir=:Project dir'
        '--project=:Project name'
        '--help:Help'
        '--debug:Enables debug mode (set -x)'
    )"
}

function _whypending {
    SQUEUE_OUTPUT=$(squeue -o "%i:%j" -u $USER | grep -v "JOBID:NAME")
    
    SCANCEL_COMMANDS=(
    )
    
    while IFS= read -r line; do
        if [[ ! -z $line ]]; then
            SCANCEL_COMMANDS+=("$line")
        fi
    done <<< "$SQUEUE_OUTPUT"
    
    SCANCEL_COMMANDS_STR=$(printf "\n'%s'" "${SCANCEL_COMMANDS[@]}")
    
    eval "_describe 'command' \"($SCANCEL_COMMANDS_STR)\""
}

function _slurmlogpath {
    SQUEUE_OUTPUT=$(squeue -o "%i:%j" -u $USER | grep -v "JOBID:NAME")
    
    SCANCEL_COMMANDS=(
    )
    
    while IFS= read -r line; do
        if [[ ! -z $line ]]; then
            SCANCEL_COMMANDS+=("$line")
        fi
    done <<< "$SQUEUE_OUTPUT"
    
    SCANCEL_COMMANDS_STR=$(printf "\n'%s'" "${SCANCEL_COMMANDS[@]}")
    
    eval "_describe 'command' \"($SCANCEL_COMMANDS_STR)\""
}


function _scancel {
    SQUEUE_OUTPUT=$(squeue -o "%i:%j" -u $USER | grep -v "JOBID:NAME")
    
    SCANCEL_COMMANDS=(
        '--signal=:Signal type (USR1, USR2, INT etc.)'
        '--batch:Send signal to all batch steps'
    )
    
    while IFS= read -r line; do
        if [[ ! -z $line ]]; then
            SCANCEL_COMMANDS+=("$line")
        fi
    done <<< "$SQUEUE_OUTPUT"
    
    SCANCEL_COMMANDS_STR=$(printf "\n'%s'" "${SCANCEL_COMMANDS[@]}")
    
    eval "_describe 'command' \"($SCANCEL_COMMANDS_STR)\""
}

function _ml {
    ML_COMMANDS=(
        '-t:Show computer parsable output'
        'unload:Unload a Module'
        'spider:Search for a module'
        'avail:Show available modules'
        'list:List loaded modules'
    )
    
    ML_COMMANDS_STR=$(printf "\n'%s'" "${ML_COMMANDS[@]}")
    
    eval "_describe 'command' \"($ML_COMMANDS_STR)\""
    _values -s ' ' 'flags' $(ml -t avail | sed -e 's#/$##' | tr '\n' ' ')
}

function _ftails {
    SQUEUE_OUTPUT=$(squeue -o "%i:%j" -u $USER | grep -v "JOBID:NAME")
    
    SCANCEL_COMMANDS=(
    )
    
    while IFS= read -r line; do
        if [[ ! -z $line ]]; then
            SCANCEL_COMMANDS+=("$line")
        fi
    done <<< "$SQUEUE_OUTPUT"
    
    SCANCEL_COMMANDS_STR=$(printf "\n'%s'" "${SCANCEL_COMMANDS[@]}")
    
    eval "_describe 'command' \"($SCANCEL_COMMANDS_STR)\""
}

function _ws_release {
    if echo "$words" | egrep "ws_release\s*-F\s*(ssd|scratch|beegfs)" >/dev/null 2>/dev/null; then
        CHOSEN_FS=$(echo "$words" | sed -e 's/.*-F\s*//' | sed -e 's/\s*//g')
        OUTPUT=$(ws_list 2>&1 | sed -e "s#$USER-##" | grep -v "is empty" | egrep "directory|filesystem" | sed -e 's#.*:\s*##' | perl -lne "
                use Data::Dumper; 
                use strict;
                use warnings;
                use autodie;    # 5
                my (\$i, \$path, \$fs) = (0, undef, undef); 
                my %val = ();
                while (<>) { 
                    chomp; 
                    if(\$i % 2 == 1) { 
                        \$path = \$_;
                    } else { 
                        if(\$path && \$_) { 
                            \$val{\$path} = \$_; 
                        } 
                    }; 
                    \$i++ 
                }; 
                delete \$val{''}; 
                foreach (keys %val) {
                    my \$key = \$_;
                    my \$v = \$val{\$key};
                    if(\$v eq qq#$CHOSEN_FS#) {
                        \$key =~ s#.*/##g;
                        print qq#\$key\n#;
                    }
                }
            "
        )

        eval "_describe 'command' \"($OUTPUT)\""
    else
        _arguments '-F[Filesystem]:filename:((ssd\:"Use SSD filesystem" scratch\:"Use Scratch filesystem" beegfs\:"Use beegfs"))'
    fi
}

function _ws_extend {
    if echo "$words" | egrep "ws_extend\s*-F\s*(ssd|scratch|beegfs)" >/dev/null 2>/dev/null; then
        CHOSEN_FS=$(echo "$words" | sed -e 's/.*-F\s*//' | sed -e 's/\s*//g')
        OUTPUT=$(ws_list 2>&1 | sed -e "s#$USER-##" | grep -v "is empty" | egrep "directory|filesystem" | sed -e 's#.*:\s*##' | perl -lne "
                use Data::Dumper; 
                use strict;
                use warnings;
                use autodie;    # 5
                my (\$i, \$path, \$fs) = (0, undef, undef); 
                my %val = ();
                while (<>) { 
                    chomp; 
                    if(\$i % 2 == 1) { 
                        \$path = \$_;
                    } else { 
                        if(\$path && \$_) { 
                            \$val{\$path} = \$_; 
                        } 
                    }; 
                    \$i++ 
                }; 
                delete \$val{''}; 
                foreach (keys %val) {
                    my \$key = \$_;
                    my \$v = \$val{\$key};
                    if(\$v eq qq#$CHOSEN_FS#) {
                        \$key =~ s#.*/##g;
                        print qq#\$key\n#;
                    }
                }
            "
        )

        eval "_describe 'command' \"($OUTPUT)\""
    else
        _arguments '-F[Filesystem]:filename:((ssd\:"Use SSD filesystem" scratch\:"Use Scratch filesystem" beegfs\:"Use beegfs"))'
    fi
}

function _ws_restore {
    if echo "$words" | egrep "ws_restore\s*-F\s*(ssd|scratch|beegfs)" >/dev/null 2>/dev/null; then
        CHOSEN_FS=$(echo "$words" | sed -e 's/.*-F\s*//' | sed -e 's/\s*//g')
        OUTPUT=$(ws_restore -l | grep -v unavailable | perl -e "
                use Data::Dumper; 
                \$l = q##; 
                %h = (); 
                while (<>) { 
                    chomp; 
                    if(/(.*):\s*$/) { 
                        \$l = \$1; 
                    } else { 
                        push @{\$h{\$l}}, \$_; 
                    } 
                }; 
                foreach (@{\$h{$CHOSEN_FS}}) {
                    print qq#\$_\n#;
                }
        "

        )

        eval "_describe 'command' \"($OUTPUT)\""
    else
        _arguments '-F[Filesystem]:filename:((ssd\:"Use SSD filesystem" scratch\:"Use Scratch filesystem" beegfs\:"Use beegfs"))'
    fi
}

compdef _ml "ml"
compdef _scancel "scancel"
compdef _sbatch_pl "sbatch.pl"
compdef _evaluate_run "evaluate-run.sh"
compdef _get_wallclock_time_sh "get_wallclock_time.sh"
compdef _whypending "whypending"
compdef _ftails "ftails"
compdef _ws_extend "ws_extend"
compdef _ws_release "ws_release"
compdef _ws_restore "ws_restore"
compdef _slurmlogpath "slurmlogpath"
