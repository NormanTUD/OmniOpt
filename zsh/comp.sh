function _sbatch_pl (){
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

function _evaluate_run () {
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

compdef _sbatch_pl "sbatch.pl"
compdef _evaluate_run "evaluate-run.sh"
compdef _get_wallclock_time_sh "get_wallclock_time.sh"
