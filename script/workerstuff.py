import mydebug
from hyperopt.mongoexp import MongoTrials
import subprocess
import filestuff
import json
import socket
import mypath
import myregexps
import re
import mongostuff
import linuxstuff
import omnioptstuff
import filestuff
import shlex
from pprint import pprint

def dier(data):
    pprint(data)
    exit(1)

def module_load_string(modules):
    module_load = ''
    module_load = module_load + "export MODULEPATH=/sw/modules/taurus/applications\n"
    module_load = module_load + "export MODULEPATH=$MODULEPATH:/sw/modules/taurus/tools\n"
    module_load = module_load + "export MODULEPATH=$MODULEPATH:/sw/modules/taurus/libraries\n"
    module_load = module_load + "export MODULEPATH=$MODULEPATH:/sw/modules/taurus/compilers\n"
    module_load = module_load + "export MODULEPATH=$MODULEPATH:/opt/modules/modulefiles\n"
    module_load = module_load + "export MODULEPATH=$MODULEPATH:/sw/modules/taurus/environment\n"

    for modname in modules:
        module_load = module_load + "eval `/usr/share/lmod/lmod/libexec/lmod $SHELL load " + modname + "`\n"
    return module_load

def initialize_mongotrials_object(projectname, data):
    mydebug.debug('MongoTrials object will be initialized')
    mongourl = 'mongo://' + data['mongodbmachine'] + ':' + str(data['mongodbport']) + '/' + data['mongodbdbname'] + '/jobs'
    mydebug.debug("Mongo-URL: " + mongourl)
    trials = MongoTrials(mongourl, exp_key=projectname)
    mydebug.debug('Initialization of MongoTrials objects done')
    return trials

def run_program(program, logfiles):
    errstr = ''
    retcode = None

    programconverted = shlex.split(program)

    try:
        fstdout = open(logfiles['stdout'], 'w+')
        fstderr = open(logfiles['stderr'], 'w+')

        sp = subprocess.Popen(
            programconverted,
            stdout=fstdout,
            stderr=fstderr,
            universal_newlines=True, 
            close_fds=True,
            bufsize=65536
        )
        sp.wait()
        retcode = sp.returncode

        fstdout.close()
        fstderr.close()
    except Exception as e:
        errstr = errstr + "!!! Forking process failed !!!"
        errstr = errstr + "!!! " + str(e) + " !!!"


    out = filestuff.get_whole_file(logfiles['stdout'])
    err = filestuff.get_whole_file(logfiles['stderr'])

    res = str("STDOUT:\n" + out + "\n\nSTDERR:\n" + err + "\n\nRETURNCODE:\n" + str(retcode) + "\n\nERRSTR:\n" + str(errstr) + "\n")

    array = {
        'res': res,
        'stdout': out,
        'stderr': err,
        'retcode': retcode,
        'errstr': errstr
    }

    return array

def get_keras_results_json(filename_log):
    return {'kerasresults': '{}', 'every_epoch_data': '{}'}

    '''
    every_epoch_data = None
    kerasresults = subprocess.run(
        [
            'perl',
            mypath.mainpath + '/keras_output_parser.pl',
            '--filename=' + filename_log
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    try:
        every_epoch_data = json.loads(kerasresults.stdout.decode('utf-8'))
    except:
        print("kerasresults coult not be decoded:\n" + str(kerasresults.stdout))
    return {'kerasresults': kerasresults, 'every_epoch_data': every_epoch_data}
    '''

def get_result_from_output(res, donealready=0):
    re_search = 'RESULT: (' + myregexps.floating_number + ')'
    if donealready == 0:
        re_search = 'RESULT: (' + myregexps.floating_number + ')\\n'
    m = re.search(re_search, res)
    if m:
        res = m.group(1)
        mydebug.debug('Found through regex: ' + res)
        floated = None
        try:
            floated = float(res)
        except:
            mydebug.error("The code `" + res + "` could not be converted into a float")
        return floated
    else:
        if donealready == 0:
            return get_result_from_output(res, 1)
        else:
            return float('inf')

def get_result_from_output_file(res):
    file1 = open(res, 'r') 
    Lines = file1.readlines() 

    count = 0
    re_search = 'RESULT: (' + myregexps.floating_number + ')'
    for line in Lines: 
        line = line.splitlines()[0]
        groups = re.search('RESULT: (' + myregexps.floating_number + ')', line, re.IGNORECASE)
        #groups = re.search(rf"{re_search}", line, re.IGNORECASE)
        if groups:
            floated = float(groups.group(1))
            return floated
    return float('inf')

def get_data_from_output(res):
    data = {}
    if res is None:
        mydebug.warning("res was None at get_data_from_output!")
        return data

    mydebug.debug("get_data_from_output:\n" + str(res))
    lines = re.findall('[a-zA-Z]+:\\s*' + myregexps.floating_number, res)

    for line in lines:
        this_match = re.match('^([a-zA-Z]+):\\s*(' + myregexps.floating_number + ")", line)
        if this_match is not None:
            data[this_match.group(1)] = this_match.group(2)
    return data


def start_worker(data, start_worker_command, myconf, slurmid, projectdir, project):
    start_worker_command = get_start_worker_command(start_worker_command, project, myconf, data, projectdir)

    startworkerbashfolder = omnioptstuff.get_project_folder(data['mongodbdbname']) + '/ipfiles/'
    startworkerbashfile = startworkerbashfolder + "startworker-" + str(slurmid)
    startworkerbashfile = linuxstuff.normalize_path(startworkerbashfile)

    try:
        filehandle = open(startworkerbashfile, 'w')
        filehandle.write(start_worker_command)
        filehandle.close()
    except Exception as this_error: 
        print("Unable to create file %s on disk: %s" % (startworkerbashfile, this_error))

def get_start_worker_command(start_worker_command, project, myconf, data, projectdirdefault=None):
    kill_after_n_no_results = myconf.int_get_config('MONGODB', 'kill_after_n_no_results')
    if kill_after_n_no_results:
        mydebug.debug('>>> kill_after_n_no_results defined: >>> ' + str(kill_after_n_no_results) + ", changed start_worker_command accordingly")
        ipfilesdir = omnioptstuff.get_project_folder(data['mongodbdbname'], projectdirdefault) + '/ipfiles/'
        colorprintfgray = 'true'

        show_live_output = 0
        info_filter = "cat"

        try:
            show_live_output = myconf.int_get_config('DEBUG', 'show_live_output')
        except: 
            pass
        try:
            if(myconf.int_get_config('DEBUG', 'stack')):
                info_filter = "grep -v '^INFO:hyperopt:'"
        except: 
            pass
        
        if str(show_live_output) == "1":
            colorprintfgray = '    printf "\\e[100m$1\\e[0m\\n"'

        #->varname<- will be replaced with python's varname at the end of this multiline-string
        # if you add any variables, you need to add them manually after that long string for replacement!!!

        # Also, DO NOT ADD MONGODB HERE!!! It will mess up everything

        start_worker_bash = '''#!/bin/bash -l

LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
module () {
    eval `$LMOD_CMD sh "$@"`
}

if hostname | grep -i ml > /dev/null; then
    module load modenv/ml
    module load TensorFlow/2.0.0-PythonAnaconda-3.7
else
    module load modenv/scs5
    module load Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4
fi

THISUUID=$(uuidgen)
THISUUIDFILE=->ipfilesdir<-/GPU_${SLURM_JOB_ID}_$(hostname)_${THISUUID}
GPUFILE=->ipfilesdir<-/GPU_${SLURM_JOB_ID}

ml TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4 2>/dev/null
PCIBUS=$(python3 -c "import tensorflow as tf; print(tf.test.gpu_device_name())" 2>&1)
echo $PCIBUS > $THISUUIDFILE
cat $THISUUIDFILE | grep pciBusID | sed -e "s/^/Node: $(hostname), CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES, physical GPU -> /"
echo "$(hostname):$(cat $THISUUIDFILE | sed -e 's/:.*, pci bus id: /:/' | sed -e 's/, .*//' | sort | sed -e 's/:/-0000/')" >> $GPUFILE
ml unload TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4 2>/dev/null

function colorprintfyellow () {
    printf "\\e[93m$1\\e[0m\\n"
}

function colorprintfred () {
    printf "\\e[91m$1\\e[0m\\n"
}

function colorprintfgray () {
    ->colorprintfgray<-
}

CUDAFILE=->ipfilesdir<-/omniopt_CUDA_VISIBLE_DEVICES_${SLURM_JOB_ID}_$(hostname) 

let mongotrycount=0
while ! nc -zvv ->mongodbmachine<- ->mongodbport<-; do 
    echo "DB not running, sleeping 1 second to try again"
    sleep 1
    let mongotrycount++
    if [[ "$mongotrycount" -eq "300" ]]; then
        echo "Waited 5 minutes without any progress on waiting for MongoDB. Exiting."
        exit 1
    fi
done

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES
export SLURM_JOB_ID

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export CURRENT_SCRIPT_DIR
WORKERBASHPID=$$
export WORKERBASHPID

#### Perl script start ####

eval $(perl -e "
use strict;
use warnings;
use Sys::Hostname;
use feature 'say';

my \$CUDA = \$ENV{CUDA_VISIBLE_DEVICES};
my \$CUR = \$ENV{CURRENT_SCRIPT_DIR};
my \$pid = \$ENV{WORKERBASHPID};
my \$done = 0;
my \$cuda_available = 0;
my \$hostname = hostname();

# If CUDA_VISIBLE_DEVICES is set
my \$enable_advanced_gpu_allocation = 0;
if (\$enable_advanced_gpu_allocation && defined \$CUDA && \$CUDA =~ m#^(\\d+,?)+\$#) {
    # Set cuda_available so that I know that at least GPUs are allocated
    \$cuda_available = 1;
    # Cycle through all available GPUs
    foreach my \$gpu (split /,/, \$CUDA) {
            next if \$done;
            my \$gpu_file = qq/\$ENV{SLURM_JOB_ID}-\$hostname-\$gpu/;
            my \$full_path_gpu_file = qq#\$CUR/\$gpu_file#;

            my \$tmpfile = \$CUR.q#/#.rand();
            system(q/touch /.\$tmpfile);
            unlink \$tmpfile;

            warn qq/--------------------------------------------------------------------------------/;
            warn qq/--------------------------------------------------------------------------------/;
            warn qq/THIS->CUDA_VISIBLE_DEVICES -> \$CUDA/;
            warn qq/--------------------------------------------------------------------------------/;
            warn qq/--------------------------------------------------------------------------------/;
            warn qq/ls: ----------------------------------------------------------------------------/;
            warn qx(ls \$CUR);
            warn qq/--------------------------------------------------------------------------------/;
            warn qq/--------------------------------------------------------------------------------/;
            warn qq/CHECKING \$full_path_gpu_file/;
            warn qq/--------------------------------------------------------------------------------/;
            warn qq/--------------------------------------------------------------------------------/;

            if(-e \$gpu_file) {
                    warn qq/The GPU \$gpu on \$hostname seems to be used already/;
            } else {
                    open my \$fh, q/>/, \$full_path_gpu_file or die \$!;
                    say \$fh qq/\$hostname-workerpid:\$pid/;
                    close \$fh;
                    print qq/export CUDA_VISIBLE_DEVICES=\$gpu/;
                    \$done = 1;
            }
    }
}

if(\$enable_advanced_gpu_allocation && \$cuda_available && !\$done) {
        warn qq/Every allocatable GPU is already allocated to a job. Setting CUDA_VISIBLE_DEVICES to empty.\n/;
        print qq/export CUDA_VISIBLE_DEVICES=""/;
}")

#### Perl script done ####

if [[ -e $CUDAFILE ]]; then
    echo -n ",${CUDA_VISIBLE_DEVICES}" >> $CUDAFILE
else
    echo -n ${CUDA_VISIBLE_DEVICES} >> $CUDAFILE
fi

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    echo "CUDA_VISIBLE_DEVICES could not be set. Continuing without specified GPUs." >&2
else
    echo "CUDA_VISIBLE_DEVICES for this worker: $CUDA_VISIBLE_DEVICES"
fi

export max_kill=->kill_after_n_no_results<-;
export counter=0;

#export PYTHONPATH=/sw/installed/Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4/lib/python3.7/site-packages/:->mainpath<-:$PYTHONPATH
export PYTHONPATH=->mainpath<-:$PYTHONPATH

#echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
#pip3 install --user future 
#echo "======================================================================"
#python3 -c "import hyperopt; import six; import pprint; import past; pprint.pprint(past)"
#echo "======================================================================"
#echo "PYTHONPATH -> $PYTHONPATH"
#echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

->start_worker_command<- 2>&1 | ->info_filter<- 2>&1 | {
    while IFS= read -r line;
    do
        colorprintfgray "$line";
        if [[ "$(ls $CURRENT_SCRIPT_DIR | grep still_has_jobs | wc -l)" -ne "0" ]]; then
            counter=0
        elif [[ $line =~ .*no\ job\ found.* ]]; then
            colorprintfyellow "no job found number $counter";
            counter=$((counter+1));
            if [[ "$counter" -gt "$max_kill" ]]; then
                if [[ "$SECONDS" -gt 600 ]]; then
                    for i in `ps -ef| awk \'$3 == \'$$\' { print $2 }\'`;
                    do
                        colorprintfred "killing $i";
                        kill $i;
                    done;
                else
                    counter=0
                fi
            fi
        else
            counter=0
        fi;
    done
}
'''

        start_worker_bash = start_worker_bash.replace("->mainpath<-", str(mypath.mainpath))
        start_worker_bash = start_worker_bash.replace("->ipfilesdir<-", str(ipfilesdir))
        start_worker_bash = start_worker_bash.replace("->mongodbport<-", str(data["mongodbport"]))
        start_worker_bash = start_worker_bash.replace("->mongodbmachine<-", str(data["mongodbmachine"]))
        start_worker_bash = start_worker_bash.replace("->colorprintfgray<-", str(colorprintfgray))
        start_worker_bash = start_worker_bash.replace("->kill_after_n_no_results<-", str(kill_after_n_no_results))
        start_worker_bash = start_worker_bash.replace("->info_filter<-", str(info_filter))
        start_worker_bash = start_worker_bash.replace("->start_worker_command<-", str(start_worker_command))

        return start_worker_bash
    else:
        return start_worker_command

def get_main_start_worker_command(data, project, enable_strace=0, enable_python_trace=0):
    python3path = linuxstuff.normalize_path('python3.7')

    python_trace = ''
    if enable_python_trace:
        python_trace = " -m trace --count "

    start_worker_command = python3path + python_trace + ' ' + mypath.mainpath + \
        "/hyperopt-mongo-worker --reserve-timeout=3600 --max-consecutive-failures=10000 --mongo=" +  data['mongodbmachine'] + \
        ':' + str(data['mongodbport']) + '/' + data['mongodbdbname'] + " --poll-interval=" + str(data['worker_poll_interval']) + \
        " --last-job-timeout=" + str(data['worker_last_job_timeout']) + " --workdir=" + omnioptstuff.get_project_folder(data['mongodbdbname']) + \
        " --max-jobs=" + str(data['max_evals'])

    tmp_basefolder = '/tmp'
    worker_command_file = tmp_basefolder + '/worker_start_command'
    mydebug.debug('worker_command_file = ' + worker_command_file)
    start_worker_command_bash = "#!/bin/bash\n\n" + start_worker_command + "\n\nrm " + worker_command_file + "\n"
    filestuff.overwrite_file(worker_command_file, start_worker_command_bash)

    return start_worker_command
