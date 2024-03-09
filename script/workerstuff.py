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
import os
import sys
import os.path

def dier(data):
    pprint(data)
    exit(1)

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
            universal_newlines=True, 
            stdout=fstdout,
            stderr=fstderr,
            encoding='utf-8',
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

def get_re_beginning (projectdir, projectname):
    if projectdir is not None and projectdir != "" and projectname is not None and projectname != "":
        re_start = projectdir + "/" + projectname + "/re_start"
        if os.path.exists(re_start):
            fline = open(re_start).readline().rstrip()
            return fline
    return "RESULT"

def get_result_from_output_file(res, projectdir="", projectname=""):
    file1 = open(res, 'r') 
    Lines = file1.readlines() 

    count = 0
    for line in Lines: 
        line = line.splitlines()[0]
        groups = re.search(get_re_beginning(projectdir, projectname) + ': (' + myregexps.floating_number + ')', line, re.IGNORECASE)
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
    lines = re.findall('[a-zA-Z0-9]+:\\s*' + myregexps.floating_number, res)

    for line in lines:
        this_match = re.match('^([a-zA-Z0-9]+):\\s*(' + myregexps.floating_number + ")", line)
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

def get_start_worker_command(start_worker_command, project, myconf, data, projectdirdefault=None, waitsecondsnz=300):
    kill_after_n_no_results = myconf.int_get_config('MONGODB', 'kill_after_n_no_results')
    if kill_after_n_no_results:
        mydebug.debug('>>> kill_after_n_no_results defined: >>> ' + str(kill_after_n_no_results) + ", changed start_worker_command accordingly")
        ipfilesdir = omnioptstuff.get_project_folder(data['mongodbdbname'], projectdirdefault) + '/ipfiles/'
        colorprintfgray = 'true'

        show_live_output = 0
        info_filter = "grep -v '^INFO:hyperopt:'"

        try:
            show_live_output = myconf.int_get_config('DEBUG', 'show_live_output')
        except: 
            pass
        
        if str(show_live_output) == "1":
            colorprintfgray = '    printf "\\e[100m$1\\e[0m\\n"'

        #->varname<- will be replaced with python's varname at the end of this multiline-string
        # if you add any variables, you need to add them manually after that long string for replacement!!!

        # Also, DO NOT ADD MONGODB HERE!!! It will mess up everything

        start_worker_bash = '''#!/bin/bash -l

function calltracer () {
        echo 'Last file/last line:'
        caller
}
trap 'calltracer' ERR

_term() { 
    echo "Caught SIGTERM signal!" 
}

trap _term SIGTERM

echo "Hostname of this worker: $(hostname)"

LMOD_CMD=/usr/share/lmod/lmod/libexec/lmod
module () {
    eval `$LMOD_CMD sh "$@"`
}

IPFILESDIR=->ipfilesdir<-
THISUUID=$(uuidgen)
THISUUIDFILE=$IPFILESDIR/GPU_${SLURM_JOB_ID}_$(hostname)_${THISUUID}
GPUFILE=$IPFILESDIR/GPU_${SLURM_JOB_ID}
NZLOGFILE=${IPFILESDIR}/nz_log_${SLURM_JOB_ID}_${THISUUID}
GENERALLOGFILE=${IPFILESDIR}/general_log_${SLURM_JOB_ID}_${THISUUID}

function myload {
    if ! module is-loaded $1; then
        module load $1 2>> $GENERALLOGFILE >> $GENERALLOGFILE 2>> $GENERALLOGFILE
    fi       
}

function myunload {
    if module is-loaded $1; then
        module unload $1 2>> $GENERALLOGFILE >> $GENERALLOGFILE 2>> $GENERALLOGFILE
    fi       
}

if hostname | grep -i ml > /dev/null; then
    myload modenv/ml
    myload Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4
    myload TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4
else
    myload release/23.04 2>&1 | grep -v loaded
    myload GCC/11.3.0 2>&1 | grep -v loaded
    myload OpenMPI/4.1.4 2>&1 | grep -v loaded
    myload Hyperopt/0.2.7 2>&1 | grep -v loaded
    myload TensorFlow/2.11.0-CUDA-11.7.0 2>&1 | grep -v loaded
fi

PCIBUS=$(python3 -c "import tensorflow as tf; print(tf.test.gpu_device_name())" 2>&1)
echo $PCIBUS > $THISUUIDFILE
cat $THISUUIDFILE | grep pciBusID | sed -e "s/^/Node: $(hostname), CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES, physical GPU -> /" >> $GENERALLOGFILE
echo "$(hostname):$(cat $THISUUIDFILE | sed -e 's/:.*, pci bus id: /:/' | sed -e 's/, .*//' | sort | sed -e 's/:/-0000/')" >> $GPUFILE

myunload TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4

function colorprintfyellow () {
    printf "\\e[93m$1\\e[0m\\n"
}

function colorprintfred () {
    printf "\\e[91m$1\\e[0m\\n"
}

function colorprintfgray () {
    ->colorprintfgray<-
}

CUDAFILE=$IPFILESDIR/omniopt_CUDA_VISIBLE_DEVICES_${SLURM_JOB_ID}_$(hostname) 

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" >> $GENERALLOGFILE

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

echo "HOSTNAME: " >> $GENERALLOGFILE
hostname >> $GENERALLOGFILE

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    echo "CUDA_VISIBLE_DEVICES could not be set. Continuing without specified GPUs." >&2 >> $GENERALLOGFILE
else
    echo "CUDA_VISIBLE_DEVICES for this worker: $CUDA_VISIBLE_DEVICES" >> $GENERALLOGFILE
fi

export max_kill=->kill_after_n_no_results<-;
export counter=0;

export PYTHONPATH=->mainpath<-:$PYTHONPATH

export MONGODBMACHINE=->mongodbmachine<-
export MONGODBPORT=->mongodbport<-

warningcounter=$(echo "scale=0; ($max_kill*0.8)/1" | bc)

let mongotrycount=0
while ! nc -z $MONGODBMACHINE $MONGODBPORT 2>&1 >> $NZLOGFILE; do 
    echo "DB not running, sleeping 1 second to try again"
    sleep 1
    let mongotrycount++
    if [[ "$mongotrycount" -eq "->waitsecondsnz<-" ]]; then
        echo "Waited ->waitsecondsnz<- seconds without any progress on waiting for MongoDB on $MONGODBMACHINE:$MONGODBPORT. Exiting."
        exit 1
    fi
done

set -e
set -o pipefail
set -u

->start_worker_command<- 2>&1 | ->info_filter<- 2>&1 | {
    while IFS= read -r line;
    do
        colorprintfgray "$line";
        if [[ "$(ls $CURRENT_SCRIPT_DIR | grep still_has_jobs | wc -l)" -ne "0" ]]; then
            counter=0
        elif [[ $line =~ .*no\ job\ found.* ]]; then
            counter=$((counter+1));
            if [[ "$counter" -gt "$warningcounter" ]]; then
                colorprintfyellow "no job found number $counter";
            fi
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

        try:
            start_worker_bash = start_worker_bash.replace("->mainpath<-", str(mypath.mainpath))
            start_worker_bash = start_worker_bash.replace("->ipfilesdir<-", str(ipfilesdir))
            start_worker_bash = start_worker_bash.replace("->waitsecondsnz<-", str(os.getenv('waitsecondsnz', waitsecondsnz)))
            start_worker_bash = start_worker_bash.replace("->mongodbport<-", str(data["mongodbport"]))
            start_worker_bash = start_worker_bash.replace("->mongodbmachine<-", str(data["mongodbmachine"]))
            start_worker_bash = start_worker_bash.replace("->colorprintfgray<-", str(colorprintfgray))
            start_worker_bash = start_worker_bash.replace("->kill_after_n_no_results<-", str(kill_after_n_no_results))
            start_worker_bash = start_worker_bash.replace("->info_filter<-", str(info_filter))
            start_worker_bash = start_worker_bash.replace("->start_worker_command<-", str(start_worker_command))
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)

        return start_worker_bash
    else:
        return start_worker_command

def get_main_start_worker_command(data, project, enable_strace=0, enable_python_trace=0):
    python3path = linuxstuff.normalize_path('python3')

    python_trace = ''
    if enable_python_trace:
        python_trace = " -m trace --count "

    start_worker_command = python3path + python_trace + ' ' + mypath.mainpath + \
        "/hyperopt-mongo-worker --reserve-timeout=3600 --max-consecutive-failures=10000 --mongo=" +  data['mongodbmachine'] + \
        ':' + str(data['mongodbport']) + '/' + data['mongodbdbname'] + " --poll-interval=" + str(data['worker_poll_interval']) + \
        " --workdir=" + omnioptstuff.get_project_folder(data['mongodbdbname']) + \
        " --max-jobs=" + str(data['max_evals'])

    tmp_basefolder = omnioptstuff.get_project_folder(data['mongodbdbname']) 
    worker_command_file = tmp_basefolder + '/worker_start_command'
    mydebug.debug('worker_command_file = ' + worker_command_file)
    start_worker_command_bash = "#!/bin/bash -l\n\n" + start_worker_command + "\n\nrm " + worker_command_file + "\n"
    filestuff.overwrite_file(worker_command_file, start_worker_command_bash)

    return start_worker_command
