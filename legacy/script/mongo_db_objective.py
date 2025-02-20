"""
This file contains the function for calling programs that should be minimized. The
only function here is the objective_function_mongodb, which gets called by the
hyperopt-mongo-worker.
"""

from os.path import expanduser
import signal
import sys
import os
import stat
import time
import socket
import mydebug
import logstuff
import omnioptstuff
import workerstuff
import re
import myregexps
import os
import pathlib
from pathlib import Path
import uuid
import hashlib

import myfunctions
from hyperopt import STATUS_OK, STATUS_FAIL
from pprint import pprint

def dier (msg):
    pprint(msg)
    sys.exit(1)

def cprint (param, show_live_output=1):
    if str(show_live_output) == "1":
        print('\x1b[1;31m' + param + '\x1b[0m')

def is_number (strnum):
    this_match = re.match(myregexps.floating_number_limited, str(strnum))
    if this_match is None:
        return False
    else:
        return True

def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()

def objective_function_mongodb(parameter):
    """
    This is the function called from the hyperopt-mongo-workers. It gets, as parameters,
    different x-values to be tested and converts them to a command-line-call. This is then
    executed in a virtual console, and parsed for outputs like
    RESULT: 0.523
    These results are then returned and put into the MongoDB by the workers.
    """

    times = {}
    times["start"] = time.time()
    projectname = parameter[0]
    xvars = parameter[1]
    projectdir = parameter[2]
    params = parameter[3]

    data = mydebug.get_data(projectname, None, projectdir)

    basepath = pathlib.Path(__file__).parent.absolute()
    still_running_file = os.path.abspath(projectdir + "/" + str(projectname) + "/ipfiles/still_has_jobs_" + uuid.uuid4().hex)
    touch(still_running_file)

    try:
        f = open(still_running_file)
    except IOError:
        print("File not accessible: " + str(still_running_file))
    finally:
        f.close()

    try:
        cprint("Defining log files", data["show_live_output"])

        res = None

        cprint("Getting parameter code", data["show_live_output"])
        parameter_code = omnioptstuff.replace_dollar_with_variable_content(
            data["objective_program"],
            'x_',
            *xvars,
            projectname
            )

        cprint("replace $x_0 and $firstname with value", data["show_live_output"])
        i = 0
        for thisvar in xvars:
            parameter_code = parameter_code.replace("($" + 'x_' + str(i) + ")", str(xvars[i]))
            parameter_code = parameter_code.replace("($" + str(thisvar) + ")", str(xvars[i]))
            i = i + 1

        cprint("Replacing $STDOUT and $STDERR", data["show_live_output"])

        maindir = os.path.dirname(os.path.realpath(__file__)) + "/../"

        md5ofjob = hashlib.md5(parameter_code.encode('utf-8')).hexdigest()

        specific_log_file = logstuff.create_log_folder_and_get_log_file_path(projectdir, projectname, md5ofjob)

        logfolder = omnioptstuff.get_project_folder(projectname, projectdir)

        log_files = {
            "stderr": specific_log_file + ".stderr",
            "stdout": specific_log_file + ".stdout",
            "debug": specific_log_file + ".debug",
            "general_filename_log": logfolder + "/program.log"
        }

        print("TRYING parameter_code = " + parameter_code, file=open(log_files["general_filename_log"], "a"))
        print("Code: " + parameter_code, file=open(log_files["general_filename_log"], "a"))

        singlelogs_path = logfolder + "/singlelogs/"

        program_file = singlelogs_path + md5ofjob + ".command"

        with open(program_file, 'w') as f:
            print("#!/bin/bash", file=f)
            print("echo $SLURM_JOB_NODELIST >> %s/nodes.txt" % logfolder, file=f)
            bash_code = parameter_code
            print(bash_code, file=f)

        st = os.stat(program_file)
        os.chmod(program_file, st.st_mode | stat.S_IEXEC)

        debug_string = ""
        home = expanduser("~")
        if os.path.exists(home + "/.oo_multigpu_debug"):
            debug_string = " --debug "

        run_code = "bash %s/tools/multigpu.sh --projectfolder=%s --logfolder=%s --account=%s --reservation=%s --num_gpus=%d --programfile='%s' --jobname=%s --cpus_per_task=%d %s" % (maindir, logfolder, singlelogs_path, params["account"], params["reservation"], int(params["num_gpus_per_worker"]), program_file, md5ofjob, params["cpus_per_task"], debug_string)

        try:
            res = workerstuff.run_program(run_code, log_files)
        except Exception as e:
            logstuff.print_visible(e)

        print("DONE RUNNING CODE", file=open(log_files["general_filename_log"], "a"))
        print("res: " + str(res), file=open(log_files["general_filename_log"], "a"))

        status = STATUS_FAIL

        cprint("Logfile: " + log_files["general_filename_log"], data["show_live_output"])

        cprint("RES: " + str(res), data["show_live_output"])

        cprint("Writing general log file", data["show_live_output"])
        logstuff.write_general_log_file(
            run_code,
            log_files["general_filename_log"],
            specific_log_file,
            res,
            log_files
        )

        cprint("Getting output from stdout", data["show_live_output"])
        orig_res = res
        if not res is None:
            res = workerstuff.get_result_from_output_file(log_files["stdout"], projectdir, projectname)
            cprint(">>>>>>RESULT found as " + str(res), data["show_live_output"])

        cprint("Getting all_results", data["show_live_output"])
        all_results = workerstuff.get_data_from_output(orig_res["stdout"])
        all_results["loss"] = res
        if is_number(all_results["loss"]):
            status = STATUS_OK

        ### Output-Header:
        output_header = '>>>OUTPUTHEADER>>>loss'
        for key in sorted(all_results.keys()):
            output_header = output_header + "," + key

        j = 0
        for thisvar in xvars:
            output_header = output_header + ",x_" + str(j)
            j = j + 1

        cprint(output_header, data["show_live_output"])
        ### Output-Header-Ende

        output_for_file = '>>>OUTPUT>>>' + str(res)

        for key in sorted(all_results.keys()):
            output_for_file = output_for_file + "," + str(all_results[key])

        j = 0
        for thisvar in xvars:
            output_for_file = output_for_file + "," + str(xvars[j])
            j = j + 1

        cprint(output_for_file, data["show_live_output"])

        times["end"] = time.time()

        all_results["starttime"] = times["start"]
        all_results["endtime"] = times["end"]
        all_results["hostname"] = socket.gethostname()
        all_results["logfile_path"] = specific_log_file

        cprint("Returning data", data["show_live_output"])

        try:
            os.remove(still_running_file)
        except Exception:
            cprint("Could not unlink " + str(still_running_file))

        return {
            'starttime': times["start"],
            'endtime': times["end"],
            'loss' : res,
            'status': status,
            'all_outputs': all_results,
            'calculation_time': times["end"] - times["start"],
            'output': '',
            'every_epoch_data': '',
            'parameters': xvars
        }

    except Exception as e:
        os.remove(still_running_file)
        print('In mongo_db_objective, an error occured: ' + str(e))
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
