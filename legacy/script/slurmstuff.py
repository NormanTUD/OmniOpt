import re
import mydebug
import subprocess

def is_slurm_id(slurmid):
    if slurmid is None:
        return 0
    pattern = re.compile(r"^\d+(_\d+)?$")
    if pattern.match(slurmid):
        return 1
    else:
        return 0

def slurm_job_is_running(projectname, data=None):
    notthisjob = None
    if data is not None:
        if 'slurmid' in data:
            notthisjob = data['slurmid']
    command = "squeue -u $USER | grep ' " + projectname + " ' | wc -l"
    if notthisjob is not None:
        command = command + " | grep -v " + notthisjob
    mydebug.debug(command)
    output = subprocess.check_output(command, shell=True)
    output = str(output.decode("utf-8"))
    output = output.rstrip("\n")
    mydebug.info(output)
    if output == '0':
        return 0
    else:
        print("(Not necessarily with the same slurm-id!)")
        return 1
