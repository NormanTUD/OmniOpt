import os
import uuid
import pathlib
from pathlib import Path
import omnioptstuff

def print_to_log(string, logfile):
    folder = os.path.dirname(logfile)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    append_write = 'a'
    if not os.path.exists(logfile):
        append_write = 'a'
    else:
        append_write = 'w'

    logfilehandler = open(logfile, append_write)

    print(string, file=logfilehandler)

    logfilehandler.close()

def print_visible (string):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(string)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

def write_general_log_file(parameter_code, general_filename_log, specific_log_file, res, log_files):
    write_program_log(
        parameter_code,
        general_filename_log,
        res
    )

    write_program_log(
        parameter_code,
        specific_log_file,
        res
    )

    write_program_logs(parameter_code, log_files, res)

def write_program_logs(parameter_code, log_files, res):
    write_program_log_text(parameter_code, log_files['stderr'], res, 'stderr')
    write_program_log_text(parameter_code, log_files['stdout'], res, 'stdout')

def write_program_log(parameter_code, filename_log, res):
    seperator = "\n=========================================\n"
    seperator2 = "\n<=======================================>\n"

    log_string = 'code: ' + str(parameter_code) + seperator
    log_string = log_string + ">>>>>>>>>>>>>>>>>\nProgram-Output:\n" + str(res) + seperator2

    logfilehandler = open(filename_log, 'a+')
    logfilehandler.write(log_string)
    logfilehandler.close()

def write_program_log_text(parameter_code, filename_log, res, typeofresults):
    seperator = "\n=========================================\n"
    seperator2 = "\n<=======================================>\n"

    log_string = 'code: ' + str(parameter_code) + seperator + ">>>>>>>>>>>>>>>>>\nProgram-Output:\n" + str(res) + seperator2
    if typeofresults == 'stdout':
        log_string = str(res['stdout'])
    else:
        log_string = str(res['stderr'])

    logfilehandler = open(filename_log, 'a')
    logfilehandler.write(log_string)
    logfilehandler.close()

def get_and_create_specific_log_folder(projectdir, projectname):
    specific_log_folder = omnioptstuff.get_project_folder(projectname, projectdir) + '/singlelogs/'
    if not os.path.exists(specific_log_folder):
        try:
            os.makedirs(specific_log_folder)
        except Exception as e:
            print('An error occured: ' + str(e))
    return specific_log_folder


def get_specific_log_file(projectdir, specific_log_folder, md5ofjob):
    specific_log_file = specific_log_folder + md5ofjob
    while os.path.exists(specific_log_file):
        specific_log_file = specific_log_folder + str(uuid.uuid1())
    return specific_log_file

def create_log_folder_and_get_log_file_path(projectdir, projectname, md5ofjob):
    specific_log_folder = get_and_create_specific_log_folder(projectdir, projectname)
    specific_log_file = get_specific_log_file(projectdir, specific_log_folder, md5ofjob)
    return specific_log_file
