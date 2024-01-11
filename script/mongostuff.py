from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 
from pymongo.errors import ConnectionFailure
import signal
from termcolor import colored
from pymongo import MongoClient
import mydebug
import os
import mypath
import linuxstuff
import omnioptstuff
import subprocess
import socket
import sys
from pathlib import Path
import slurmstuff

class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()

'''
get_mongo_db_error_code(code) accepts a code of mongodb as parameter
and returns a description and, if possible, instructions on how to solve
the error to stdout and as text as return value
'''

def get_mongo_db_error_code(code):
    mydebug.debug("get_mongo_db_error_code(" + str(code) + ")")
    codes = [
        [0, 'Returned by MongoDB applications upon successful exit.'],
        [2, 'The specified options are in error or are incompatible with other options.'],
        [3, 'Returned by mongod if there is a mismatch between hostnames specified on the command line and in the local.sources collection, in master/slave mode.'],
        [4, 'The version of the database is different from the version supported by the mongod (or mongod.exe) instance. The instance exits cleanly.'],
        [5, 'Returned by mongos if a problem is encountered during initialization.'],
        [12, 'Returned by the mongod.exe process on Windows when it receives a Control-C, Close, Break or Shutdown event.'],
        [14, 'Returned by MongoDB applications which encounter an unrecoverable error, an uncaught exception or uncaught signal. The system exits without performing a clean shutdown.'],
        [48, 'A newly started mongod or mongos could not start listening for incoming connections, due to an error.'],
        [62, "Returned by mongod if the datafiles in --dbpath are incompatible with the version of mongod currently running.", "delete " + os.path.normpath(mypath.mainpath) + "/./mongodb directory"],
        [100, "Returned by mongod when the process throws an uncaught exception.", "\n\t- Check if the process is already running and if so, use that one or kill that one to start a new one\n\t- check if the mongo db key has the right permissions (usually 400)"],
        [127, "Error 127 occurs when there is a fatal flaw in starting MongoDB, for example, a library file is missing. Please run the MongoDB command manually and check the standard error for more details"]
    ]

    res_text = "Unkown error code (" + str(code) + ")!"

    for ret in codes:
        if ret[0] == code:
            res_text = ret[1]
            if len(ret) == 3:
                res_text = res_text + "\n" + colored("possible solution: ", 'green') + ret[2]

    if linuxstuff.is_tool('ps'):
        if linuxstuff.is_tool('grep'):
            psauxfcommand = 'ps auxf | grep mongod | grep -v "mongodb.py" | grep -v grep'
            proc = subprocess.Popen(psauxfcommand, stdout=subprocess.PIPE, shell=True)
            (out, err) = proc.communicate()

            out = str(out)

            if out.count("\n") == 0:
                sys.stderr.write("No MongoDB process found\n")
            else:
                sys.stderr.write("\n\n" + psauxfcommand + ":\n\n" + out + "\n\n")
        else:
            sys.stderr.write("grep not installed!")
    else:
        sys.stderr.write("ps not installed!")

    return res_text

"""
mongo_db_already_up accepts a single parameter with a connection-string to a
mongodb-server. It returns true, if the server is reachable, and false, if not
"""

def mongo_db_already_up(conStr):
    mydebug.debug_xtreme("mongo_db_already_up(" + conStr + ")...")
    client = MongoClient(conStr)
    ret = True
    try:
        # The ismaster command is cheap and does not require auth.
        mongo_timeout = int(os.getenv("MONGOTIMEOUT", 86400 * 7))
        with Timeout(mongo_timeout):
            client.admin.command('ismaster')
    except ConnectionFailure:
        ret = False
    except Timeout.Timeout:
        mydebug.debug('Timeout')
        ret = False

    mydebug.debug_xtreme("mongo_db_already_up() == " + str(ret))
    return ret

def create_mongo_db_connection_string(data):
    machine = str(data['mongodbmachine'])
    port = str(data['mongodbport'])
    directory = str(data['mongodbdir'])
    url = 'mongodb://' + machine + ':' + str(port) + '/' + directory
    return url


def shut_down_mongodb(projectname):
    folder = omnioptstuff.get_project_folder(projectname) + '/mongodb/'
    command = "mongod --dbpath " + folder + " --shutdown >/dev/null 2>/dev/null"
    mydebug.debug(command)
    output = subprocess.check_output(command, shell=True)
    if str(output) != "b''":
        sys.stderr.write(str(output))
    os.remove(folder + '/mongod.lock')


def backup_mongo_db(projectname, data):
    connstr = create_mongo_db_connection_string(data)
    sys.stderr.write(str(connstr))
    if not mongo_db_already_up(connstr):
        sys.stderr.write('connstr: ' + str(connstr) + ", DB *****NOT***** up!!!!")
    else:
        machine = data['mongodbmachine']
        port = data['mongodbport']
        url = machine + ':' + str(port)
        outdir = omnioptstuff.get_project_folder(projectname) + '/backup/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '/'
        os.makedirs(outdir)

        command = "mongodump --host=" + url + ' --out=' + outdir
        mydebug.debug(command)
        output = subprocess.check_output(command, shell=True)
        sys.stderr.write(output)

def file_is_writable (filename):
    returnvalue = 1
    try:
        with open(filename, 'w') as f:
            pass
    except IOError as x:
        returnvalue = 0
        print('error, file ' + filename + " is not writable; " + str(x))
    return returnvalue

def start_mongo_db(projectname, data, check_for_running_db=0, tried_again=0):
    if "MONGODBALREADYSTARTED" in os.environ:
        return
    if not projectname is None:
        if check_for_running_db == 1:
            if slurmstuff.slurm_job_is_running(projectname):
                sys.stderr.write('ERROR: Job is running right now')
                return 1
    port = os.getenv("mongodbport", data["mongodbport"] or None)
    if port is None or port == "": 
        port = os.getenv("mongodbport", 56741)
    directory = projectname
    logpath = 'mongodb.log'
    projectfolder = omnioptstuff.get_project_folder(projectname)
    if linuxstuff.is_tool('mongo') and linuxstuff.is_tool('mongod'):
        url = create_mongo_db_connection_string(data)
        if mongo_db_already_up(url):
            mydebug.info("Mongo-DB already running. Not starting it again.")
        else:
            mydebug.debug("Mongo-DB *NOT* already running. Trying to start it...")
            # Wenn kein absoluter Pfad, dann relativer in den Projektdaten
            if not directory.startswith('/'):
                dbpath = omnioptstuff.get_project_folder(projectname) + '/mongodb'
                mydebug.debug("The path is relative, and thus, it's put into the project's folder: " + dbpath)
            else:
                dbpath = directory
            ipfilesdir = omnioptstuff.get_project_folder(projectname) + '/ipfiles/'
            mydebug.debug("Checking for path `" + dbpath + "`")
            if not os.path.exists(dbpath):
                mydebug.debug("`" + dbpath + "` did not exist, creating it now!")
                os.makedirs(dbpath)
            else:
                mydebug.debug("`" + dbpath + "` did exist, doing nothing!")

            exclamstring = "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            if not os.access(dbpath, os.W_OK):
                sys.stderr.write(exclamstring + "`" + dbpath + "` is not writable!!!" + exclamstring)

            mongo_lock_file = dbpath + '/mongod.lock'
            if os.path.exists(mongo_lock_file):
                sys.stderr.write(exclamstring + "The file `" + mongo_lock_file + "` already exists!!!" + exclamstring)
                return 100

            premongodb = 'ulimit -f unlimited; ulimit -n 64000; ulimit -u 32000; '

            mongo_db_parameter = ' --bind_ip_all --smallfiles '

            if not logpath.startswith(projectfolder):
                logpath = omnioptstuff.get_project_folder(projectname) + '/' + logpath
            logfilepath = linuxstuff.normalize_path(logpath) 
            if not file_is_writable(logfilepath):
                print("File " + logfilepath + " is not writeable. Trying to fix that...")
                try:
                    if os.path.exists(logpath):
                        os.chmod(logpath, 0o644);
                    else:
                        print("Logpath '%s' could not be found. Trying to create it..." % str(logpath))
                        Path(logpath).touch()
                        os.chmod(logpath, 0o644);

                    if os.path.exists(logfilepath):
                        os.chmod(logfilepath, 0o644);
                    else:
                        print("The path '%s' could not be found." % str(logfilepath))
                except Exception as e:
                    print("Chmodding the file failed. Trying anyways...")
                    print(str(e))
            dbpath = linuxstuff.normalize_path(dbpath) 
            startDB = premongodb + 'mongod ' + mongo_db_parameter + " --fork --dbpath " + dbpath + " --port " + str(port) + " --logpath " + logfilepath + " --logappend 1>&2 2>/dev/null >/dev/null"
            mydebug.debug(startDB + "\n")
            mongo = subprocess.Popen(startDB, shell=True, preexec_fn=os.setsid)
            mongo.wait()
            if mongo.returncode:
                if tried_again == 0:
                    os.system("bash tools/repair_database.sh " + str(dbpath) + " 1")
                    return start_mongo_db(projectname, data, check_for_running_db, tried_again + 1)
                else:
                    return_code = mongo.returncode
                    sys.stderr.write("`" + startDB + "` exited with error code " + str(return_code) + ", that means:\n====================\n" + get_mongo_db_error_code(return_code) + "\n====================\n")
                    return return_code
            else:
                mydebug.debug("Starting mongodb with `" + startDB + "` done")
    else:
        mydebug.error("Mongo is not installed! Try loading ml MongoDB/4.0.3. It should be available on all types of nodes, regardless of the architecture.")
    return 0

