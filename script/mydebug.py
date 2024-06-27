import sys
import os
import textwrap
import inspect
import traceback
import logging
from pprint import pprint
from termcolor import colored

from getOpts import getOpts
import myfunctions
import mypath
import range_generator
from hyperopt import fmin, tpe
from hyperopt.rand import suggest
import networkstuff
import linuxstuff
import omnioptstuff

from simulated_annealing import validate_space_simulated_annealing, simulated_annealing
from gridsearch import gridsearch

global myconf
myconf = None

def dier(data):
    pprint(data)
    exit(1)

def cprint (param, show_live_output=1):
    if str(show_live_output) == "1":
        print('\x1b[1;31m' + str(param) + '\x1b[0m')


def set_myconf (projectname, projectdirdefault=None):
    config_path = omnioptstuff.get_config_path_by_projectname(projectname, projectdirdefault)
    myconf = getOpts(config_path)
    return myconf

global set_debug
set_debug = False

global set_debug_xtreme
set_debug_xtreme = False

global set_info
set_info = True

global set_stack
set_stack = False

global set_warning
set_warning = True

global set_success
set_success = True

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def get_data(projectname=None, params=None, projectdirdefault=None):
    projectfolder = ""
    debug("projectname: " + str(projectname))
    params = myfunctions.parse_params(sys.argv)
    projectdir = None

    if "projectdir" in params:
        projectdir = params["projectdir"]

    if projectdirdefault is not None:
        projectdir = projectdirdefault

    global myconf
    projectfolder = omnioptstuff.get_project_folder(projectname, projectdir)

    config_path = omnioptstuff.get_config_path_by_projectfolder(projectfolder)
    myconf = getOpts(config_path)

    set_debug = myconf.bool_get_config('DEBUG', 'debug')
    set_debug_xtreme = myconf.bool_get_config('DEBUG', 'debug_xtreme')
    set_info = myconf.bool_get_config('DEBUG', 'info')
    set_stack = myconf.bool_get_config('DEBUG', 'stack')
    set_warning = myconf.bool_get_config('DEBUG', 'warning')
    set_success = myconf.bool_get_config('DEBUG', 'success')
    projectfolder = projectfolder + "/"

    debug("Getting config-data")
    data = {}
    data["objective_program"] = None
    data["seed"] = None
    debug("Objective code WAS none!")
    data["objective_program"] = myconf.get_cli_code('DATA', 'objective_program')
    if data["objective_program"]:
        debug("objective_program was ok!")
    else:
        debug("objective_program was NOT ok!")

    mongodbipfolder = projectfolder + "/ipfiles/"
    try:
        os.stat(mongodbipfolder)
    except:
        os.mkdir(mongodbipfolder)

    mongodbipfile = mongodbipfolder + "mongodbserverip-" + str(os.getenv("SLURM_JOB_ID"))
    mongodbipfile = linuxstuff.normalize_path(mongodbipfile)
    if os.path.isfile(mongodbipfile):
        saved_ip = ''
        with open(mongodbipfile) as f:
            saved_ip = f.readline()
        if networkstuff.is_valid_ipv4_address(saved_ip):
            data["mongodbmachine"] = saved_ip
        else:
            print("The IP `" + saved_ip + "` is not a valid one!")
    else:
        if(str(os.getenv("SLURM_JOB_ID")) != "None"):
            sys.stderr.write("The file `" + mongodbipfile + "` could not be found! Using 127.0.0.1 instead\n")
        data["mongodbmachine"] = "127.0.0.1"

    mongodbportfile = mongodbipfolder + "mongodbportfile-" + str(os.getenv("SLURM_JOB_ID"))
    mongodbportfile = linuxstuff.normalize_path(mongodbportfile)
    if os.path.isfile(mongodbportfile):
        savedport = ''
        with open(mongodbportfile) as f:
            savedport = f.readline()
        data["mongodbport"] = int(savedport)
        #print("mongodbport from file: " + str(savedport))
    else:
        default_port = str(os.getenv("mongodbport", 56741))
        if(str(os.getenv("SLURM_JOB_ID")) != "None"):
            sys.stderr.write("The file `" + mongodbportfile + "` could not be found! Using " + default_port + " instead\n")
        data["mongodbport"] = default_port
    try:
        data["mongodbmachine"] = myconf.str_get_config('MONGODB', 'machine')
    except: 
        pass
    if not networkstuff.ping(data["mongodbmachine"]):
        warning("Server `" + str(data["mongodbmachine"]) + "` not reachable via ping!")
    data["mongodbdir"] = projectname
    data["mongodbdbname"] = projectname
    data["mongodblogpath"] = "mongodb.log"
    data["worker_last_job_timeout"] = myconf.int_get_config('MONGODB', 'worker_last_job_timeout')
    data["worker_poll_interval"] = myconf.float_get_config('MONGODB', 'poll_interval')

    data["max_evals"] = myconf.int_get_config('DATA', 'max_evals')
    if type(data["max_evals"]) is None:
        raise Exception("The max_evals Option in DATA must be set!")
    if not data["max_evals"] > 0:
        raise Exception("max_evals has to be at least one!")
    data["algo_name"] = myconf.str_get_config('DATA', 'algo_name')
    data["range_generator_name"] = myconf.str_get_config('DATA', 'range_generator_name')
    data["dimensions"] = myconf.int_get_config('DIMENSIONS', 'dimensions')
    if not data["dimensions"] > 0:
        raise Exception("Dimensions must at least be 1 or higher and integer-only!")

    data["precision"] = myconf.int_get_config("DATA", "precision")

    algorithms_desc = range_generator.get_algorithms_list()
    data["algo"] = None
    if data["algo_name"] in algorithms_desc:
        if data["algo_name"] == "hyperopt.rand.suggest":
            data["algo"] = suggest
        else:
            data["algo"] = eval(data["algo_name"])

    if data["algo"] is None:
        possible_algorithms = ''
        for i in algorithms_desc:
            possible_algorithms = possible_algorithms + "\n\t- " + i + "\n\t\t " + algorithms_desc[i] + "\n"
        raise Exception("Algorithm has to be set! Possible algorithms:\n" + possible_algorithms)

    data["show_stack"] = myconf.int_get_config('DEBUG', 'stack')

    data["show_live_output"] = 0
    try:
        data["show_live_output"] = myconf.int_get_config('DEBUG', 'show_live_output')
    except: 
        pass

    try:
        data["seed"] = myconf.int_get_config('DATA', 'seed')
    except: 
        pass

    if data["show_stack"] == 0:
        sys.tracebacklimit = 0

    if not params is None:
        if "mongodbmachine" in params:
            if not(networkstuff.is_valid_ipv4_address(params["mongodbmachine"]) or networkstuff.is_valid_ipv6_address(params["mongodbmachine"])):
                raise Exception("Invalid IP `" + params["mongodbmachine"] + "`. Must be either IPv4 or IPv6-adress!")
            else:
                data["mongodbmachine"] = params["mongodbmachine"]

    return data

def debug_xtreme(message):
    if set_debug_xtreme:
        if set_stack:
            traceback.print_stack()
        logging.debug(colored(message, 'yellow'))

def debug(message):
    if set_debug:
        if set_stack:
            traceback.print_stack()
        logging.debug(colored(message, 'yellow'))

def error(message):
    if set_stack:
        traceback.print_stack()
    logging.error(colored(message, 'red'))

def warning(message):
    if set_warning:
        if set_stack:
            traceback.print_stack()
        logging.warning(colored(message, 'magenta'))

def info(message):
    if set_info:
        if set_stack:
            traceback.print_stack()
        logging.info(colored(message, 'green'))

def success(message):
    if set_stack:
        traceback.print_stack()
        if set_success:
            successstr = colored(message, 'green') + "\n"
            sys.stderr.write(successstr)

def range_generator_info(chosen_range_generator):
    string = "Chosen range generator: `" + chosen_range_generator['name'] + "`"
    string = string + "- Range generator description: `" + chosen_range_generator['description'] + "`\n"
    string = string + "- Needed variables: `" + str(chosen_range_generator['options']) + "`"
    return string

def get_valid_range_generators():
    range_generator_dict = range_generator.get_range_generator_list()
    valid_generators = ''
    for this in range_generator_dict:
        valid_generators = valid_generators + "\t- " + this["parameters"] + "\n"
        desc = textwrap.fill(this["description"], 60) + "\n"
        desc = '\t\t\t'.join(desc.splitlines(True))
        valid_generators = valid_generators + "\t\t\t" + desc + "\n"
    return valid_generators

def module_warnings():
    warnings_list = [
        {
            'name': 'networkx',
            'error_version': '2.0',
            'text': "may contain errors resulting in `TypeError: 'generator' object is not subscriptable line`",
            'use_version': '1.11'
        }
    ]

    for item in warnings_list:
        exec("import " + item["name"])
        used_ver = eval(item["name"] + ".__version__")
        if used_ver == item["error_version"]:
            warning("W A R N I N G ! ! ! The module `" + item["name"] + "` with the version " + \
                item["error_version"] +  " " + item["text"] + ", use version " + \
                item["use_version"] + \
                " if any errors occur! Maybe you need to use a virtualenvironment. See the manual for more details.")
