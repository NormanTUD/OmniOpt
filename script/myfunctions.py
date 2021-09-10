from signal import signal, SIGPIPE, SIG_DFL
import sys
import re
import mydebug
import slurmstuff
import omnioptstuff
import mypath
from pprint import pprint
from random import randint
import socket
import traceback
import os

signal(SIGPIPE, SIG_DFL)

def dier (msg):
    pprint(msg)
    traceback.print_stack()
    sys.exit(1)

def parse_all_arguments(all_args):
    parsed_args = {}
    #traceback.print_stack()
    #print("===================")

    for i in range(len(all_args)):
        if i != 0:
            this_parsed = parse_single_argument(all_args[i])
            if this_parsed is not None:
                parsed_args[this_parsed['name']] = this_parsed['value']
    return parsed_args

def parse_single_argument(str_arg):
    #print("str_arg = %s" % str_arg)
    prog = re.compile(r"^--(.*?)(?:=(.*))?$")
    result = prog.match(str_arg)

    ret = None

    if result is not None and result.group(0) is not None:
        ret = {
            'name': result.group(1),
            'value': result.group(2)
        }
    else:
        mydebug.warning("WARNING: Invalid argument `" + str_arg + "`, cannot be parsed!")
    return ret

def parse_params(argvd):
    data = {
        'parameter': None,
        'int': False,
        'filename': None,
        'slurmid': None,
        'seperator': ",",
        'projectdir': mypath.mainpath + '/../projects/',
        'project': None,
        'minvalue': None,
        'switchaxes': False,
        'maxvalue': None,
        'maxtime': None,
        'mongodbmachine': None,
        'mongodbport': None,
        'debug': None,
        'setmongoperparameter': None
    }

    parsed_args = parse_all_arguments(argvd)

    parameter_names = [
        "mongodbport",
        "projectdir",
        "parameter",
        "slurmid",
        "seperator",
        "filename",
        "project",
        "minvalue",
        "switchaxes",
        "maxvalue",
        "maxtime",
        "mongodbmachine",
        "mongodbport",
        'debug'
    ]

    default_ip = "127.0.0.1"

    for pname in parameter_names:
        if pname in parsed_args:
            data[pname] = parsed_args[pname]

    if data["mongodbmachine"] is None:
        data["mongodbmachine"] = default_ip
    else:
        data["setmongoperparameter"] = 1;

    if data["mongodbport"] is None:
        data["mongodbport"] = str(os.getenv("mongodbport", 56741))
    else:
        data["setmongoperparameter"] = 1;

    if data["projectdir"]:
        data["projectdir"] = data["projectdir"] + "/"

    if 'int' in parsed_args:
        data['int'] = True
    else:
        data['int'] = False

    if 'maxvalue' in parsed_args:
        try:
            data['maxvalue'] = float(parsed_args['maxvalue'])
        except Exception as e: 
            sys.stderr.write(e)

    if 'maxtime' in parsed_args:
        try:
            data['maxtime'] = float(parsed_args['maxtime'])
        except Exception as e: 
            sys.stderr.write(e)

    if 'switchaxes' in parsed_args:
        data['switchaxes'] = True
    else:
        data['switchaxes'] = False

    # wenn slurm id und node count, dann schaue in die writeip log datei für mongodbmachine

    if 'slurmid' in data:
        if not data['slurmid'] is None:
            if slurmstuff.is_slurm_id(data['slurmid']):
                if 'project' in data:
                    mongoinfo = omnioptstuff.find_mongo_info(
                        data['project'],
                        data['slurmid'],
                        data,
                        data['projectdir']
                    )
                    data['mongodbport'] = mongoinfo['mongodbport']
                    data['mongodbmachine'] = mongoinfo['mongodbmachine']
                else:
                    sys.stderr.write("Slurmid is defined, but no project! That's quite weird.")
            else:
                sys.stderr.write("`" + str(data['slurmid']) + "` is not a valid slurm id!")
                data['slurmid'] = None
    return data
