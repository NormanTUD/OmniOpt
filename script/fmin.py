#!/usr/bin/env python3
"""
This will start fmin, the main process feeding the workers with new positions in n-dimensional
space to try out.
"""

import pprint
import sys

from gridsearch import gridsearch, validate_space_exhaustive_search, ExhaustiveSearchError
from simulated_annealing import validate_space_simulated_annealing, simulated_annealing
from mydebug import debug, warning, myconf, get_data, module_warnings, set_myconf
from mongo_db_objective import objective_function_mongodb
import hyperopt
from hyperopt import fmin, hp
import myfunctions
import omnioptstuff
import workerstuff
import mypath
import atexit
import numpy as np
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def dier (msg):
    pprint.pprint(msg)

debug('Starting...')
module_warnings()

if sys.version_info[0] < 3:
    raise Exception('Must be using python3 or higher')

params = myfunctions.parse_params(sys.argv)
projectname = None
projectdir = None
if 'project' in params:
    projectname = params['project']
    myconf = set_myconf(projectname)
if 'projectdir' in params:
    projectdir = params['projectdir']
data = get_data(projectname, params)
space = hp.choice('a', [omnioptstuff.get_choices(myconf, data)])

#print('>>>>>>>>>>>>>>>>>> PROJECTNAME:::: ' + str(projectname))

debug("Defining empty `best_data`")
best_data = None

debug('Beginning fmin')
debug('algo = ' + str(data['algo']))
debug('max_evals = ' + str(data['max_evals']))

if data["algo"] == "gridsearch":
    validate_space_exhaustive_search(space)
elif data["algo"] == "annealing":
    validate_space_simulated_annealing(space)

fmin_parameters = {
    'fn':                       objective_function_mongodb,
    'trials':                   workerstuff.initialize_mongotrials_object(projectname, data),
    'space':                    [projectname, space, projectdir, params],
    'algo':                     data['algo'],
    'max_evals':                data['max_evals'],
    'catch_eval_exceptions':    True,
    'max_queue_len':            10,
    'verbose':                  0
}

if data["seed"] is not None:
    os.environ['HYPEROPT_FMIN_SEED'] = str(data["seed"])
    print("Using seed: %s" % data["seed"])
    # fmin_parameters["rstate"] = np.random.default_rng(int(data["seed"]))

try:
    best = fmin(**fmin_parameters)
    debug('Ending fmin')
    best_data = hyperopt.space_eval(space, best)
except Exception as e: 
    print(f"{bcolors.FAIL}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{bcolors.ENDC}")
    print(f"{bcolors.FAIL}OmniOpt encountered an error.{bcolors.ENDC}")
    if len(str(e)):
        print(e)
    print("This is probably caused by wrong hyperparameters.")
    print("Possible error sources:")
    print("- You have negative values in hp.loguniform entries")
    print("- hp.choice or hp.uniform you have a lower bound less than 0")
    print(f"{bcolors.FAIL}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{bcolors.ENDC}")



def end_code ():
    global myconf
    global best
    global best_data

    if best_data is not None:
        omnioptstuff.print_best(myconf, best, best_data)
    else:
        warning("Could not get best_data!")

atexit.register(end_code)
