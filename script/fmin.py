#!/usr/bin/env python3
"""
This will start fmin, the main process feeding the workers with new positions in n-dimensional
space to try out.
"""

import pprint
import sys

from mydebug import debug, warning, myconf, get_data, module_warnings, set_myconf
from mongo_db_objective import objective_function_mongodb
import hyperopt
from hyperopt import fmin, hp
import myfunctions
import omnioptstuff
import workerstuff
import mypath

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

fmin_parameters = {
    'fn':                       objective_function_mongodb,
    'trials':                   workerstuff.initialize_mongotrials_object(projectname, data),
    'space':                    [projectname, space, projectdir],
    'algo':                     data['algo'],
    'max_evals':                data['max_evals'],
    'catch_eval_exceptions':    True,
    'max_queue_len':            10
}

best = fmin(**fmin_parameters)
debug('Ending fmin')
best_data = hyperopt.space_eval(space, best)
if best_data is not None:
    print("Best result data:")
    pprint.pprint(best)
    pprint.pprint(best_data)
else:
    warning("Could not get best_data!")
