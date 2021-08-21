#!/usr/bin/env python3
"""
This will start the workers on individual servers, so that jobs are being
processed.
"""

import sys
import os
import os.path
import workerstuff

from mydebug import debug, myconf, get_data, module_warnings, set_myconf
import myfunctions

debug('Starting...')
module_warnings()

if sys.version_info[0] < 3:
    raise Exception('Must be using python3 or higher')

params = myfunctions.parse_params(sys.argv)
projectname = None
if 'project' in params:
    projectname = params['project']
    myconf = set_myconf(projectname)
data = get_data(projectname, params)

debug("Defining empty `best_data`")
best_data = None

start_worker_command = workerstuff.get_main_start_worker_command(data, params['project'], 0, 0)
debug(workerstuff.start_worker(data, start_worker_command, myconf, params['slurmid'], params['projectdir'], params['project']))
