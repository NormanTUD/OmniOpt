#!/usr/bin/env python3
"""
This will start the mongodb database on the main server.
"""

import sys

from mydebug import debug, myconf, get_data, module_warnings, set_myconf
import myfunctions
import mongostuff

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
exit_code = mongostuff.start_mongo_db(projectname, data)
sys.exit(exit_code)
