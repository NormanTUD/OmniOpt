#!/usr/bin/env python3

import pprint
import sys

from mydebug import debug, error, warning, info, myconf, get_data, module_warnings, set_myconf
from mongo_db_objective import objective_function_mongodb
import myfunctions
import mongostuff

if sys.version_info[0] < 3:
    raise Exception("Must be using python3 or higher")

params = myfunctions.parse_params(sys.argv)
projectname = None
if "project" in params:
    projectname = params["project"]
    myconf = set_myconf(projectname)
data = get_data(projectname, params)

mongostuff.start_mongo_db(projectname, data, 1)
