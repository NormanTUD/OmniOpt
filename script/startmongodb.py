#!/usr/bin/env python3

import pprint
import sys

from mydebug import myconf, get_data, set_myconf
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
