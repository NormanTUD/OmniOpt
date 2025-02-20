#!/usr/bin/env python3

import pprint
import sys

from mydebug import debug, error, warning, info, myconf, get_data, module_warnings, set_myconf
from hyperopt import fmin, tpe, hp
import myfunctions
import mongostuff

if sys.version_info[0] < 3:
    raise Exception("Must be using python3 or higher")

params = myfunctions.parse_params(sys.argv)
projectname = None
if "project" in params:
    projectname = params["project"]
    myconf = set_myconf(projectname)
try:
    data = get_data(projectname, params)

    mongostuff.shut_down_mongodb(projectname)
except Exception as e:
    print(str(e))
    sys.exit(1)
