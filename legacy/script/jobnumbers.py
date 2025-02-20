#!/usr/bin/env python3

import pprint
import os
import sys
import datetime
import time
from termcolor import colored

from mydebug import debug, error, warning, info, myconf, get_data, module_warnings, set_myconf
import myfunctions
import mongostuff

import pymongo

params = myfunctions.parse_params(sys.argv)
projectname = None
if "project" in params:
    projectname = params["project"]
    myconf = set_myconf(projectname)
data = get_data(projectname, params)

mongostuff.start_mongo_db(projectname, data)

connect_string = "mongodb://" + data['mongodbmachine'] + ":" + str(data["mongodbport"]) + "/"
debug("Connecting to: " + connect_string)

myclient = pymongo.MongoClient(connect_string)
connect_to_db = params["project"]
print("Connecting to DB: " + connect_to_db + ", " + connect_string)
testdb = myclient[connect_to_db]
jobs = testdb["jobs"]

while (1):
    i = 0
    for line in jobs.find({}, {"result": 1}):
        i = i + 1

    print(i)

    max_evals = data["max_evals"]

    print ("max_evals: " + str(max_evals) + "/" + str(i) + " = " + str(i / max_evals))

    print(os.environ.get("subjobs"))
    if (i / max_evals) >= 0.99:
        exit(0)
    time.sleep(500)
