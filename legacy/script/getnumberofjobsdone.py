#!/usr/bin/env python3

import pprint
import os
import sys
import time
from termcolor import colored

from mydebug import debug, error, warning, info, myconf, get_data, set_myconf
import hyperopt
import myfunctions

import inspect
import re

import pymongo

params = myfunctions.parse_params(sys.argv)
projectname = None
if "project" in params:
    projectname = params["project"]
    myconf = set_myconf(projectname)
data = get_data(projectname, params)

connect_string = "mongodb://" + data['mongodbmachine'] + ":" + str(data["mongodbport"]) + "/"
debug("Connecting to: " + connect_string)

myclient = pymongo.MongoClient(connect_string)
connect_to_db = params["project"]
print("Connecting to DB: " + connect_to_db + ", " + connect_string)
testdb = myclient[connect_to_db]
jobs = testdb["jobs"]

results = []
for line in jobs.find({}, {"result": 1, "misc.vals": 1, "state": 2}):
    results.append(line)

print("jobsindb: " + str(len(results)) + "\n")
print("maxevals: " + str(data["max_evals"]) + "\n")
