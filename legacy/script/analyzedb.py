#!/usr/bin/env python3

import pprint
import os
import sys
import time
from termcolor import colored

from mydebug import debug, error, myconf, get_data, set_myconf
import hyperopt
import myfunctions

import inspect
import re

import pymongo
import mongostuff

def donothing():
    return 1

params = myfunctions.parse_params(sys.argv)
projectname = None
if "project" in params:
    projectname = params["project"]
    myconf = set_myconf(projectname)
data = get_data(projectname, params)

data["mongodbport"] = 32335

mongostuff.start_mongo_db(projectname, data, 1)

connect_string = "mongodb://" + data['mongodbmachine'] + ":" + str(data["mongodbport"]) + "/"
debug("Connecting to: " + connect_string)

myclient = pymongo.MongoClient(connect_string)
connect_to_db = params["project"]
print("Connecting to DB: " + connect_to_db + ", " + connect_string)
testdb = myclient[connect_to_db]
jobs = testdb["jobs"]

jobtimes = []
times = []
results = []
for line in jobs.find({}, {"result": 1, "misc.vals": 1}):
    calculation_time = -1
    starttime = None
    endtime = None
    result = None
    try:
        calculation_time = line["result"]["calculation_time"]
        starttime = line["result"]["starttime"]
        endtime = line["result"]["endtime"]
        result = {"all_outputs": line["result"]["all_outputs"], "vals": line["misc"]["vals"]} 
    except Exception as e:
        donothing()
        print("ERROR: " + str(e))
    if starttime is not None:
        #print("starttime: " + str(starttime) + ", endtime: " + str(endtime))
        jobtimes.append(calculation_time)
        times.append(starttime)
        times.append(endtime)
        results.append(result)

if len(jobtimes):
    avg_runtime = sum(jobtimes) / float(len(jobtimes))
    print("avg: " + str(round(avg_runtime)))
    mintime = min(times)
    maxtime = max(times)
    print("min-time: " + str(round(mintime)))
    print("max-time: " + str(round(maxtime)))
    timediff = maxtime - mintime
    print("wholetime: " + str(round(timediff)))

if len(results):
    best_result = {}
    i = 0
    minimum_result = float("inf")
    while i < len(results):
        try:
            if float(results[i]["all_outputs"]["RESULT"]) < float(minimum_result):
                minimum_result = float(results[i]["all_outputs"]["RESULT"])
                best_result = results[i]
        except Exception as e:
            donothing()
            print("ERROR: " + str(e))
        i = i + 1

    maxdim = 0
    for key in best_result["all_outputs"]:
        print(key + ": " + str(best_result["all_outputs"][key]))
    for key in best_result["vals"]:
        if key.startswith("x"):
            thisdim = int(re.match('x_([0-9]+)$', key).group(1))
            if thisdim > maxdim:
                maxdim = thisdim
            print(key + ": " + str(float(best_result["vals"][key][0])))
    
    print("maxdim: " + str(maxdim))
    print("numofjobs: " + str(i))

mongostuff.shut_down_mongodb(projectname)
