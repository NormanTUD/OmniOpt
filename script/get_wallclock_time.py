#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
ml release/23.04 2>&1 | grep -v load
ml MongoDB/4.0.3 2>&1 | grep -v load
ml GCC/11.3.0 2>&1 | grep -v load
ml OpenMPI/4.1.4 2>&1 | grep -v load
ml Hyperopt/0.2.7 2>&1 | grep -v load
ml matplotlib/3.5.2 2>&1 | grep -v load
"""

import sys
import os
import math

import pymongo
import myfunctions
import mydebug
from mydebug import debug, get_data, debug_xtreme
import numberstuff
import omnioptstuff
import mongostuff
import time

global myconf
params = myfunctions.parse_params(sys.argv)
if "project" in params:
    projectname = params["project"]
    myconf = mydebug.set_myconf(projectname)
myconf = mydebug.set_myconf(projectname)

"""
#LOAD:
export MODULEPATH=/sw/modules/taurus/applications
export MODULEPATH=$MODULEPATH:/sw/modules/taurus/tools
export MODULEPATH=$MODULEPATH:/sw/modules/taurus/libraries
export MODULEPATH=$MODULEPATH:/sw/modules/taurus/compilers
export MODULEPATH=$MODULEPATH:/opt/modules/modulefiles
export MODULEPATH=$MODULEPATH:/sw/modules/taurus/environment

eval `/usr/share/lmod/lmod/libexec/lmod $SHELL load modenv/both`
eval `/usr/share/lmod/lmod/libexec/lmod $SHELL load modenv/eb`
eval `/usr/share/lmod/lmod/libexec/lmod $SHELL load modenv/hyperopt`

"""

def readable_time (seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def p (percent, status):
    if os.getenv('DISPLAYGAUGE', None) is not None:
        stderr("PERCENTGAUGE: %d" % percent)
        stderr("GAUGESTATUS: %s" % status)

def stderr (msg):
    sys.stderr.write(msg + "\n")
    sys.stdout.flush()

def main():
    p(20, "Loading main script")
    params = myfunctions.parse_params(sys.argv)
    if "project" in params:
        projectname = params["project"]
        myconf = mydebug.set_myconf(projectname)
    else:
        raise Exception("No project name found! Use --project=abcdef as parameter!")

    projectname = None
    if "project" in params:
        projectname = params["project"]
    data = get_data(projectname, params)
    start_mongo_db = True
    if "mongodbmachine" in params and params["mongodbmachine"] is not None and params["mongodbmachine"] != "127.0.0.1":
        data["mongodbmachine"] = params["mongodbmachine"]
        start_mongo_db = False
    if "mongodbport" in params and params["mongodbport"] is not None:
        data["mongodbport"] = params["mongodbport"]

    if start_mongo_db:
        mongostuff.start_mongo_db(projectname, data)

    p(30, "Connecting to Database")
    connect_string = "mongodb://" + data['mongodbmachine'] + ":" + str(data["mongodbport"]) + "/"
    debug("Connecting to: " + connect_string)

    myclient = pymongo.MongoClient(connect_string)
    connect_to_db = params["project"]
    sys.stderr.write("Connecting to DB: " + connect_to_db)
    db = myclient[connect_to_db]
    myclient.admin.command({'setParameter': 1, 'internalQueryExecMaxBlockingSortBytes': 10*335544320})
    jobs = db["jobs"]
    p(40, "Connected to Database")

    earliest_time = None
    latest_time = None

    p(50, "Finding earliest and latest jobs")
    earliest = jobs.find({"result.endtime": { "$exists": "true" }, "result.starttime": { "$exists": "true" }}, {"result.starttime": 1}).sort("result.starttime", 1).limit(1)
    for doc in earliest:
        earliest_time = int(doc["result"]["starttime"])
    latest = jobs.find({"result.endtime": { "$exists": "true" }}, {"result.endtime": 1}).sort("result.endtime", -1).limit(1)
    for doc in latest:
        latest_time = int(doc["result"]["endtime"])

    if earliest_time is None:
        raise Exception("earliest_time is None")

    if latest_time is None:
        raise Exception("latest_time is None")
    p(90, "Found earliest and latest jobs")

    runtime_in_seconds = latest_time - earliest_time

    wallclock_time = readable_time(runtime_in_seconds)

    print("WallclockTime: %s (%s seconds)" % (wallclock_time, str(runtime_in_seconds)))

main()
