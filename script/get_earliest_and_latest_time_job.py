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

def main():
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

    connect_string = "mongodb://" + data['mongodbmachine'] + ":" + str(data["mongodbport"]) + "/"
    debug("Connecting to: " + connect_string)

    myclient = pymongo.MongoClient(connect_string)
    connect_to_db = params["project"]
    print("Connecting to DB: " + connect_to_db)
    db = myclient[connect_to_db]
    myclient.admin.command({'setParameter': 1, 'internalQueryExecMaxBlockingSortBytes': 10*335544320})
    jobs = db["jobs"]

    earliest = jobs.find({"result.endtime": { "$exists": "true" }}, {"result.endtime": 1}).sort("result.endtime", 1).limit(1)
    for doc in earliest:
        print("EARLIEST: " + str(int(doc["result"]["endtime"])) + "\n")
    latest = jobs.find({"result.endtime": { "$exists": "true" }}, {"result.endtime": 1}).sort("result.endtime", -1).limit(1)
    for doc in latest:
        print("LATEST: " + str(int(doc["result"]["endtime"])) + "\n")

main()
