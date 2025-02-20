#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import pprint

import sys
import os

import pymongo
import myfunctions
import mydebug
from mydebug import debug, get_data, debug_xtreme
import numberstuff
import omnioptstuff
import mongostuff
import traceback

import decimal

ctx = decimal.Context()
import re

import logging
import time
import datetime

import os

def dier (msg):
    pprint.pprint(msg)
    sys.exit(1)

def p (percent, status):
    if os.getenv('DISPLAYGAUGE', None) is not None:
        stderr("PERCENTGAUGE: %d" % percent)
        stderr("GAUGESTATUS: %s" % status)

def stderr (msg):
    sys.stderr.write(msg + "\n")
    sys.stdout.flush()

def readable_time (seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

class REMatcher(object):
    def __init__(self, matchstring):
        self.matchstring = matchstring

    def match(self,regexp):
        self.rematch = re.match(regexp, self.matchstring)
        return bool(self.rematch)

    def group(self,i):
        return self.rematch.group(i)

def split(word):
    return [char for char in word] 

def float_to_str(f):
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

global myconf
params = myfunctions.parse_params(sys.argv)
if "project" in params:
    projectname = params["project"]
    myconf = mydebug.set_myconf(projectname)
myconf = mydebug.set_myconf(projectname)

def main():
    global params
    p(60, "Started main-function, collecting data")
    parameter = "loss"
    if "parameter" in params and params["parameter"] is not None:
        parameter = params["parameter"]
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
    if "setmongoperparameter" in params and params["setmongoperparameter"] is not None and params["setmongoperparameter"] != "":
        if "mongodbmachine" in params and params["mongodbmachine"] is not None and params["mongodbmachine"] != "":
            data["mongodbmachine"] = params["mongodbmachine"]
            start_mongo_db = False
            if "mongodbport" in params and params["mongodbport"] is not None and params["mongodbport"] != "":
                data["mongodbport"] = params["mongodbport"]
                start_mongo_db = False

    if start_mongo_db:
        debug("Start mongo DB")
        mongostuff.start_mongo_db(projectname, data)

    p(70, "Started MongoDB if neccessary")

    connect_string = "mongodb://" + data['mongodbmachine'] + ":" + str(data["mongodbport"]) + "/"
    debug("Connecting to: " + connect_string)

    client = pymongo.MongoClient(connect_string)
    connect_to_db = params["project"]
    debug("Connecting to DB: " + connect_to_db)
    db = client[connect_to_db]
    client.admin.command({'setParameter': 1, 'internalQueryExecMaxBlockingSortBytes': 10*335544320})
    jobs = db["jobs"]
    p(80, "Connected to MongoDB")

    default_value_for_defective_runs = float('inf')
    search_dict = {"result.status": "ok"}

    number_of_frames = int(os.getenv("NUMBEROFFRAMES", 1))

    earliest_time = None
    latest_time = None
    timeskip_per_frame = None
    runtime = None

    earliest = jobs.find({"result.endtime": { "$exists": "true" }, "result.starttime": { "$exists": "true" }}, {"result.starttime": 1}).sort("result.starttime", 1).limit(1)

    for doc in earliest:
        earliest_time = int(doc["result"]["starttime"])

    latest_real = None

    latest = jobs.find({"result.endtime": { "$exists": "true" }}, {"result.endtime": 1}).sort("result.endtime", -1).limit(1)
    for doc in latest:
        latest_real = int(doc["result"]["endtime"])

    if number_of_frames == 1:
        latest_time = earliest_time
    else:
        latest_time = latest_real
    if earliest_time is None or latest_time is None:
        stderr("Earliest time or latest_time is none (which means that either the DB is defect, got deleted between the two requests or something went horribly wrong some way I cannot think about right now)")
        sys.exit(3)

    runtime = int(latest_real - earliest_time)

    if number_of_frames < 1:
        stderr("ERROR: number of frames cannot be smaller than 1")
        sys.exit(5)
    elif number_of_frames == 1:
        timeskip_per_frame = 1
    else:
        timeskip_per_frame = int(runtime / number_of_frames)

    p(90, "Initialied variables")

    try:
        values = []
        i = 0
        #stderr("Getting values from DB")
        for x in jobs.find(search_dict, {"misc.vals": 1, "result.starttime": 1, "result.endtime": 1, "result.all_outputs": 1, "misc.vals": 1}).sort("result.all_outputs." + str(parameter), pymongo.DESCENDING):
            values.append(x)               
            i = i + 1

        client.close()

        j = 0
        current_time = earliest_time
        times = []
        p(95, "Got data")
        while j < number_of_frames:
            e = []
            dimensions = {}

            if number_of_frames > 1:
                stderr("time remaining: %s (%d), time_skip: %s" % (readable_time(latest_time - current_time), latest_time - current_time, readable_time(timeskip_per_frame)))

            start_time = time.time()
            number_of_values = 0
            j += 1
            plot_path = os.getenv("PLOTPATH", None)
            if plot_path == "":
                plot_path = None
            plot_path_j_str = None
            if plot_path is not None: 
                j_str = str("{:05d}".format(int(j)))
                try:
                    plot_path_j_str = plot_path % j_str
                except Exception as error:
                    plot_path_j_str = plot_path
                if os.path.isfile(plot_path_j_str):
                    stderr("The file " + str(plot_path_j_str) + " already exists. Skipping it")
                    continue

            best_value_data = None
            best_value = float("inf")

            if number_of_frames > 1:
                stderr("\n")
                if len(times) == 0:
                    stderr("%d of %d..." % (j, number_of_frames))
                else:
                    frames_left = number_of_frames - j
                    avg_time = int((sum(times) / len(times)) * 1.5)
                    seconds_left = frames_left * avg_time

                    time_left = readable_time(seconds_left)

                    stderr("%d of %d, AVG-time: %d, frames left: %d, ETA: %s..." % (j, number_of_frames, avg_time, frames_left, time_left))
            current_time += timeskip_per_frame

            data_collection_start = time.time()
            for x in values:
                value = default_value_for_defective_runs
                thistime = None

                try:
                    thistime = x["result"]["starttime"]
                    if float(thistime) < 0:
                        thistime = None
                except KeyError:
                    thistime = None

                try:
                    value = x["result"]["all_outputs"][parameter]
                    if float(value) < 0:
                        value = default_value_for_defective_runs
                except KeyError:
                    value = default_value_for_defective_runs

                if (params["maxvalue"] is None or value < params["maxvalue"]) and (params["maxtime"] is None or int(thistime) <= params["maxtime"]) and (number_of_frames == 1 or int(thistime) <= current_time):
                    debug("Appending " + str(value) + " to e")
                    if value < best_value:
                        best_value_data = x
                    e.append(float(value))

                    debug("Starting going through vals!")
                    i = 1
                    vals = x["misc"]["vals"]
                    for val in sorted(vals):
                        if val.startswith("x"):
                            mydebug.debug_xtreme(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> val: " + val)
                            thisval = vals[val][0]
                            if "int" in params and params["int"]:
                                thisval = int(round(thisval)) 
                            debug("thisval = " + str(thisval))
                            if str(i) not in dimensions:
                                dimensions[str(i)] = []
                            temp = dimensions[str(i)]
                            temp.append(thisval)
                            i = i + 1
                    number_of_values = number_of_values + 1

                    debug("End of data-collection")

            data = {"dimensions": dimensions, parameter: e}

            title = "f(x_0, x_1, x_2, ...) = " + parameter + ", min at f(";

            title_creation_start = time.time()
            title = 'Best value not known!'
            if best_value_data is not None:
                title = projectname + '('
                var_for_title = []
                x_number = 1
                for item in sorted(best_value_data["misc"]["vals"]):
                    if item.startswith("x"):
                        this_value = omnioptstuff.get_parameter_value(myconf, x_number, best_value_data["misc"]["vals"][item][0])
                        try:
                            rounded_number = float(this_value)
                            scientific_notation = os.getenv("SCIENTIFICNOTATION", 0)
                            if scientific_notation != 0 and scientific_notation != "0":
                                rounded_number = format(rounded_number, "." + str(scientific_notation) + "E")

                        except Exception as error:
                            rounded_number = this_value
                        var_for_title.append(str(omnioptstuff.get_axis_label(myconf, x_number)) + " = " + str(rounded_number))
                        x_number = x_number + 1
                title = title + ', '.join(var_for_title) + ") = " + str(float(best_value_data["result"]["all_outputs"][parameter]))
                title = title + "\nProject: " + str(projectname)
                title = title + ", Number of evals: " + str(number_of_values) + ", " + "Number of dimensions: " + str(len(dimensions.keys()))

            if runtime is not None and runtime != 0:
                title = title + ", Runtime: " + str(readable_time(runtime))
            
            nodes = myfunctions.get_nodes_from_project(params["projectdir"], params["project"])
            if nodes is not None:
                title = title + "\nNodes: " + ", ".join(nodes)


            print(title)

    except Exception as exception:
        stderr("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        stderr("STACK TRACE\n%s\nSTACK TRACE ENDED" % traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        stderr(str(exc_type) + " " + str(fname) + " " + str(exc_tb.tb_lineno))
        stderr(str(exception))
        sys.exit(100)

main()
