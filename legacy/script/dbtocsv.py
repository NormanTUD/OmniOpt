#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys
from pprint import pprint

import pymongo
import myfunctions
import mydebug
from mydebug import debug, get_data, debug_xtreme
import re
import omnioptstuff
import filestuff
import mongostuff
import os

def stderr (msg):
    sys.stderr.write(msg + "\n")
    sys.stdout.flush()

def p (percent, status):
    if os.getenv('DISPLAYGAUGE', None) is not None:
        stderr("PERCENTGAUGE: %d" % percent)
        stderr("GAUGESTATUS: %s" % status)
        sys.stderr.flush()

seperator = ", "

global myconf
params = myfunctions.parse_params(sys.argv)
if "project" in params:
    projectname = params["project"]
    myconf = mydebug.set_myconf(projectname)
myconf = mydebug.set_myconf(projectname)

def die(x):
    pprint(x)
    sys.exit(1)

def remove_prefix(s, prefix):
    return s[len(prefix):] if s.startswith(prefix) else s

def main():
    p(25, "In Python-Script")
    params = myfunctions.parse_params(sys.argv)
    if "project" in params:
        projectname = params["project"]
        myconf = mydebug.set_myconf(projectname)
    else:
        raise Exception("No project name found! Use --project=abcdef as parameter!")

    if "seperator" in params:
        seperator = params["seperator"]

    projectname = None
    if "project" in params:
        projectname = params["project"]
    try:
        data = get_data(projectname, params)
    except Exception as e:
        print(e)
        sys.exit(1)
    if "mongodbmachine" in params:
        data["mongodbmachine"] = params["mongodbmachine"]
    if "mongodbport" in params:
        data["mongodbport"] = params["mongodbport"]

    p(30, "Starting MongoDB")
    mongostuff.start_mongo_db(projectname, data)
    p(40, "Started MongoDB")

    connect_string = "mongodb://" + data['mongodbmachine'] + ":" + str(data["mongodbport"]) + "/"
    myclient = pymongo.MongoClient(connect_string)
    connect_to_db = data["mongodbdbname"]
    testdb = myclient[connect_to_db]
    p(50, "Connected to MongoDB")

    makeint = params["int"]

    jobs = testdb["jobs"]

    times = []
    e = []
    dimensions = {}
    number_of_values = 0
    keys_for_header = []

    i = 0
    p(51, "Getting data")
    for x in jobs.find({"result.loss": { "$ne": float("inf") }}, {"result.all_outputs": 1, "misc.vals": 1, "book_time": 1}):
        if "misc" in x:
            if "vals" in x["misc"]:
                vals = x["misc"]["vals"]
                if "result" in x:
                    if "all_outputs" in x["result"]:
                        vals_2 = x["result"]["all_outputs"]

                        keys_for_header = keys_for_header + [*vals, *vals_2]
                        i = i + 1

    keys_for_header = set(keys_for_header)
    try:
        keys_for_header.remove("a")
    except Exception as error:
        True

    p(52, "Reading data")

    for x in jobs.find({"result.all_outputs": {"$exists":"true"}}, {"result.all_outputs": 1, "misc.vals": 1, "book_time": 1}):
        t = float("-inf")
        try:
            t = str(x["book_time"])
            t = re.sub(
                r"\..*$",
                "",
                t
            )
        except Exception as Error:
            t = float('-inf')
        times.append(t)

        value = float('inf')
        try:
            value = x["result"]["all_outputs"]
        except Exception as error:
            sys.stderr.write(pprint.pformat(x))
            value = float('inf')
        e.append(value)

        i = 1

        input_vals = x["misc"]["vals"]
        for key in input_vals:
            if key.startswith("x"):
                #old_input_key = input_vals[key][0]
                x_counter = int(remove_prefix(key, "x_")) + 1
                input_vals[key] = omnioptstuff.get_parameter_value(myconf, x_counter, input_vals[key][0])
                #sys.stderr.write("x_counter: %s, key: %s, input_vals[key] old: %s, input_vals[key] new: %s\n" % (str(x_counter), str(key), str(old_input_key), str(input_vals[key])))
        output_vals = x["result"]["all_outputs"]

        all_vals = {**input_vals,  **output_vals}

        for header in sorted(keys_for_header):
            thisval = "NaN"
            try:
                thisval = all_vals[header]
                try:
                    if isinstance(thisval, list):
                        thisval = thisval[0]
                except:
                    pass
            except:
                pass
            if header not in dimensions:
                dimensions[header] = []
            temp = dimensions[header]
            temp.append(thisval)
            i = i + 1

        number_of_values = number_of_values + 1

    p(80, "Got data")

    rename_keys = []
    for oldkey in dimensions:
        match = re.match('^x_(\d+)$', oldkey)
        if match:
            rename_keys.append(oldkey)

    if makeint:
        for xkey in rename_keys:
            dimensions[xkey] = [ '%.0f' % elem for elem in dimensions[xkey] ]

    for oldkey in rename_keys:
        match = re.match('^x_(\d+)$', oldkey)
        if match:
            newkey = omnioptstuff.get_axis_label(myconf, int(match.group(1)) + 1)
            if newkey not in dimensions:
                dimensions[newkey] = dimensions.pop(oldkey)

    heading = "time" + seperator + seperator.join(str(x) for x in sorted(dimensions.keys())) + "\n"
    csv_file = heading

    p(99, "Printing data")

    for row_id in range(0, number_of_values - 1):
        time = 'NaN'
        try:
            time = times[row_id]
        except KeyError:
            time = 'NaN'
        except (e): 
            sys.stderr.write("EXCEPTION 1: " + e)
        this_line_array = []
        for title in sorted(dimensions.keys()):
            try:
                this_line_array.append(dimensions[title][row_id])
            except Exception as e:
                sys.stderr.write("EXCEPTION 2: " + str(e))
        this_line = time + seperator +  seperator.join(str(x) for x in this_line_array)
        csv_file = csv_file + this_line + "\n"

    print(csv_file)
    if "filename" in params and params["filename"] is not None:
        filestuff.overwrite_file(params["filename"], csv_file)
        p(100, "Done")

main()
