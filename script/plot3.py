#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import pprint

import sys
import os
import math

import matplotlib

import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import pymongo
import myfunctions
import mydebug
from mydebug import debug, get_data, debug_xtreme
import numberstuff
import omnioptstuff
import mongostuff

import decimal

ctx = decimal.Context()
import re

import logging
import time
import datetime

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True

import math
import os
import psutil

def p (percent, status):
    if os.getenv('DISPLAYGAUGE', None) is not None:
        stderr("PERCENTGAUGE: %d" % percent)
        stderr("GAUGESTATUS: %s" % status)

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def show_mem_usage():
    process = psutil.Process(os.getpid())
    return convert_size(process.memory_info().rss)

def stderr (msg):
    sys.stderr.write(msg + "\n")
    sys.stdout.flush()

def readable_time (x):
    return str(datetime.timedelta(seconds=x))

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

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

def autoround (number):
    if number == float("inf") or number == float("-inf"):
        return number
    else:
        if number - int(number) == 0:
            return int(number)
        else:
            number_str = float_to_str(number)
            m = REMatcher(number_str)

            max_number_of_digits = 2
            if os.environ.get('NONZERODIGITS') is not None:
                max_number_of_digits = os.environ.get('NONZERODIGITS')

            if m.match(r"(.*\.0*[1-9]{1," + str(max_number_of_digits) + "}).*"):
                return float(m.group(1))
        return number

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

def _plotthis_single_axis(x, loss, axarr):
    index_of_min_value = numberstuff.get_index_of_minimum_value(loss)

    min_x_str = str(autoround(x[index_of_min_value]))

    if os.getenv("HIDEMAXVALUESINPLOT", None) is not None:
        min_at_desc = " (min: " + min_x_str + ")"
    else:
        index_of_max_value = numberstuff.get_index_of_maximum_value(loss)
        max_x_str = str(autoround(x[index_of_max_value]))
        min_at_desc = " (min: " + min_x_str + ", max: " + max_x_str + ")"

    try:
        for ax in axarr.flat:
            ax.set_xlabel(omnioptstuff.get_axis_label(myconf, 1))
            ax.set_ylabel("loss")
    except Exception as error:
        print(error)

    x = omnioptstuff.get_parameter_name_from_value(myconf, 1, x)

    plt.plot(x, loss, 'x', color='black', alpha=0.3);

def _plotthis(axarr, axis, x, y, e, parameter, switchaxes):
    if switchaxes:
        x, y = y, x
        axis[0], axis[1] = axis[1], axis[0]

    # Correct for hp.choice und hp.choiceint-pseudo selector
    if os.environ.get('SWITCHXY') == 1:
        x = omnioptstuff.get_parameter_value(myconf, axis[1], x)
        y = omnioptstuff.get_parameter_value(myconf, axis[0], y)
    else:
        x = omnioptstuff.get_parameter_value(myconf, axis[0], x)
        y = omnioptstuff.get_parameter_value(myconf, axis[1], y)

    string_desc = omnioptstuff.get_axis_label(myconf, axis[0]) + ' - ' + omnioptstuff.get_axis_label(myconf, axis[1])

    index_of_min_value = numberstuff.get_index_of_minimum_value(e)

    min_x_str = str(autoround(x[index_of_min_value]))
    min_y_str = str(autoround(y[index_of_min_value]))

    if os.getenv("HIDEMAXVALUESINPLOT", None) is not None:
        min_at_desc = " (min: " + min_x_str + ", " + min_y_str + ")"
    else:
        index_of_max_value = numberstuff.get_index_of_maximum_value(e)
        max_x_str = str(autoround(x[index_of_max_value]))
        max_y_str = str(autoround(y[index_of_max_value]))
        min_at_desc = " (min: " + min_x_str + ", " + min_y_str + ", max: " + max_x_str + ", " + max_y_str + ")"

    size_in_px = 7
    if os.environ.get('BUBBLESIZEINPX') is not None:
        size_in_px = int(os.environ.get('BUBBLESIZEINPX'))

    sc = axarr.scatter(x, y, c=e, s=size_in_px, cmap='jet', edgecolors="none")
    plt.colorbar(sc, ax=axarr)
    axarr.grid()
    axarr.autoscale(enable=True, axis='both', tight=True)
    axarr.relim()
    axarr.autoscale_view()

    stretch_factor = 0.1

    min_x = numberstuff.get_min_value(x)
    max_x = numberstuff.get_max_value(x)
    width = max_x - min_x
    additional_width = width * stretch_factor

    min_y = numberstuff.get_min_value(y)
    max_y = numberstuff.get_max_value(y)
    height = max_y - min_y
    additional_height = height * stretch_factor

    axarr.set_xlim(min_x - additional_width, max_x + additional_width);
    axarr.set_ylim(min_y - additional_height, max_y + additional_height);

    axarr.set_xlabel(omnioptstuff.get_axis_label(myconf, axis[0]))
    axarr.set_ylabel(omnioptstuff.get_axis_label(myconf, axis[1]))
    axarr.set_title(string_desc + ', ' + parameter + ": " + min_at_desc, fontsize=9)

def __check_data(axis, data, parameter):
    if axis[0] == 0 or axis[1] == 0:
        raise Exception("Axis must begin at 1, not 0!")

    if not "dimensions" in data:
        raise Exception("The data must contain a subdictionary called `dimensions`")

    if not str(axis[0]) in data["dimensions"]:
        raise Exception("The data must contain a subdictionary called `dimensions` with values for the axis " + str(axis[0]) + " (" + omnioptstuff.get_axis_label(myconf, axis[0]) + ")")

    if not str(axis[1]) in data["dimensions"]:
        raise Exception("The data must contain a subdictionary called `dimensions` with values for the axis " + str(axis[1]) + " (" + omnioptstuff.get_axis_label(myconf, axis[1]) + ")")

    if not parameter in data:
        raise Exception("The data must contain a subarray called `" + parameter + "`")

    if not len(axis) == 2:
        raise Exception("Can only plot 2 dimensions, not more or less!")

    data1 = data["dimensions"][str(axis[0])]
    data2 = data["dimensions"][str(axis[1])]
    heatmap = data[parameter]

    if len(data1) != len(data2):
        raise Exception("Both axis must contain the same number of data!")

    if len(data1) != len(heatmap):
        raise Exception("All axes must contain the same number of data as the parameter-array!")

def plotthis(axarr, axis, data, parameter, switchaxes):
    __check_data(axis, data, parameter)
    data1 = data["dimensions"][str(axis[0])]
    debug_xtreme("plotthis, data1: " + str(data1))
    data2 = data["dimensions"][str(axis[1])]
    debug_xtreme("plotthis, data2: " + str(data2))
    heatmap = data[parameter]
    _plotthis(axarr, [int(axis[0]), int(axis[1])], data1, data2, heatmap, parameter, switchaxes)

def plotdata(data, parameter, switchaxes):
    keys = list(data["dimensions"].keys())

    permutations = numberstuff.findsubsets(keys, 2)
    number_of_permutations = len(permutations)

    layout = numberstuff.get_largest_divisors(number_of_permutations)
    if layout is None:
        raise Exception("ERROR getting the layout: get_largest_divisors(" + str(number_of_permutations) + ") = " + str(layout))

    if len(keys) == 1 or len(keys) == 2:
        layout["x"] = 1
        layout["y"] = 1
    else:
        if layout["x"] == 1:
            layout["x"] = 2

        if layout["y"] == 1:
            layout["y"] = 2

    stderr("x: %d, y: %d" % (layout["y"], layout["x"]))
    f, axarr = plt.subplots(layout["x"], layout["y"], squeeze=False)
    f.set_size_inches(20, 15, forward=True)
    plt.subplots_adjust(wspace=0.15, hspace=0.75)

    permutations_array = []
    for item in sorted(permutations):
        permutations_array.append(item)

    i = 0
    if len(keys) == 1:
        _plotthis_single_axis(data["dimensions"]["1"], data[parameter], axarr)
    else:
        for row in axarr:
            for col in row:
                if i < len(permutations_array):
                    try:
                        plotthis(col, permutations_array[i % len(permutations_array)], data, parameter, switchaxes)
                    except Exception as e: 
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        stderr(e)
                        col.axis('off')
                        stderr("%d not in permutations_array" % i)
                        sys.exit(4)
                else:
                        col.axis('off')
                i = i + 1

def main():
    global params
    p(60, "Started main-function, collecting data")
    stderr("main")
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

    stderr("StartMongoDB: " + str(start_mongo_db))
    stderr(params["mongodbmachine"])
    stderr(params["mongodbport"])

    #mydebug.set_myconf(projectname)

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

    show_failed_jobs_in_plot = os.getenv("SHOWFAILEDJOBSINPLOT", "0")
    stderr("show_failed_jobs_in_plot: " + show_failed_jobs_in_plot)

    default_value_for_defective_runs = float('inf')
    search_dict = {"result.status": "ok"}

    number_of_frames = int(os.getenv("NUMBEROFFRAMES", 1))

    earliest_time = None
    latest_time = None
    timeskip_per_frame = None
    runtime = None

    earliest = jobs.find({"result.endtime": { "$exists": "true" }}, {"result.endtime": 1}).sort("result.endtime", 1).limit(1)

    for doc in earliest:
        earliest_time = int(doc["result"]["endtime"])

    if number_of_frames == 1:
        latest_time = earliest_time
    else:
        latest = jobs.find({"result.endtime": { "$exists": "true" }}, {"result.endtime": 1}).sort("result.endtime", -1).limit(1)
        for doc in latest:
            latest_time = int(doc["result"]["endtime"])

    if earliest_time is None or latest_time is None:
        stderr("Earliest time or latest_time is none (which means that either the DB is defect, got deleted between the two requests or something went horribly wrong some way I cannot think about right now)")
        sys.exit(3)

    if number_of_frames < 1:
        stderr("ERROR: number of frames cannot be smaller than 1")
        sys.exit(5)
    elif number_of_frames == 1:
        timeskip_per_frame = 1
    else:
        runtime = int(latest_time - earliest_time)
        timeskip_per_frame = int(runtime / number_of_frames)

    if show_failed_jobs_in_plot == "1":
        search_dict = {}
        default_value_for_defective_runs = 999999

    p(90, "Initialied variables")

    try:
        values = []
        i = 0
        stderr("Getting values from DB")
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
            stderr("Current's script memory footprint: " + str(show_mem_usage()))
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
            #stderr("TAKEN TIME of data_collection: %f" % float(float(time.time()) - float(data_collection_start)))

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
                        this_val = omnioptstuff.get_parameter_value(myconf, x_number, best_value_data["misc"]["vals"][item][0])
                        try:
                            rounded_number = str(autoround(float(this_val)))
                        except Exception as error:
                            rounded_number = this_val
                        var_for_title.append(omnioptstuff.get_axis_label(myconf, x_number) + " = " + rounded_number)
                        x_number = x_number + 1
                title = title + ', '.join(var_for_title) + ") = " + str(autoround(float(best_value_data["result"]["all_outputs"][parameter])))
                title = title + "\nNumber of evals: " + str(number_of_values) + ", " + "Number of dimensions: " + str(len(dimensions.keys()))
                title = title + ", Project: " + str(projectname)

            if plot_path is not None:
                matplotlib.use("Agg") # No x11 needed
            stderr("TAKEN TIME of title_creation: %f" % float(float(time.time()) - float(title_creation_start)))

            plot_data_start_time = time.time()

            plotdata(data, parameter, params["switchaxes"])
            stderr("TAKEN TIME of plotdata: %f" % float(float(time.time()) - float(plot_data_start_time)))

            stderr(title)

            plt.suptitle(title)

            write_data_start = time.time()
            if number_of_values != 0:
                if plot_path is None:
                    stderr("No PLOTPATH defined")
                    if os.getenv('DISPLAY', None) is None:
                        stderr("!!!!!!!!!!!!!!!!!!! No DISPLAY-Variable found. Is ssh running with -x?")
                        sys.exit(6)
                    p(99, "Finished data collection, starting Interface")
                    stderr("Showing graph")
                    plt_show_value = plt.show()
                else:
                    stderr("Writing to %s" % plot_path_j_str)
                    graph_width = int(os.getenv("SVGEXPORTSIZE", 2000))
                    dpi_number = int(os.getenv("DPISIZE", 600))
                    try:
                        p(99, "Finished data collection, writing file")
                        plt.savefig(plot_path_j_str, format=os.getenv("EXPORT_FORMAT", "svg"), width=graph_width, dpi=dpi_number)
                    except Exception as exception:
                        stderr("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        stderr(e)
                        stderr("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        sys.exit(10)
                    stderr("Saved file to %s" % (plot_path_j_str))
            #else:
            #    sys.stderr("No values found")
            #stderr("TAKEN TIME of write_data: %f" % float(float(float(time.time()) - float(write_data_start))))
            plt.close('all')
            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)
            times = times[-10:]

    except Exception as exception:
        stderr("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        stderr(str(exc_type) + " " + str(fname) + " " + str(exc_tb.tb_lineno))
        stderr(str(exception))
        sys.exit(100)

main()
