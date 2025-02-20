#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import pprint

import sys
import os

import matplotlib

import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import pymongo
import myfunctions
import myregexps
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

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True

import os

def set_axlims(axarr, series, name, marginfactor):
    """
    Fix for a scaling issue with matplotlibs scatterplot and small values.
    Takes in a pandas series, and a marginfactor (float).
    A marginfactor of 0.2 would for example set a 20% border distance on both sides.
    Output:[bottom,top]
    To be used with .set_ylim(bottom,top)
    """

    p(98, "Getting min(series)")
    minv = min(i for i in series if i is not None)
    p(98, "Getting max(series)")
    maxv = max(i for i in series if i is not None)

    p(98, "Returning axarr if minv or maxv are string, else continue")
    if type(minv) == str or type(maxv) == str:
        p(98, "Returning axarr if minv or maxv are string")
        return axarr

    p(98, "datarange: maxv - minv")
    datarange = maxv - minv
    p(98, "Defining border")
    border = abs(datarange * marginfactor)
    p(98, "Setting maxlim")
    maxlim = maxv + border
    p(98, "Setting minlim")
    minlim = minv - border

    p(98, "Defining maxlim as float for .20f")
    maxlim = float(format(maxlim, ".20f"))
    p(98, "Defining minlim as float for .20f")
    minlim = float(format(minlim, ".20f"))

    stderr(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\nmin: %s, max: %s\n" % (str(minlim), str(maxlim)))

    if name == "x":
        axarr.set_xlim(minlim, maxlim);
    else:
        axarr.set_ylim(minlim, maxlim);

    return axarr

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

def is_number (strnum):
    this_match = re.match(myregexps.floating_number_limited, str(strnum))
    if this_match is None:
        return False
    else:
        return True

def is_integer (strnum):
    this_match = re.match(myregexps.integer_limited, str(strnum))
    if this_match is None:
        return False
    else:
        return True

def int_or_float_if_possible(value):
    if is_number(value):
        if is_integer(value):
            return int(value)
        else:
            return float(value)
    else:
        return value

def autoround_if_possible (value):
    if is_number(value):
        return str(autoround(float(value)))
    else:
        return value

global myconf
params = myfunctions.parse_params(sys.argv)
if "project" in params:
    projectname = params["project"]
    myconf = mydebug.set_myconf(projectname)
myconf = mydebug.set_myconf(projectname)

def _plotdata_singleaxis (x, loss, axarr):
    p(98, "Getting index of min data")
    index_of_min_value = numberstuff.get_index_of_minimum_value(loss)
    p(98, "Got index of min data")

    p(98, "Trying autorounded x string if possible")
    min_x_str = autoround_if_possible(x[index_of_min_value])
    p(98, "Autorounded x string if possible")

    if os.getenv("HIDEMAXVALUESINPLOT", None) is not None:
        p(98, "Defining min_at_desc (1)")
        min_at_desc = " (min: " + min_x_str + ")"
        p(98, "Defined min_at_desc (1)")
    else:
        p(98, "Defining index_of_max_value (2)")
        index_of_max_value = numberstuff.get_index_of_maximum_value(loss)
        p(98, "Defining max_x_str (2)")
        max_x_str = autoround_if_possible(x[index_of_max_value])
        p(98, "Defining min_at_desc (2)")
        min_at_desc = " (min: " + str(min_x_str) + ", max: " + str(max_x_str) + ")"
        p(98, "Defined min_at_desc (2)")

    try:
        p(98, "Trying to set axis data")
        for ax in axarr.flat:
            ax.set_xlabel(omnioptstuff.get_axis_label(myconf, 1))
            ax.set_ylabel("loss")
        p(98, "Setting axis data done")
    except Exception as error:
        stderr("STACK TRACE\n%s\nSTACK TRACE ENDED" % traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        stderr(str(exc_type) + " " + str(fname) + " " + str(exc_tb.tb_lineno))
        stderr(error)

    p(98, "Trying go get_parameter_name_from_value()")
    x = omnioptstuff.get_parameter_name_from_value(myconf, 1, x)
    p(98, "Got get_parameter_name_from_value()")

    plt.plot(x, loss, 'x', color='black', alpha=0.3);

def _plotdata(axarr, axis, data, parameter, switchaxes):
    p(98, "Starting basic-plot checks")
    if axis[0] == 0 or axis[1] == 0:
        raise Exception("Axis must begin at 1, not 0!")

    if not "dimensions" in data:
        raise Exception("The data must contain a subdictionary called `dimensions`")

    axis_warning = "The data must contain a subdictionary called `dimensions` with values for the axis "
    if not str(axis[0]) in data["dimensions"]:
        raise Exception(axis_warning + str(axis[0]) + " (" + omnioptstuff.get_axis_label(myconf, axis[0]) + ")")

    if not str(axis[1]) in data["dimensions"]:
        raise Exception(axis_warning + str(axis[1]) + " (" + omnioptstuff.get_axis_label(myconf, axis[1]) + ")")

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

    p(98, "Done with basic-plot checks")

    p(98, "Setting x, y and e data for this plot")

    x = data["dimensions"][str(axis[0])]
    y = data["dimensions"][str(axis[1])]
    d = data[parameter]
    axis = [int(axis[0]), int(axis[1])]

    if switchaxes or os.environ.get('SWITCHXY') == 1:
        x, y = y, x
        axis[0], axis[1] = axis[1], axis[0]

    string_desc = omnioptstuff.get_axis_label(myconf, axis[0]) + ' - ' + omnioptstuff.get_axis_label(myconf, axis[1])
    p(98, "Got axis description")

    index_of_min_value = numberstuff.get_index_of_minimum_value(d)
    p(98, "Got min value")

    for x_i in range(0, len(x)):
        if is_number(x[x_i]):
            x[x_i] = float(x[x_i])
        if is_number(y[x_i]):
            y[x_i] = float(y[x_i])

    min_x_str = autoround_if_possible(x[int(index_of_min_value)])
    min_y_str = autoround_if_possible(y[int(index_of_min_value)])

    p(98, "Got min_[xy]_str")

    if os.getenv("HIDEMAXVALUESINPLOT", None) is not None:
        min_at_desc = " (min: " + str(min_x_str) + ", " + str(min_y_str) + ")"
    else:
        index_of_max_value = numberstuff.get_index_of_maximum_value(d)
        max_x_str = autoround_if_possible(x[index_of_max_value])
        max_y_str = autoround_if_possible(y[index_of_max_value])
        min_at_desc = " (min: " + str(min_x_str) + ", " + str(min_y_str) + ", max: " + str(max_x_str) + ", " + str(max_y_str) + ")"

    size_in_px = 7
    p(98, "Got size_in_px")
    if os.environ.get('BUBBLESIZEINPX') is not None:
        size_in_px = int(os.environ.get('BUBBLESIZEINPX'))

    p(98, "Defining matplotlib data")
    sc = axarr.scatter(x, y, c=d, s=size_in_px, cmap='RdYlGn', edgecolors="none")

    p(98, "Defining colorbar")
    plt.colorbar(sc, ax=axarr)

    p(98, "Defining grid")
    axarr.grid()

    p(98, "Defining autoscale")
    axarr.autoscale(enable=True, axis='both', tight=True)

    p(98, "Defining relim")
    axarr.relim()

    p(98, "Defining autoscale_view")
    axarr.autoscale_view()

    p(98, "Defining axlims x")
    axarr = set_axlims(axarr, x, 'x', 0.1)

    p(98, "Defining axlims y")
    axarr = set_axlims(axarr, y, 'y', 0.1)

    try:
        p(98, "Defining ticklabels x")
        axarr.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

        p(98, "Defining ticklabels y")
        axarr.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    except Exception as e:
        print("!!! WARNING !!!")
        print(e)

    plt.margins(0)

    p(98, "Setting labels")
    axarr.set_xlabel(omnioptstuff.get_axis_label(myconf, axis[0]))
    axarr.set_ylabel(omnioptstuff.get_axis_label(myconf, axis[1]))
    axarr.set_title(string_desc + ', ' + parameter + ": " + min_at_desc, fontsize=9)
    p(98, "For these axis, everything has been set")

def plotdata(data, parameter, switchaxes):
    p(98, "Plotting data")
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
    p(98, "Got layout")

    #stderr("x: %d, y: %d" % (layout["y"], layout["x"]))
    f, axarr = plt.subplots(layout["x"], layout["y"], squeeze=False)
    f.set_size_inches(20, 15, forward=True)
    plt.subplots_adjust(wspace=0.15, hspace=0.75)

    permutations_array = []
    for item in sorted(permutations):
        permutations_array.append(item)

    i = 0
    if len(keys) == 1:
        p(98, "Plot single axis")
        _plotdata_singleaxis(data["dimensions"]["1"], data[parameter], axarr)
    else:
        p(98, "Plotting all axes")
        for row in axarr:
            for col in row:
                if i < len(permutations_array):
                    try:
                        _plotdata(col, permutations_array[i % len(permutations_array)], data, parameter, switchaxes)
                    except Exception as e: 
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        stderr(str(exc_type) + " " + str(fname) + " " + str(exc_tb.tb_lineno))
                        stderr(e)
                        col.axis('off')
                        stderr("%d not in permutations_array" % i)
                        sys.exit(4)
                else:
                        col.axis('off')
                i = i + 1

def main():
    global params
    p(60, "Started main-function")
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

    #stderr("StartMongoDB: " + str(start_mongo_db))
    #stderr(params["mongodbmachine"])
    #stderr(params["mongodbport"])

    #mydebug.set_myconf(projectname)

    if(start_mongo_db):
        p(65, "MongoDB needs to be started")
    else:
        p(65, "MongoDB doesn't need to be started")

    if start_mongo_db:
        p(67, "Starting mongo DB")
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
    #stderr("show_failed_jobs_in_plot: " + show_failed_jobs_in_plot)

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

    if show_failed_jobs_in_plot == "1":
        search_dict = {}
        default_value_for_defective_runs = 999999

    p(90, "Initialized variables")

    values = []
    i = 0
    p(92, "Getting values from DB")
    for x in jobs.find(search_dict, {"misc.vals": 1, "result.starttime": 1, "result.endtime": 1, "result.all_outputs": 1, "misc.vals": 1}).sort("result.all_outputs." + str(parameter), pymongo.DESCENDING):
        values.append(x)               
        i = i + 1

    client.close()

    j = 0
    current_time = earliest_time
    times = []
    p(93, "Got data")
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

        p(94, "Going through values")
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
            except KeyError:
                value = default_value_for_defective_runs

            if (params["maxvalue"] is None or value < params["maxvalue"]) and (params["maxtime"] is None or int(thistime) <= params["maxtime"]) and (number_of_frames == 1 or int(thistime) <= current_time):
                debug("Appending " + str(value) + " to e")
                if value < best_value:
                    best_value = float(x["result"]["all_outputs"]["RESULT"])
                    best_value_data = x
                e.append(float(value))

                debug("Starting going through vals!")
                i = 1
                vals = x["misc"]["vals"]
                for val in sorted(vals):
                    if val.startswith("x"):
                        dim_nr = int(val.replace("x_", ""))
                        mydebug.debug_xtreme(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> val: " + val)
                        thisval = vals[val][0]

                        thisval = omnioptstuff.get_parameter_value(myconf, dim_nr + 1, thisval)

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

        p(95, "Got all data")
        data = {"dimensions": dimensions, parameter: e}

        p(96, "Creating title")

        title = 'Best value not known!'
        if best_value_data is not None:
            p(96, "Got best_value_data")
            title = projectname + '('
            var_for_title = []
            for item in sorted(best_value_data["misc"]["vals"]):
                p(96, "Sorting through items")
                if item.startswith("x"):
                    x_number = int(item.replace("x_", ""))
                    p(96, "Got x_number")

                    value = best_value_data["misc"]["vals"][item][0]
                    p(96, "Got value")
                    this_value = int_or_float_if_possible(omnioptstuff.get_parameter_value(myconf, x_number + 1, value))
                    p(96, "Got this_value")

                    try:
                        rounded_number = autoround_if_possible(this_value)
                        scientific_notation = os.getenv("SCIENTIFICNOTATION", 0)
                        if scientific_notation != 0 and scientific_notation != "0":
                            rounded_number = format(float(rounded_number), "." + str(scientific_notation) + "E")
                        p(96, "Got rounded_number")
                    except Exception as error:
                        rounded_number = str(this_value)
                        p(96, "Got rounded_number (exception)")

                    this_axis_label = omnioptstuff.get_axis_label(myconf, x_number + 1)
                    p(96, "Got this_axis_label");

                    var_for_title.append(str(this_axis_label) + " = " + str(rounded_number))
                    p(96, "Got var_for_title");
            if best_value_data["result"] == {}:
                title = title + ', '.join(var_for_title) + ") = Not determinable"
            else:
                title = title + ', '.join(var_for_title) + ") = " + autoround_if_possible(best_value_data["result"]["all_outputs"][parameter])
            p(96, "Title Part I");
            title = title + "\nProject: " + str(projectname)
            p(96, "Title Part II");
            title = title + ", Number of evals: " + str(number_of_values) + ", " + "Number of dimensions: " + str(len(dimensions.keys()))
            p(96, "Title Part III");

        if runtime is not None and runtime != 0:
            title = title + ", Runtime: " + str(readable_time(runtime))
            p(96, "Title Part IV");

        if plot_path is not None:
            matplotlib.use("Agg") # No x11 needed

        p(97, "Title was created")
        plotdata(data, parameter, params["switchaxes"])

        stderr(title)

        plt.suptitle(title)
        plt.gcf().canvas.set_window_title(projectname)

        if number_of_values != 0:
            if plot_path is None:
                #stderr("No PLOTPATH defined")
                if os.getenv('DISPLAY', None) is None:
                    stderr("!!!!!!!!!!!!!!!!!!! No DISPLAY-Variable found. Is ssh running with -x?")
                    sys.exit(6)
                p(99, "Finished data collection, starting Interface")
                #stderr("Showing graph")
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
        else:
            sys.stderr("No values found")
        plt.close('all')
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        times = times[-10:]
main()
