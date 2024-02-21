import logging
logging.getLogger('matplotlib.font_manager').disabled = True
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import pprint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
import csv
import datetime
import time
import re
import dateutil
import sys
import os
import omnioptstuff
from glob import glob

def stderr (msg):
    sys.stderr.write(msg + "\n")
    sys.stdout.flush()

def p (percent, status):
    if os.getenv('DISPLAYGAUGE', None) is not None:
        stderr("PERCENTGAUGE: %d" % percent)
        stderr("GAUGESTATUS: %s" % status)

p(11, "Using TkAgg")
try:
    matplotlib.use('TkAgg')
except:
    print("Using TkAgg failed. Trying to use Agg...")
    matplotlib.use('Agg')

p(12, "Used TkAgg")
print_to_svg = 0


projectname = sys.argv[1]
search_folder = sys.argv[2]
try:
    print_to_svg = sys.argv[3]
except Exception as e:
    print_to_svg = os.getenv("PLOTPATH", None)
    if print_to_svg is None:
        print("No SVG file given, using X11 Output")

csvs = glob(search_folder + "/nvidia-*/gpu_usage.csv")
all_gpus = 0
if os.environ.get('SHOWALLGPUS') is not None:
    all_gpus = 1

p(30, "Getting data")
valid_gpus = omnioptstuff.get_valid_gpus_from_logfolder(search_folder)
if len(valid_gpus) == 0:
    all_gpus = 1

graph_data = {}

def die (data):
    pprint.pprint(data)
    sys.exit(1)

def dier (data):
    die(data)

if 0 == len(csvs):
    die("No CSV files under " + search_folder)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def get_nearest_prime_factors (n):
    if n == 1:
        return [1, 1]
    if n == 2:
        return [2, 1]
    pf = prime_factors(n)
    if len(pf) == 2:
        return pf
    if len(pf) == 3:
        pf = (pf[0]*pf[1], pf[2])
        if pf[0] < pf[1]:
            pf[0], pf[1] = pf[1], pf[0]
        return pf
      
    pf = get_nearest_prime_factors(n + 1)
    if pf[0] < pf[1]:
        pf[0], pf[1] = pf[1], pf[0]
    return pf


def clean_percentage_data (data):
    data = re.sub(r"^\s*", "", data)
    data = re.sub(r"\s*$", "", data)
    data = re.sub(r"\s*%\s*", "", data)
    return data

def clean_mib_data (data):
    data = re.sub("\s*MiB", "", data)
    return data

def get_servername_from_path (path):
    servername = re.sub(r".*/nvidia-", "", path)
    servername = re.sub(r"/gpu_usage.csv", "", servername)
    return servername

number_of_servers = 0
for csvpath in csvs:
    with open(csvpath, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if not re.match(r"timestamp", row[0]): # skip headlines
                timestamp = row[0]
                gpu_name = row[1]
                bus_id = row[2]
                driver_version = row[3]
                pstate = row[4]
                link_gen_max = row[5]
                link_gen_current = row[6]
                temperature = row[7]
                utilization_gpu = clean_percentage_data(row[8])
                utilization_memory = clean_percentage_data(row[9])
                memory_total = clean_mib_data(row[10])
                memory_free = clean_mib_data(row[11])
                memory_used = clean_mib_data(row[12])
                bus_id = str.replace(bus_id, " ", "")
                servername = get_servername_from_path(csvpath)
                gpuservername = servername + "-" + bus_id

                if (len(valid_gpus) and omnioptstuff.is_valid_gpu(valid_gpus, servername, bus_id)) or all_gpus == 1:
                    if not gpuservername in graph_data:
                        graph_data[gpuservername] = {}
                        graph_data[gpuservername]["dates"] = []
                        graph_data[gpuservername]["gpu_utilization"] = []
                        graph_data[gpuservername]["mem_utilization"] = []

                    graph_data[gpuservername]["dates"].append(dateutil.parser.parse(timestamp))
                    graph_data[gpuservername]["gpu_utilization"].append(int(utilization_gpu))
                    graph_data[gpuservername]["mem_utilization"].append(int(utilization_memory))

empty_plots = []
p(70, "Got data")
for gpuservername in graph_data:
    if 0 == sum(graph_data[gpuservername]["mem_utilization"]) == sum(graph_data[gpuservername]["gpu_utilization"]):
        empty_plots.append(gpuservername)

for gpuservername in empty_plots:
    del graph_data[gpuservername]

number_of_servers = len(graph_data)
max_subplot_x, max_subplot_y = get_nearest_prime_factors(number_of_servers)
print("max_subplot_x: %s, max_subplot_y: %s" % (max_subplot_x, max_subplot_y))

p(99, "Starting plot")
fig, axs = plt.subplots(max_subplot_x, max_subplot_y, squeeze=False, figsize=(18,8))
if print_to_svg:
    fig.set_size_inches(20, 15, forward=True)

fig.canvas.set_window_title(projectname)

fig.suptitle(projectname)

x, y = 0, 0
plotted_graphs = 0

shown_legend = 0
for gpuservername in graph_data:
    plotted_graphs += 1
    if plotted_graphs <= number_of_servers:
        if len(graph_data[gpuservername]["gpu_utilization"]) and len(graph_data[gpuservername]["mem_utilization"]):
            xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
            axs[x, y].xaxis.set_major_formatter(xfmt)
            if len(graph_data[gpuservername]["gpu_utilization"]):
                axs[x, y].plot(graph_data[gpuservername]["dates"], graph_data[gpuservername]["gpu_utilization"], label='GPU-Utilization', linewidth=0.3)
            if len(graph_data[gpuservername]["mem_utilization"]):
                axs[x, y].plot(graph_data[gpuservername]["dates"], graph_data[gpuservername]["mem_utilization"], label='Mem-Utilization', linewidth=0.3)
            axs[x, y].set_title(gpuservername)
            axs[x, y].set_xlabel("time")
            axs[x, y].set_ylabel("in %")
            if shown_legend == 0:
                axs[x, y].legend()
                shown_legend = 1
        else:
            axs[x, y].axis('off')
            axs[x, y].set_visible(False)
    else:
        print("DONT SHOW ME x = %d, y = %d" % (x, y))
        axs[x, y].axis('off')
        axs[x, y].set_visible(False)

    x += 1
    if x >= max_subplot_x:
        x = 0
        y += 1

fig.autofmt_xdate()
fig.tight_layout()
if print_to_svg:
    plt.savefig(print_to_svg, format="svg")
else:
    plt.show()
p(100, "Done")
