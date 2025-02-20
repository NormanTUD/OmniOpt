from hyperopt import hp
import mypath
import os
import mydebug
import linuxstuff
import range_generator
import re
import myfunctions
import sys
import networkstuff
import myregexps
from os import path

import pprint
from collections import defaultdict
import re

class REMatcher(object):
    def __init__(self, matchstring):
        self.matchstring = matchstring

    def match(self,regexp):
        self.rematch = re.match(regexp, self.matchstring)
        return bool(self.rematch)

    def group(self,i):
        return self.rematch.group(i)

def get_slurm_id_from_logfolder (logfolder):
    regex = re.compile('^\d+$')

    for root, dirs, files in os.walk(logfolder):
        for thisdir in dirs:
            if regex.match(thisdir):
                return thisdir
    return None

def get_ipfiles_path_from_logfolder(logfolder):
    ipfiles_path = os.path.abspath(logfolder + "/../../ipfiles")
    if path.isdir(ipfiles_path):
        return ipfiles_path
    return None

def get_valid_gpus_from_logfolder(logfolder):
    slurm_id = get_slurm_id_from_logfolder(logfolder)
    valid_gpus = defaultdict(list)
    if slurm_id is not None:
        ipfiles_path = get_ipfiles_path_from_logfolder(logfolder)
        if ipfiles_path is not None:
            gpu_log_file = ipfiles_path + "/GPU_" + slurm_id
            if path.isfile(gpu_log_file):
                filehandle = open(gpu_log_file, "r")
                lines = filehandle.readlines()
                for line in lines:
                    #line like taurusml6-00000004:04:00.0
                    m = REMatcher(line)
                    if m.match(r"\s*(.*)-(.*)\s*"):
                        #print("HOSTNAME %s, GPU-Bus-ID: %s" % (m.group(1), m.group(2)))
                        hostname = m.group(1)
                        gpu_bus_id = m.group(2)
                        valid_gpus[hostname].append(gpu_bus_id)
    return valid_gpus

def is_valid_gpu(gpus, hostname, gpu):
        if hostname in gpus:
                if gpu in gpus[hostname]:
                    return 1
        return 0

def cprint (param, show_live_output=1):
    if str(show_live_output) == "1":
        print('\x1b[1;31m' + param + '\x1b[0m')


def get_project_folder(projectname, projectdir=None):
    path = mypath.mainpath + "/../projects/"
    params = myfunctions.parse_params(sys.argv)

    if not projectname is None:
        basepath = mypath.mainpath + "/../projects/"
        if "projectdir" in params:
            basepath = params["projectdir"]
        if projectdir is not None:
            basepath = projectdir
        tmp_path = linuxstuff.normalize_path(basepath + "/" + str(projectname) + '/')
        mydebug.debug("Proposed project-folder: " + tmp_path)
        if os.path.isdir(tmp_path):
            path = tmp_path
    else:
        sys.stderr.write("Empty project name. Using `" + path + "` as project folder")
        return None

    normalized_path = linuxstuff.normalize_path(path)
    return normalized_path

def get_parameter_name_from_value (myconf, axis_number, value):
    axis_type = get_axis_type(myconf, axis_number)
    if axis_type == "hp.choice" or axis_type == "hp.choiceint":
        this_options = myconf.str_get_config('DIMENSIONS', 'options_' + str(axis_number - 1))
        values = [x.strip() for x in this_options.split(',')]
        result_array = []
        for item in value:
            this_value = values[item]
            result_array.append(this_value)
        return result_array
    else:
        return value

def get_parameter_value (myconf, axis_number, value):
    axis_type = get_axis_type(myconf, axis_number - 1)
    #print("axis-type:", axis_type, "value:", value)
    if axis_type == "hp.choice" or axis_type == "hp.choiceint" or axis_type == "hp.pchoice" or axis_type == "hp.choicestep":
        values = []

        if axis_type == "hp.choicestep":
            min_val = myconf.int_get_config('DIMENSIONS', 'min_dim_' + str(axis_number - 1))
            max_val = myconf.int_get_config('DIMENSIONS', 'max_dim_' + str(axis_number - 1))
            step_val = myconf.int_get_config('DIMENSIONS', 'step_dim_' + str(axis_number - 1))

            for x in range(min_val, max_val + step_val, step_val):
                values.append(x)

        else:
            this_options = myconf.str_get_config('DIMENSIONS', 'options_' + str(axis_number - 1))
            values = [x.strip() for x in this_options.split(',')]

            if axis_type == "hp.pchoice":
                original_values = values
                values = []
                for item in original_values:
                    x, prob = item.split('=')
                    values.append(x)

        if hasattr(value, "__len__"):
            result_array = []
            for item in value:
                this_value = values[int(item)]
                if not re.match(r'^-?\d+(?:\.\d+)?$', this_value) is None:
                    this_value = float(this_value)
                result_array.append(this_value)
            return result_array
        else:
            try:
                return values[value]
            except:
                return None
    else:
        return value

def get_axis_type (myconf, number):
    thistype = None

    number = int(number)

    try:
        thistype = myconf.str_get_config('DIMENSIONS', 'range_generator_' + str(number))
    except Exception:
        sys.stderr.write("Could not get range_generator_%s\n" % str(number))
        thistype = myconf.str_get_config('DATA', 'range_generator_name')
    if thistype is not None:
        return thistype
    else:
        default_type = myconf.str_get_config('DIMENSIONS', 'range_generator_name')
        if thistype is not None:
            return default_type
        else:
            return "hp.randint"


def get_axis_label(myconf, number):
    thisname = "*unknown label*"

    try:
        thisname = myconf.str_get_config('DIMENSIONS', 'dim_' + str(number - 1) + '_name')
    except Exception:
        print("Could not get label name from axis " + str(number - 1) + "\n")

    if thisname is not None:
        return thisname
    else:
        return 'x_' + str(number - 1)

    axis_labels = ()

    if len(axis_labels) < number:
        return 'x_' + str(number - 1)
    else:
        return axis_labels[number - 1]

def get_config_path_by_projectfolder(projectfolder):
    path = projectfolder + '/config.ini'
    return linuxstuff.normalize_path(path)

def dier (msg):
    pprint.pprint(msg)
    sys.exit(1)

def get_config_path_by_projectname(projectname, projectdirdefault=None):
    mainpath = mypath.mainpath

    params = myfunctions.parse_params(sys.argv)

    if "projectdir" in params:
        mainpath = params["projectdir"]

    if projectdirdefault is not None:
        mainpath = projectdirdefault

    path = mainpath + '/config.ini'
    if not projectname is None:
        if mainpath == mypath.mainpath:
            tmp_path = linuxstuff.normalize_path(mainpath + '/../projects/' + str(projectname) + '/config.ini')
        else:
            tmp_path = linuxstuff.normalize_path(mainpath + str(projectname) + '/config.ini')
        mydebug.debug_xtreme('====> ' + linuxstuff.normalize_path(tmp_path))
        if os.path.isfile(tmp_path):
            path = tmp_path
        else:
            sys.stderr.write("The file `" + tmp_path + "` did not exist!\n")
    else:
        sys.stderr.write("Project name was empty!\n")
        return None
    return linuxstuff.normalize_path(path)

def create_space(algo_name, myconf, i):
    label = 'x_' + str(i)
    space = None
    if algo_name == 'hp.choice':
        options_string = myconf.str_get_config('DIMENSIONS', 'options_' + str(i))
        options = []
        for x in options_string.split(','):
            options.append(x)
        space = hp.choice(label, options)
    elif algo_name == 'hp.pchoice':
        options_string = myconf.str_get_config('DIMENSIONS', 'options_' + str(i))
        options = []

        sum_prob = 0
        sum_prob = int(sum_prob)

        for item in options_string.split(','):
            x, prob = item.split('=')
            prob = int(prob)

            options.append((prob, x))

            sum_prob = sum_prob + prob

        if sum_prob != 100:
            mydebug.warning("The sum of all probabilites in %s is not 100, but %f. It gets normalized, so that %f is becoming equal to 100%%." % (label, sum_prob, sum_prob))

            original_options = options
            options = []

            for item in original_options:
                normalized_prob = int((item[0] / sum_prob) * 100)
                options.append((normalized_prob, item[1]))

        
        original_options = options
        options = []

        # normalize them back to numbers between 0 and 1
        for item in original_options:
            options.append((item[0] / 100, item[1]))

        space = hp.pchoice(label, options)
    elif algo_name == 'hp.choicestep':
        low = myconf.float_get_config('DIMENSIONS', 'min_dim_' + str(i))
        upper = myconf.int_get_config('DIMENSIONS', 'max_dim_' + str(i))
        step = myconf.int_get_config('DIMENSIONS', 'step_dim_' + str(i))
        options = []
        for x in range(int(low), int(upper) + 1, step):
            options.append(x)
        space = hp.choice(label, options)
    elif algo_name == 'hp.choiceint':
        low = myconf.float_get_config('DIMENSIONS', 'min_dim_' + str(i))
        upper = myconf.int_get_config('DIMENSIONS', 'max_dim_' + str(i))
        options = []
        for x in range(int(low), int(upper) + 1):
            options.append(x)
        space = hp.choice(label, options)
    elif algo_name == 'hp.randint':
        upper = myconf.int_get_config('DIMENSIONS', 'max_dim_' + str(i))
        space = hp.randint(label, upper)
    elif algo_name == 'hp.uniform':
        low = myconf.float_get_config('DIMENSIONS', 'min_dim_' + str(i))
        high = myconf.float_get_config('DIMENSIONS', 'max_dim_' + str(i))
        space = hp.uniform(label, low, high)
    elif algo_name == 'hp.quniform':
        low = myconf.float_get_config('DIMENSIONS', 'min_dim_' + str(i))
        high = myconf.float_get_config('DIMENSIONS', 'max_dim_' + str(i))
        q = myconf.float_get_config('DIMENSIONS', 'q_' + str(i))
        space = hp.quniform(label, low, high, q)
    elif algo_name == 'hp.loguniform':
        low = myconf.float_get_config('DIMENSIONS', 'min_dim_' + str(i))
        high = myconf.float_get_config('DIMENSIONS', 'max_dim_' + str(i))
        space = hp.loguniform(label, low, high)
    elif algo_name == 'hp.uniformint':
        min_dim = myconf.float_get_config('DIMENSIONS', 'min_dim_' + str(i))
        max_dim = myconf.float_get_config('DIMENSIONS', 'max_dim_' + str(i))
        space = hp.uniformint(label, min_dim, max_dim)
    elif algo_name == 'hp.qloguniform':
        low = myconf.float_get_config('DIMENSIONS', 'min_dim_' + str(i))
        high = myconf.float_get_config('DIMENSIONS', 'max_dim_' + str(i))
        q = myconf.float_get_config('DIMENSIONS', 'q_' + str(i))
        space = hp.qloguniform(label, low, high, q)
    elif algo_name == 'hp.normal':
        mu = myconf.float_get_config('DIMENSIONS', 'mu_' + str(i))
        sigma = myconf.float_get_config('DIMENSIONS', 'sigma_' + str(i))
        space = hp.normal(label, mu, sigma)
    elif algo_name == 'hp.qnormal':
        mu = myconf.float_get_config('DIMENSIONS', 'mu_' + str(i))
        sigma = myconf.float_get_config('DIMENSIONS', 'sigma_' + str(i))
        q = myconf.float_get_config('DIMENSIONS', 'q_' + str(i))
        space = hp.qnormal(label, mu, sigma, q)
    elif algo_name == 'hp.lognormal':
        mu = myconf.float_get_config('DIMENSIONS', 'mu_' + str(i))
        sigma = myconf.float_get_config('DIMENSIONS', 'sigma_' + str(i))
        space = hp.lognormal(label, mu, sigma)
    elif algo_name == 'hp.qlognormal':
        mu = myconf.float_get_config('DIMENSIONS', 'mu_' + str(i))
        sigma = myconf.float_get_config('DIMENSIONS', 'sigma_' + str(i))
        q = myconf.float_get_config('DIMENSIONS', 'q_' + str(i))
        space = hp.qlognormal(label, mu, sigma, q)
    else:
        raise Exception("Unknown algorithm `" + str(algo_name) + "`")

    return space

def get_choices(myconf, data):
    maxnumofdims = myconf.int_get_config('DIMENSIONS', 'dimensions')
    chosen_range_generator = get_chosen_range_generator(data)
    mydebug.debug('Defining variables for dimension')
    choices = []
    i = 0
    for this_dim in range(0, maxnumofdims):
        this_range_generator = chosen_range_generator['name']
        try:
            this_range_generator = myconf.str_get_config('DIMENSIONS', 'range_generator_' + str(i))
        except Exception as e:
            sys.stderr.write('No special range generator for axis ' + str(i) + " chosen! Using the default (which is " + str(this_range_generator) + ")")
        choices.append(create_space(this_range_generator, myconf, this_dim))
        i = i + 1
    return choices

def get_chosen_range_generator(data):
    mydebug.debug('Getting range_generator_dict')
    range_generator_dict = range_generator.get_range_generator_list()
    chosen_range_generator = None
    for item in range_generator_dict:
        mydebug.debug_xtreme(">> Checking `" + item['name'] + "`")
        if item['name'] == data['range_generator_name']:
            mydebug.debug(item['name'] + ' == ' + data['range_generator_name'])
            chosen_range_generator = item
        else:
            mydebug.debug_xtreme('>> ' + item['name'] + " != " + data['range_generator_name'])
    if chosen_range_generator is None:
        mydebug.debug("Couldn't get range_generator_name. Showing valid generators.")
        valid_generators = get_valid_range_generators()
        raise Exception(colored("Range generator named `" + data['range_generator_name'] + "` not found!", 'red') + " Valid range-generator-names:\n" + valid_generators)
    else:
        mydebug.debug(mydebug.range_generator_info(chosen_range_generator))

    return chosen_range_generator

def replace_dollar_with_variable_content(code, varname, *variables, projectname=None):
    code_copy = code
    thisvars = {}
    i = 0
    for thisvar in variables:
        thisvars[varname + str(i)] = variables[i]
        i = i + 1

    thisvars['mainpath'] = mypath.mainpath
    thisvars['homepath'] = mypath.homepath
    thisvars['projectdir'] = linuxstuff.normalize_path(mypath.mainpath + '/../projects/')
    thisvars['thisprojectdir'] = linuxstuff.normalize_path(thisvars['projectdir'] + '/' + str(projectname) + '/program/')
    if os.path.isdir(str(projectname)):
        thisvars['thisprojectpath'] = linuxstuff.normalize_path(str(projectname))

    for thisvar in thisvars:
        key = thisvar
        value = thisvars[key]

        nonintstr = str(value).split('.')[0]
        if nonintstr is not None:
            searchforint = "int($" + key + ")"
            code_copy = code_copy.replace(searchforint, str(nonintstr))
        else:
            searchforint = "int($" + key + ")"
            code_copy = code_copy.replace(searchforint, "$" + key)

        searchfor = "($" + key + ")"
        code_copy = code_copy.replace(searchfor, str(value))
    if code_copy == code:
        mydebug.warning("No parameters were replaced! Please make sure that you have a dollar sign in front of the variable names, e.g.: `perl script.pl --par1=($" + varname + "0) --par2=int($" + varname + "1) ...` in your config-file!")
    check_for_unreplaced = re.search('($' + varname + '\\d)', code_copy)
    if check_for_unreplaced is not None:
        mydebug.warning("There are more parameters than defined dimensions! Not replacing " + check_for_unreplaced.group(0) + " from string `" + code + "`, turning it into `" + code_copy + "`")

    sys.stderr.write(code_copy)
    mydebug.debug('Code that will get executed: ' + code_copy)
    return code_copy

def find_mongo_info (project, slurmid, data=None, projectdir=None):
    result = {}

    path = linuxstuff.normalize_path(mypath.mainpath + "/../projects/")
    if not project is None:
        basepath = mypath.mainpath + "/../projects/"
        if "projectdir" in data:
            basepath = data["projectdir"]
        if projectdir is not None:
            basepath = projectdir
        tmp_path = linuxstuff.normalize_path(basepath + str(project) + '/')
        mydebug.debug("Proposed project-folder: " + tmp_path)
        if os.path.isdir(tmp_path):
            path = tmp_path
        else:
            sys.stderr.write("find_mongo_info() !!!! The folder \n\t" + tmp_path + "\ndoes not exist. Using \n\t" + path + "\n as path !!!!") 
    else:
        sys.stderr.write("Empty project name. Using `" + path + "` as project folder")

    retval = linuxstuff.normalize_path(path)

    mongodbipfolder = retval + '/ipfiles/'

    try:
        os.stat(mongodbipfolder)
    except:
        os.mkdir(mongodbipfolder)

    mongodbipfile = mongodbipfolder + 'mongodbserverip-' + str(slurmid)
    mongodbipfile = linuxstuff.normalize_path(mongodbipfile)
    if os.path.isfile(mongodbipfile):
        saved_ip = ''
        with open(mongodbipfile) as f:
            saved_ip = f.readline()
        if networkstuff.is_valid_ipv4_address(saved_ip):
            mydebug.debug('IP for MongoDB: ' + saved_ip)
            result['mongodbmachine'] = saved_ip
        else:
            sys.stderr.write("The IP `" + saved_ip + "` is not a valid one!")
    else:
        if(slurmid != 'None'):
            sys.stderr.write("The file `" + mongodbipfile + "` could not be found! Using 127.0.0.1 instead\n")
        result['mongodbmachine'] = '127.0.0.1'

    mongodbportfile = mongodbipfolder + "mongodbportfile-" + str(slurmid)
    mongodbportfile = linuxstuff.normalize_path(mongodbportfile)
    mydebug.debug('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PORTFILE: ' + mongodbportfile)
    if os.path.isfile(mongodbportfile):
        savedport = ''
        with open(mongodbportfile) as f:
            savedport = f.readline()
        result['mongodbport'] = int(savedport)
    else:
        default_port = str(os.getenv("mongodbport", 56741))
        if(slurmid != 'None'):
            sys.stderr.write("The file `" + mongodbportfile + "` could not be found! Using " + default_port + " instead\n")
        result['mongodbport'] = default_port

    return result

def print_best (myconf, best, best_data):
    string = "Best result data:\n"
    string = string + "=========================================\n"
    for key in best:
        if key.startswith("x_"):
            key_nr = key
            key_nr = key_nr.replace("x_", "")
            label = get_axis_label(myconf, int(key_nr) + 1)
            this_value = get_parameter_value(myconf, int(key_nr) + 1, best[key])
            string = string + str(label) + " = " + str(this_value) + "\n"
    string = string + "=========================================\n"
    SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", None)
    if SLURM_JOB_ID is not None:
        best_file = mypath.mainpath + "/../." + str(SLURM_JOB_ID) + ".log"
        best_file_handler = open(best_file, 'w')
        print(string, file=best_file_handler)
        best_file_handler.close()
    else:
        print(string)
