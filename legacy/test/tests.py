from time import sleep
import unittest
import math
import re
import os
import sys
import glob
import inspect
from pathlib import Path
from subprocess import Popen, PIPE
import random
import string

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe()
        )
    )
) + "/../script/"
parentdir = os.path.dirname(currentdir)
mainpath = parentdir + "/.."
sys.path.insert(0, parentdir)

import myfunctions
import codechecker
import mydebug
import range_generator
import mypath
import myregexps
import mongo_db_objective
import hyperopt
import json
import networkstuff
import numberstuff
import linuxstuff
import slurmstuff
import omnioptstuff
import mongostuff
import workerstuff
import pprint
import socket
from getOpts import getOpts

global myconf

myconf = mydebug.set_myconf("DONOTDELETE_testcase", "test/projects/")
thisdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
floating_number_limited = myregexps.floating_number_limited
intregex = myregexps.integer_limited

testprojectdir = "test/projects/"

testprojectname = "DONOTDELETE_testcase"

generallogfile = {
    "stdout": linuxstuff.normalize_path(mypath.mainpath + "/../test/TESTLOGS/testlog.stdout"),
    "stderr": linuxstuff.normalize_path(mypath.mainpath + "/../test/TESTLOGS/testlog.stderr"),
    "debug": linuxstuff.normalize_path(mypath.mainpath + "/../test/TESTLOGS/testlog.debug")
}

def shellquote(s):
    s = s.replace("#!", "#")
    return s

def run_command_get_output(cli_command):
    output = ''
    f = os.popen(cli_command)
    for line in f.readlines():
        output = output + line
    return output

params = {
    "num_gpus_per_worker": 0,
    "cpus_per_task": 4,
    "partition": "haswell",
    "reservation": None,
    "account": None
}

objective_function_mongodb_output = mongo_db_objective.objective_function_mongodb([testprojectname, [5, 6, 7, 8], testprojectdir, params])
#pprint.pprint(objective_function_mongodb_output)
#sys.exit(0)

class TestStringMethods(unittest.TestCase):
    def test_worker_file_creation(self):
        start_worker_command = "hallo"
        projectdir = mypath.mainpath + "/../test/projects/"
        config_path = omnioptstuff.get_config_path_by_projectname(testprojectname, projectdir)
        myconf = getOpts(config_path)

        data = {}
        data['mongodbdbname'] = testprojectname
        data['mongodbport'] = 12345
        data['mongodbmachine'] = "localhost" 

        command_output = workerstuff.get_start_worker_command(start_worker_command, testprojectname, myconf, data, testprojectdir, 1)
        shell_check = "((cat <<'EOL'\n" + shellquote(command_output) + '\nEOL\n) | bash -n) && echo "OK" || echo "FAILED"'
        filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        f = open(filename, "w+")
        f.write(shell_check)
        f.close()

        output = run_command_get_output('bash ' + filename)

        os.remove(filename)

        if "->" in shell_check and "<-" in shell_check:
            print("NOT ALL ->VARIABLES<- have been replaced")
            self.assertTrue(False)
        else:
            self.assertTrue(re.match(r"OK", str(output)))

    def test_double_test_names(self):
        code_for_check = "cat " + mainpath + "/test/tests.py | grep def | sort | less | uniq -c | egrep -v '^\s+1\s+' | wc -l"
        self.assertEqual(run_command_get_output(code_for_check), "0\n")

    def test_re(self):
        ipadress = networkstuff.get_ip()
        self.assertTrue(networkstuff.is_valid_ipv4_address(ipadress))

    def test_valid_python_1(self):
        self.assertTrue(codechecker.is_valid_python_code('x + y'))

    def test_valid_python_2(self):
        self.assertFalse(codechecker.is_valid_python_code('asdasdasdwqewqqs ****** DAsd * *A*SD*AS*DA *SD* A*SD* WWRE"§asd'))

    def test_dollar_replacement_2(self):
        tuples = ('2', '3', '5')
        to_replace = 'string ($x0) ($x1) ($x2) $x3 $x4 ($x0)'
        final = 'string 2 3 5 $x3 $x4 2'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement(self):
        tuples = ('2', '3', '5', '7')
        to_replace = 'string ($x0) ($x1) ($x2) ($x3) $x4 ($x0)'
        final = 'string 2 3 5 7 $x4 2'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_3(self):
        tuples = ('2', '3', '5', '7')
        to_replace = '($mainpath) string ($x0) ($x1) ($x2) ($x3) $x4 ($x0)'
        final = mypath.mainpath + ' string 2 3 5 7 $x4 2'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_4(self):
        tuples = ('2.6', '3.9')
        to_replace = '($mainpath) string ($x0) int($x1)'
        final = mypath.mainpath + ' string 2.6 3'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_5(self):
        tuples = ('2.6', '3.9', '3')
        to_replace = '($mainpath) string ($x0) int($x1) int($x2)'
        final = mypath.mainpath + ' string 2.6 3 3'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_6(self):
        tuples = ('2.6', '-3.9', '-3')
        to_replace = '($mainpath) string ($x0) int($x1) int($x2)'
        final = mypath.mainpath + ' string 2.6 -3 -3'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_7(self):
        tuples = ('2.6', '-3.9', '3')
        to_replace = '($mainpath) string ($x0) int($x1) int($x2)'
        final = mypath.mainpath + ' string 2.6 -3 3'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_8(self):
        tuples = ('2.6', '-3.9')
        to_replace = '($mainpath) string ($x_0) int($x_1) int($x_2)'
        final = mypath.mainpath + ' string 2.6 -3 int($x_2)'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x_', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_9(self):
        tuples = ('26', '3')
        to_replace = '($mainpath) string int($x_0)xint($x_1)'
        final = mypath.mainpath + ' string 26x3'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x_', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_10(self):
        tuples = ('0', '1', '2', '3', '4', '5', '6', '7', '8', 'a')
        to_replace = '($mainpath) string ($x_0)x($x_1)x($x_2)x($x_3)x($x_4)x($x_5)x($x_6)x($x_7)x($x_8)x($x_9)'
        final = mypath.mainpath + ' string 0x1x2x3x4x5x6x7x8xa'
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x_', *tuples)
        self.assertEqual(final, replaced)

    def test_dollar_replacement_11(self):
        tuples = ('0', '1', '2', '3', '4', '5', '6', '7', '8', 'a')
        to_replace = 'objective_program=bash ($mainpath)/../projects/Hist4D_ML_TEST/program/start_training.sh --generate-heatmaps=false --input-directory=/projects/p_scads/hist4d/300data/ --model-structure-config-file-path=($mainpath)/../projects/Hist4D_ML_TEST/program/model-structure-config.csv --batch-size=int($x_0) --epochs=int($x_1) --binary-mode=false --match-subdirectories=zwinger,semperoper,nicht_zwinger,dresden_hofkirche --square=int($x_2)xint($x_3)'
        final = mypath.mainpath + ' string 0x1x2x3x4x5x6x7x8xa'
        final = 'objective_program=bash ' + mypath.mainpath + '/../projects/Hist4D_ML_TEST/program/start_training.sh --generate-heatmaps=false --input-directory=/projects/p_scads/hist4d/300data/ --model-structure-config-file-path=' + mypath.mainpath + '/../projects/Hist4D_ML_TEST/program/model-structure-config.csv --batch-size=0 --epochs=1 --binary-mode=false --match-subdirectories=zwinger,semperoper,nicht_zwinger,dresden_hofkirche --square=2x3';
        replaced = omnioptstuff.replace_dollar_with_variable_content(to_replace, 'x_', *tuples)
        self.assertEqual(final, replaced)

    def test_output_13(self):
        tstring = json.dumps(workerstuff.get_data_from_output("q: -62.6213"), sort_keys=True)
        self.assertEqual(tstring, '{"q": "-62.6213"}')

    def test_output_14(self):
        self.assertEqual(json.dumps(workerstuff.get_data_from_output("hallo: -32.6213\ntest\nblubb: 3222\n"), sort_keys=True), '{"blubb": "3222", "hallo": "-32.6213"}')

    def test_output_15(self):
        self.assertEqual(json.dumps(workerstuff.get_data_from_output("hallo\n\nhalloblabla"), sort_keys=True), '{}')

    def test_output_16(self):
        self.assertEqual(json.dumps(workerstuff.get_data_from_output("\n\n\n\n\nresult: 0.4500\n\n\n\n\n\n"), sort_keys=True), '{"result": "0.4500"}')

    def test_output_17(self):
        self.assertEqual(json.dumps(workerstuff.get_data_from_output("\n\n\n\n\nresult:0.4500\n\n\n\n\n\n"), sort_keys=True), '{"result": "0.4500"}')

    def test_output_18(self):
        self.assertEqual(json.dumps(workerstuff.get_data_from_output("\n\n\n[1]\n\n\nresult:0.5000\n\n\n"), sort_keys=True), '{"result": "0.5000"}')

    def test_output_21(self):
        self.assertTrue(workerstuff.get_result_from_output_file("test/example.stdout", "", ""), "0.232519665992964")

    def test_output_23(self):
        self.assertEqual(json.dumps(workerstuff.get_data_from_output("\n\n\n[1]\n\n\nresult1:0.5000\n\n\n"), sort_keys=True), '{"result1": "0.5000"}')

    def test_valid_ip4(self):
        self.assertTrue(networkstuff.is_valid_ipv4_address('8.9.10.11'))

    def test_valid_ip4_2(self):
        self.assertFalse(networkstuff.is_valid_ipv4_address('abc.def'))

    def test_valid_ip4_3(self):
        self.assertFalse(networkstuff.is_valid_ipv4_address('500.125.125.125'))

    def test_valid_ip6(self):
        self.assertTrue(networkstuff.is_valid_ipv6_address('2001:0db8:85a3:0000:0000:8a2e:0370:7334'))

    def test_valid_ip6_2(self):
        self.assertFalse(networkstuff.is_valid_ipv6_address('abc.def'))

    def test_cli_output(self):
        output = run_command_get_output('echo "hallo"')
        self.assertEqual(output, "hallo\n")

    def test_valid_range_generators(self):
        tstring = mydebug.get_valid_range_generators()
        self.assertGreaterEqual(len(tstring), 50)

    def test_get_algorithms_list(self):
        algorithms_list = range_generator.get_algorithms_list()
        self.assertEqual(len(algorithms_list), 4)

    def test_range_generator(self):
        range_generator_dict = range_generator.get_range_generator_list()
        self.assertGreaterEqual(len(range_generator_dict), 10)

    def test_range_generator_info(self):
        range_generator_dict = range_generator.get_range_generator_list()
        chosen_range_generator = None
        for item in range_generator_dict:
            if item['name'] == "hp.choice":
                chosen_range_generator = item

        tstring = mydebug.range_generator_info(chosen_range_generator)
        self.assertGreaterEqual(len(tstring), 20)

    def test_get_data(self):
        data = mydebug.get_data(testprojectname, None, testprojectdir)
        self.assertGreaterEqual(len(data), 5)

    def test_is_tool(self):
        self.assertTrue(linuxstuff.is_tool('ls'))

    def test_is_tool_2(self):
        self.assertFalse(linuxstuff.is_tool('lsksdfknsjdfnwkjfnbfjwjhn'))

    def test_ping_1(self):
        self.assertTrue(networkstuff.ping("localhost"))

    def test_ping_2(self):
        self.assertFalse(networkstuff.ping("546.5648.564.5646"))

    def test_ping_3(self):
        self.assertTrue(networkstuff.ping("127.0.0.1"))

    def test_mongo_db_error_code(self):
        self.assertEqual(type(mongostuff.get_mongo_db_error_code(0)), str)

    def test_mongo_db_error_code2(self):
        self.assertNotEqual(type(mongostuff.get_mongo_db_error_code(100)), int)

    def test_normalize_path(self):
        self.assertEqual(linuxstuff.normalize_path('/home/../../'), '/')

    def test_normalize_path2(self):
        self.assertEqual(linuxstuff.normalize_path('/home/../../home'), '/home')

    def test_normalize_path3(self):
        self.assertNotEqual(linuxstuff.normalize_path('/home'), '/')

    def test_python_files(self):
        for filename in glob.glob("./*/*.py"):
            f = open(filename, 'r')
            s = f.read()
            f.close()
            programvalue = codechecker.is_valid_python_code(s)
            if not programvalue:
                print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!! File " + filename + " has syntax errors!")
            self.assertTrue(programvalue)

    def test_bash_files(self):
        for filename in glob.glob("./*.sh"):
            self.assertTrue(codechecker.is_valid_bash_file(filename))

    def test_bash_files_2(self):
        for filename in glob.glob("./script/fmin.py"):
            self.assertFalse(codechecker.is_valid_bash_file(filename))

    def test_is_integer_1(self):
        self.assertEqual(numberstuff.is_integer('/home'), 0)

    def test_is_integer_2(self):
        self.assertEqual(numberstuff.is_integer('1'), 1)

    def test_is_integer_3(self):
        self.assertEqual(numberstuff.is_integer('143252'), 1)

    def test_is_integer_4(self):
        self.assertEqual(numberstuff.is_integer('3.14159'), 0)

    def test_float_re_1(self):
        self.assertTrue(re.match(floating_number_limited, "5"))

    def test_float_re_2(self):
        self.assertTrue(re.match(floating_number_limited, "+5"))

    def test_float_re_3(self):
        self.assertTrue(re.match(floating_number_limited, "-5"))

    def test_float_re_4(self):
        self.assertTrue(re.match(floating_number_limited, "5.05"))

    def test_float_re_5(self):
        self.assertTrue(re.match(floating_number_limited, "-5.054"))

    def test_float_re_6(self):
        self.assertTrue(re.match(floating_number_limited, "+5.055"))

    def test_float_re_7(self):
        self.assertTrue(re.match(floating_number_limited, "+52.055"))

    def test_float_re_8(self):
        self.assertFalse(re.match(floating_number_limited, "hallo"))

    def test_float_re_9(self):
        self.assertTrue(re.match(floating_number_limited, "+52.055"))

    def test_is_prime_1(self):
        self.assertFalse(numberstuff.is_prime(1))

    def test_is_prime_2(self):
        self.assertTrue(numberstuff.is_prime(2))

    def test_is_prime_3(self):
        self.assertTrue(numberstuff.is_prime(3))

    def test_is_prime_4(self):
        self.assertFalse(numberstuff.is_prime(4))

    def test_is_prime_5(self):
        self.assertTrue(numberstuff.is_prime(5))

    def test_is_prime_7(self):
        self.assertFalse(numberstuff.is_prime(9999999968))

    def test_is_prime_8(self):
        self.assertTrue(numberstuff.is_prime(101))

    def test_nearest_non_prime_1(self):
        self.assertTrue(numberstuff.nearest_non_prime(3) == 4)

    def test_nearest_non_prime_2(self):
        self.assertTrue(numberstuff.nearest_non_prime(11) == 12)

    def test_nearest_non_prime_3(self):
        self.assertTrue(numberstuff.nearest_non_prime(15) == 15)

    def test_nearest_non_prime_4(self):
        self.assertTrue(numberstuff.nearest_non_prime(19) == 20)

    def test_nearest_non_prime_5(self):
        self.assertTrue(numberstuff.nearest_non_prime(45) == 45)

    def test_mongo_db_connection_string(self):
        data = {}
        data["mongodbmachine"] = "localhost"
        data["mongodbport"] = "1234"
        data["mongodbdir"] = "testdb"
        url = 'mongodb://' + data["mongodbmachine"] + ':' + str(data["mongodbport"]) + '/' + data["mongodbdir"]
        self.assertEqual(mongostuff.create_mongo_db_connection_string(data), url)

    def test_mongo_db_connection_string_2(self):
        data = {}
        data["mongodbmachine"] = "127.0.0.1"
        data["mongodbport"] = "1234"
        data["mongodbdir"] = "testdb"
        url = 'mongodb://' + data["mongodbmachine"] + ':' + str(data["mongodbport"]) + '/' + data["mongodbdir"]
        self.assertEqual(mongostuff.create_mongo_db_connection_string(data), url)

    def test_is_valid_url_1(self):
        self.assertTrue(networkstuff.is_valid_url("http://google.de"))

    def test_is_valid_url_2(self):
        self.assertTrue(networkstuff.is_valid_url("https://google.de/google.de/google.de/"))

    def test_is_valid_url_3(self):
        self.assertFalse(networkstuff.is_valid_url("ICHBINDOCHKEINEECHTEURL!"))

    def test_is_valid_url_4(self):
        self.assertTrue(networkstuff.is_valid_url("192.168.0.103"))

    def test_is_valid_url_5(self):
        self.assertTrue(networkstuff.is_valid_url("http://scads.de/test/test/mwerfwsd&2342aa3=322"))

    def test_is_valid_url_6(self):
        self.assertTrue(networkstuff.is_valid_url("https://www2.scads.de/test/test/mwerfwsd&2342aa3=322"))

    def test_normalize_path_1(self):
        self.assertEqual(linuxstuff.normalize_path("/home/../test"), "/test")

    def test_normalize_path_2(self):
        self.assertEqual(linuxstuff.normalize_path("/home/test"), "/home/test")

    def test_normalize_path_3(self):
        self.assertEqual(linuxstuff.normalize_path("/home/test/../../home/test/../../home/test"), "/home/test")

    def test_normalize_path_4(self):
        self.assertEqual(linuxstuff.normalize_path("/home/test///////test"), "/home/test/test")

    def test_slurm_id_1(self):
        self.assertEqual(slurmstuff.is_slurm_id("3242342"), 1)

    def test_slurm_id_2(self):
        self.assertEqual(slurmstuff.is_slurm_id("3242342_5"), 1)

    def test_slurm_id_3(self):
        self.assertEqual(slurmstuff.is_slurm_id("32423a42_5"), 0)

    def test_slurm_id_4(self):
        self.assertEqual(slurmstuff.is_slurm_id("a"), 0)

    def test_slurm_id_5(self):
        self.assertEqual(slurmstuff.is_slurm_id(""), 0)

    def test_slurm_id_6(self):
        self.assertEqual(slurmstuff.is_slurm_id(None), 0)

    def test_parse_single_argument_1(self):
        output = myfunctions.parse_single_argument("--test=123")
        tstring = json.dumps(output, sort_keys=True)
        self.assertEqual(tstring, '{"name": "test", "value": "123"}')

    def test_parse_single_argument_2(self):
        output = myfunctions.parse_single_argument("--test_data=abc")
        tstring = json.dumps(output, sort_keys=True)
        self.assertEqual(tstring, '{"name": "test_data", "value": "abc"}')

    def test_parse_single_argument_3(self):
        output = myfunctions.parse_single_argument("test_data=abc")
        tstring = json.dumps(output, sort_keys=True)
        self.assertEqual(tstring, 'null')

    def test_parse_single_argument_4(self):
        output = myfunctions.parse_single_argument("test_data")
        tstring = json.dumps(output, sort_keys=True)
        self.assertEqual(tstring, 'null')

    def test_perl_files(self):
        for filename in glob.glob("./*.pl"):
            self.assertTrue(codechecker.is_valid_perl_file(filename))

    def test_perl_files_2(self):
        for filename in glob.glob("./*/*.pl"):
            self.assertTrue(codechecker.is_valid_perl_file(filename))

    def test_mongo_db_objective_1(self):
        isok = 0
        if (
            "all_outputs" in objective_function_mongodb_output and
            "calculation_time" in objective_function_mongodb_output and
            "loss" in objective_function_mongodb_output and
            objective_function_mongodb_output["status"] == "ok"
        ):
            isok = 1
        self.assertEqual(1, isok)

    def test_mongo_db_objective_2(self):
        self.assertTrue(re.match(floating_number_limited, str(objective_function_mongodb_output["loss"])))

    def test_mongo_db_objective_3(self):
        self.assertTrue(re.match(floating_number_limited, str(objective_function_mongodb_output["calculation_time"])))

    def test_mongo_db_objective_4(self):
        self.assertEqual(objective_function_mongodb_output["all_outputs"]["summiert"], "26")

    def test_wrong_mongo_db_is_down(self):
        self.assertFalse(mongostuff.mongo_db_already_up("mongodb://127.0.0.1:6666/ICHEXISTIERENICHTMALINMEINEMEIGENENTRAUM"))

    def test_get_project_folder_1(self):
        self.assertEqual(omnioptstuff.get_project_folder(None, testprojectdir), None)

    def test_get_project_folder_2(self):
        self.assertEqual(linuxstuff.normalize_path(mainpath + "/" + omnioptstuff.get_project_folder(testprojectname, testprojectdir)), linuxstuff.normalize_path(mypath.mainpath + "/../test/projects/" + testprojectname))

    def test_get_config_path_by_project_name_1(self):
        str_is = linuxstuff.normalize_path(mainpath + "/" + omnioptstuff.get_config_path_by_projectname(testprojectname, testprojectdir))
        str_should = linuxstuff.normalize_path(mypath.mainpath + "/../test/projects/" + testprojectname + "/config.ini")
        self.assertEqual(str_is, str_should)

    def test_get_config_path_by_project_name_2(self):
        self.assertEqual(omnioptstuff.get_config_path_by_projectname(None, testprojectdir), None)

    def test_testproject_exists(self):
        self.assertTrue(os.path.exists(mypath.mainpath + "/../test/projects/" + testprojectname + "/config.ini"))

    def test_parse_single_argument(self):
        output = myfunctions.parse_single_argument("--argtest1=5")
        tstring = json.dumps(output, sort_keys=True)
        self.assertEqual(tstring, '{"name": "argtest1", "value": "5"}')

    def test_parse_all_arguments(self):
        output = myfunctions.parse_all_arguments(["pythonprogram.py", "--argtest2=10", "--argtest3=20"])
        self.assertEqual(output, {'argtest2': '10', 'argtest3': '20'})

    def test_get_index_of_maximum_value(self):
        testarray = [10, 5, 1, 43432, 43]
        maxindex = 3
        self.assertEqual(maxindex, numberstuff.get_index_of_maximum_value(testarray))

    def test_get_index_of_maximum_value_2(self):
        testarray = [10, 5, 1, 43432, 433234234234234234, 43]
        maxindex = 4
        self.assertEqual(maxindex, numberstuff.get_index_of_maximum_value(testarray))

    def test_get_index_of_minimum_value(self):
        testarray = [10, 5, 1, 43432, 43]
        minindex = 2
        self.assertEqual(minindex, numberstuff.get_index_of_minimum_value(testarray))

    def test_get_index_of_minimum_value_2(self):
        testarray = [10, 5, 54334534, 1, 43432, 43]
        minindex = 3
        self.assertEqual(minindex, numberstuff.get_index_of_minimum_value(testarray))

    def test_get_index_of_minimum_value_3(self):
        testarray = [0, 0, 0, 0, 0, 0]
        minindex = 0
        self.assertEqual(minindex, numberstuff.get_index_of_minimum_value(testarray))

    def test_get_axis_label_1(self):
        self.assertEqual("layer1", omnioptstuff.get_axis_label(myconf, 1))

    def test_get_axis_label_2(self):
        self.assertEqual("layer2", omnioptstuff.get_axis_label(myconf, 2))

    def test_get_parameter_value_1(self):
        self.assertEqual(str(2), str(omnioptstuff.get_parameter_value(myconf, 2, 0)))

    def test_get_parameter_value_2(self):
        self.assertEqual(str(5), str(omnioptstuff.get_parameter_value(myconf, 2, 1)))

    def test_get_parameter_value_3(self):
        self.assertEqual(str(6), str(omnioptstuff.get_parameter_value(myconf, 2, 2)))

    def test_get_parameter_value_4(self):
        self.assertEqual(str(11), str(omnioptstuff.get_parameter_value(myconf, 2, 3)))

    def test_get_parameter_value_5(self):
        self.assertEqual(None, omnioptstuff.get_parameter_value(myconf, 2, 4))

    def test_get_largest_divisors(self):
        self.assertEqual(numberstuff.get_largest_divisors(0), {"x": 2, "y": 1})

    def test_get_largest_divisors_2(self):
        self.assertEqual(numberstuff.get_largest_divisors(1), {"x": 2, "y": 1})

    def test_get_largest_divisors_3(self):
        self.assertEqual(numberstuff.get_largest_divisors(2), {"x": 2, "y": 1})

    def test_get_largest_divisors_4(self):
        self.assertEqual(numberstuff.get_largest_divisors(4), {"x": 2, "y": 2})

    def test_get_largest_divisors_5(self):
        self.assertEqual(numberstuff.get_largest_divisors(5), {"x": 3, "y": 2})

    def test_get_largest_divisors_6(self):
        self.assertEqual(numberstuff.get_largest_divisors(9), {"x": 3, "y": 3})

    def test_get_random_number_between_x_and_y_1(self):
        limita = 10
        limitb = 50
        res = numberstuff.random_number_between_x_and_y(limita, limitb)
        is_between_a_and_b = 0
        if res >= limita and res <= limitb:
            is_between_a_and_b = 1
        self.assertEqual(is_between_a_and_b, 1)

    def test_get_random_number_between_x_and_y_2(self):
        limita = 50543
        limitb = 700000
        res = numberstuff.random_number_between_x_and_y(limita, limitb)
        is_between_a_and_b = 0
        if res >= limita and res <= limitb:
            is_between_a_and_b = 1
        self.assertEqual(is_between_a_and_b, 1)

    def test_get_random_number_between_x_and_y_3(self):
        limita = 1000000
        limitb = 1000001
        res = numberstuff.random_number_between_x_and_y(limita, limitb)
        is_between_a_and_b = 0
        if res >= limita and res <= limitb:
            is_between_a_and_b = 1
        self.assertEqual(is_between_a_and_b, 1)


    def test_integer_regex_1(self):
        self.assertTrue(re.match(intregex, str(132)))

    def test_integer_regex_2(self):
        self.assertTrue(re.match(intregex, str(324)))

    def test_integer_regex_3(self):
        self.assertFalse(re.match(intregex, "abcdef"))

    def test_integer_regex_4(self):
        self.assertFalse(re.match(intregex, ""))

    def test_run_program_1(self):
        res = workerstuff.run_program("echo hallo", generallogfile)
        self.assertEqual(res["res"], "STDOUT:\nhallo\n\n\nSTDERR:\n\n\nRETURNCODE:\n0\n\nERRSTR:\n\n")

    def test_run_program_2(self):
        res = workerstuff.run_program("bash " + thisdir + "/outputtest2.sh", generallogfile)
        self.assertEqual(res["res"], "STDOUT:\nstdout\n\n\nSTDERR:\nstderr\n\n\nRETURNCODE:\n5\n\nERRSTR:\n\n")

if __name__ == '__main__':
    unittest.main()
