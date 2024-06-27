import argparse
import os
import os.path
import re
import sys

import configparser
import mydebug
import myfunctions
import mypath

class getOpts:
    def __init__(self, config_file):
        self.filename = config_file
        if self.filename is None:
            sys.stderr.write("The config file was empty!\n")
            sys.exit(1)
        else:
            if not os.path.exists(self.filename):
                sys.stderr.write("The config-file could not be found (searched for it in the path " + self.filename + ")\n")
                self.filename = None
        try:
            self.myconfig = self.get_program_config()
        except Exception as e:
            print(str(e))

    def log_file_exists(self, path):
        if path and os.path.isfile(path):
            return True
        else:
            return False

    def get_config_path(self):
        return self.filename

    def get_program_config(self):
        my_config_file = self.filename
        if self.log_file_exists(my_config_file):
            myconfig = configparser.ConfigParser()
            myconfig.read(my_config_file)
            return myconfig
        else:
            raise Exception("`" + my_config_file + "` not found! Cannot continue!")

    def bool_get_config(self, category, name):
        data = None
        if data is None:
            # Cannot output debug-messages here, since at initialization, this function is used to get whether to debug or not in the first place
            try:
                myconfig = self.get_program_config()
            except Exception as e:
                print(str(e))
                return False
            data = int(myconfig.get(category, name))
        if re.match('[yj1]', str(data), flags=re.IGNORECASE):
            data = True
        else:
            data = False
        return data

    def float_get_config(self, category, name):
        data = None
        if data is None:
            try:
                myconfig = self.get_program_config()
            except Exception as e:
                print(str(e))
                return False
            data = float(myconfig.get(category, name))
        return data
    def int_get_config(self, category, name):
        data = None
        if data is None:
            try:
                myconfig = self.get_program_config()
            except Exception as e:
                print(str(e))
                return False
            data = int(myconfig.get(category, name))
        return data

    def str_get_config(self, category, name):
        data = None
        if data is None:
            try:
                myconfig = self.get_program_config()
            except Exception as e:
                print(str(e))
                return False
            data = str(myconfig.get(category, name))
        return data

    def get_cli_code(self, category, name):
        data = None
        if data is None:
            try:
                myconfig = self.get_program_config()
            except Exception as e:
                print(str(e))
                return False
            try:
                data = str(myconfig.get(category, name))
            except Exception as e: 
                print("Unexpected error:")
                print(e)
                return None
        if not data is None:
            return data
        else:
            return None
