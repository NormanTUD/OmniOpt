import os
import os.path
from os.path import expanduser

developer_machine = "alanwatts"
mainpath = os.path.dirname(os.path.realpath(__file__))
config_file_name = 'config.ini'
config_file = mainpath + '/' + config_file_name
homepath = expanduser("~")
if os.getenv("singularity") and os.path.isfile(homepath + "/" + config_file_name):
    config_file = homepath + "/" + config_file_name
