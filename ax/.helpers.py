import os
import sys

def check_environment_variable(variable_name):
    try:
        value = os.environ[variable_name]
        return True
    except KeyError:
        return False

if not check_environment_variable("RUN_VIA_RUNSH"):
    print("Must be run via the bash script, cannot be run as standalone.")

    sys.exit(16)

def in_venv():
    return sys.prefix != sys.base_prefix


if not in_venv():
    print("No venv loaded. Cannot continue.")
    sys.exit(19)
