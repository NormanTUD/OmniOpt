import os
from importlib.metadata import version
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

def warn_versions():
    wrns = []

    supported_versions = {
        "ax": ["0.3.7", "0.3.8.dev133", "0.52.0"],
        "botorch": ["0.10.0", "0.10.1.dev46+g7a844b9e", "0.11.0", "0.9.5"],
        "torch": ["2.3.0"],
        "seaborn": ["0.13.2"],
        "pandas": ["1.5.3", "2.2.2"],
        "numpy": ["1.26.4"],
        "matplotlib": ["3.6.3", "3.9.0"],
        "plotext": ["5.2.8"],
        "submitit": ["1.5.1"],
        "tqdm": ["4.66.4"]
    }

    for key in supported_versions.keys():
        _supported_versions = supported_versions[key]
        _real_version = version(key)
        if _real_version not in _supported_versions:
            wrns.append(f"Possibly unsupported {key}-version: {_real_version} not in supported {', '.join(_supported_versions)}")

    if len(wrns):
        print("- " + ("\n- ".join(wrns)))

warn_versions()
