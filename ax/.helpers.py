import numpy as np
import re
import os
from importlib.metadata import version
import sys
from pprint import pprint

def check_environment_variable(variable_name):
    try:
        value = os.environ[variable_name]
        return True
    except KeyError:
        return False

if not check_environment_variable("RUN_VIA_RUNSH"):
    print("Must be run via the bash script, cannot be run as standalone.")

    sys.exit(16)

#def in_venv():
#    return sys.prefix != sys.base_prefix


#if not in_venv():
#    print("No venv loaded. Cannot continue.")
#    sys.exit(19)

def warn_versions():
    wrns = []

    supported_versions = {
        "ax": ["0.36.0", "0.3.7", "0.3.8.dev133", "0.52.0"],
        "botorch": ["0.10.0", "0.10.1.dev46+g7a844b9e", "0.11.0", "0.8.5", "0.9.5", "0.11.3"],
        "torch": ["2.3.0", "2.3.1", "2.4.0"],
        "seaborn": ["0.13.2"],
        "pandas": ["1.5.3", "2.0.3", "2.2.2"],
        "numpy": ["1.24.4", "1.26.4"],
        "matplotlib": ["3.6.3", "3.7.5", "3.9.0", "3.9.1", "3.9.1.post1"],
        "submitit": ["1.5.1"],
        "tqdm": ["4.66.4", "4.66.5"]
    }

    for key in supported_versions.keys():
        _supported_versions = supported_versions[key]
        try:
            _real_version = version(key)
            if _real_version not in _supported_versions:
                wrns.append(f"Possibly unsupported {key}-version: {_real_version} not in supported version(s): {', '.join(_supported_versions)}")
        except Exception as e:
            pass

    if len(wrns):
        print("- " + ("\n- ".join(wrns)))

def looks_like_float(x):
    if isinstance(x, (int, float)):
        return True  # int and float types are directly considered as floats
    elif isinstance(x, str):
        try:
            float(x)  # Try converting string to float
            return True
        except ValueError:
            return False  # If conversion fails, it's not a float-like string
    return False  # If x is neither str, int, nor float, it's not float-like

def looks_like_int(x):
    if isinstance(x, bool):
        return False
    elif isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    elif isinstance(x, str):
        return bool(re.match(r'^\d+$', x))
    else:
        return False

def looks_like_number (x):
    return looks_like_float(x) or looks_like_int(x) or type(x) == int or type(x) == float or type(x) == np.int64

def to_int_when_possible(val):
    # Überprüfung, ob der Wert ein Integer ist oder ein Float, der eine ganze Zahl sein könnte
    if type(val) == int or (type(val) == float and val.is_integer()) or (type(val) == str and val.isdigit()):
        return int(val)

    # Überprüfung auf nicht-numerische Zeichenketten
    if type(val) == str and re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return val

    try:
        # Versuche, den Wert als Float zu interpretieren
        val = float(val)
        # Bestimmen der Anzahl der Dezimalstellen, um die Genauigkeit der Ausgabe zu steuern
        if '.' in str(val):
            decimal_places = len(str(val).split('.')[1])
            # Formatieren des Floats mit der exakten Anzahl der Dezimalstellen, ohne wissenschaftliche Notation
            formatted_value = format(val, f'.{decimal_places}f').rstrip('0').rstrip('.')
            return formatted_value if formatted_value else '0'
        else:
            return str(int(val))
    except:
        # Falls ein Fehler auftritt, gebe den ursprünglichen Wert zurück
        return val

def dier (msg):
    pprint(msg)
    sys.exit(1)

def assert_condition(condition, error_text):
    if not condition:
        raise AssertionError(error_text)

warn_versions()
