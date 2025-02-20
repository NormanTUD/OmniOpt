import subprocess
import shlex
import ast
from subprocess import Popen, PIPE
import mydebug

def is_valid_perl_file(filename):
    process = Popen(['perl', "-c", filename], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()

    if exit_code == 0:
        return True
    else:
        return False

def is_valid_bash_file(filename):
    process = Popen(['bash', "-n", filename], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()

    if exit_code == 0:
        return True
    else:
        return False

def is_valid_python_code(code):
    mydebug.debug_xtreme("Trying to check code `" + code + "`")
    retval = True
    try:
        ast.parse(code)
    except SyntaxError:
        retval = False
    if retval:
        mydebug.debug("Code has no syntax errors!")
    else:
        mydebug.error("Code has syntax errors!")
    return retval
