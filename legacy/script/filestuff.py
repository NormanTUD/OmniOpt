import linuxstuff
import sys

def get_whole_file(filename):
    all_of_it = '!!!NO DATA READ!!!'
    with open(filename, 'r') as fh:
        all_of_it = fh.read()
    return str(all_of_it)

def overwrite_file(filename, content):
    filename = linuxstuff.normalize_path(filename)
    try:
        with open(filename, 'w') as f:
            print(content, file=f)
        f.close()
    except Exception as e:
        sys.stderr.write("WARNING!!! overwrite_file(filename=" + str(filename) + ", content=" + str(content) + ") gave the following error message: " + str(e))

