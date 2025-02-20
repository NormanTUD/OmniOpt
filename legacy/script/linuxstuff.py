import re
import os
from distutils.spawn import find_executable

def normalize_path(path):
    path = re.sub(
        r"/{2,}",
        '/',
        path
    )
    return os.path.expanduser(os.path.normpath(path))

def is_tool(name):
    """Check whether `name` is on PATH."""
    return find_executable(name) is not None
