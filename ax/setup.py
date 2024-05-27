from pprint import pprint
import os
import site
import sys
from distutils.sysconfig import get_python_lib

from setuptools import setup

def dier (msg):
    pprint(msg)
    sys.exit(1)

# Allow editable install into user site directory.
# See https://github.com/pypa/pip/issues/7953.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Warn if we are installing over top of an existing installation. This can
# cause issues where files that were deleted from a more recent OmniOpt2 are
# still present in site-packages. See #18115.
overlay_warning = False
if "install" in sys.argv:
    lib_paths = [get_python_lib()]
    if lib_paths[0].startswith("/usr/lib/"):
        print("You need to be in a virtuel environment or something similiar to install this package")
    for lib_path in lib_paths:
        existing_path = os.path.abspath(os.path.join(lib_path, "omniopt2"))
        if os.path.exists(existing_path):
            # We note the need for the warning here, but present it after the
            # command is run, so it's more likely to be seen.
            overlay_warning = True
            break

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()


setup(name='omniopt',
    version='0.2',
    description='Automatic hyperparameter optimizer based on Ax/Botorch',
    author='Norman Koch',
    author_email='norman.koch@tu-dresden.de',
    url='https://scads.ai/transfer-2/verfuegbare-software-dienste-en/omniopt/',
    install_requires=install_requires,
    packages=[
        '.',
    ],
    data_files=[('bin', [
        "omniopt_evaluate", 
        "omniopt", 
        "omniopt_plot", 
        ".helpers.py",
        ".omniopt_plot_general.py",
        ".omniopt_plot_gpu_usage.py",
        ".omniopt_plot_kde.py",
        ".omniopt_plot_scatter.py",
        ".omniopt_plot_worker.py",
        ".omniopt.py",
        ".shellscript_functions", 
        ".general.sh", 
    ])],
    include_package_data=True,
)

if overlay_warning:
    sys.stderr.write(
        """

========
WARNING!
========

You have just installed OmniOpt2 over top of an existing
installation, without removing it first. Because of this,
your install may now include extraneous files from a
previous version that have since been removed from
OmniOpt2. This is known to cause a variety of problems. You
should manually remove the

%(existing_path)s

directory and re-install OmniOpt2.

"""
        % {"existing_path": existing_path}
    )
