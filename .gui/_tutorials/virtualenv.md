# <span class="tutorial_icon invert_in_dark_mode">🧑‍💻</span> VirtualEnv

<!-- What are Virtual Environments and how OmniOpt2 uses them -->

<!-- Category: Preparations, Basics and Setup -->

<div id="toc"></div>

## What is a virtual environment?

Most software depends on other software to work. This software usually needs to be there in a very specific version, and may conflict with other programs that need other versions of the same program. To solve this, *virtual environments* have been invented. They are a single folder containing all the dependencies your program needs.

## Where will OmniOpt2 install its dependencies to?

By default, OmniOpt2 tries to install the dependencies in a folder in your home folder, so that you only need to install OmniOpt2 once, and all other installations can access this virtualenv.

The default folder will be created like this:

```bash
.omniax_$(uname -m)_$(python3 --version | sed -e 's# #_#g')$_cluster
```

while `uname -m` yields to your CPU architecture (usually `x86_64`),
`python --version | sed -e 's# #_#g'` will yield to your python version (e.g. `3.11.2`), and `$_cluster` contains the cluster name, taken from the environment variable `CLUSTERHOST`.

## How to install further modules manually

Sometimes you need to install more modules, ie. when you want to export to a certain database like [Oracle or PostGRES](tutorials?tutorial=sqlite#other-db-systems-than-sqlite3). You can do that by first activating that virtualenv:

```bash
source ~/.omniax_x86_64_Python_3.11.2/bin/activate
```

and then installing modules with `pip install ...`.

Make sure you are running the same python-Version. That means, if you do it on a System with Lmod or similar systems, `module load ...` the proper python-version and modules first.

On the `capella` partition of the TU Dresden HPC System, this would be:

```bash
ml release/24.04 GCCcore/12.3.0 Python/3.11.3 Tkinter/3.11.3 PostgreSQL/16.1
```

On the ML-Power9-Partition:

```bash
ml release/24.04 GCCcore/12.3.0 Python/3.11.3 Tkinter/3.11.3 PostgreSQL/16.1 zlib/1.2.12 GCC/12.2.0 OpenBLAS/0.3.21
```

On all others:

```bash
ml release/23.04 GCCcore/12.2.0 Python/3.10.8 GCCcore/11.3.0 Tkinter/3.10.4 PostgreSQL/14.4
```

## Change the directory the virtual environment will be installed to

If you cannot write in your home, for example, because it is too full, you can set the path where the virtual environment will be installed and used to. Simply set the variable:

```bash
export root_venv_dir=/path/where/it/should/be/installed/to
```

You need to do this every time before you start OmniOpt2 as well, since without that shell variable (that has to be exported!) it will try to do it from your home again.
