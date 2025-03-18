# VirtualEnv

<!-- What are Virtual Environments and how OmniOpt2 uses them -->

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

## Change the directory the virtual environment will be installed to

If you cannot write in your home, for example, because it is too full, you can set the path where the virtual environment will be installed and used to. Simply set the variable:

```bash
export root_venv_dir=/path/where/it/should/be/installed/to
```

You need to do this every time before you start OmniOpt2 as well, since without that shell variable (that has to be exported!) it will try to do it from your home again.
