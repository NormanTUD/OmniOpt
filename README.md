# What is OmniOpt?

# The new version of OmniOpt is in the ax subdirectory! Everything outside of that directory is deprecated!

## CI Status for OmniOpt2

![Current build status](https://github.com/NormanTUD/OmniOpt/actions/workflows/main.yml/badge.svg?event=push)
![Latest Release](https://img.shields.io/github/v/release/NormanTUD/OmniOpt)
![Open Issues](https://img.shields.io/github/issues/NormanTUD/OmniOpt)
![Open Pull Requests](https://img.shields.io/github/issues-pr/NormanTUD/OmniOpt)
![License](https://img.shields.io/badge/license-GNU-blue.svg)
![Bug Issues](https://img.shields.io/github/issues/NormanTUD/OmniOpt/bug)
![GitHub Repo stars](https://img.shields.io/github/stars/NormanTUD/OmniOpt)
![Pull Requests](https://img.shields.io/github/issues-pr/NormanTUD/OmniOpt)
![Stars](https://img.shields.io/github/stars/NormanTUD/OmniOpt)
![Forks](https://img.shields.io/github/forks/NormanTUD/OmniOpt)
![Contributors](https://img.shields.io/github/contributors/NormanTUD/OmniOpt)
![Last Commit](https://img.shields.io/github/last-commit/NormanTUD/OmniOpt)
[![Coverage Status](https://coveralls.io/repos/github/NormanTUD/OmniOpt/badge.svg?branch=main)](https://coveralls.io/github/NormanTUD/OmniOpt?branch=main)

## Old OmniOpt

-- deprecated, not supported anymore --

OmniOpt is a hyperparameter minimizer tool. This means you can give it a
program that accepts hyperparameters as parameters via the command-line, like

```bash
python your_program.py --param1=$PARAM1 --param2=$PARAM2 ...
```

and returns lines like:

```bash
RESULT: 0.1234
```

in it's standard output. OmniOpt will try to find the lowest result-value in
an intelligent way automatically.

## How to run OmniOpt?

It's probably easiest to use the GUI available under
<https://imageseg.scads.de/omnioptgui/>. Running it manually without the GUI
is not recommended.

## Something has gone wrong. Where can I check what and fix it?

If something has gone wrong, your first step should be looking into the
`projects/PROJECTNAME/singlelogs`-folder.

If it does exist, choose any random file that has no extension and look into
it. It will have the command that was executed and it's `STDOUT` and `STDERR`.
It is very probable that, when executing the command in the first line
manually, you will get an error which is similiar to the error in the file. If
so, then try fixing it in your program.

If this folder does not exist or you could not find an error, check the `.out`
file in the main directory of OmniOpt. Errors will be printed in red. If there
are no errors, look at the first line, it looks like this:

```bash
Log file: /home/h8/s3811141/test/randomtest_71361/omniopt/debuglogs/0
```

This log (the number at the end may vary) contains all debug-outputs.

A common error may be:

```bash
The Job is running right now. It's slurm ID is 19027181. Cannot run repair
with a database that is running.
...
The file `/path/test/projects/gpu_test/mongodb/mongod.lock` already exists!!!

```

This error is resolved by waiting for the other job (in thise case 19027181)
to finish and re-starting the current job.

Once a job is started, another one cannot be started on the same
`projects`-folder again.

If this does not help, do not hesitate to contact me at
<mailto:norman.koch@tu-dresden.de>.

## What do the files and folders here mean?

## debuglogs

For every optimization run with OmniOpt, in this folder a new file with an
auto-incrementing number will be created. It contains all debug-outputs of
OmniOpt, even if `--debug` is not set, so one can always look here if
something goes wrong.

## documentation

This contains the documentation of OmniOpt.

## evaluate-run.sh

Run this with

```bash
bash evaluate-run.sh
```

To get access to the results in many different formats.

## gui

This contains the source code of the OmniOpt-GUI, available at
<https://imageseg.scads.de/omnioptgui/>.

## logfile.txt (only exists if at least one optimization has run)

This file is created by Hyperopt and is always empty. It can be ignored.

## perllib

This directory contains the Perl-Modules needed to run OmniOpt.

## projects

This is the default directory, in which projects reside. A project must
consist of a folder named like the project, which contains a `config.ini`
file. When a project has run, the folders `mongodb`, `singlelogs` and `logs`
are  contained within the specific project's folder. `mongodb` contains the
database on which OmniOpt operates, `singlelogs` contains 3 files for each
single run, each starting with a GUID, namely the `.stdout`-file, which
contains the `STDOUT` of the run, the `.stderr`-file, which contains it's
`STDERR` and a file without file-ending, which contains both, the `STDOUT` and
the `STDERR` also also the command-line with the parsed arguments.

Check these files if something goes wrong first. This is where your programs
are called.

## README.md

Well... if you don't know what this file is, I cannot help you.

## sbatch.pl

This is the main-script that loads all modules, installs all dependencies,
starts the MongoDB-instance, the workers and all logging tools.

## script

In this directory, all the python-scripts abstracting away from HyperOpt
reside. You probably don't need to touch any of them, ever.

## test

This directory contains several tests to ensure that OmniOpt runs properly. It
also contains a different `projects`-folder with very small and simple
test-projects that only test OmniOpt. You probably don't need them. If you
want to test OmniOpt, run

```bash
perl sbatch.pl --debug --run_full_tests
```

for the complete test-suite (with starting `sbatch`-jobs) or

```bash
perl sbatch.pl --debug --run_tests
```

for the faster, but incomplete test-suite, which does not start any
`sbatch`-jobs. This is also accessable via `evaluate-run.sh`.

## tools

This directory contains different scripts to abstract away Taurus-specifics.
Like, for plotting graphs, you need to load a certain number of modules in a
certain order, before `python3 script/plot3.py` works. But because that's
really annoying to remember, just use

```bash
perl tools/plot.pl --project=testproject --projectdir=./projects/
```

or use the `evaluate-run.sh`.

## zsh

Autocompletetion for zsh. Use

```bash
bash zsh/install.sh
```

for enabling auto-completion on ZSH after re-starting the ZSH.
This must not be done for every repo, once is enough.
