#!/bin/bash

source debug.sh

module_load modenv/classic
module_load mongodb/3.6.3

perl check_slurm_files.pl $*
