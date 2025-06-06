#!/usr/bin/env bash

export SLURM_CPUS_ON_NODE=$(cat /proc/cpuinfo | grep processor | wc -l)
sudo sed -i "s/REPLACE_IT/${SLURM_CPUS_ON_NODE}/g" /etc/slurm/slurm.conf

sudo service munge start
sudo slurmctld
sudo service ssh start

tail -f /dev/null
