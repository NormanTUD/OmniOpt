#!/bin/bash

set -e
set -x

if [ ! -f /etc/slurm/slurm.conf ]; then
    echo "ERROR: slurm.conf not found in /etc/slurm!"
    exit 1
fi

echo "Starting MUNGE..."
service munge start

echo "Starting MariaDB in debug mode..."
mysqld --user=mysql --skip-grant-tables --log-error-verbosity=3 --debug


echo "Starting Slurm..."
service slurmd start
service slurmctld start

tail -f /dev/null
