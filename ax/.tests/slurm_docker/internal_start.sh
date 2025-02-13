#!/bin/bash
set -e

# Munge starten
echo "Starting MUNGE..."
service munge start

# MariaDB starten (statt mysql)
echo "Starting MariaDB..."
service mariadb start

# Prüfen, ob slurm.conf existiert
if [ ! -f /etc/slurm/slurm.conf ]; then
    echo "ERROR: slurm.conf not found in /etc/slurm!"
    exit 1
fi

# Slurm-Dienste starten
echo "Starting Slurm..."
service slurmd start
service slurmctld start

# Container am Leben halten
tail -f /dev/null
