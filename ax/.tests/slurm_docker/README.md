[![Docker Slurm](https://github.com/NOAA-GSL/DockerSlurmCluster/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/NOAA-GSL/DockerSlurmCluster/actions/workflows/docker.yml)

# Slurm Cluster in Ubuntu Docker Images Using Docker Compose
This is an installation of a Slurm cluster inside Docker.

This is an adaptation of the work done by Rodrigo Ancavil del Pino:

https://medium.com/analytics-vidhya/slurm-cluster-with-docker-9f242deee601

There are three containers:

* A front-end container that acts as a Slurm cluster front-end node
* A master container that acts as a Slurm master node
* A node container that acts as a Slurm compute node

These containers are launched using Docker Compose to build
a fully functioning Slurm cluster.  A `docker-compose.yml`
file defines the cluster, specifying ports and volumes to
be shared.  Multiple instances of the node container can be
used to create clusters of different sizes.  The cluster
behaves as if it were running on multiple nodes even if the
containers are all running on the same host machine.

# Quick Start

To start the slurm cluster environment:
```
docker-compose -f docker-compose.yml up -d
```
To stop the cluster:
```
docker-compose -f docker-compose.yml stop
```
To check the cluster logs:
```
docker-compose -f docker-compose.yml logs -f
```
(stop logs with CTRL-c")

To check status of the cluster containers:
```
docker-compose -f docker-compose.yml ps
```
To check status of Slurm:
```
docker exec slurm-frontend sinfo
```
To run a Slurm job:
```
docker exec slurm-frontend srun hostname
```

## OmniOpt use case:

Build with `bash install_docker`. Then:

```
docker exec slurm-frontend bash /oo_dir/.tests/main --max_eval=2 --num_random_steps=1
```

## OmniOpt Run:

```
docker exec slurm-frontend bash /oo_dir/omniopt --live_share --send_anonymized_usage_stats --partition alpha --experiment_name __main__tests__BOTORCH_MODULAR___nogridsearch --mem_gb=4 --time 60 --worker_timeout=5 --max_eval 2 --num_parallel_jobs 1 --gpus 0 --run_program Li8udGVzdHMvb3B0aW1pemF0aW9uX2V4YW1wbGUgLS1pbnRfcGFyYW09JyUoaW50X3BhcmFtKScgLS1mbG9hdF9wYXJhbT0nJShmbG9hdF9wYXJhbSknIC0tY2hvaWNlX3BhcmFtPSclKGNob2ljZV9wYXJhbSknICAtLWludF9wYXJhbV90d289JyUoaW50X3BhcmFtX3R3byknIC0tbnJfcmVzdWx0cz0x --parameter int_param range -100 10 int --parameter float_param range -100 10 float --parameter choice_param choice 1,2,4,8,16,hallo --parameter int_param_two range -100 10 int --follow --num_random_steps 1 --model BOTORCH_MODULAR --auto_exclude_defective_hosts --hide_ascii_plots
```
