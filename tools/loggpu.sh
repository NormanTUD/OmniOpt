#!/bin/bash

export LOGFILE=$1
export HOST=$(hostname)
export IPFILESDIR=$2
export CUDA_FILE=$IPFILESDIR/omniopt_CUDA_VISIBLE_DEVICES_${3}_$(hostname)

if command -v nvidia-smi 2>/dev/null >/dev/null; then
	if [ -f $CUDA_FILE ]; then
	    touch $LOGFILE
	    if [[ -s $LOGFILE ]]; then
		    # File is NOT empty
		    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader >> $LOGFILE
	    else
		    # File IS empty
		    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv > $LOGFILE
	    fi
#    else
#        echo "CUDA_FILE $CUDA_FILE not found" >&2
	fi
#else
#    echo "nvidia-smi not found" >&2
fi
