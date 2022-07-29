#!/bin/bash

#print some environment information for diagnostics:
echo "Job ${JOBNAME} Starting at: "`date`
echo "Running with command \"compute run_single_training.sh "$@"\" on host: "`hostname`
#use specific version of python
source /opt/rh/rh-python38/enable
#use local python packages, installed with pip3 install --user <package>:
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7/lib/python3.8/site-packages:/nfs/opt/python3.8/packages.sl7:/home/kvk23/.local/lib/python3.8/site-packages 
#run the python script:
python3 ./code/$1.py $@ #the $@ passes remaining arguments to python