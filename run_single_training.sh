#!/bin/bash
#
####################################################
##### Superconductivity Single Training Script #####
####################################################
# This bash script runs the superconductivy model training in ./code/training_single.py on the Cornell CLASSE compute farm
# and emails kvk23@cornell.edu when completed
#
# Author: Kirk Kleinsasser
####################################################
#
# Passes variable to set the run time name of the job
#$ -N train_single.results
#
# Set the queue
#$ -q all.q
#
# Set shell, but the default is /bin/bash
#$ -S /bin/bash
#
# To make sure that the error.e and output.o files arrive in the current working directory
#$ -cwd
#
# Send an email when the job ends
#$ -m e -M kvk23@cornell.edu
#
# Set thread requirements
#$ -pe sge_pe 32
#
# Combine error and output files into single file
#$ -j y
#
# Print some environment information - for reporting diagnostics only.
echo "Job ${JOBNAME} Starting at: "`date`
echo "Running with command \"compute run_single_training.sh "$@"\" on host: "`hostname`
#
# Run the python script
source /opt/rh/rh-python38/enable
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7/lib/python3.8/site-packages:/nfs/opt/python3.8/packages.sl7:/home/kvk23/.local/lib/python3.8/site-packages #use local python packages, installed with pip3 install --user <package>
python3 ./code/training_single.py $@ # pass inputs to python (limits and enabled types)

# Documentation:
# https://wiki.classe.cornell.edu/Computing/GridEngine - CLASSE wiki
# https://mail.google.com/mail/u/0/#inbox/FMfcgzGpGTHKnnfqXlTLRRSKVzDfDqrP - Cornell IT explaination of submiting items to queue while using python libraries
