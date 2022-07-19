#!/bin/bash
#
###############################################
##### Superconductivity Featurizer Script #####
###############################################
# This bash script runs the superconductivy featurizers in ./build_features.py on the Cornell CLASSE compute farm
# and emails kvk23@cornell.edu when completed
#
# Author: Kirk Kleinsasser
###############################################
#
# Could do "qsub -e `pwd` -m e -M kvk23@cornell.edu -N featurizer.results -o `pwd` -pe sge_pe 12 -q all.q -S /bin/bash"
#
# Passes variable to set the run time name of the job
#$ -N featurizer.results
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
# Print some environment information - for reporting diagnostics only.
echo "Job ${JOBNAME} Starting at: "`date`
echo "Running with arguments \""$@"\" on host: "`hostname`
#
# Run the python script
source /opt/rh/rh-python38/enable
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7
python3 ./code/build_features.py $@

# Documentation:
# https://wiki.classe.cornell.edu/Computing/GridEngine - CLASSE wiki
# https://mail.google.com/mail/u/0/#inbox/FMfcgzGpGTHKnnfqXlTLRRSKVzDfDqrP - Cornell IT explaination of submiting items to queue while using python libraries
