#!/bin/bash
#
# To set the run time name of the job
#$ -N get_featurizers
#
# Set the queue
#$ -q all.q
#
# Set shell, but the default is /bin/bash
#$ -S /bin/bash
#
# To make sure that the .e and .o files arrive in the current working directory
#$ -cwd
#
# Send an email when the job ends
#$ -m e -M kvk23@cornell.edu
#
# Print some environment information - for reporting diagnostics only.
echo "Job ${JOBNAME} Starting at: "`date`
echo "Running on host: "`hostname`
echo "In directory: "`pwd`
#
# Run the python script
source /opt/rh/rh-python38/enable
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7
python ./get_featurizers.py #my code
#
#
