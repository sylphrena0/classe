#!/bin/bash
#
# Passes variable to set the run time name of the job
#$ -N supercon_optimize_results
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
# Send an email when the job starts, ends, or errors out
#$ -m bea -M kvk23@cornell.edu
#
# Set thread requirements
#$ -pe sge_pe 12
# Print some environment information - for reporting diagnostics only.
echo "Job ${JOBNAME} Starting at: "`date`
echo "Running on host: "`hostname`
echo "In directory: "`pwd`
#
# Run the python script
source /opt/rh/rh-python38/enable
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7
python3 ./supercon_optimize.py
#
#
