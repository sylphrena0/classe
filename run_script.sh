#!/bin/bash
#
######################################################
##### Bash Script to Submit Python Files to Qsub #####
######################################################
# This bash script runs python scripts on the Cornell CLASSE compute farm
# and accepts arguments for qsub and for the python script. 
# Syntax (using ./.bashrc) is "compute --pythonscript <scriptname> <scriptarguments>"
#
# Author: Kirk Kleinsasser
######################################################
#
# Passes variable to set the run time name of the job
#$ -N script.results
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
echo "Running with command \"compute run_optimizer "$@"\" on host: "`hostname`

#read the options
TEMP=`getopt -o '' --long pythonscript: -- "$@" command 2> /dev/null` #command 2> /dev/null suppresses unknown argument errors here
eval set -- "$TEMP"

#extract options and their arguments into variables.
while true ; do
    case "$1" in
        --pythonscript)
            pythonscript=$2 ; shift 2 ;;
        "") shift ; break ;; #breaks while loop when we run out of arguments
        *) break ;; #at last argument defined for bash, rest are sent to python
    esac
done

# Run the python script
source /opt/rh/rh-python38/enable
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7/lib/python3.8/site-packages:/nfs/opt/python3.8/packages.sl7:/home/kvk23/.local/lib/python3.8/site-packages #use local python packages, installed with pip3 install --user <package>
python3 ./code/$pythonscript $@ #pass inputs to python (limits and enabled types)

# Documentation:
# https://wiki.classe.cornell.edu/Computing/GridEngine - CLASSE wiki
# https://mail.google.com/mail/u/0/#inbox/FMfcgzGpGTHKnnfqXlTLRRSKVzDfDqrP - Cornell IT explaination of submiting items to queue while using python libraries
# https://www.tutorialspoint.com/unix_commands/getopt.htm - tutorial on using getopt
