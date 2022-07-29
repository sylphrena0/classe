#!/bin/bash
######################################################
##### Bash Script to Submit Python Files to Qsub #####
######################################################
# This bash script runs python scripts on the Cornell CLASSE compute farm
# and accepts arguments for qsub and for the python script
#
# Author: Kirk Kleinsasser
######################################################

#read the options
TEMP=`getopt -o pf:c::m::n::e:: --long pythonfile:,cores::,memory::,notify::,email:: -- "$@" command 2> /dev/null` #command 2> /dev/null suppresses unknown argument errors here
echo "start loop"
#extract options and their arguments into variables.
while true ; do
    case "$1" in
        -pf|--pythonfile)
            file=$2 ; echo $1; shift 2 ;;
        -c|--cores)
            case "$2" in
                "") cores='32';  echo $1; shift 2 ;;
                 *) cores=$2; echo $1; shift 2 ;;
            esac ;;
        -m|--memory)
            case "$2" in
                "") memory='32G' ; echo $1; shift 2 ;;
                 *) memory=$2 ; echo $1; shift 2 ;;
            esac ;;
        -n|--notify)
            case "$2" in
                "") notify='e' ; echo $1; shift 2 ;;
                 *) notify=$2 ; echo $1; shift 2 ;;
            esac ;;
        -e|--email)
            case "$2" in
                "") email='kvk23@cornell.edu' ; echo $1; shift 2 ;;
                 *) email=$2 ; echo $1; shift 2 ;;
            esac ;;
        --) shift ; echo "break '--'"; break ;; #breaks while loop when we run out of arguments
        "") shift ; echo "break ''"; break ;; #breaks while loop when we run out of arguments
        *) echo "last defined argument"; break ;;
    esac
done
echo "loop done"
#passes variable to set the run time name of the job
#$ -N compute.results
#
#set the queue
#$ -q all.q
#
#set shell, but the default is /bin/bash
#$ -S /bin/bash
#
#to make sure that the error.e and output.o files arrive in the current working directory
#$ -cwd
#
#combine error and output files into single file
#$ -j y
#
#print some environment information - for diagnostics
echo "Job ${JOBNAME} Starting at: "`date`
echo "Running with command \"compute run_single_training.sh "$@"\" on host: "`hostname`
#
#run the python script
source /opt/rh/rh-python38/enable
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7/lib/python3.8/site-packages:/nfs/opt/python3.8/packages.sl7:/home/kvk23/.local/lib/python3.8/site-packages #use local python packages, installed with pip3 install --user <package>
#python3 ./code/$file $@ # pass inputs to python (limits and enabled types)

# Documentation:
# https://wiki.classe.cornell.edu/Computing/GridEngine - CLASSE wiki
# https://mail.google.com/mail/u/0/#inbox/FMfcgzGpGTHKnnfqXlTLRRSKVzDfDqrP - Cornell IT explaination of submiting items to queue while using python libraries
# https://www.tutorialspoint.com/unix_commands/getopt.htm - tutorial on using getopt
