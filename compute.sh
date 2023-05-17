#!/bin/bash
######################################################
##### Bash Script to Submit Python Files to Qsub #####
######################################################
# This bash script runs python scripts on the Cornell CLASSE compute farm
# and accepts arguments for qsub and for the python script [attempts to accept qsub args]
#
# Author: Sylphrena Kleinsasser
######################################################
#TODO: Figure out why this won't run

#read the options
TEMP=`getopt -o pf:c::m::n::e:: --long pythonfile:,cores::,memory::,notify::,email:: -- "$@" command 2> /dev/null` #command 2> /dev/null suppresses unknown argument errors here
eval set -- "$TEMP"

#extract options and their arguments into variables.
while true ; do
    case "$1" in
        -pf|--pythonfile) #IMPORTANT: file arg should not include .py extension
            file=$2 ; shift 2 ;;
        -c|--cores)
            case "$2" in
                "") cores='32' ; shift 2 ;;
                 *) cores=$2 ; shift 2 ;;
            esac ;;
        -m|--memory)
            case "$2" in
                "") memory='32G' ; shift 2 ;;
                 *) memory=$2 ; shift 2 ;;
            esac ;;
        -n|--notify)
            case "$2" in
                "") notify='e' ; shift 2 ;;
                 *) notify=$2 ; shift 2 ;;
            esac ;;
        -e|--email)
            case "$2" in
                "") email='kvk23@cornell.edu' ; shift 2 ;;
                 *) email=$2 ; shift 2 ;;
            esac ;;
        "") shift ; break ;; #breaks while loop when we run out of arguments
        *) break ;; #at last argument defined for bash, rest are sent to python
    esac
done

exec "qsub" -N $file -q all.q -S /bin/bash -cwd -m $notify -M $email -pe sge_pe $cores -l mem_free=$memory -j y compute_run.sh $@

==> compute_run.sh <== #would be in a seperate file, but this doesn't work (I don't know why) so I don't want the clutter!
#print some environment information for diagnostics:
echo "Job ${JOBNAME} Starting at: "`date`
echo "Running with command \"compute run_single_training.sh "$@"\" on host: "`hostname`
#use specific version of python
source /opt/rh/rh-python38/enable
#use local python packages, installed with pip3 install --user <package>:
export PYTHONPATH=/nfs/opt/python3.8/packages.sl7/lib/python3.8/site-packages:/nfs/opt/python3.8/packages.sl7:/home/kvk23/.local/lib/python3.8/site-packages 
#run the python script:
python3 ./code/$1.py $@ #the $@ passes remaining arguments to python
==>               <== 

# Documentation:
# https://wiki.classe.cornell.edu/Computing/GridEngine - CLASSE wiki
# https://mail.google.com/mail/u/0/#inbox/FMfcgzGpGTHKnnfqXlTLRRSKVzDfDqrP - Cornell IT explaination of submiting items to queue while using python libraries
# https://www.tutorialspoint.com/unix_commands/getopt.htm - tutorial on using getopt