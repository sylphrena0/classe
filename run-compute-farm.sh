#!/bin/bash
#This is a template script for use in the batch queueing system. Please only edit the user sections.
LOCAL_DIR=${TMPDIR}
################USER SETTINGS###########################
#You can edit the variables here, but something valid must be present.
# SOURCE_DIR defaults to a files subdirectory of the directory you submitted the job from
#Comment out to not import any files for the job. Edit for your specific needs.
SOURCE_DIR=${SGE_O_WORKDIR}   
DESTINATION_DIR=/cdat/tem/sge/${USER}/${JOB_ID}
########################################################
if [ ! -z "${SOURCE_DIR}" ]; then
   rsync -avz ${SOURCE_DIR}/ ${LOCAL_DIR}
fi

#Put your code in the user section below. You can delete the entire
#section between USER SECTION and END USER SECTION - it is a very simple
#example script that does a loop and echos some job data for testing.
#################USER SECTION###########################
python3
#################END USER SECTION#######################
mkdir -p ${DESTINATION_DIR}
ls ${DESTINATION_DIR}
rsync -avz ${LOCAL_DIR}/ ${DESTINATION_DIR}