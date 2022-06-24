#!/bin/bash
#this is a template script for use in the batch queueing system. Please only edit the user sections.
#send files to compute farm: https://wiki.classe.cornell.edu/Computing/GridEngine
LOCAL_DIR=${TMPDIR}
################USER SETTINGS###########################
#you can edit the variables here, but something valid must be present.
#SOURCE_DIR defaults to a files subdirectory of the directory you submitted the job from
#comment out to not import any files for the job. Edit for your specific needs.
SOURCE_DIR=${SGE_O_WORKDIR}   
DESTINATION_DIR=/cdat/tem/sge/${USER}/${JOB_ID}
########################################################
if [ ! -z "${SOURCE_DIR}" ]; then
   rsync -avz ${SOURCE_DIR}/ ${LOCAL_DIR}
fi

#put your code in the user section below. 
#################USER SECTION########################### 

echo "Starting Script!"
python3 ./build_features.py

#################END USER SECTION#######################
mkdir -p ${DESTINATION_DIR}
ls ${DESTINATION_DIR}
rsync -avz ${LOCAL_DIR}/ ${DESTINATION_DIR}