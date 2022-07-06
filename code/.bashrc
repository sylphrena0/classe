defaultopt() {
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
    # Print some environment information - for reporting diagnostics only.
    echo "Job ${JOBNAME} Starting at: "`date`
    echo "Running on host: "`hostname`
    echo "In directory: "`pwd`
    #
    # Run the python script
    source /opt/rh/rh-python38/enable
    export PYTHONPATH=/home/kvk23/.local/lib/python3.8/site-packages #use local python packages, installed with pip3 install --user <package>
    python3 ./model_optimizer.py --samplesize 1000 # pass inputs to python (limits and enabled types)
}

optimize() {
    defaultopt()
    cd /cdat/tem/kvk23/queue
    qsub $@
    qstat

    local arg from to
    while getopts 'nt:cr:' arg
    do
        case ${arg} in
            nt) notify=${OPTARG};;
            cr) cores=${OPTARG};;
            *) return 1 # illegal option
        esac
    done

}
