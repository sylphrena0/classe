alias cstat="clear; cd /cdat/tem/kvk23/queue; qstat; ls -lXr" #shows queue status and ls
alias wcompute="while true; do cstat; sleep 5; done" #runs cstat every five seconds till interupt
alias rpip3="source /opt/rh/rh-python38/enable; export PYTHONPATH=/nfs/opt/python3.8/packages.sl7/lib/python3.8/site-packages:/nfs/opt/python3.8/packages.sl7" #sources python 3.8 and libraries

compute() { #starts a job
    cd /cdat/tem/kvk23/queue/
    qsub $@
    qstat
}