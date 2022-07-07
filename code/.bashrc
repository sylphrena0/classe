alias cstat="clear; cd /cdat/tem/kvk23/queue; qstat; ls -lXr" #shows queue status and ls
alias wcompute="while true; do cstat; sleep 5; done" #runs cstat every five seconds till interupt

compute() { #starts a job
    cd /cdat/tem/kvk23/queue
    qsub $@
    qstat
}