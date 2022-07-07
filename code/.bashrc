alias cstat="clear; cd /cdat/tem/kvk23/queue; qstat; ls -l"

compute() {
    cd /cdat/tem/kvk23/queue
    qsub $@
    qstat
}