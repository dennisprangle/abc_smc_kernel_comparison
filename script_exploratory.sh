#!/bin/bash
# Runs the jobs below in parallel using GNU parallel.
# The job list, job log and each job's stdout/stderr are saved in $LOGDIR.
# Each run saves its summary to its own file in $RUNDIR, and these are combined at the end.
NJOBS=8
LOGDIR=logs/exploratory
RUNDIR=results/runs/exploratory
# One thread per process, so parallel jobs don't compete for cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

rm -rf $RUNDIR
mkdir -p $LOGDIR $RUNDIR

# Print one command per line
make_jobs() {
for kernel in ind_1_hit 1_hit naive r_hit_er multi_r_hit_er
do
	args="--seed 1 --kernel $kernel --save_history --save_wloss --results_dir $RUNDIR"
	# classic_rw isn't run with the independence kernel
	[ $kernel != ind_1_hit ] && echo "python quadratic_example.py --proposal classic_rw $args"
	echo "python quadratic_example.py --proposal classic_ind $args"
	echo "python quadratic_example.py --proposal mix $args"
	echo "python quadratic_example.py --proposal flow $args"
done
}

make_jobs > $LOGDIR/jobs.txt
parallel -j $NJOBS --joblog $LOGDIR/joblog.txt --results "$LOGDIR/{#}/" < $LOGDIR/jobs.txt

python combine_results.py $RUNDIR results/exploratory.csv
