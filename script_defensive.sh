#!/bin/bash
# Runs the jobs below in parallel using GNU parallel.
# The job list, job log and each job's stdout/stderr are saved in $LOGDIR.
# Each run saves its summary to its own file in $RUNDIR, and these are combined at the end.
NJOBS=8
LOGDIR=logs/defensive
RUNDIR=results/runs/defensive
# One thread per process, so parallel jobs don't compete for cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

rm -rf $RUNDIR
mkdir -p $LOGDIR $RUNDIR

# Print one command per line
make_jobs() {
for seed in {1..5}
do
	for kernel in naive 1_hit ind_1_hit r_hit_er multi_r_hit_er
	do
		args="--proposal mix --seed $seed --kernel $kernel --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60 --results_dir $RUNDIR"
		echo "python quadratic_example.py $args"
		echo "python MG1_example.py $args --quantile_summaries"
		echo "python slcp_example.py $args"
		echo "python SEIR_example.py $args"
		echo "python gaussian_mixture_example.py $args"
	done
done
}

make_jobs > $LOGDIR/jobs.txt
parallel -j $NJOBS --joblog $LOGDIR/joblog.txt --results "$LOGDIR/{#}/" < $LOGDIR/jobs.txt

python combine_results.py $RUNDIR results/defensive.csv
