#!/bin/bash
# Runs the jobs below in parallel using GNU parallel.
# The job list, job log and each job's stdout/stderr are saved in $LOGDIR.
# Each run saves its summary to its own file in $RUNDIR, and these are combined at the end.
NJOBS=8
LOGDIR=logs/fallback
RUNDIR=results/runs/fallback
# One thread per process, so parallel jobs don't compete for cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

rm -rf $RUNDIR
mkdir -p $LOGDIR $RUNDIR

# Print one command per line
make_jobs() {
for seed in {1..5}
do
	for fallback_flag in "" " --disable_fallback"
	do
		for proposal in mix flow
		do
			args="--proposal $proposal --seed $seed --kernel 1_hit --results_dir $RUNDIR"
			[ $proposal = mix ] && args+=" --mix_components 5"
			args+="$fallback_flag"
			echo "python quadratic_example.py $args"
			echo "python MG1_example.py $args --quantile_summaries"
			echo "python slcp_example.py $args"
			echo "python SEIR_example.py $args"
			echo "python gaussian_mixture_example.py $args"
		done
	done
done
}

make_jobs > $LOGDIR/jobs.txt
parallel -j $NJOBS --joblog $LOGDIR/joblog.txt --results "$LOGDIR/{#}/" < $LOGDIR/jobs.txt

python combine_results.py $RUNDIR results/fallback.csv
