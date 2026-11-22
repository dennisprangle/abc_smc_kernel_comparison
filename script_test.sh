#!/bin/bash
# Quick test of the parallel setup: short runs of every example and proposal, of the r-hit kernel variants,
# and of non-default numbers of particles and epsilon reduction parameter,
# followed by some checks on the output.
# The job list, job log and each job's stdout/stderr are saved in $LOGDIR.
# Each run saves its summary to its own file in $RUNDIR, and these are combined at the end.
NJOBS=8
LOGDIR=logs/test
RUNDIR=results/runs/test
OUTPUT=results/test.csv
# Quadratic example histories are saved in results/histories_$NAME
NAME=test
# One thread per process, so parallel jobs don't compete for cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

rm -rf $LOGDIR $RUNDIR results/histories_$NAME $OUTPUT
mkdir -p $LOGDIR $RUNDIR

# Standard arguments for each proposal
proposal_args() {
	case $1 in
		mix) echo " --disable_fallback --mix_components 5" ;;
		flow) echo " --disable_fallback" ;;
	esac
}

# Print one command per line
make_jobs() {
for seed in 1 2
do
	for proposal in mix flow classic_ind
	do
		args="--proposal $proposal --seed $seed --kernel 1_hit$(proposal_args $proposal) --max_mins 1 --results_dir $RUNDIR"
		# History file names don't include the seed, so only save histories for seed 1
		history_args=""
		[ $seed = 1 ] && history_args=" --name $NAME --save_history --save_wloss"
		echo "python quadratic_example.py $args$history_args"
		echo "python MG1_example.py $args --quantile_summaries"
		echo "python slcp_example.py $args"
		echo "python SEIR_example.py $args"
		echo "python gaussian_mixture_example.py $args"
	done
done
# r-hit kernel variants. r=3 tests the early acceptance bound, and multi early rejection with more than one remaining hit
for kernel in r_hit multi_r_hit r_hit_ea multi_r_hit_ea r_hit_er multi_r_hit_er
do
	args="--proposal mix --seed 1 --kernel $kernel --hits 3$(proposal_args mix) --max_mins 1 --results_dir $RUNDIR"
	echo "python quadratic_example.py $args"
	echo "python MG1_example.py $args --quantile_summaries"
	echo "python slcp_example.py $args"
	echo "python SEIR_example.py $args"
	echo "python gaussian_mixture_example.py $args"
done
# Non-default number of particles and epsilon reduction parameter (as in script_tuning.sh).
# Each proposal sets up its particles differently, so test them all
for proposal in mix flow classic_ind
do
	args="--proposal $proposal --seed 1 --kernel 1_hit$(proposal_args $proposal) --n_particles 200 --erp 0.25 --max_mins 1 --results_dir $RUNDIR"
	echo "python quadratic_example.py $args"
	echo "python MG1_example.py $args --quantile_summaries"
	echo "python slcp_example.py $args"
	echo "python SEIR_example.py $args"
	echo "python gaussian_mixture_example.py $args"
done
}

make_jobs > $LOGDIR/jobs.txt
parallel -j $NJOBS --joblog $LOGDIR/joblog.txt --results "$LOGDIR/{#}/" < $LOGDIR/jobs.txt

python combine_results.py $RUNDIR $OUTPUT

# Checks
echo
echo "== Failed jobs (exit code: command). See $LOGDIR/<job number>/stderr for details:"
awk -F'\t' 'NR>1 && $7 != 0 {print $7": "$9}' $LOGDIR/joblog.txt
echo "== Job run times (seconds): should all be a bit over 60, not much more"
awk -F'\t' 'NR>1 {printf "%.0f ", $4} END {print ""}' $LOGDIR/joblog.txt
echo "== Rows in $OUTPUT: $(($(wc -l < $OUTPUT) - 1)), expected $(wc -l < $LOGDIR/jobs.txt)"
echo "== First rows of $OUTPUT: should have hits, defensive_proportion and seed columns"
head -3 $OUTPUT
echo "== Training proportions: should be between 0 and 1, and highest for flow"
python -c "import pandas as pd; print(pd.read_csv('$OUTPUT')[['model', 'proposal', 'seed', 'training_proportion']].to_string(index=False))"
echo "== Particles and erp: should be 200 and 0.25 for the 15 tuning-test rows, 1000 and 0.5 for the rest"
python -c "import pandas as pd; print(pd.read_csv('$OUTPUT').groupby(['n_particles', 'erp']).size().to_string())"
echo "== Quadratic histories: each should have wloss and training_proportion columns"
for f in results/histories_$NAME/*_history.csv
do
	echo "$f: $(head -1 $f)"
done
