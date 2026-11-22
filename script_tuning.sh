#!/bin/bash
# Effect of the number of particles and the epsilon reduction parameter (erp), on a 3x3 grid,
# for the MG1 model, with the mixture, flow and classic random walk proposals and the kernels that use a single hit (or none).
# (classic_rw isn't run with the independence kernel.)
# (1000 particles and erp 0.5 are the defaults used by the other scripts.)
# Runs the jobs below in parallel using GNU parallel.
# The job list, job log and each job's stdout/stderr are saved in $LOGDIR.
# Each run saves its summary to its own file in $RUNDIR, and these are combined at the end.
NJOBS=8
LOGDIR=logs/tuning_mg1
RUNDIR=results/runs/tuning_mg1
# One thread per process, so parallel jobs don't compete for cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

rm -rf $RUNDIR
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
for seed in {1..5}
do
	for n_particles in 500 1000 2000
	do
		for erp in 0.25 0.5 0.75
		do
			for proposal in mix flow classic_rw
			do
				for kernel in naive 1_hit ind_1_hit
				do
					# classic_rw isn't run with the independence kernel
					[ $proposal = classic_rw ] && [ $kernel = ind_1_hit ] && continue
					args="--proposal $proposal --seed $seed --kernel $kernel$(proposal_args $proposal) --n_particles $n_particles --erp $erp --results_dir $RUNDIR"
					echo "python MG1_example.py $args --quantile_summaries"
				done
			done
		done
	done
done
}

make_jobs > $LOGDIR/jobs.txt
parallel -j $NJOBS --joblog $LOGDIR/joblog.txt --results "$LOGDIR/{#}/" < $LOGDIR/jobs.txt

python combine_results.py $RUNDIR results/tuning_mg1.csv
