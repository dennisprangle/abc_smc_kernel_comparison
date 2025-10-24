#!/bin/bash
rm results/results.csv

for seed in {1..5}
do
    for flow in 'spline' 'spline_two_layers' 'coupling'
    do
        python quadratic_example.py --proposal flow --seed $seed --kernel 1_hit --flow $flow --disable_fallback
        python MG1_example.py --proposal flow --seed $seed --kernel 1_hit --flow $flow --quantile_summaries --disable_fallback
        python slcp_example.py --proposal flow --seed $seed --kernel 1_hit --flow $flow --disable_fallback
        python SEIR_example.py --proposal flow --seed $seed --kernel 1_hit --flow $flow --disable_fallback
        python gaussian_mixture_example.py --proposal flow --seed $seed --kernel 1_hit --flow $flow --disable_fallback
    done
done

mv results/results.csv results/flow.csv
