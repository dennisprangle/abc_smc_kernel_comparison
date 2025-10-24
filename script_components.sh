#!/bin/bash
rm results/results.csv

for seed in {1..5}
do
        for comp in {3..10}
        do
                python quadratic_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components $comp --disable_fallback
                python MG1_example.py --proposal mix --seed $seed --kernel 1_hit --quantile_summaries --mix_components $comp --disable_fallback
                python slcp_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components $comp --disable_fallback
                python SEIR_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components $comp --disable_fallback
                python gaussian_mixture_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components $comp --disable_fallback
        done

        python quadratic_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components 3 10 --disable_fallback
        python MG1_example.py --proposal mix --seed $seed --kernel 1_hit --quantile_summaries --mix_components 3 10 --disable_fallback
        python slcp_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components 3 10 --disable_fallback
        python SEIR_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components 3 10 --disable_fallback
        python gaussian_mixture_example.py --proposal mix --seed $seed --kernel 1_hit --mix_components 3 10 --disable_fallback
done

mv results/results.csv results/components.csv
