#!/bin/bash
rm results/results.csv

for seed in {1..5}
do
        for hits in {2..5}
        do
                python quadratic_example.py --proposal mix --seed $seed --kernel multi_r_hit --hits $hits --mix_components 5 --disable_fallback
                python MG1_example.py --proposal mix --seed $seed --kernel multi_r_hit --hits $hits --quantile_summaries --mix_components 5 --disable_fallback
                python slcp_example.py --proposal mix --seed $seed --kernel multi_r_hit --hits $hits --mix_components 5 --disable_fallback
                python SEIR_example.py --proposal mix --seed $seed --kernel multi_r_hit --hits $hits --mix_components 5 --disable_fallback
                python gaussian_mixture_example.py --proposal mix --seed $seed --kernel multi_r_hit --hits $hits --mix_components 5 --disable_fallback
        done
done

mv results/results.csv results/hits.csv