#!/bin/bash
rm results/results.csv

for seed in {1..5}
do
	python quadratic_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python quadratic_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python quadratic_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python quadratic_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python MG1_example.py --proposal mix --seed $seed --kernel naive --quantile_summaries --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python MG1_example.py --proposal mix --seed $seed --kernel 1_hit --quantile_summaries --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python MG1_example.py --proposal mix --seed $seed --kernel ind_1_hit --quantile_summaries --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python MG1_example.py --proposal mix --seed $seed --kernel multi_r_hit --quantile_summaries --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python slcp_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python slcp_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python slcp_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python slcp_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python SEIR_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python SEIR_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python SEIR_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python SEIR_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5 --defensive_proportion 0.1 --max_mins 60
done

mv results/results.csv results/defensive.csv
