#!/bin/bash
rm results/results.csv

for seed in {1..5}
do
	python quadratic_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5
	python quadratic_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5
	python quadratic_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5
	python quadratic_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5
	python quadratic_example.py --proposal flow --seed $seed --kernel naive --disable_fallback
	python quadratic_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python quadratic_example.py --proposal flow --seed $seed --kernel ind_1_hit --disable_fallback
	python quadratic_example.py --proposal flow --seed $seed --kernel multi_r_hit --disable_fallback
	python quadratic_example.py --proposal classic_ind --seed $seed --kernel naive
	python quadratic_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python quadratic_example.py --proposal classic_ind --seed $seed --kernel ind_1_hit
	python quadratic_example.py --proposal classic_ind --seed $seed --kernel multi_r_hit
	python quadratic_example.py --proposal classic_rw --seed $seed --kernel naive
	python quadratic_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python quadratic_example.py --proposal classic_rw --seed $seed --kernel multi_r_hit
	python MG1_example.py --proposal mix --seed $seed --kernel naive --quantile_summaries --disable_fallback --mix_components 5
	python MG1_example.py --proposal mix --seed $seed --kernel 1_hit --quantile_summaries --disable_fallback --mix_components 5
	python MG1_example.py --proposal mix --seed $seed --kernel ind_1_hit --quantile_summaries --disable_fallback --mix_components 5
	python MG1_example.py --proposal mix --seed $seed --kernel multi_r_hit --quantile_summaries --disable_fallback --mix_components 5
	python MG1_example.py --proposal flow --seed $seed --kernel naive --quantile_summaries --disable_fallback
	python MG1_example.py --proposal flow --seed $seed --kernel 1_hit --quantile_summaries --disable_fallback
	python MG1_example.py --proposal flow --seed $seed --kernel ind_1_hit --quantile_summaries --disable_fallback
	python MG1_example.py --proposal flow --seed $seed --kernel multi_r_hit --quantile_summaries --disable_fallback
	python MG1_example.py --proposal classic_ind --seed $seed --kernel naive --quantile_summaries
	python MG1_example.py --proposal classic_ind --seed $seed --kernel 1_hit --quantile_summaries
	python MG1_example.py --proposal classic_ind --seed $seed --kernel ind_1_hit --quantile_summaries
	python MG1_example.py --proposal classic_ind --seed $seed --kernel multi_r_hit --quantile_summaries
	python MG1_example.py --proposal classic_rw --seed $seed --kernel naive --quantile_summaries
	python MG1_example.py --proposal classic_rw --seed $seed --kernel 1_hit --quantile_summaries
	python MG1_example.py --proposal classic_rw --seed $seed --kernel multi_r_hit --quantile_summaries
	python slcp_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5
	python slcp_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5
	python slcp_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5
	python slcp_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5
	python slcp_example.py --proposal flow --seed $seed --kernel naive --disable_fallback
	python slcp_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python slcp_example.py --proposal flow --seed $seed --kernel ind_1_hit --disable_fallback
	python slcp_example.py --proposal flow --seed $seed --kernel multi_r_hit --disable_fallback
	python slcp_example.py --proposal classic_ind --seed $seed --kernel naive
	python slcp_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python slcp_example.py --proposal classic_ind --seed $seed --kernel ind_1_hit
	python slcp_example.py --proposal classic_ind --seed $seed --kernel multi_r_hit
	python slcp_example.py --proposal classic_rw --seed $seed --kernel naive
	python slcp_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python slcp_example.py --proposal classic_rw --seed $seed --kernel multi_r_hit
	python SEIR_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5
	python SEIR_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5
	python SEIR_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5
	python SEIR_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5
	python SEIR_example.py --proposal flow --seed $seed --kernel naive --disable_fallback
	python SEIR_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python SEIR_example.py --proposal flow --seed $seed --kernel ind_1_hit --disable_fallback
	python SEIR_example.py --proposal flow --seed $seed --kernel multi_r_hit --disable_fallback
	python SEIR_example.py --proposal classic_ind --seed $seed --kernel naive
	python SEIR_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python SEIR_example.py --proposal classic_ind --seed $seed --kernel ind_1_hit
	python SEIR_example.py --proposal classic_ind --seed $seed --kernel multi_r_hit
	python SEIR_example.py --proposal classic_rw --seed $seed --kernel naive
	python SEIR_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python SEIR_example.py --proposal classic_rw --seed $seed --kernel multi_r_hit
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel naive --disable_fallback --mix_components 5
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel 1_hit --disable_fallback --mix_components 5
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel ind_1_hit --disable_fallback --mix_components 5
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel multi_r_hit --disable_fallback --mix_components 5
	python gaussian_mixture_example.py --proposal flow --seed $seed --kernel naive --disable_fallback
	python gaussian_mixture_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python gaussian_mixture_example.py --proposal flow --seed $seed --kernel ind_1_hit --disable_fallback
	python gaussian_mixture_example.py --proposal flow --seed $seed --kernel multi_r_hit --disable_fallback
	python gaussian_mixture_example.py --proposal classic_ind --seed $seed --kernel naive
	python gaussian_mixture_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python gaussian_mixture_example.py --proposal classic_ind --seed $seed --kernel ind_1_hit
	python gaussian_mixture_example.py --proposal classic_ind --seed $seed --kernel multi_r_hit
	python gaussian_mixture_example.py --proposal classic_rw --seed $seed --kernel naive
	python gaussian_mixture_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python gaussian_mixture_example.py --proposal classic_rw --seed $seed --kernel multi_r_hit
done
