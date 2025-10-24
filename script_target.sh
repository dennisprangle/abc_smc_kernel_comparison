#!/bin/bash
rm results/results.csv

for seed in {1..5}
do
	python quadratic_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python quadratic_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python quadratic_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback
	python quadratic_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python MG1_example.py --proposal classic_rw --seed $seed --kernel 1_hit --quantile_summaries
	python MG1_example.py --proposal classic_ind --seed $seed --kernel 1_hit --quantile_summaries
	python MG1_example.py --proposal mix --seed $seed --kernel 1_hit --quantile_summaries  --mix_components 5 --disable_fallback
	python MG1_example.py --proposal flow --seed $seed --kernel 1_hit --quantile_summaries --disable_fallback
	python slcp_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python slcp_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python slcp_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback
	python slcp_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python SEIR_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python SEIR_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python SEIR_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback
	python SEIR_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python gaussian_mixture_example.py --proposal classic_rw --seed $seed --kernel 1_hit
	python gaussian_mixture_example.py --proposal classic_ind --seed $seed --kernel 1_hit
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback
	python gaussian_mixture_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback
	python quadratic_example.py --proposal classic_rw --seed $seed --kernel 1_hit --target_prev
	python quadratic_example.py --proposal classic_ind --seed $seed --kernel 1_hit --target_prev
	python quadratic_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback --target_prev
	python quadratic_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback --target_prev
	python MG1_example.py --proposal classic_rw --seed $seed --kernel 1_hit --quantile_summaries --target_prev
	python MG1_example.py --proposal classic_ind --seed $seed --kernel 1_hit --quantile_summaries --target_prev
	python MG1_example.py --proposal mix --seed $seed --kernel 1_hit --quantile_summaries  --mix_components 5 --disable_fallback --target_prev
	python MG1_example.py --proposal flow --seed $seed --kernel 1_hit --quantile_summaries --disable_fallback --target_prev
	python slcp_example.py --proposal classic_rw --seed $seed --kernel 1_hit --target_prev
	python slcp_example.py --proposal classic_ind --seed $seed --kernel 1_hit --target_prev
	python slcp_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback --target_prev
	python slcp_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback --target_prev
	python SEIR_example.py --proposal classic_rw --seed $seed --kernel 1_hit --target_prev
	python SEIR_example.py --proposal classic_ind --seed $seed --kernel 1_hit --target_prev
	python SEIR_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback --target_prev
	python SEIR_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback --target_prev
	python gaussian_mixture_example.py --proposal classic_rw --seed $seed --kernel 1_hit --target_prev
	python gaussian_mixture_example.py --proposal classic_ind --seed $seed --kernel 1_hit --target_prev
	python gaussian_mixture_example.py --proposal mix --seed $seed --kernel 1_hit  --mix_components 5 --disable_fallback --target_prev
	python gaussian_mixture_example.py --proposal flow --seed $seed --kernel 1_hit --disable_fallback --target_prev
done
mv results/results.csv results/target.csv
