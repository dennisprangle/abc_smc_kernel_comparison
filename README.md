This is code for the paper [A Comparison of Kernels for ABC-SMC](https://arxiv.org/abs/2511.06351) by Dennis Prangle, Cecilia Viscardi and Sammy Ragy.

The `*example.py` files run individual experiments on different models.

The `script*.sh` files run the main analyses in the paper, by calling the `*example.py` files repeatedly.

The results are stored in the `results` directory, together with `analysis*.py` files which analyse the results.

The main ABC-SMC code is in the `algorithms` directory. The `models` directory has code needed by some of the simulator models. The `ref_samples` directory contains reference samples from the posteriors or good approximation to them, as well as the code to produce these.

The `requirements.txt` file shows the package versions used for the paper's results.
