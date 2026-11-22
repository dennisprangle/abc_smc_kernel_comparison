# Transport ABC

Code for "A Comparison of Kernels for ABC-SMC" by Dennis Prangle, Cecilia Viscardi and Sammy Ragy (https://arxiv.org/abs/2511.06351). To reproduce the analysis, run the scripts in the Experiments section below.

## Setup

    pip install -r requirements.txt

Note `requirements.txt` pins the full environment used for the paper's experiments (Linux with CUDA 13 builds of PyTorch), including all indirect dependencies. On other platforms, installing the directly imported packages (torch, nflows, numpy, scipy, pandas, matplotlib, seaborn, statsmodels, POT, graspologic, wrapt_timeout_decorator) should be enough.
Recreating the reference samples would additionally need the `pystan` and `particles` python packages.

Running the experiment scripts also needs [GNU parallel](https://www.gnu.org/software/parallel/).


## Code

### `algorithms/`
* `abc_smc.py`: classic ABC SMC
* `transport_abc.py`: ABC SMC with a normalising flow proposal
* `transport_abc_mixture.py`: ABC SMC with a Gaussian mixture proposal
* `proposals.py`: ABC-SMC proposal distributions
* `mcmc.py`: ABC-MCMC kernels
* `nflowsutils.py`: extensions, variations and fixes for `nflows` code
* `utils.py`: general utilities (Wasserstein distance, saving run summaries, ...)

### `models/`
* `MG1.py`: M/G/1 queueing model
* `epimodels.py`: SEIR epidemic model

### `ref_samples/`
Reference posterior samples (`*_ref_sample.txt`), used to compute the loss, and the scripts that made them.

### Examples
Each of these runs one model:
* `quadratic_example.py`: a toy quadratic model
* `gaussian_mixture_example.py`: a Gaussian mixture model
* `MG1_example.py`: the M/G/1 queueing model
* `slcp_example.py`: SLCP (simple likelihood, complex posterior)
* `SEIR_example.py`: the SEIR epidemic model

They all accept the same main options, including `--proposal {classic_ind,classic_rw,mix,flow}`, `--kernel`, `--seed`, `--max_mins`, `--n_particles` and `--erp`. Run `python <example> --help` to see them all.
Each run appends a summary to `results/results.csv`, or with `--results_dir`, writes it to its own file in that directory.

### Other python code
* `quadratic_illustration.py`: illustration plots of the proposals (saved in `plots/`)
* `combine_results.py`: utility for experiment scripts below

## Experiments and analysis

Each `script*.sh` runs its jobs in parallel (`NJOBS=8`; logs go to `logs/`) and then combines the results.
Outputs and analysis scripts are in `results/`. Run the analysis scripts from inside that folder.

| Script | Experiment | Output | Analysis |
|---|---|---|---|
| `script.sh` | main comparison of proposals and kernels | `results.csv` | `analysis_main.py`, `analysis_training.py` |
| `script_defensive.sh` | defensive mixture proportion | `defensive.csv` | `analysis_defensive.py` |
| `script_hits.sh` | number of hits for the r-hit kernels | `hits.csv` | `analysis_hits.py` |
| `script_exploratory.sh` | quadratic model histories | `exploratory.csv`, `histories_quadratic/` | `analysis_exploratory.py` |
| `script_components.sh` | number of mixture components | `components.csv` | `analysis_components.py` |
| `script_flow.sh` | flow architecture | `flow.csv` | `analysis_flow.py` |
| `script_fallback.sh` | fallback to the classic proposal | `fallback.csv` | `analysis_fallback.py` |
| `script_target.sh` | training on the current or previous target | `target.csv` | `analysis_target.py` |
| `script_tuning.sh` | number of particles and erp (MG1) | `tuning_mg1.csv` | `analysis_tuning.py` |
| `script_test.sh` | quick check of the parallel setup | `test.csv` | — |
