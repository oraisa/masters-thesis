# Differentially Private Markov Chain Monte Carlo

## Installing Dependencies

Run 
``` sh
pip install -r requirements.txt
```
to install the required dependencies. On systems with the `pip3` command, 
it should be used instead.

## Running the Experiments

To run the experiments and generate each figure, run
``` sh
snakemake -j4
```
The number after `-j` sets the number of concurrent jobs Snakemake runs.
The figures are placed in the `Thesis/figures` directory. The results of each
MCMC chain are placed in `code/results`.

## External code
The code in the directory `code/dp_mcmc_module/` is taken from 
[https://github.com/DPBayes/DP-MCMC-NeurIPS2019](https://github.com/DPBayes/DP-MCMC-NeurIPS2019)
under the MIT license and modified to work with the other code in this project.
