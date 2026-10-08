# Out-of-bag prediction balls for random forests in metric spaces

This repository contains the code to reproduce all figures and tables from the paper "Out-of-bag prediction balls for random forests in metric spaces", as well as additional simulation results.

## Overview

The code builds on [pyfrechet](https://github.com/matthieubulte/pyfrechet) (with commit ID: 3e02448261f47f95649693b968a47c9a95be3e6e), which contains the implementation of random forests in metric spaces. Several additions are made to implement OOB predictions, new metric spaces, and other features required in the paper.

## Repository Structure

For each metric space (`simulations_metricspace/`), there is a common structure:

- Files to generate the data, under the names: `metricspace_gen_data.py` (to generate the data to train the random forests), `metricspace_gen_type_i.py` (generate data to test Type I coverage) and `metricspace_gen_type_iii.py` (generate data to test Type III coverage). 
- The file `metricspace_results.py` generates all the results for the metric space (Type II and IV data are generated inside this file for convenience).
- Folders: "data" and "results".

In the Euclidean space, there are additional files to generate the results to compare the radii and volumes of OOB prediction balls and split-conformal balls for different dimensionalities of the response. On the hyperboloid, the data is generated using R, since the library `rotasym` is used to sample from the von Mises-Fisher distribution. For this reason, the data for Types II and IV is also generated in R using a separate script.

To generate **all** the figures and tables displayed in the paper, please run `main_results.ipynb`. Even though the results and data files are reproducible, they are also included in the repository, so that the figures can be generated easily (some results are computationally expensive to obtain).

Below is a simple scheme of the distribution of the library:

```
├── main_results.ipynb          # Main notebook reproducing all paper results.
├── support_functions.py        # Support functions for main_results.ipynb.
├── pyfrechet/                  # Modification and extension of pyfrechet to implement random forests in metric spaces.
├── simulations_*/              # Raw simulation data and scripts for each metric space.
└── requirements.txt            # Python (3.9.9) dependencies.
```

## How to reproduce the result files
Many of the experiments were conducted in a cluster. For this reason, the data and the results are divided by blocks. To run `metricspace_results.py` files, do
```
python metricspace_results.py b
```
where b is the block number.