import sys, os
from pathlib import Path

# Get the root directory (parent of simulations_H2)
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from pyfrechet.metric_spaces import MetricData, H2
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
from pyfrechet.metrics import mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import contextlib

np.random.seed(1000)
param_grid = {
    'estimator__min_split_size': [1, 5, 10]
}

# Custom scorer (negative mean squared error)
neg_mse = make_scorer(mse, greater_is_better=False)

# By-blocks execution
SIMULATION_DIR = Path(__file__).parent
DATA_DIR = SIMULATION_DIR / 'data'
RESULTS_DIR = SIMULATION_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
TYPE_I_DIR = SIMULATION_DIR / 'TypeIdata'
TYPE_II_DIR = SIMULATION_DIR / 'TypeIIdata'
TYPE_III_DIR = SIMULATION_DIR / 'TypeIIIdata'
TYPE_IV_DIR = SIMULATION_DIR / 'TypeIVdata'

n_samples = len(list(DATA_DIR.glob('*.csv')))
current_block = int(sys.argv[1])

base = Tree(split_type='2means', mtry=None, impurity_method='cart')
base_forest = BaggedRegressor(estimator=base, n_estimators=200, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=-1, seed = 5)

M = H2(2)

def tune_forest(X, y, forest = base_forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """
    tuned_forest = GridSearchCV(estimator=forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=-1, verbose=0)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Main task
def task(file) -> None:
    """Processes a single file for hyperboloid data regression."""
    # Check if results file already exists
    results_filename = RESULTS_DIR / f'{file[:-4]}_results.npy'
    if results_filename.exists():
        print(f"Results file already exists for {file}, skipping...")
        return
    
    with open(DATA_DIR / file, 'rb') as f:
        sample = pd.read_csv(f)
        
    sample.drop(columns = ['Unnamed: 0'], inplace = True)
    #name columns 1, 2 and 4 as V1, V2 and V3
    sample.columns = ['t', 'V1', 'V2', 'V3']

    X = sample['t'].values.reshape(-1, 1)
    y = MetricData(M, sample.iloc[:, 1:4].values)
    kappa = int(file.split('_')[3][5:])
    N = int(file.split('_')[2][1:])
    samp = int(file.split('_')[1][4:])
    
    # Perform hyperparameter tuning
    forest = tune_forest(X, y, base_forest, param_grid)
    oob_quantile = np.percentile(forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    
    ############################################################################################################
    # TYPE I COVERAGE RESULTS
    MC = 1000
    type_i_filename = TYPE_I_DIR / f'H2_type_i_N{N}_kappa{kappa}.csv'
    
    hyp_data = pd.read_csv(type_i_filename)

    hyp_data.columns = ['t', 'V1', 'V2', 'V3']

    # Randomly select rows from the dataframe
    thetas = hyp_data['t'].values.reshape(-1, 1)
    new_ys = hyp_data[['V1', 'V2', 'V3']].values

    pb_new_pred = forest.predict(thetas)
    pb_i_cov = np.repeat(M.d(pb_new_pred, MetricData(M, new_ys.squeeze()))[:, np.newaxis], 3, axis=1) <= np.tile(oob_quantile, (MC, 1))


############################################################################################################            
    # TYPE II COVERAGE RESULTS
    filename = TYPE_II_DIR / f'samp_{samp}_N_{N}_kappa{kappa}.csv'
    hyp_data = pd.read_csv(filename)
    hyp_data.drop(columns = ['Unnamed: 0'], inplace = True)
    #name columns 1, 2 and 3 as V1, V2 and V3
    hyp_data.columns = ['t', 'V1', 'V2', 'V3']

    thetas = hyp_data['t'].values
    new_ys = hyp_data[['V1', 'V2', 'V3']].values
    pb_new_preds = forest.predict(thetas.reshape(-1, 1))
    pb_ii_cov = np.sum(M.d(new_ys, pb_new_preds.data).reshape(MC,1) <= np.tile(oob_quantile, (MC,1)), axis = 0) / MC
    
############################################################################################################
    # TYPE III COVERAGE RESULTS
    type_iii_filename = TYPE_III_DIR / f'H2_type_iii_N{N}_kappa{kappa}.csv'
    
    hyp_data = pd.read_csv(type_iii_filename)
    hyp_data.columns = ['t', 'V1', 'V2', 'V3']

    # Randomly select rows from the dataframe
    thetas = hyp_data['t'].values.reshape(-1, 1)
    new_ys = hyp_data[['V1', 'V2', 'V3']].values

    pb_new_pred = forest.predict(thetas[0].reshape(-1,1))

    pb_iii_cov = np.repeat(M.d(pb_new_pred, MetricData(M, new_ys.squeeze()))[:, np.newaxis], 3, axis=1) <= np.tile(oob_quantile, (MC, 1))

############################################################################################################
    # TYPE IV COVERAGE RESULTS
    filename = TYPE_IV_DIR / f'samp_{samp}_N_{N}_kappa{kappa}.csv'
    hyp_data = pd.read_csv(filename)
    hyp_data.drop(columns = ['Unnamed: 0'], inplace = True)
    #name columns 1, 2 and 3 as V1, V2 and V3
    hyp_data.columns = ['t', 'V1', 'V2', 'V3']

    thetas = hyp_data['t'].values
    new_ys = hyp_data[['V1', 'V2', 'V3']].values

    pb_new_preds = forest.predict(thetas.reshape(-1,1))
    pb_iv_cov = np.sum(M.d(new_ys, pb_new_preds.data).reshape(MC,1) <= np.tile(oob_quantile, (MC,1)), axis = 0) / MC

    # Store results
    results = {
        'i_cov': pb_i_cov,
        'ii_cov': pb_ii_cov,
        'iii_cov': pb_iii_cov,
        'iv_cov': pb_iv_cov,
        'OOB_quantile': oob_quantile
        }
    results_filename = RESULTS_DIR / f'{file[:-4]}_results.npy'
    np.save(results_filename, results)

# Get list of files to process
file_list = [f.name for f in DATA_DIR.glob('*.csv') if f.name.endswith(f'block_{current_block}.csv')]
total_files = len(file_list)

with tqdm_joblib(tqdm(desc="Percentage of tasks completed:", total = total_files)) as progress_bar:
    Parallel(n_jobs=-1, verbose=2)(delayed(task)(file) for file in file_list)