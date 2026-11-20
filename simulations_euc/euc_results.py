import sys, os
import pickle
import numpy as np
from sc_radius import split_conformal_radii
from scipy.stats import beta
import time
import joblib
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import contextlib
from sklearn.metrics import mean_squared_error as mse

np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])
betas = np.array([1, -1, 1])  # Define the true beta values

data_dir = os.path.join(os.getcwd(), 'simulations_euc', 'data')
results_dir = os.path.join(os.getcwd(), 'simulations_euc', 'results')
os.makedirs(results_dir, exist_ok=True)

# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_euc/' 'data')))
current_block = int(sys.argv[1])

# Define parameter grid for tuning
param_grid = {
    'min_samples_leaf': [1, 5, 10],
    'max_features': [1, 2, 3]
    }

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

# Function to simulate regression data
def simulate_data(sigma, X_design, betas):
    Ys = []
    sample_size = X_design.shape[0]

    # Generate the error term epsilon, which follows a normal distribution
    epsilon = np.random.normal(0, sigma, size = sample_size).reshape(sample_size, 1)

    # Step 2: Apply the model transformations
    X_1 = X_design[:, 0]  # X_1 corresponds to the first column (without intercept)
    X_2 = X_design[:, 1]  # X_2 corresponds to the second column
    X_3 = X_design[:, 2]  # X_3 corresponds to the third column

    # Calculate the response vector Y = beta_0 + beta_1*X_1 + beta_2*X_2 + beta_3*X_3 + epsilon
    Ys = betas[0] * X_1 + betas[1] * X_2 + betas[2] * X_3
    # Convert list to array for easier manipulation
    Ys = Ys.reshape(sample_size, 1) + epsilon
    
    return X_design, Ys

def tune_forest(X, y, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    base_forest = RandomForestRegressor(n_jobs=-1, random_state=1000, n_estimators=200, oob_score=True)
    grid_search = GridSearchCV(estimator = base_forest, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def task(file):
    # Load data
    with open(os.path.join(data_dir, file), 'rb') as f:
        sample = pickle.load(f)

    sigma_approx = float(file.split('_')[3][5:])
    N = int(file.split('_')[2][1:])
    if sigma_approx == 0.9:
        true_sigma = np.sqrt(3)/2
    elif sigma_approx == 1.7:
        true_sigma = np.sqrt(3)
    # elif sigma_approx == 1.7:
    #     true_sigma = np.sqrt(3)
    else:
        raise ValueError("Sigma value not found.")

    X = sample['X']
    y = sample['Y']
    y = y.ravel()
    n_predictors = X.shape[1]

    # Conformal data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    # Measure time for tuning and fitting (prediction balls)
    start_time = time.time()
    # Perform hyperparameter tuning
    pb_forest = tune_forest(X, y, param_grid) 
    # Radii   
    oob_quantile = np.percentile(np.abs(pb_forest.oob_prediction_ - y), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    end_time = time.time()
    pb_time = end_time - start_time
    # Measure time for tuning and fitting (SC regions)
    start_time = time.time()
    # Perform hyperparameter tuning
    conf_forest = tune_forest(X_train, y_train, param_grid) 
    # Radii 
    test_preds = conf_forest.predict(X_test)
    quantile = split_conformal_radii(np.abs(y_test - test_preds))
    end_time = time.time()
    conf_time = end_time - start_time

############################################################################################################
    # TYPE I COVERAGE RESULTS
    MC = 1000
    type_i_filename = f'euc_type_i_N{N}_sigma{np.round(true_sigma, 1)}.pkl'
    with open(os.path.join(os.getcwd(), 'simulations_euc', 'type_i_data', type_i_filename), 'rb') as f:
        type_i_sample = pickle.load(f)

    # Use the pre-generated data
    Xs = type_i_sample['X'].reshape(-1, 3)
    new_ys = type_i_sample['Y']

    # Predict the new observations
    conf_new_pred = conf_forest.predict(Xs.reshape(-1,3))
    pb_new_pred = pb_forest.predict(Xs.reshape(-1,3))
    #pb_new_pred = np.tile(pb_new_pred, (MC, 1))  # Repeat the prediction for MC samples
    pb_i_cov = np.repeat(np.abs(pb_new_pred - new_ys.squeeze())[:, np.newaxis], 3, axis=1) <= np.tile(oob_quantile, (MC, 1))
    conf_i_cov = np.repeat(np.abs(conf_new_pred - new_ys.squeeze())[:, np.newaxis], 3, axis=1) <= np.tile(quantile, (MC, 1))

############################################################################################################            
    # TYPE II COVERAGE RESULTS
    #Generate observations to estimate the probability
    new_X = 2*np.sqrt(5)*(np.random.beta(2, 2, (MC, n_predictors)) - 1/2)
    new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas = betas)

    pb_new_pred = pb_forest.predict(new_X)
    conf_new_pred = conf_forest.predict(new_X)

    pb_ii_cov = np.sum(np.abs(pb_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC
    conf_ii_cov = np.sum(np.abs(conf_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(quantile, (MC, 1)), axis = 0) / MC

############################################################################################################
    # TYPE III COVERAGE RESULTS
    type_iii_filename = f'euc_type_iii_N{N}_sigma{np.round(true_sigma, 1)}.pkl'
    with open(os.path.join(os.getcwd(), 'simulations_euc', 'type_iii_data', type_iii_filename), 'rb') as f:
        type_iii_sample = pickle.load(f)

    # Use the pre-generated data
    Xs = type_iii_sample['X'].reshape(-1, 3)
    new_ys = type_iii_sample['Y']

    # Predict the new observations
    pb_new_pred = pb_forest.predict(Xs[0].reshape(-1,3))
    conf_new_pred = conf_forest.predict(Xs[0].reshape(-1,3))

    #pb_new_pred = np.tile(pb_new_pred, (MC, 1))  # Repeat the prediction for MC samples
    pb_iii_cov = np.repeat(np.abs(pb_new_pred - new_ys.squeeze())[:, np.newaxis], 3, axis=1) <= np.tile(oob_quantile, (MC, 1))
    conf_iii_cov = np.repeat(np.abs(conf_new_pred - new_ys.squeeze())[:, np.newaxis], 3, axis=1) <= np.tile(quantile, (MC, 1))


############################################################################################################
    # TYPE IV COVERAGE RESULTS
    #Generate observations to estimate the probability
    q_25 = 2*np.sqrt(5)*(beta(2,2).ppf(.25)-1/2)
    new_X = np.repeat(q_25, MC * n_predictors).reshape(MC, n_predictors)
    new_X, new_y = simulate_data(sigma = true_sigma, X_design=new_X, betas = betas)

    pb_new_pred = pb_forest.predict(new_X)
    conf_new_pred = conf_forest.predict(new_X)

    pb_iv_cov = np.sum(np.abs(pb_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC
    conf_iv_cov = np.sum(np.abs(conf_new_pred.reshape(-1,1) - new_y).reshape(-1,1) <= np.tile(quantile, (MC, 1)), axis = 0) / MC

############################################################################################################
    # MSE
    test_size = 1000
    new_X_design = 2*np.sqrt(5)*(np.random.beta(2, 2, (test_size, n_predictors)) - 1/2)
    X_test, y_test = simulate_data(sigma = true_sigma, X_design=new_X_design, betas = betas)

    pb_new_pred = pb_forest.predict(X_test)
    conf_new_pred = conf_forest.predict(X_test)

    pb_mse = mse(pb_new_pred, y_test)
    conf_mse = mse(conf_new_pred, y_test)

    # Store results
    results = {
        'pb_i_cov': pb_i_cov,
        'conf_i_cov': conf_i_cov,
        'pb_ii_cov': pb_ii_cov,
        'conf_ii_cov': conf_ii_cov,
        'pb_iii_cov': pb_iii_cov,
        'conf_iii_cov': conf_iii_cov,
        'pb_iv_cov': pb_iv_cov,
        'conf_iv_cov': conf_iv_cov,
        'OOB_quantile': oob_quantile,
        'quantile': quantile,
        'pb_time': pb_time,
        'conf_time': conf_time,
        'pb_mse': pb_mse,
        'conf_mse': conf_mse
    }
    filename = os.path.join(results_dir, file[:-4] + '_results.npy')
    np.save(filename, results)

file_list = list(filter(lambda file: file.endswith(f'block_{current_block}.pkl'), filter(lambda file: file.endswith('.pkl'), os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'data/')))))
total_files = len(file_list)

with tqdm_joblib(tqdm(desc="Percentage of tasks completed:", total = total_files)) as progress_bar:
    Parallel(n_jobs=4, verbose=2)( delayed(task)(file) for file in file_list)