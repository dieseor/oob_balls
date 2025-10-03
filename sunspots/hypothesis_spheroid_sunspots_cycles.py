from joblib import Parallel, delayed
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from pyfrechet.metric_spaces import MetricData, Sphere, Spheroid, two_euclidean, sphere_to_spheroid, spheroid_to_sphere
from pyfrechet.metrics import mse
from sklearn.model_selection import train_test_split, KFold
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.d_trees import d_Tree
from scipy import stats
from statsmodels.stats.multitest import multipletests
import time
import itertools

def custom_GCV(M_response, structure, X_train, y_train, param_grid, seed=5, n_splits=5):
    """
    Manual Grid Search CV for Fréchet forests.
    
    Parameters:
    - M_response: Metric space object for response
    - structure: Structure for predictors
    - X_train: array-like, shape (n_samples, n_features)
    - y_train: array-like, shape (n_samples, 3)
    - param_grid: dict with keys 'min_split_size', 'mtry'
    - seed: int, random seed for reproducibility
    - n_splits: int, number of CV folds

    Returns:
    - final_forest: fitted BaggedRegressor on full training data with best parameters
    - best: dict, best hyperparameters
    - cv_results: list of dicts with CV performance
    """
    grid = list(itertools.product(param_grid['min_split_size'],
                                  param_grid['mtry']))#
    cv_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for min_split_size, mtry in grid:
        fold_errors = []#
        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr_raw, y_val_raw = y_train[train_index], y_train[val_index]

            # Define metric and structure
            y_tr = MetricData(M_response, y_tr_raw.reshape(-1, 3))
            y_val = MetricData(M_response, y_val_raw.reshape(-1, 3))#
            # Define forest
            base = d_Tree(split_type='2means', impurity_method='medoid', structure=structure,
                        min_split_size=min_split_size, mtry=mtry)
            forest = BaggedRegressor(estimator=base, n_estimators=200,
                                     bootstrap_fraction=1, bootstrap_replace=True,
                                     seed=seed, n_jobs=12)
            forest.fit(X_tr, y_tr)
            preds = forest.predict(X_val)
            error = mse(y_val, preds)
            fold_errors.append(error)

        avg_error = np.mean(fold_errors)
        cv_results.append({
            'min_split_size': min_split_size,
            'mtry': mtry,
            'cv_error': avg_error
        })
        print(f"  CV - min_split_size={min_split_size}, mtry={mtry}, CV error={avg_error:.6f}")#
    best = min(cv_results, key=lambda x: x['cv_error'])
    print(f"  Best CV params: {best}")
    # Final refit on full training data
    y_train_metric = MetricData(M_response, y_train.reshape(-1, 3))
    final_tree = d_Tree(split_type='2means', impurity_method='medoid',
                      structure=structure,
                      #min_split_size=best['min_split_size'], mtry=best['mtry'])
                      min_split_size=1, mtry=1)
    final_forest = BaggedRegressor(estimator=final_tree, n_estimators=200,
                                   bootstrap_fraction=1, bootstrap_replace=True,
                                   seed=seed, n_jobs=12)
    final_forest.fit(X_train, y_train_metric)

    return final_forest, best, cv_results

# Add area calculation functions
def canonical_lattice(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a canonical lattice on the unit sphere.
    
    Parameters:
    - n: int, number of points to generate

    Returns:
    - sphere_points: array of shape (n, 3) with coordinates on sphere
    """
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T

def sphere_area_pred_ball(M_spheroid, radius, spheroid_centers, sphere_points=None, n_points=5000, n_jobs=12):
    """
    Estimate areas of prediction balls for multiple reference points after mapping to sphere.
    
    Parameters:
    - M_spheroid: Spheroid metric space
    - radius: float, radius of the balls
    - spheroid_centers: array of shape (n_refs, 3), centers of balls on spheroid
    - sphere_points: optional, precomputed lattice points on sphere
    - n_points: int, number of lattice points if sphere_points not provided
    
    Returns:
    - areas: array of areas for each reference point
    """
    # Generate or use provided lattice points
    if sphere_points is None:
        sphere_points = canonical_lattice(n_points)
    
    # Map lattice points to spheroid (do this once)
    spheroid_points = sphere_to_spheroid(sphere_points, M_spheroid.a, M_spheroid.c)
    
    # Function to process one reference point
    def process_ref_point(center_spheroid):
        # Count points inside ball
        inside_ball = np.sum(M_spheroid.d(spheroid_points, center_spheroid) < radius)
        
        # Calculate area
        return 4 * np.pi * inside_ball / len(sphere_points)
    
    # Process all reference points in parallel
    areas = Parallel(n_jobs=n_jobs)(
        delayed(process_ref_point)(center) 
        for center in spheroid_centers
    )
    
    return np.array(areas)

np.random.seed(1000)

def train_and_predict(X_train, X_test, y_train, y_test, a, c, seed=5):
    """
    Train a model with specific spheroid parameters and return predictions, errors, and areas
    """
    print(f"Training model for {'sphere' if a == 1 and c == 1 else f'spheroid (a={a}, c={c})'}...")
    
    y_train_spheroid = sphere_to_spheroid(y_train, a, c)
    M = Spheroid(a=a, c=c)
    
    # Define structure of predictors
    structure = [(Sphere(dim=2), list(range(3)))]
    
    # Cross-validation for hyperparameter tuning
    param_grid = {'min_split_size': [1, 5, 10], 'mtry': [1]}
    print(f"  Performing CV for hyperparameter tuning...")
    
    forest, best_params, cv_results = custom_GCV(
        M_response=M,
        structure=structure,
        X_train=X_train,
        y_train=y_train_spheroid,
        param_grid=param_grid, 
        seed=seed, 
        n_splits=5
    )
    
    # Get predictions
    preds = forest.predict(X_test)
    
    # For sphere case, predictions are already on sphere
    if a == 1 and c == 1:
        sphere_preds = preds.data
    else:
        sphere_preds = spheroid_to_sphere(preds.data, a, c, R=1)
    
    # Calculate point-wise squared errors (always on sphere for comparison)
    sphere_preds_metric = MetricData(Sphere(2), sphere_preds)
    test_metric = MetricData(Sphere(2), y_test)
    point_errors = np.array([mse(test_metric[i:i+1], sphere_preds_metric[i:i+1])
                            for i in range(len(y_test))])
    
    # Calculate overall MSE
    overall_mse = mse(test_metric, sphere_preds_metric)
    print(f"Overall MSE for {('sphere' if a == 1 and c == 1 else f'spheroid (a={a}, c={c})')}: {overall_mse:.6f}")
    
    # Calculate areas using OOB quantile (90th percentile for 0.1 significance)
    oob_quantile = np.percentile(forest.oob_errors(), 90, method='inverted_cdf')
    can_lat = canonical_lattice(5000)
    
    # Calculate areas for each test point - ALWAYS measured on the sphere via spheroid mapping
    test_point_areas = sphere_area_pred_ball(M, oob_quantile, spheroid_centers=preds.data, 
                                           sphere_points=can_lat, n_jobs=12)
    
    # Calculate Type II coverage (proportion of test points within ball)
    test_points_spheroid = sphere_to_spheroid(y_test, a, c)
    distances = M.d(MetricData(M, test_points_spheroid), preds)
    type_II_coverage = np.mean(distances <= oob_quantile)
    
    return sphere_preds, point_errors, test_point_areas, type_II_coverage, oob_quantile, best_params

def task(cyc):
    filename = f'sunspots_births_{cyc}_deaths.csv'
    print(f'Processing {filename}...')
    filepath = os.path.join(os.getcwd(), 'sunspots/data', filename)

    # Load data
    sample = pd.read_csv(filepath)
    X = np.vstack([sample['births_X.1'], sample['births_X.2'], sample['births_X.3']]).T
    y = np.vstack([sample['deaths_X.1'], sample['deaths_X.2'], sample['deaths_X.3']]).T
    
    # Same train/test split for all configurations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

    # Define spheroid configurations
    configs = [
        (0.1, 1),
        (0.2, 1),
        (0.3, 1),
        (0.4, 1),
        (0.5, 1),
        (0.6, 1),
        (0.7, 1),
        (0.8, 1),
        (0.9, 1),
        (1.1, 1),
        (1.25, 1),
        (1.5, 1),
    ]

    results = {
        'cycle': cyc,
        'n_test_points': len(X_test),
        'train_data': {'X': X_train, 'y': y_train},
        'test_data': {'X': X_test, 'y': y_test},
        'predictions': {},
        'point_errors': {},
        'point_areas': {},
        'point_coverage': {},
        'best_params': {},
        'config_params': [],
        'oob_quantile': {}
    }

    # Get sphere predictions (baseline)
    sphere_preds, sphere_errors, sphere_areas, sphere_coverage, sphere_best_params, oob_quantile = train_and_predict(X_train, X_test, y_train, y_test, 1, 1)
    # sphere_preds, sphere_errors, sphere_areas, sphere_coverage, oob_quantile = train_and_predict(X_train, X_test, y_train, y_test, 1, 1)

    results['predictions']['sphere'] = sphere_preds
    results['point_errors']['sphere'] = sphere_errors
    results['point_areas']['sphere'] = sphere_areas
    results['point_coverage']['sphere'] = sphere_coverage
    results['best_params']['sphere'] = sphere_best_params
    results['oob_quantile']['sphere'] = oob_quantile

    # For each spheroid configuration
    for a, c in configs:  # exclude sphere case which we already did
        print(f"Processing spheroid with a={a}, c={c}")
        
        # Train and get predictions
        spheroid_preds, spheroid_errors, spheroid_areas, spheroid_coverage, spheroid_best_params, oob_quantile = train_and_predict(X_train, X_test, y_train, y_test, a, c)
        # spheroid_preds, spheroid_errors, spheroid_areas, spheroid_coverage, oob_quantile = train_and_predict(X_train, X_test, y_train, y_test, a, c)

        # Store predictions, errors, areas, coverage, and best params
        config_key = f'spheroid_{a}_{c}'
        results['predictions'][config_key] = spheroid_preds
        results['point_errors'][config_key] = spheroid_errors
        results['point_areas'][config_key] = spheroid_areas
        results['point_coverage'][config_key] = spheroid_coverage
        results['best_params'][config_key] = spheroid_best_params
        results['oob_quantile'][config_key] = oob_quantile

        
        results['config_params'].append((a, c))

    # Save results for this cycle
    output_path = f'sunspots/results/hypothesis_results_cycle_{cyc}.npy'
    np.save(output_path, results)
    print("Results saved to", output_path)
    
    return results

def combine_cycle_results(all_cycle_results):
    """
    Combine results from multiple cycles and perform statistical tests.
    
    Parameters:
    - all_cycle_results: list of results dictionaries from each cycle
    
    Returns:
    - combined_results: dictionary with combined statistics and p-values
    """
    # Get configuration parameters from first cycle
    configs = all_cycle_results[0]['config_params']
    
    combined_results = {
        'config_params': configs,
        'weighted_mse_sphere': 0,
        'weighted_mse_spheroids': [],
        'weighted_area_sphere': 0,
        'weighted_area_spheroids': [],
        'weighted_coverage_sphere': 0,
        'weighted_coverage_spheroids': [],
        'delta_mse_percent': [],
        'delta_area_percent': [],
        'p_values_mse': [],
        'p_values_area': [],
        'p_adjusted_mse': [],
        'p_adjusted_area': [],
        'reject_h0_mse': [],
        'reject_h0_area': []
    }
    
    # Calculate total test points across all cycles for weighting
    total_test_points = sum(cycle_result['n_test_points'] for cycle_result in all_cycle_results)
    
    # Combine point-wise errors and areas across all cycles
    combined_sphere_errors = []
    combined_sphere_areas = []
    combined_spheroid_errors = {i: [] for i in range(len(configs))}
    combined_spheroid_areas = {i: [] for i in range(len(configs))}
    
    # Collect weighted means for MSE, areas, and coverage
    weighted_sphere_mse = 0
    weighted_sphere_area = 0
    weighted_sphere_coverage = 0
    weighted_spheroid_mse = [0] * len(configs)
    weighted_spheroid_area = [0] * len(configs)
    weighted_spheroid_coverage = [0] * len(configs)
    
    for cycle_result in all_cycle_results:
        weight = cycle_result['n_test_points'] / total_test_points
        
        # Collect sphere data
        sphere_errors = cycle_result['point_errors']['sphere']
        sphere_areas = cycle_result['point_areas']['sphere']
        sphere_coverage = cycle_result['point_coverage']['sphere']
        combined_sphere_errors.extend(sphere_errors)
        combined_sphere_areas.extend(sphere_areas)
        
        # Add to weighted means
        weighted_sphere_mse += weight * np.mean(sphere_errors)
        weighted_sphere_area += weight * np.mean(sphere_areas)
        weighted_sphere_coverage += weight * sphere_coverage
        
        # Collect spheroid data
        for i, (a, c) in enumerate(configs):
            config_key = f'spheroid_{a}_{c}'
            spheroid_errors = cycle_result['point_errors'][config_key]
            spheroid_areas = cycle_result['point_areas'][config_key]
            spheroid_coverage = cycle_result['point_coverage'][config_key]
            
            combined_spheroid_errors[i].extend(spheroid_errors)
            combined_spheroid_areas[i].extend(spheroid_areas)
            
            # Add to weighted means
            weighted_spheroid_mse[i] += weight * np.mean(spheroid_errors)
            weighted_spheroid_area[i] += weight * np.mean(spheroid_areas)
            weighted_spheroid_coverage[i] += weight * spheroid_coverage
    
    # Store weighted means
    combined_results['weighted_mse_sphere'] = weighted_sphere_mse
    combined_results['weighted_mse_spheroids'] = weighted_spheroid_mse
    combined_results['weighted_area_sphere'] = weighted_sphere_area
    combined_results['weighted_area_spheroids'] = weighted_spheroid_area
    combined_results['weighted_coverage_sphere'] = weighted_sphere_coverage
    combined_results['weighted_coverage_spheroids'] = weighted_spheroid_coverage
    
    # Convert to numpy arrays for statistical tests
    combined_sphere_errors = np.array(combined_sphere_errors)
    combined_sphere_areas = np.array(combined_sphere_areas)
    
    # Perform t-tests for each spheroid configuration
    p_values_mse = []
    p_values_area = []
    
    for i, (a, c) in enumerate(configs):
        spheroid_errors = np.array(combined_spheroid_errors[i])
        spheroid_areas = np.array(combined_spheroid_areas[i])
        
        # Calculate relative differences
        delta_mse_percent = 100 * (weighted_spheroid_mse[i] - weighted_sphere_mse) / weighted_sphere_mse
        delta_area_percent = 100 * (weighted_spheroid_area[i] - weighted_sphere_area) / weighted_sphere_area
        
        combined_results['delta_mse_percent'].append(delta_mse_percent)
        combined_results['delta_area_percent'].append(delta_area_percent)
        
        # Perform paired t-test for MSE
        # H0: mu_sphere >= mu_spheroid vs H1: mu_sphere < mu_spheroid (we want to see that there is no evidence that the spheroid performs worse)
        t_stat_mse, p_val_mse = stats.ttest_rel(combined_sphere_errors, spheroid_errors, alternative='less')
        p_values_mse.append(p_val_mse)
        
        # Perform paired t-test for areas
        # H0: mu_sphere_area = mu_spheroid_area vs H1: mu_sphere_area > mu_spheroid_area (one-sided test)
        t_stat_area, p_val_area = stats.ttest_rel(combined_sphere_areas, spheroid_areas, alternative='greater')
        p_values_area.append(p_val_area)
        
        print(f"Spheroid (a={a}, c={c}):")
        print(f"Delta MSE: {delta_mse_percent:.1f}%")
        print(f"Delta Area: {delta_area_percent:.1f}%")
        print(f"MSE t-test p-value: {p_val_mse:.4f}")
        print(f"Area t-test p-value: {p_val_area:.4f}")
    
    # Store p-values
    combined_results['p_values_mse'] = p_values_mse
    combined_results['p_values_area'] = p_values_area
    
    # BY correction for multiple testing
    if len(p_values_mse) > 0:
        # Correct MSE p-values
        reject_mse, p_adjusted_mse, _, _ = multipletests(p_values_mse, alpha=0.01, method='fdr_by')
        combined_results['reject_h0_mse'] = reject_mse
        combined_results['p_adjusted_mse'] = p_adjusted_mse
        
        # Correct area p-values
        reject_area, p_adjusted_area, _, _ = multipletests(p_values_area, alpha=0.01, method='fdr_by')
        combined_results['reject_h0_area'] = reject_area
        combined_results['p_adjusted_area'] = p_adjusted_area
    
    return combined_results

def generate_latex_table(combined_results, save_to_file=True):
    """
    Generate LaTeX table from combined results and save to file.
    
    Parameters:
    - combined_results: dictionary with combined statistics and p-values
    - save_to_file: bool, whether to save the table to a file
    
    Returns:
    - table_data: dictionary with structured table data
    - latex_table: string with complete LaTeX table
    """
    configs = combined_results['config_params']
    
    # Separate oblate and prolate configurations
    oblate_configs = [(i, a, c) for i, (a, c) in enumerate(configs) if a < 1]
    prolate_configs = [(i, a, c) for i, (a, c) in enumerate(configs) if a >= 1 and not (a == 1 and c == 1)]
    
    # Structure the table data
    table_data = {
        'oblate_configs': oblate_configs,
        'prolate_configs': prolate_configs,
        'oblate_lambdas': [a for i, a, c in oblate_configs],
        'prolate_lambdas': [c for i, a, c in prolate_configs],
        'sphere_data': {
            'weighted_mse': combined_results['weighted_mse_sphere'],
            'weighted_area': combined_results['weighted_area_sphere'],
            'weighted_coverage': combined_results['weighted_coverage_sphere']
        },
        'oblate_data': {
            'delta_mse_percent': [combined_results['delta_mse_percent'][i] for i, a, c in oblate_configs],
            'delta_area_percent': [combined_results['delta_area_percent'][i] for i, a, c in oblate_configs],
            'weighted_coverage': [combined_results['weighted_coverage_spheroids'][i] for i, a, c in oblate_configs],
            'p_adjusted_mse': [combined_results['p_adjusted_mse'][i] for i, a, c in oblate_configs],
            'p_adjusted_area': [combined_results['p_adjusted_area'][i] for i, a, c in oblate_configs]
        },
        'prolate_data': {
            'delta_mse_percent': [combined_results['delta_mse_percent'][i] for i, a, c in prolate_configs],
            'delta_area_percent': [combined_results['delta_area_percent'][i] for i, a, c in prolate_configs],
            'weighted_coverage': [combined_results['weighted_coverage_spheroids'][i] for i, a, c in prolate_configs],
            'p_adjusted_mse': [combined_results['p_adjusted_mse'][i] for i, a, c in prolate_configs],
            'p_adjusted_area': [combined_results['p_adjusted_area'][i] for i, a, c in prolate_configs]
        }
    }
    
    # Generate LaTeX table string
    latex_lines = []
    latex_lines.append("\\begin{table}[hpbt]")
    latex_lines.append("\\setlength{\\tabcolsep}{1.5pt}")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{lccccccccc|c|ccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multicolumn{1}{c}{} & \\multicolumn{9}{c}{Oblate ($S_{1, \\lambda}$)} &  \\multicolumn{1}{c}{Sphere} & \\multicolumn{3}{c}{Prolate ($S_{\\lambda, 1}$)} \\\\")
    latex_lines.append(" \\cmidrule(lr){2-10} \\cmidrule(lr){11-11}  \\cmidrule(lr){12-14}")
    
    # Lambda header
    lambda_header = "\\multicolumn{1}{c}{$\\lambda$}"
    for i, a, c in oblate_configs:
        if a == 0.5:
            lambda_header += f" & $\\mathbf{{{a}}}$"
        else:
            lambda_header += f" & ${a}$"
    lambda_header += " & $1.0$"  # sphere
    for i, a, c in prolate_configs:
        lambda_header += f" & ${c}$"
    latex_lines.append(lambda_header + " \\\\")
    latex_lines.append("\\midrule")
    
    # MSE row
    mse_row = "$\\Delta_{\\mathrm{MSE}}$"
    for delta in table_data['oblate_data']['delta_mse_percent']:
        mse_row += f" & ${delta:.1f}$"
    mse_row += " & $0.0$"  # sphere reference
    for delta in table_data['prolate_data']['delta_mse_percent']:
        mse_row += f" & ${delta:.1f}$"
    latex_lines.append(mse_row + " \\\\")
    
    # Area row
    area_row = "$\\Delta_{\\mathrm{area}}$"
    for j, delta in enumerate(table_data['oblate_data']['delta_area_percent']):
        if table_data['oblate_lambdas'][j] == 0.5:  # Bold for lambda = 0.5
            area_row += f" & $\\mathbf{{{delta:.1f}}}$"
        else:
            area_row += f" & ${delta:.1f}$"
    area_row += " & $0.0$"  # sphere reference
    for delta in table_data['prolate_data']['delta_area_percent']:
        area_row += f" & ${delta:.1f}$"
    latex_lines.append(area_row + " \\\\")
    
    # Coverage row
    coverage_row = "Coverage"
    for coverage in table_data['oblate_data']['weighted_coverage']:
        coverage_row += f" & ${coverage * 100:.1f}$"
    coverage_row += f" & ${table_data['sphere_data']['weighted_coverage'] * 100:.1f}$"  # sphere
    for coverage in table_data['prolate_data']['weighted_coverage']:
        coverage_row += f" & ${coverage * 100:.1f}$"
    latex_lines.append(coverage_row + " \\\\")
    
    # MSE p-value row
    p_mse_row = "$p_{\\mathrm{MSE}}$"
    for p_val in table_data['oblate_data']['p_adjusted_mse']:
        p_mse_row += f" & ${p_val:.10f}$"
    p_mse_row += " & ---"  # sphere
    for p_val in table_data['prolate_data']['p_adjusted_mse']:
        p_mse_row += f" & ${p_val:.10f}$"
    latex_lines.append(p_mse_row + " \\\\")
    
    # Area p-value row
    p_area_row = "$p_{\\mathrm{area}}$"
    for p_val in table_data['oblate_data']['p_adjusted_area']:
        p_area_row += f" & ${p_val:.10f}$"
    p_area_row += " & ---"  # sphere
    for p_val in table_data['prolate_data']['p_adjusted_area']:
        p_area_row += f" & ${p_val:.10f}$"
    latex_lines.append(p_area_row + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{For each $\\lambda$ specification, the errors, areas, and coverages were calculated on the test points of cycles $21$--$23$, and the reported values are weighted means across the three cycles, with weights proportional to the number of test points on each cycle. The areas correspond to OOB balls centered on the test points (measured in $100 \\times R_\\odot^2$). The relative MSE difference $\\Delta_{\\mathrm{MSE}}$ and the relative mean area difference $\\Delta_{\\mathrm{area}}$ (both in $\\%$) were calculated with respect to the balls on the sphere. The reported coverage corresponds to Type II (in \\%) for $\\alpha=0.10$. The $p$-value of the two-sided paired $t$-test with null hypothesis $H_0: \\mathrm{MSE}_{\\mathbb{S}^2} = \\mathrm{MSE}_{S_{1, \\lambda}}$ is $p_{\\mathrm{MSE}}$. For $p_{\\mathrm{area}}$, we considered a one-sided paired $t$-test with alternative hypothesis $H_1: \\mathrm{area}_{\\mathbb{S}^2} > \\mathrm{area}_{S_{1, \\lambda}}$, to test the equality of mean areas.}\\label{tab:spheroid_metrics}")
    latex_lines.append("\\end{table}")
    
    latex_table = '\n'.join(latex_lines)
    
    # Save to file if requested
    if save_to_file:
        # Save structured data as NumPy file
        table_output_path = 'sunspots/results/table_data.npy'
        np.save(table_output_path, table_data)
        print(f"Table data saved to {table_output_path}")
        
        # Save LaTeX table as text file
        latex_output_path = 'sunspots/results/table_latex.tex'
        with open(latex_output_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {latex_output_path}")
    
    return table_data, latex_table


if __name__ == "__main__":
    cycles = [21, 22, 23]

    # Collect results from all cycles
    all_cycle_results = []
    
    print("Processing all cycles...")
    for cyc in cycles:
        print(f"Processing cycle {cyc}")
        cycle_result = task(cyc)
        all_cycle_results.append(cycle_result)
    
    print("\nCombining results across cycles...")
    combined_results = combine_cycle_results(all_cycle_results)
    
    # Save combined results
    output_path = 'sunspots/results/combined_hypothesis_results.npy'
    np.save(output_path, combined_results)
    print(f"Combined results saved to {output_path}")
    
    print("\nGenerating LaTeX table...")
    table_data, latex_table = generate_latex_table(combined_results)