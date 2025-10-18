import numpy as np
from pyswarm import pso
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from config import get_config
from typing import Optional
CFG = get_config()
GS = float(CFG["soil"]["Gs"])

# SageMaker adapter
import os
import json
import boto3

class SageMakerPredictorAdapter:
    """Adapter that mimics scikit-learn's predict API, calling a SageMaker endpoint.

    Expected behavior:
    - predict(X) -> np.array of shape (n_samples,)
    - predict(X, return_std=True) -> (np.array, np.array) where std is approximated
    """
    def __init__(self, endpoint_name: str, region: Optional[str] = None):
        self.endpoint_name = endpoint_name
        self.client = boto3.client('sagemaker-runtime', region_name=region or os.getenv('AWS_REGION'))

    def predict(self, X, return_std: bool = False):
        # Ensure X is a 2D list
        if hasattr(X, 'tolist'):
            payload = X.tolist()
        else:
            payload = X
        body = json.dumps({"instances": payload})
        resp = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=body,
        )
        result = json.loads(resp['Body'].read().decode('utf-8'))
        # Expect {"predictions": [y1, y2, ...]} or raw list
        if isinstance(result, dict) and 'predictions' in result:
            preds = result['predictions']
        else:
            preds = result
        y = np.array(preds).reshape(-1)
        if return_std:
            # If endpoint does not return std, provide a small constant as placeholder
            std = np.full_like(y, 0.05, dtype=float)
            return y, std
        return y


def calculate_rmse(y_true, y_pred):
    """Compute root mean squared error (RMSE).

    Args:
        y_true: Array-like of ground-truth target values.
        y_pred: Array-like of predicted target values.

    Returns:
        float: The RMSE between predictions and ground truth.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def golden_section_search(f, a, b, tol=1e-5):
    """Minimize a univariate function on [a, b] via Golden Section search.

    Args:
        f: Callable taking a single float argument and returning a float objective.
        a: Lower bound of the search interval.
        b: Upper bound of the search interval.
        tol: Termination tolerance on the bracket size.

    Returns:
        float: Approximate minimizer location within [a, b].
    """
    invphi = (np.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1/phi^2
    
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
    
    return (b + a) / 2
 


# General helper functions (unchanged)
 

def calculate_suction(void_ratio, moisture, alpha):
    """Calculate suction using a mixed dry/wet SWCC formulation.

    Uses parameters from config: swcc.n, swcc.m, swcc.a, swcc.omega and soil.Gs.
    The mixing between dry and wet curves is controlled by alpha in [0, 1].

    Args:
        void_ratio: Soil void ratio e.
        moisture: Moisture content in percent (0-100).
        alpha: Mixing factor between dry and wet suction (1=dry, 0=wet).

    Returns:
        float: Suction value under given state.
    """
    n_swcc = float(CFG["swcc"]["n"]) 
    m_swcc = float(CFG["swcc"]["m"]) 
    a_swcc = float(CFG["swcc"]["a"]) 
    a_w_swcc = a_swcc / 2
    omega = float(CFG["swcc"]["omega"]) 
    sr = max(min(GS * (moisture/100) / void_ratio, 0.9999999999), 0.001)
    suc_d = a_swcc * (sr ** (-1 / m_swcc) - 1) ** (1 / n_swcc) / void_ratio ** omega
    suc_w = a_w_swcc * (sr ** (-1 / m_swcc) - 1) ** (1 / n_swcc) / void_ratio ** omega
    return suc_d * alpha + (1-alpha) * suc_w

def void_ratio_to_density(void_ratio, moisture):
    """Convert void ratio and moisture to bulk density using Gs from config.

    Args:
        void_ratio: Soil void ratio e.
        moisture: Moisture content in percent (0-100).

    Returns:
        float: Bulk density.
    """
    dry_density = GS / (1 + void_ratio)
    return dry_density * (1.0 + moisture / 100)
 

def objective_function(void_ratio, lwd_modulus, moisture_numeric, alpha, model):
    """PSO objective for void ratio fit using surrogate model and suction physics.

    Args:
        void_ratio: Candidate void ratio (scalar or array-like from PSO).
        lwd_modulus: LWD modulus at the point.
        moisture_numeric: Moisture content (%) at the point.
        alpha: Mixing factor between dry and wet suction.
        model: Trained surrogate model mapping features to void ratio.

    Returns:
        float: Squared error between candidate and predicted void ratio.
    """
    # Accept vector input from PSO and extract scalar
    try:
        void_ratio_scalar = float(np.atleast_1d(void_ratio)[0])
    except Exception:
        void_ratio_scalar = void_ratio
    suction = calculate_suction(void_ratio_scalar, moisture_numeric, alpha)
    X_new = np.array([[moisture_numeric / 100, lwd_modulus, suction]])
    predicted_void_ratio = model.predict(X_new)[0]
    return (void_ratio_scalar - predicted_void_ratio)**2

def predict_void_ratio(lwd_modulus, moisture_numeric, alpha, initial_void_ratio_guess, model):
    """Estimate void ratio via PSO by minimizing objective_function.

    Args:
        lwd_modulus: LWD modulus value.
        moisture_numeric: Moisture content (%) value.
        alpha: Mixing factor between dry and wet suction.
        initial_void_ratio_guess: Historical parameter (kept for API compatibility).
        model: Trained surrogate model mapping features to void ratio.

    Returns:
        tuple[float, float]: (optimal_void_ratio, std_of_model_prediction).
    """
    cfg = get_config()
    lb = [cfg["pso"]["lb_void_ratio"]]
    ub = [cfg["pso"]["ub_void_ratio"]]
    xopt, fopt = pso(
        objective_function,
        lb,
        ub,
        args=(lwd_modulus, moisture_numeric, alpha, model),
        swarmsize=cfg["pso"]["swarmsize"],
        omega=cfg["pso"]["omega"],
        phip=cfg["pso"]["phip"],
        phig=cfg["pso"]["phig"],
        maxiter=cfg["pso"]["maxiter"],
        debug=cfg["pso"]["debug"],
    )
    suction = calculate_suction(xopt[0], moisture_numeric, alpha)
    X_new = np.array([[moisture_numeric / 100, lwd_modulus, suction]])
    predicted_void_ratio, std_predicted_void_ratio = model.predict(X_new, return_std=True)
    return xopt[0], std_predicted_void_ratio
 

def physical_model(lwd_modulus, moisture_input, initial_moisture_values, initial_void_ratio_guess, wd, wo, wl, alpha,gpr_model):
    """Run the physics + surrogate model to predict void ratio and density.

    Note: The wd/wo/wl parameters are ignored; per-row moisture values are
    provided via initial_moisture_values dict keyed by moisture label.

    Args:
        lwd_modulus: LWD modulus value.
        moisture_input: Moisture label ('low'|'medium'|'high').
        initial_moisture_values: Dict mapping label to numeric moisture (%).
        initial_void_ratio_guess: Kept for API compatibility; unused by PSO backend.
        wd: Deprecated.
        wo: Deprecated.
        wl: Deprecated.
        alpha: Mixing factor between dry and wet suction.
        gpr_model: Surrogate model for void ratio prediction.

    Returns:
        tuple[float, float, float, float]: (void_ratio, std_void_ratio, density, moisture_numeric).
    """
    # 'wd/wo/wl' kept for signature compatibility; values come from per-row initial_moisture_values
    moisture_numeric = initial_moisture_values[moisture_input]
    predicted_void_ratio, std_predicted_void_ratio = predict_void_ratio(lwd_modulus, moisture_numeric, alpha, initial_void_ratio_guess, gpr_model)
    predicted_density = void_ratio_to_density(predicted_void_ratio, moisture_numeric)
    return predicted_void_ratio, std_predicted_void_ratio, predicted_density, moisture_numeric
# Load GPR model
def load_gpr_model(filepath):
    """Load a trained GPR model or return a SageMaker endpoint adapter if configured."""
    endpoint = os.getenv('SAGEMAKER_ENDPOINT_NAME')
    if endpoint:
        return SageMakerPredictorAdapter(endpoint, os.getenv('AWS_REGION'))
    return joblib.load(filepath)

# Function to load data
 
 

def train_and_evaluate_gpr_kernels(kernels, X_a, residuals, physical_density, NDG_density, lane_size, option, cv_splits=3):
    """
    Train and evaluate Gaussian Process Regressor models with different kernels using cross-validation.

    Parameters:
    - kernels: A dictionary of kernel names and corresponding kernel objects.
    - X_a: The input matrix for the GPR model (e.g., LWD, x_coord, y_coord, moisture, suction).
    - residuals: The residuals from the physical model (observed - predicted).
    - physical_density: The predicted physical density values.
    - NDG_density: The observed NDG density values.
    - lane_size: The number of data points per lane.
    - option: The option number (1 or 2) for distinguishing between different experiments.
    - cv_splits: Number of cross-validation splits (default is 5).

    Returns:
    - best_gpr: The GPR model with the highest R² score.
    - best_kernel_name: The name of the best-performing kernel.
    - best_r2: The highest cross-validated R² score achieved.
    - best_rmse: The lowest cross-validated RMSE achieved.
    """
    # Convert lists to NumPy arrays if they aren't already
    X_a = np.array(X_a)
    residuals = np.array(residuals)
    physical_density = np.array(physical_density)
    NDG_density = np.array(NDG_density)

    best_r2 = -np.inf
    best_rmse = np.inf
    best_kernel_name = None
    best_gpr = None

    # Define the cross-validation scheme
    cfg = get_config()
    kf = KFold(n_splits=int(cfg["cv"]["splits"]), shuffle=True, random_state=42)

    # Open the file for logging R² and RMSE scores for this option
    with open(f"performance_metrics_option_{option}.txt", "w") as file:
        for kernel_name, kernel in kernels.items():
            r2_scores = []
            rmse_scores = []

            for train_index, test_index in kf.split(X_a):
                X_train, X_test = X_a[train_index], X_a[test_index]
                residuals_train, residuals_test = residuals[train_index], residuals[test_index]
                physical_density_train, physical_density_test = physical_density[train_index], physical_density[test_index]
                NDG_density_test = NDG_density[test_index]

                # Debugging: Print shapes and types
                print(f"Kernel: {kernel_name}")
                print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
                print(f"residuals_train shape: {residuals_train.shape}, residuals_test shape: {residuals_test.shape}")
                print(f"physical_density_train shape: {physical_density_train.shape}, physical_density_test shape: {physical_density_test.shape}")
                print(f"NDG_density_test shape: {NDG_density_test.shape}")

                # Train GPR model
                gpr_option1 = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=int(cfg["gpr"]["n_restarts_optimizer"]),
                    alpha=float(cfg["gpr"]["alpha"]),
                )
                gpr_option1.fit(X_train, residuals_train)
                
                # Predict on the test data
                y_pred = gpr_option1.predict(X_test) + physical_density_test

                # Debugging: Print predictions and actual values
                print(f"y_pred shape: {y_pred.shape}, NDG_density_test shape: {NDG_density_test.shape}")
                print(f"y_pred: {y_pred}")
                print(f"NDG_density_test: {NDG_density_test}")

                # Calculate R² score
                r2 = r2_score(NDG_density_test, y_pred)
                r2_scores.append(r2)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(NDG_density_test, y_pred))
                rmse_scores.append(rmse)

            # Average the cross-validation scores
            avg_r2 = np.mean(r2_scores)
            avg_rmse = np.mean(rmse_scores)

            # Print and log the results
            print(f'{kernel_name} Kernel Avg R²: {avg_r2:.4f}, Avg RMSE: {avg_rmse:.4f}')
            file.write(f'{kernel_name} Kernel Avg R²: {avg_r2:.4f}, Avg RMSE: {avg_rmse:.4f}\n')
            
            # Check if this is the best model
            if avg_r2 > best_r2 or (avg_r2 == best_r2 and avg_rmse < best_rmse):
                best_r2 = avg_r2
                best_rmse = avg_rmse
                best_kernel_name = kernel_name
                best_gpr = gpr_option1
        
        file.write(f'\nBest Kernel: {best_kernel_name} with Avg R²: {best_r2:.4f}, Avg RMSE: {best_rmse:.4f}\n')

    print(f'\nBest Kernel: {best_kernel_name} with Avg R²: {best_r2:.4f}, Avg RMSE: {best_rmse:.4f}')
    
    return best_gpr, best_kernel_name, best_r2, best_rmse
 

def calculate_ndg_suction(NDG_void, moisture_numeric_array, alpha_list, moisture_input_list):
    """
    Calculate NDG suction for each data point.

    Parameters:
    - NDG_void: Array of NDG void ratios.
    - moisture_numeric_array: Array of moisture numeric values.
    - alpha_list: List of alpha values for each data point.
    - moisture_input_list: List of moisture inputs corresponding to each data point.

    Returns:
    - NDG_suction_array: Array of calculated NDG suction values.
    """
    NDG_suction_list = []

    for i in range(len(NDG_void)):
        # Use the alpha value provided in the alpha_list
        alpha = alpha_list[i]
        
        # Calculate suction using the NDG_void, moisture_numeric_array, and alpha
        NDG_suction = calculate_suction(NDG_void[i], moisture_numeric_array[i], alpha)
        
        # Append the calculated suction to the list
        NDG_suction_list.append(NDG_suction)

    # Convert the list to a numpy array
    NDG_suction_array = np.array(NDG_suction_list)
    
    return NDG_suction_array


def write_optimal_alpha_to_file(optimal_alpha, option):
    """Persist the optimized alpha value for a given option to text file.

    Args:
        optimal_alpha: The optimized alpha scalar.
        option: Option name/identifier used in file naming.
    """
    with open(f"optimal_alpha_option_{option}.txt", "w") as file:
        file.write(f"Optimal alpha for Option {option}: {optimal_alpha:.6f}\n")
