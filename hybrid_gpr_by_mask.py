import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from helper_hybrid import (
    golden_section_search,
    void_ratio_to_density,
    physical_model,
    load_gpr_model,
    train_and_evaluate_gpr_kernels,
    calculate_ndg_suction,
    write_optimal_alpha_to_file,
    calculate_rmse,
)
from hybrid_initial_settings import (
    threshold,
    kernels,
)
from config import get_config
import sys
import os
from data_interface import ensure_json_from_csv, load_dataset_json


def read_coords(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read relative coordinates from dataset frame.

    Expects `rel_X (m)` and `rel_Y (m)` columns to be present.

    Args:
        df: Input DataFrame from JSON data.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y coordinate arrays.
    """
    return df['rel_X (m)'].to_numpy(), df['rel_Y (m)'].to_numpy()


def prepare_dataset(json_path: str, default_moisture_label: str):
    """Load and transform dataset for model training and evaluation.

    Loads JSON, ensures required columns, computes derived moisture numeric per row
    using OMC and the configured offset, and returns arrays used downstream.

    Args:
        json_path: Path to dataset JSON.
        default_moisture_label: Fallback moisture label when missing in data.

    Returns:
        Tuple of arrays: (ndg_density, lwd_av, moisture_labels, moisture_numeric,
        ndg_void, x_coord, y_coord, used_mask, omc)
    """
    payload = load_dataset_json(json_path)
    df = pd.DataFrame(payload['data'])
    ndg_density = df['NDG-density'].to_numpy()
    # Use precomputed LWD_av
    lwd_av = df['LWD_av'].to_numpy()
    used_mask = df['usedfortraining'].astype(bool).to_numpy()
    x_coord, y_coord = read_coords(df)
    # Derive per-row moisture numeric from OMC: w_d=OMC-0.5, w_o=OMC, w_l=OMC+0.5
    omc = df['OMC'].astype(float).to_numpy()
    moisture_input_list = df.get('moisture_label', pd.Series([default_moisture_label]*len(df))).astype(str).str.lower().tolist()
    derived_values = []
    omc_offset = float(get_config()['omc']['offset'])
    for i, label in enumerate(moisture_input_list):
        if label == 'low':
            derived_values.append(omc[i] - omc_offset)
        elif label == 'high':
            derived_values.append(omc[i] + omc_offset)
        else:
            derived_values.append(omc[i])
    moisture_numeric_list = derived_values
    moisture_numeric_array = np.array(moisture_numeric_list)
    cfg = get_config()
    Gs = float(cfg['soil']['Gs'])
    ndg_void = Gs / (ndg_density / (1 + moisture_numeric_array / 100.0)) - 1.0
    return ndg_density, lwd_av, moisture_input_list, moisture_numeric_array, ndg_void, x_coord, y_coord, used_mask, omc


def optimize_alpha_for_dataset(best_kernel, lwd_av, moisture_input_list, omc, alpha_init, gpr_model):
    """Coarse optimization of alpha using physics-only variance heuristic.

    This function proposes an alpha by maximizing separation (variance) of
    physics-only densities across data points. It uses a golden-section search
    over [alpha.min, alpha.max] from config. The result is later refined when
    retraining the GPR on residuals.

    Args:
        best_kernel: The selected GPR kernel (unused here but kept for API symmetry).
        lwd_av: Array of LWD averages per row.
        moisture_input_list: List of moisture labels per row.
        omc: Array of OMC values per row.
        alpha_init: Initial alpha (unused in heuristic but kept for API).
        gpr_model: GPR surrogate model instance.

    Returns:
        float: Proposed alpha value.
    """
    def objective(alpha: float) -> float:
        phys_dens = []
        for i in range(len(lwd_av)):
            cfg_local = get_config()
            initial_void_ratio_guess = float(cfg_local['physics']['initial_void_ratio_guess'])
            moisture_input = moisture_input_list[i]
            omc_offset_local = float(cfg_local['omc']['offset'])
            m_phys_void_ratio, _, _, moisture_numeric = physical_model(
                lwd_av[i], moisture_input, {'low': omc[i]-omc_offset_local, 'medium': omc[i], 'high': omc[i]+omc_offset_local}, initial_void_ratio_guess, None, None, None, alpha, gpr_model
            )
            phys_dens.append(void_ratio_to_density(m_phys_void_ratio, moisture_numeric))
        # Dummy R2 on physical only to guide alpha; we refine after training. Negate to minimize
        y = np.array(phys_dens)
        return -np.var(y)  # placeholder objective favoring variance separation

    cfg = get_config()
    alpha_min = float(cfg['alpha']['min'])
    alpha_max = float(cfg['alpha']['max'])
    tol = float(cfg['golden_section']['tol'])
    return golden_section_search(objective, alpha_min, alpha_max, tol)


def run_option(option_name: str, csv_path: str, json_path: str, cfg, gpr_model):
    """Run full pipeline for an option: prepare, train, optimize, predict, plot.

    Args:
        option_name: Name to tag outputs (e.g., 'option1').
        csv_path: Source CSV path.
        json_path: Target JSON path used for loading.
        cfg: Loaded configuration dict.
        gpr_model: Pre-loaded GPR surrogate model.
    """
    json_ready = ensure_json_from_csv(csv_path, json_path)
    ndg_density, lwd_av, moisture_input_list, moisture_numeric_array, ndg_void, x_coord, y_coord, used_mask, omc = prepare_dataset(
        json_ready, cfg['moisture']['default_label']
    )

    # Initial alpha
    alpha = float(cfg['alpha']['initial'])

    # Physical density for all rows
    physical_density = []
    omc_offset = float(cfg['omc']['offset'])
    for i in range(len(lwd_av)):
        initial_void_ratio_guess = float(cfg['physics']['initial_void_ratio_guess'])
        # Use derived moisture numeric directly with moisture label to satisfy API
        m_phys_void_ratio, _, _, moisture_numeric = physical_model(
            lwd_av[i], moisture_input_list[i], {'low': omc[i]-omc_offset, 'medium': omc[i], 'high': omc[i]+omc_offset}, initial_void_ratio_guess, None, None, None, alpha, gpr_model
        )
        physical_density.append(void_ratio_to_density(m_phys_void_ratio, moisture_numeric))
    physical_density = np.array(physical_density)

    # Residuals (observed - physical)
    residuals = ndg_density - physical_density

    # NDG suction
    alpha_list = [alpha] * len(ndg_void)
    ndg_suction_array = calculate_ndg_suction(ndg_void, moisture_numeric_array, alpha_list, moisture_input_list)

    # Build training matrix
    Xa = np.column_stack((
        lwd_av[used_mask],
        x_coord[used_mask],
        y_coord[used_mask],
        moisture_numeric_array[used_mask],
        ndg_suction_array[used_mask],
    ))
    res_a = residuals[used_mask]
    ndg_a = ndg_density[used_mask]
    phys_a = physical_density[used_mask]

    # Train and select kernel
    gpr_model_opt, best_kernel_name, best_r2, best_rmse = train_and_evaluate_gpr_kernels(
        kernels=kernels,
        X_a=Xa,
        residuals=res_a,
        physical_density=phys_a,
        NDG_density=ndg_a,
        lane_size=res_a.shape[0],
        option=option_name,
    )

    # Optimize alpha (optional refinement)
    best_kernel = gpr_model_opt.kernel
    optimal_alpha = optimize_alpha_for_dataset(best_kernel, lwd_av, moisture_input_list, omc, alpha, gpr_model)
    write_optimal_alpha_to_file(optimal_alpha, option=option_name)

    # Recompute physical density with optimized alpha
    physical_density_opt = np.zeros_like(physical_density)
    omc_offset = float(cfg['omc']['offset'])
    for i in range(len(lwd_av)):
        initial_void_ratio_guess = float(cfg['physics']['initial_void_ratio_guess'])
        m_phys_void_ratio, _, _, moisture_numeric = physical_model(
            lwd_av[i], moisture_input_list[i], {'low': omc[i]-omc_offset, 'medium': omc[i], 'high': omc[i]+omc_offset}, initial_void_ratio_guess, None, None, None, optimal_alpha, gpr_model
        )
        physical_density_opt[i] = void_ratio_to_density(m_phys_void_ratio, moisture_numeric)
    residuals_opt = ndg_density - physical_density_opt

    # Retrain on training set
    ndg_suction_array = calculate_ndg_suction(ndg_void, moisture_numeric_array, [optimal_alpha] * len(ndg_void), moisture_input_list)
    Xa = np.column_stack((
        lwd_av[used_mask],
        x_coord[used_mask],
        y_coord[used_mask],
        moisture_numeric_array[used_mask],
        ndg_suction_array[used_mask],
    ))
    gpr_model_opt.fit(Xa, residuals_opt[used_mask])

    # Predict on holdout (not used for training)
    hold_mask = ~used_mask
    Xp = np.column_stack((
        lwd_av[hold_mask],
        x_coord[hold_mask],
        y_coord[hold_mask],
        moisture_numeric_array[hold_mask],
        ndg_suction_array[hold_mask],
    ))
    pred_res = gpr_model_opt.predict(Xp)
    pred_density = physical_density_opt[hold_mask] + pred_res
    actual_density = ndg_density[hold_mask]

    rmse = calculate_rmse(actual_density, pred_density)
    r2 = r2_score(actual_density, pred_density)

    # Save per-option results
    pd.DataFrame({
        'Actual_NDG_Density': actual_density,
        'Predicted_Density': pred_density,
    }).to_csv(f'prediction_vs_ndg_{option_name}.csv', index=False)

    with open(f'final_results_{option_name}.txt', 'w') as f:
        f.write(f'Final R² on holdout with optimized alpha: {r2:.4f}\n')
        f.write(f'Final RMSE on holdout with optimized alpha: {rmse:.4f}\n')

    # Plotting disabled to keep the Lambda deployment lightweight.


def main():
    """Entry point: configure logging, load model, run both options."""
    cfg = get_config()
    # Setup logging redirection
    log_path = cfg.get('logging', {}).get('file', 'run.log')
    if not os.path.isabs(log_path):
        log_path = os.path.join(os.path.dirname(__file__), log_path)
    mode = 'a' if cfg.get('logging', {}).get('append', True) else 'w'
    log_fh = open(log_path, mode)
    sys.stdout = log_fh
    sys.stderr = log_fh
    np.random.seed(int(cfg['seed']))
    # Load model
    gpr_model = load_gpr_model(cfg['model']['path'])

    # Run Option 1 and Option 2 based on CSVs
    run_option('option1', cfg['data']['csv_option1'], cfg['data']['json_option1'], cfg, gpr_model)
    run_option('option2', cfg['data']['csv_option2'], cfg['data']['json_option2'], cfg, gpr_model)
    log_fh.flush()
    log_fh.close()


if __name__ == '__main__':
    main()

