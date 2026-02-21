import numpy as np
from typing import Tuple, Dict, Any, Optional

def calculate_wls_similarity(
    x: np.ndarray, y: np.ndarray, 
    E: np.ndarray, N: np.ndarray, 
    w_sq: np.ndarray, sum_w: float, w_sqrt: np.ndarray, n: int
) -> Tuple[float, float, float, float, float, float, np.ndarray, float, np.ndarray]:
    """Calculates WLS Horizontal Similarity parameters."""
    
    # Calculate weighted centroids
    x_c = np.sum(w_sq * x) / sum_w
    y_c = np.sum(w_sq * y) / sum_w
    E_c = np.sum(w_sq * E) / sum_w
    N_c = np.sum(w_sq * N) / sum_w
    
    # Center coordinates
    x_prime = x - x_c
    y_prime = y - y_c
    E_prime = E - E_c
    N_prime = N - N_c

    # Validations for Horizontal Adjustment
    if n < 2:
        raise ValueError("Se requieren al menos 2 puntos para el ajuste horizontal.")

    coords_matrix = np.column_stack((E_prime, N_prime))
    if np.linalg.matrix_rank(coords_matrix, tol=1e-5) < 2 and n >= 3:
        raise ValueError("Geometría deficiente")

    # Solve for a and b using centered coordinates
    A = np.zeros((2 * n, 2))
    A[:n, 0] = x_prime
    A[:n, 1] = -y_prime
    A[n:, 0] = y_prime
    A[n:, 1] = x_prime

    L = np.concatenate([E_prime, N_prime])

    w_sqrt_2n = np.concatenate([w_sqrt, w_sqrt])
    A_w = A * w_sqrt_2n[:, np.newaxis]
    L_w = L * w_sqrt_2n

    params_ab, _, _, _ = np.linalg.lstsq(A_w, L_w, rcond=None)
    a = params_ab[0]
    b = params_ab[1]
    
    # --- Baarda Data Snooping (Horizontal) ---
    dof_h = 2 * n - 4
    outlier_flags_h = np.zeros(n, dtype=bool)
    
    if dof_h > 0:
        V_h = A_w @ params_ab - L_w
        sigma0_sq_h = np.dot(V_h, V_h) / dof_h
        
        if sigma0_sq_h > 1e-12:
            N_mat_h = A_w.T @ A_w
            try:
                N_inv_h = np.linalg.inv(N_mat_h)
                diag_Q_hat = np.sum((A_w @ N_inv_h) * A_w, axis=1)
                diag_Q_vv = 1.0 - diag_Q_hat
                
                diag_Q_vv_safe = np.maximum(diag_Q_vv, 1e-12)
                w_stat = V_h / (np.sqrt(sigma0_sq_h) * np.sqrt(diag_Q_vv_safe))
                
                w_stat_E = w_stat[:n]
                w_stat_N = w_stat[n:]
                max_w_stat = np.maximum(np.abs(w_stat_E), np.abs(w_stat_N))
                
                critical_value = 3.0
                outlier_flags_h = max_w_stat > critical_value
            except np.linalg.LinAlgError:
                pass

    return a, b, x_c, y_c, E_c, N_c, outlier_flags_h, sigma0_sq_h if dof_h > 0 else 0.0, np.column_stack((E_prime, N_prime))


def calculate_wls_vertical(
    Z_error: np.ndarray, E_prime: np.ndarray, N_prime: np.ndarray, 
    w_sq: np.ndarray, sum_w: float, w_sqrt: np.ndarray, n: int, 
    N_c: float, E_c: float
) -> Tuple[float, float, float, int, bool, bool, np.ndarray, float]:
    """Calculates WLS Vertical parameters."""
    bad_vertical_geometry = False
    bad_condition = False
    
    if n >= 3:
        A_v = np.ones((n, 3))
        A_v[:, 1] = N_prime
        A_v[:, 2] = E_prime
        
        A_v_w = A_v * w_sqrt[:, np.newaxis]
        Z_error_w = Z_error * w_sqrt
        
        v_params, _, rank, _ = np.linalg.lstsq(A_v_w, Z_error_w, rcond=None)
        
        if rank < 3:
            bad_vertical_geometry = True
            C = np.sum(w_sq * Z_error) / sum_w
            slope_n = 0.0
            slope_e = 0.0
        else:
            cond = np.linalg.cond(A_v)
            if cond > 1e10:
                bad_condition = True

            C = v_params[0]
            slope_n = v_params[1]
            slope_e = v_params[2]
    else:
        C = np.sum(w_sq * Z_error) / sum_w
        slope_n = 0.0
        slope_e = 0.0
        rank = 1
        A_v_w = np.ones((n, 1)) * w_sqrt[:, np.newaxis]
        v_params = np.array([C])
        Z_error_w = Z_error * w_sqrt
        
    # --- Baarda Data Snooping (Vertical) ---
    num_v_params = 3 if (n >= 3 and rank >= 3) else 1
    dof_v = n - num_v_params
    outlier_flags_v = np.zeros(n, dtype=bool)
    sigma0_sq_v = 0.0
    
    if dof_v > 0 and 'A_v_w' in locals():
        V_v = A_v_w @ v_params - Z_error_w
        sigma0_sq_v = np.dot(V_v, V_v) / dof_v
        
        if sigma0_sq_v > 1e-12:
            N_mat_v = A_v_w.T @ A_v_w
            try:
                N_inv_v = np.linalg.inv(N_mat_v)
                diag_Q_hat_v = np.sum((A_v_w @ N_inv_v) * A_v_w, axis=1)
                diag_Q_vv_v = 1.0 - diag_Q_hat_v
                
                diag_Q_vv_v_safe = np.maximum(diag_Q_vv_v, 1e-12)
                w_stat_v = V_v / (np.sqrt(sigma0_sq_v) * np.sqrt(diag_Q_vv_v_safe))
                
                critical_value = 3.0
                outlier_flags_v = np.abs(w_stat_v) > critical_value
            except np.linalg.LinAlgError:
                pass
                
    return C, slope_n, slope_e, rank, bad_vertical_geometry, bad_condition, outlier_flags_v, sigma0_sq_v

def check_extrapolation(E_trans: np.ndarray, N_trans: np.ndarray, local_pts: np.ndarray) -> Optional[float]:
    """Calculates extrapolation distance. Returns max distance if outside, else None."""
    if local_pts is not None and len(local_pts) >= 3:
        try:
            from scipy.spatial import ConvexHull
            from scipy.spatial.qhull import QhullError
            
            try:
                hull = ConvexHull(local_pts)
                equations = hull.equations
                trans_pts = np.column_stack((E_trans, N_trans))
                dists = np.dot(trans_pts, equations[:, :2].T) + equations[:, 2]
                max_dists = np.max(dists, axis=1)
                epsilon = 1e-5
                outside_mask = max_dists > epsilon
                
                if np.any(outside_mask):
                    return np.max(max_dists[outside_mask]), np.sum(outside_mask)
            except QhullError:
                pass
        except ImportError:
            pass
    return None, 0
