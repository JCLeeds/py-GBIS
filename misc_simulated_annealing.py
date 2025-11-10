import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.interpolate import griddata
import pCDM_model as pCDM_fast
import okada_model as okada_fast

import matplotlib.pyplot as plt
import os 
import pandas as pd
import llh2local as llh
import local2llh as l2llh

def simulated_annealing_optimization(u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                                   C_inv, C_logdet, los_e, los_n, los_u,starting_params=None,
                                   SA_iterations=10000, initial_temp=10.0, 
                                   cooling_rate=0.95, min_temp=0.01,
                                   step_sizes=None, bounds=None,model_type='pCDM'):
    """
    Use simulated annealing to find good initial parameter estimates.
    
    Parameters:
    -----------
    u_los_obs : array
        Observed LOS displacements
    X_obs, Y_obs : array
        Observation coordinates
    incidence_angle, heading : float
        Satellite geometry parameters
    C_inv : array
        Inverse of noise covariance matrix
    C_logdet : float
        Log determinant of noise covariance matrix
    los_e, los_n, los_u : float
        Line-of-sight unit vector components
    n_iterations : int
        Number of SA iterations
    initial_temp : float
        Initial temperature
    cooling_rate : float
        Temperature cooling rate (0 < rate < 1)
    min_temp : float
        Minimum temperature (stopping criterion)
    step_sizes : dict
        Step sizes for each parameter
    bounds : dict
        Parameter bounds
        
    Returns:
    --------
    best_params : dict
        Best parameter values found
    best_energy : float
        Best energy (negative log likelihood)
    energy_trace : list
        Energy evolution during SA
    temperature_trace : list
        Temperature evolution during SA
    """
    

    
    def energy_function(params):
        """Calculate energy (negative log likelihood)"""
        try:
            # Check bounds first
            for key, (lower, upper) in bounds.items():
                if not (lower <= params[key] <= upper):
                    return np.inf
                
            # dvs = [params['DVx'], params['DVy'], params['DVz']]
            # signs = [np.sign(dv) for dv in dvs if dv != 0]
            
            # if len(set(signs)) > 1:
            #     return np.inf
                

          
            # Debug: Print parameters being tested
            if np.random.rand() < 0.01:  # Print 1% of the time
                param_str = ", ".join([f"{key}={value:.5f}" if isinstance(value, (int, float)) else f"{key}={value}" 
                                     for key, value in params.items()])
                print(f"  Testing params: {param_str}")

            # Forward model - ensure inputs are proper types
            X_obs_arr = np.asarray(X_obs, dtype=float)
            Y_obs_arr = np.asarray(Y_obs, dtype=float)
            

            if model_type.lower() == 'pcdm':
                ue, un, uv = pCDM_fast.pCDM(
                    X_obs_arr, Y_obs_arr, 
                    float(params['X0']), float(params['Y0']), float(params['depth']),
                    float(params['omegaX']), float(params['omegaY']), float(params['omegaZ']),
                    float(params['DVx']), float(params['DVy']), float(params['DVz']), 
                    0.25
                )
            elif model_type.lower() == 'mctigue':
                # ue, un, uv = pCDM_fast.McTigue(
                #     X_obs_arr, Y_obs_arr, 
                #     float(params['X0']), float(params['Y0']), float(params['depth']),
                #     float(params['a']), float(params['c']),
                #     float(params['DV']),
                #     0.25
                # )
                pass
            elif model_type.lower() == 'mogi':
                # ue, un, uv = pCDM_fast.Mogi(
                #     X_obs_arr, Y_obs_arr, 
                #     float(params['X0']), float(params['Y0']), float(params['depth']),
                #     float(params['DV']),
                #     0.25
                # )
                pass
            elif model_type.lower() == 'yang':
                # ue, un, uv = pCDM_fast.Yang(
                #     X_obs_arr, Y_obs_arr, 
                #     float(params['X0']), float(params['Y0']), float(params['depth']),
                #     float(params['a']), float(params['c']),
                #     float(params['DV']),
                #     0.25
                # )
                pass
            elif model_type.lower() == 'okada':
                
                ue, un, uv = okada_fast.disloc3d3(
                    X_obs_arr, Y_obs_arr, 
                    xoff=float(params['X0']), yoff=float(params['Y0']), depth=float(params['depth']),
                    length=float(params['length']), width=float(params['width']),
                    strike=float(params['strike']), dip=float(params['dip']),
                    rake=float(params['rake']),
                    slip=float(params['slip']),
                    opening=float(params['opening']),
                    nu=0.25
                )
                pass
            
            # Ensure outputs are arrays
            ue = np.asarray(ue, dtype=float)
            un = np.asarray(un, dtype=float)  
            uv = np.asarray(uv, dtype=float)
            
            # Check for NaN or inf in forward model output
            if np.any(~np.isfinite(ue)) or np.any(~np.isfinite(un)) or np.any(~np.isfinite(uv)):
                return np.inf
            
            # Convert to line-of-sight
            u_los_pred = -(ue * los_e + un * los_n + uv * los_u)
            u_los_pred = np.asarray(u_los_pred, dtype=float)
            u_los_pred = u_los_pred.flatten()
            
            # Check for NaN or inf in LOS prediction
            if np.any(~np.isfinite(u_los_pred)):
                return np.inf
            
            # Calculate residuals
            residuals = np.asarray(u_los_obs.flatten(), dtype=float) - u_los_pred
            
            # Check residuals shape and finite values
            if residuals.shape != u_los_obs.shape:
                print(f"Shape mismatch: residuals {residuals.shape} vs observed {u_los_obs.shape}")
                return np.inf
                
            if np.any(~np.isfinite(residuals)):
                return np.inf
            
            # Calculate negative log likelihood (energy to minimize)
            quad_form = residuals.T @ C_inv @ residuals
            if not np.isfinite(quad_form):
                return np.inf
                
            neg_log_lik = 0.5 * (quad_form + C_logdet + len(residuals) * np.log(2 * np.pi))
            
            if not np.isfinite(neg_log_lik):
                return np.inf
            
            return neg_log_lik
            
        except Exception as e:
            if np.random.rand() < 0.001:  # Print occasional errors for debugging
                print(f"  Error in energy_function: {type(e).__name__}: {e}")
            return np.inf
    
    # Initialize with random parameters within bounds
    current_params = {}
    for key, (lower, upper) in bounds.items():
        current_params[key] = np.random.uniform(lower, upper)
    # Use provided starting parameters if available
    if starting_params is not None:
        current_params = starting_params.copy()
        print(f"Using provided starting parameters:")
        for key, val in current_params.items():
            print(f"  {key:10s}: {val:8.4f}")
    else:
        for key, (lower, upper) in bounds.items():
            current_params[key] = np.random.uniform(lower, upper)
        print(f"Generated random starting parameters:")
        for key, val in current_params.items():
            print(f"  {key:10s}: {val:8.4f}")

 

    
    current_energy = energy_function(current_params)
    
    # Track best solution
    best_params = current_params.copy()
    best_energy = current_energy
    
    # Storage for diagnostics
    energy_trace = []
    temperature_trace = []
    acceptance_count = 0
    
    temperature = initial_temp
    param_names = list(current_params.keys())
    
    print(f"\nStarting simulated annealing optimization...")
    print(f"Initial temperature: {initial_temp}, Cooling rate: {cooling_rate}")
    print(f"Initial energy: {current_energy:.3f}")
    print(f"SA bounds: ")
    for key, (lower, upper) in bounds.items():
        print(f"  {key:10s}: [{lower:8.2f}, {upper:8.2f}]")
    
    for i in range(SA_iterations):
        # Propose new parameters by perturbing one parameter
        proposed_params = current_params.copy()
        param_to_change = np.random.choice(param_names)
        
        # Generate proposal with temperature-dependent step size
        step_scale = np.sqrt(temperature / initial_temp)  # Scale step size with temperature
        proposed_params[param_to_change] += np.random.normal(0, step_sizes[param_to_change] * step_scale)
        
        # Calculate energy of proposed state
        proposed_energy = energy_function(proposed_params)
        
        # Accept or reject based on Metropolis criterion
        energy_diff = proposed_energy - current_energy
        
        if energy_diff < 0 or np.random.rand() < np.exp(-energy_diff / temperature):
            # Accept proposal
            current_params = proposed_params
            current_energy = proposed_energy
            acceptance_count += 1
            
            # Update best solution if improved
            if current_energy < best_energy:
                best_params = current_params.copy()
                best_energy = current_energy
        
        # Store diagnostics
        energy_trace.append(current_energy)
        temperature_trace.append(temperature)
        
        # Cool down temperature
        temperature *= cooling_rate
        
        # Progress updates
        if (i + 1) % (SA_iterations // 10) == 0:
            acceptance_rate = acceptance_count / (i + 1)
            print(f"SA iteration {i+1:5d}/{SA_iterations}: "
                  f"T={temperature:.4f}, E={current_energy:.3f}, "
                  f"Best E={best_energy:.3f}, Accept rate={acceptance_rate:.3f}")
        
        # Stop if temperature is too low
        if temperature < min_temp:
            print(f"Stopping SA: temperature {temperature:.6f} below minimum {min_temp}")
            break
    
    final_acceptance_rate = acceptance_count / min(i + 1, SA_iterations)
    print(f"SA completed. Final acceptance rate: {final_acceptance_rate:.3f}")
    print(f"Best energy found: {best_energy:.6f}")
    print(f"Best parameters found:")
    for key, val in best_params.items():
        print(f"  {key:10s}: {val:8.4f}")
    
    return best_params, best_energy, energy_trace, temperature_trace

