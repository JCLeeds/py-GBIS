import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.interpolate import griddata
import pCDM_model as pCDM_fast
import okada_model as okada_fast
import UNE_three_component as UNE_three

import matplotlib.pyplot as plt
import os 
import pandas as pd
import llh2local as llh
import local2llh as l2llh

def simulated_annealing_optimization(u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                                   C_inv, C_logdet, los_e, los_n, los_u, starting_params=None,
                                   SA_iterations=10000, initial_temp=10.0,
                                   cooling_rate=0.95, min_temp=0.01,
                                   step_sizes=None, bounds=None, model_type='pCDM'):
    """
    Simulated annealing optimizer supporting single-model or joint (multi-model) parameter spaces.

    - If `model_type` is a string, `step_sizes`, `bounds`, and `starting_params` may be dicts as before.
    - If `model_type` is a list (joint inversion), supply `step_sizes`, `bounds`, and `starting_params`
      as lists of dicts (same order as `model_type`). The optimizer flattens the joint parameter
      vector internally and returns `best_params` as a list of per-model dicts for joint runs.

    Returns the same diagnostics as the single-model version but `best_params` will be a list
    when a joint run was performed.

    Notes:
    - Multi‑IFG inputs supported: `u_los_obs`, `X_obs`, `Y_obs` may be lists of per‑IFG arrays and
      `incidence_angle` / `heading` may be scalars or per‑IFG lists. The function will concatenate
      observations internally and compute LOS vectors if needed.
    - If `los_e`/`los_n`/`los_u` are provided and their length matches the concatenated observations,
      they are used; otherwise the LOS vectors are computed from `incidence_angle`/`heading`.
    """
    # --- Accept multiple IFGs as input (lists of per-IFG arrays) ---
    # Support u_los_obs, X_obs, Y_obs being either single arrays or lists of arrays.
    multi_ifg_input = (
        isinstance(u_los_obs, (list, tuple)) or
        isinstance(X_obs, (list, tuple)) or
        isinstance(Y_obs, (list, tuple))
    )
    if multi_ifg_input:
        # Ensure we have lists for all three observation inputs
        u_list = [np.asarray(u).flatten() for u in (u_los_obs if isinstance(u_los_obs, (list, tuple)) else [u_los_obs])]
        X_list = [np.asarray(x).flatten() for x in (X_obs if isinstance(X_obs, (list, tuple)) else [X_obs])]
        Y_list = [np.asarray(y).flatten() for y in (Y_obs if isinstance(Y_obs, (list, tuple)) else [Y_obs])]

        if not (len(u_list) == len(X_list) == len(Y_list)):
            raise ValueError("When passing multiple IFGs, u_los_obs/X_obs/Y_obs must be lists of the same length")

        n_ifgs = len(u_list)

        # Normalize incidence_angle/heading to per-IFG lists (allow scalar to broadcast)
        inc_list = (incidence_angle if isinstance(incidence_angle, (list, tuple, np.ndarray)) else [incidence_angle] * n_ifgs)
        head_list = (heading if isinstance(heading, (list, tuple, np.ndarray)) else [heading] * n_ifgs)
        if not (len(inc_list) == len(head_list) == n_ifgs):
            raise ValueError("incidence_angle and heading must be scalar or lists with one entry per IFG")

        # Concatenate observations into single flattened arrays (matching how the MCMC code works)
        u_los_obs = np.concatenate(u_list)
        X_obs = np.concatenate(X_list)
        Y_obs = np.concatenate(Y_list)

        # Build per-point LOS direction arrays and concatenate
        los_e_arr = np.empty_like(u_los_obs, dtype=float)
        los_n_arr = np.empty_like(u_los_obs, dtype=float)
        los_u_arr = np.empty_like(u_los_obs, dtype=float)
        idx0 = 0
        for cnt, inc_a, head_a in zip([len(u) for u in u_list], inc_list, head_list):
            inc_rad_i = np.radians(inc_a)
            head_rad_i = np.radians(head_a)
            los_e_i = np.sin(inc_rad_i) * np.cos(head_rad_i)
            los_n_i = -np.sin(inc_rad_i) * np.sin(head_rad_i)
            los_u_i = -np.cos(inc_rad_i)
            idx1 = idx0 + cnt
            los_e_arr[idx0:idx1] = los_e_i
            los_n_arr[idx0:idx1] = los_n_i
            los_u_arr[idx0:idx1] = los_u_i
            idx0 = idx1

        # Prefer provided los_e/los_n/los_u only if they already match the concatenated length
        if hasattr(los_e, '__len__') and len(los_e) == len(u_los_obs):
            # caller provided LOS arrays that already match concatenated observations - keep them
            pass
        else:
            los_e = los_e_arr
            los_n = los_n_arr
            los_u = los_u_arr

        # Ensure covariance matrix matches concatenated observation length
        if hasattr(C_inv, 'shape') and C_inv.shape[0] != len(u_los_obs):
            raise ValueError(f"C_inv dimension {C_inv.shape[0]} does not match concatenated observations {len(u_los_obs)}")

    # Normalize inputs for single vs multi-model
    if isinstance(model_type, list):
        model_list = [m.lower() for m in model_type]
    else:
        model_list = [str(model_type).lower()]

    num_models = len(model_list)
    single_model_mode = (num_models == 1)

    # Normalize step_sizes / bounds / starting_params into lists for joint mode
    step_sizes_list = step_sizes if isinstance(step_sizes, list) or step_sizes is None else [step_sizes]
    bounds_list = bounds if isinstance(bounds, list) or bounds is None else [bounds]
    starting_params_list = starting_params if isinstance(starting_params, list) or starting_params is None else [starting_params]

    if not single_model_mode:
        # Validate list lengths
        if step_sizes_list is None or bounds_list is None:
            raise ValueError("For joint-SA please provide 'step_sizes' and 'bounds' as lists (one dict per model)")
        if not (len(step_sizes_list) == len(bounds_list) == num_models):
            raise ValueError("step_sizes and bounds must each be lists with one dict per model")

    # Build flattened parameter space for joint runs (or keep single dict for single-model)
    def _build_flat_space():
        flat_bounds = {}
        flat_step_sizes = {}
        flat_start = {}
        labels = []
        if single_model_mode:
            bdict = bounds_list[0] if bounds_list is not None and len(bounds_list) > 0 else bounds
            sdict = step_sizes_list[0] if step_sizes_list is not None and len(step_sizes_list) > 0 else step_sizes
            if bdict is None or sdict is None:
                raise ValueError("bounds and step_sizes must be provided for SA")
            for k, (lo, hi) in bdict.items():
                flat_bounds[k] = (lo, hi)
                flat_step_sizes[k] = sdict.get(k, 1.0)
            if starting_params_list and starting_params_list[0] is not None:
                for k, v in starting_params_list[0].items():
                    flat_start[k] = v
            return flat_bounds, flat_step_sizes, flat_start, labels

        # Joint mode
        for idx, m in enumerate(model_list):
            label = f"{m}_{idx+1}"
            labels.append(label)
            bdict = bounds_list[idx]
            sdict = step_sizes_list[idx]
            for k in bdict.keys():
                flat_key = f"{label}__{k}"
                flat_bounds[flat_key] = bdict[k]
                flat_step_sizes[flat_key] = sdict.get(k, 1.0)
            # collect starting params if provided
            if starting_params_list and starting_params_list[idx] is not None:
                for k, v in starting_params_list[idx].items():
                    flat_start[f"{label}__{k}"] = v
        return flat_bounds, flat_step_sizes, flat_start, labels

    flat_bounds, flat_step_sizes, flat_start, labels = _build_flat_space()

    # Energy function works on flattened parameter dict
    def energy_function_flat(flat_params):
        try:
            # Check bounds
            for key, (lower, upper) in flat_bounds.items():
                if key not in flat_params:
                    return np.inf
                if not (lower <= flat_params[key] <= upper):
                    return np.inf

            # Reconstruct per-model dicts if joint
            if single_model_mode:
                pdict = flat_params
                # run single forward model
                if model_list[0] == 'pcdm':
                    ue, un, uv = pCDM_fast.pCDM(
                        np.asarray(X_obs, dtype=float), np.asarray(Y_obs, dtype=float),
                        float(pdict['X0']), float(pdict['Y0']), float(pdict['depth']),
                        float(pdict.get('omegaX', 0)), float(pdict.get('omegaY', 0)), float(pdict.get('omegaZ', 0)),
                        float(pdict.get('DVx', 0)), float(pdict.get('DVy', 0)), float(pdict.get('DVz', 0)), 0.25
                    )
                elif model_list[0] == 'okada':
                    ue, un, uv = okada_fast.disloc3d3(
                        np.asarray(X_obs, dtype=float), np.asarray(Y_obs, dtype=float),
                        xoff=float(pdict['X0']), yoff=float(pdict['Y0']), depth=float(pdict['depth']),
                        length=float(pdict.get('length', 0)), width=float(pdict.get('width', 0)),
                        slip=float(pdict.get('slip', 0)), opening=float(pdict.get('opening', 0)),
                        strike=float(pdict.get('strike', 0)), dip=float(pdict.get('dip', 0)), rake=float(pdict.get('rake', 0)), nu=0.25
                    )
                elif model_list[0] == 'une':
                    uv, ue, un = UNE_three.model(
                        np.asarray(X_obs, dtype=float), np.asarray(Y_obs, dtype=float),
                        depth=float(pdict['depth']), yield_kt=float(pdict['yield_kt']),
                        dv_factor=float(pdict.get('dv_factor', 0.1)),
                        chimney_amp=float(pdict.get('chimney_amp', 0.15)),
                        chimney_height_fac=10, chimney_peck_k=0.35,
                        compact_amp=float(pdict.get('compact_amp', 0.05)),
                        anelastic_fac=5,
                        x0=float(pdict.get('X0', 0)), y0=float(pdict.get('Y0', 0)),
                        nu=0.25, mu=30e9
                    )
                else:
                    # fallback: try pcdm
                    ue, un, uv = pCDM_fast.pCDM(np.asarray(X_obs, dtype=float), np.asarray(Y_obs, dtype=float),
                                                float(pdict.get('X0', 0)), float(pdict.get('Y0', 0)), float(pdict.get('depth', 0)),
                                                float(pdict.get('omegaX', 0)), float(pdict.get('omegaY', 0)), float(pdict.get('omegaZ', 0)),
                                                float(pdict.get('DVx', 0)), float(pdict.get('DVy', 0)), float(pdict.get('DVz', 0)), 0.25)
                ue = np.asarray(ue, dtype=float)
                un = np.asarray(un, dtype=float)
                uv = np.asarray(uv, dtype=float)

            else:
                # Joint: sum contributions from each model
                ue = np.zeros(len(X_obs), dtype=float)
                un = np.zeros(len(X_obs), dtype=float)
                uv = np.zeros(len(X_obs), dtype=float)
                for idx, m in enumerate(model_list):
                    label = labels[idx]
                    # build parameter dict for this model
                    pdict = {}
                    # find keys starting with label__
                    for flat_k, val in flat_params.items():
                        if flat_k.startswith(label + '__'):
                            orig_k = flat_k.split('__', 1)[1]
                            pdict[orig_k] = val
                    mm = m
                    if 'pcdm' in mm:
                        ue_i, un_i, uv_i = pCDM_fast.pCDM(np.asarray(X_obs, dtype=float), np.asarray(Y_obs, dtype=float),
                                                         pdict.get('X0', 0), pdict.get('Y0', 0), pdict.get('depth', 0),
                                                         pdict.get('omegaX', 0), pdict.get('omegaY', 0), pdict.get('omegaZ', 0),
                                                         pdict.get('DVx', 0), pdict.get('DVy', 0), pdict.get('DVz', 0), 0.25)
                    elif 'okada' in mm:
                        ue_i, un_i, uv_i = okada_fast.disloc3d3(np.asarray(X_obs, dtype=float), np.asarray(Y_obs, dtype=float),
                                                               xoff=pdict.get('X0'), yoff=pdict.get('Y0'), depth=pdict.get('depth'),
                                                               length=pdict.get('length'), width=pdict.get('width'),
                                                               slip=pdict.get('slip', 0), opening=pdict.get('opening', 0),
                                                               strike=pdict.get('strike', 0), dip=pdict.get('dip', 0), rake=pdict.get('rake', 0), nu=0.25)
                    elif 'une' in mm:
                        uv_i, ue_i, un_i = UNE_three.model(
                            np.asarray(X_obs, dtype=float), np.asarray(Y_obs, dtype=float),
                            depth=pdict.get('depth', 1000), yield_kt=pdict.get('yield_kt', 1.0),
                            dv_factor=pdict.get('dv_factor', 0.1),
                            chimney_amp=pdict.get('chimney_amp', 0.15),
                            chimney_height_fac=10, chimney_peck_k=0.35,
                            compact_amp=pdict.get('compact_amp', 0.05),
                            anelastic_fac=5,
                            x0=pdict.get('X0', 0), y0=pdict.get('Y0', 0),
                            nu=0.25, mu=30e9)
                    else:
                        # unsupported -> zero contribution
                        ue_i = np.zeros(len(X_obs))
                        un_i = np.zeros(len(X_obs))
                        uv_i = np.zeros(len(X_obs))
                    ue += np.asarray(ue_i, dtype=float)
                    un += np.asarray(un_i, dtype=float)
                    uv += np.asarray(uv_i, dtype=float)

            # Convert to LOS
            u_los_pred = -(ue * los_e + un * los_n + uv * los_u)
            u_los_pred = np.asarray(u_los_pred, dtype=float).flatten()

            # residuals
            residuals = np.asarray(u_los_obs.flatten(), dtype=float) - u_los_pred
            if residuals.shape != u_los_obs.shape:
                return np.inf
            if np.any(~np.isfinite(residuals)):
                return np.inf

            quad_form = residuals.T @ C_inv @ residuals
            if not np.isfinite(quad_form):
                return np.inf
            neg_log_lik = 0.5 * (quad_form + C_logdet + len(residuals) * np.log(2 * np.pi))
            if not np.isfinite(neg_log_lik):
                return np.inf
            return neg_log_lik

        except Exception as e:
            if np.random.rand() < 0.001:
                print(f"Error in energy_function_flat: {type(e).__name__}: {e}")
            return np.inf


    
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
            elif model_type.lower() == 'une':
                uv, ue, un = UNE_three.model(
                    X_obs_arr, Y_obs_arr,
                    depth=float(params['depth']), yield_kt=float(params['yield_kt']),
                    dv_factor=float(params['dv_factor']),
                    chimney_amp=float(params['chimney_amp']),
                    chimney_height_fac=10, chimney_peck_k=0.35,
                    compact_amp=float(params['compact_amp']),
                    anelastic_fac=5,
                    x0=float(params['X0']), y0=float(params['Y0']),
                    nu=0.25, mu=30e9
                )
            
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
    
    # Initialize current_params in the flattened space
    if flat_start:
        # start from provided starting parameters (flattened)
        current_params = flat_start.copy()
        print("Using provided starting parameters (flattened/joint where applicable):")
        for key, val in current_params.items():
            print(f"  {key:10s}: {val:8.4f}")
    else:
        current_params = {}
        for key, (lower, upper) in flat_bounds.items():
            current_params[key] = np.random.uniform(lower, upper)
        print("Generated random starting parameters (flattened/joint where applicable):")
        for key, val in current_params.items():
            print(f"  {key:10s}: {val:8.4f}")

    # Prepare names and step sizes for proposals
    param_names = list(current_params.keys())
    step_sizes_flat = {}
    for k in param_names:
        step_sizes_flat[k] = flat_step_sizes.get(k, 1.0)


 

    
    current_energy = energy_function(current_params)
    
    # Track best solution
    best_params = current_params.copy()
    best_energy = current_energy
    
    # Storage for diagnostics
    energy_trace = []
    temperature_trace = []
    acceptance_count = 0
    
    temperature = initial_temp
    
    print(f"\nStarting simulated annealing optimization...")
    print(f"Initial temperature: {initial_temp}, Cooling rate: {cooling_rate}")
    print(f"Initial energy: {current_energy:.3f}")
    print(f"SA bounds (flattened keys): ")
    for key, (lower, upper) in flat_bounds.items():
        print(f"  {key:10s}: [{lower:8.2f}, {upper:8.2f}]")

    for i in range(SA_iterations):
        # Propose new parameters by perturbing one flattened parameter
        proposed_params = current_params.copy()
        param_to_change = np.random.choice(param_names)

        # Generate proposal with temperature-dependent step size
        step_scale = np.sqrt(temperature / initial_temp)  # Scale step size with temperature
        proposed_params[param_to_change] += np.random.normal(0, step_sizes_flat[param_to_change] * step_scale)

        # Calculate energy of proposed state
        proposed_energy = energy_function_flat(proposed_params)

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
        if (i + 1) % max(1, (SA_iterations // 10)) == 0:
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

    # Convert best_params back to per-model dicts for joint mode, keep dict for single-model
    if single_model_mode:
        best_return = best_params.copy()
    else:
        best_return = []
        for idx, label in enumerate(labels):
            pdict = {}
            for flat_k, val in best_params.items():
                if flat_k.startswith(label + '__'):
                    orig_k = flat_k.split('__', 1)[1]
                    pdict[orig_k] = val
            best_return.append(pdict)

    print(f"Best parameters found:")
    if single_model_mode:
        for key, val in best_return.items():
            print(f"  {key:10s}: {val:8.4f}")
    else:
        for idx, pd in enumerate(best_return):
            print(f" Model {idx+1} ({model_list[idx]}):")
            for key, val in pd.items():
                print(f"  {key:10s}: {val:8.4f}")

    return best_return, best_energy, energy_trace, temperature_trace

