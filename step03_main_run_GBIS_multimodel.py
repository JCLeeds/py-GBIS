import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.linalg import block_diag
import pCDM_model as pCDM_fast

import matplotlib.pyplot as plt
import os 
import pandas as pd
import llh2local as llh
import local2llh as l2llh
import misc_plotting_functions_multimodel as pCDM_BI_plotting_funcs
import misc_simulated_annealing_multimodel as pCDM_BI_simulated_annealing
import okada_model as okada
import pickle
from datetime import datetime
import gc
import UNE_three_component_fast as UNE_three

#### In this current version posative is towards the satellite ####
#### This follows GBIS conventions #####

"""
Bayesian inference for pCDM source parameters using MCMC with spatially correlated noise.

Author: John Condon
Date of edit: December 2024
"""

def convert_lat_long_2_xy(lat, lon, lat0, lon0):
    ll = [lon.flatten(), lat.flatten()]
    ll = np.array(ll, dtype=float)
    xy = llh.llh2local(ll, np.array([lon0, lat0], dtype=float))
    x = xy[0,:].reshape(lat.shape)
    y = xy[1,:].reshape(lat.shape)
    return xy

def estimate_noise_covariance(X_obs, Y_obs, u_los_obs, sill=None, nugget=None, range_param=None):
    """
    Estimate noise covariance matrix using variogram parameters.
    
    Parameters:
    -----------
    X_obs, Y_obs : array_like
        Observation coordinates
    u_los_obs : array_like
        Observed line-of-sight displacements
    sill : float, optional
        Variogram sill (total variance). If None, estimated from data.
    nugget : float, optional
        Variogram nugget (measurement error variance). If None, estimated.
    range_param : float, optional
        Variogram range (correlation length). If None, estimated.
        
    Returns:
    --------
    C : ndarray
        Covariance matrix
    """
    n = len(u_los_obs)
    
    # Calculate distances between all observation points
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt((X_obs[i] - X_obs[j])**2 + (Y_obs[i] - Y_obs[j])**2)
    
    # Estimate variogram parameters if not provided
    if sill is None:
        sill = np.var(u_los_obs)
    
    if nugget is None:
        nugget = sill * 0.01  # Assume 1% nugget effect
    
    if range_param is None:
        # Estimate range as a fraction of the maximum distance
        max_dist = np.max(distances)
        range_param = max_dist / 3.0
    
    # Construct covariance matrix using exponential model
    # C(h) = nugget + (sill - nugget) * exp(-h/range)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i, j] = sill
            else:
                h = distances[i, j]
                C[i, j] = (sill - nugget) * np.exp(-h / range_param)
    
    return C, sill, nugget, range_param

def bayesian_inference_pCDM_with_noise(u_los_obs, X_obs, Y_obs, incidence_angle, heading, 
                                            n_iterations=10000, proposal_std=None,
                                            sill=None, nugget=None, range_param=None,
                                            adaptive_interval=1000, target_acceptance=0.23,
                                            adapt_method='conservative',
                                            initial_params=None, priors=None, max_step_sizes=None,
                                            use_sa_init=True, figure_folder=None, model_type='pCDM',
                                            burn_in=None):
        """
        Bayesian inference for various source models using Metropolis-Hastings MCMC
        with spatially correlated noise model and adaptive proposal scaling.
        
        Parameters:
        -----------
        u_los_obs : array_like
            Observed line-of-sight displacements
        X_obs, Y_obs : array_like
            Observation coordinates
        incidence_angle : float
            Satellite incidence angle in degrees
        heading : float
            Satellite heading in degrees
        n_iterations : int
            Number of MCMC iterations
        proposal_std : dict
            Standard deviations for proposal distribution
        sill : float, optional
            Variogram sill parameter
        nugget : float, optional
            Variogram nugget parameter
        range_param : float, optional
            Variogram range parameter
        adaptive_interval : int
            Number of iterations between adaptation steps
        target_acceptance : float
            Target acceptance rate (0.23 is optimal for multivariate problems)
        adapt_method : str, optional
            Which adaptation routine to use during burn-in:
            - 'conservative' (default): the existing gentle per-parameter scheme.
            - 'standard'     : MCMC‑standard Robbins–Monro multiplicative updates that
                               target the acceptance probability (suitable for Bayesian inversion).
        initial_params : dict, optional
            Initial parameter values. If None, uses default values.
        priors : dict, optional
            Prior bounds for each parameter. If None, uses default bounds.
        max_step_sizes : dict, optional
            Maximum allowed step sizes for adaptive scaling. If None, uses defaults.
        model_type : str or list
            Type of forward model to use ('pCDM', 'Mogi', 'Yang', etc.) or a list of models for joint inversion. Simulated-annealing init is supported for both single- and joint-model runs.
            
        Returns:
        --------
        samples : dict
            MCMC samples for each parameter
        log_likelihood_trace : array
            Log-likelihood values at each iteration
        """
        
        # Support single-IFG (existing) OR multiple IFGs (Option A: concatenate + block-diagonal covariance).
        multi_ifg = isinstance(u_los_obs, (list, tuple, np.ndarray)) and (
            isinstance(u_los_obs, (list, tuple)) or (hasattr(u_los_obs, '__iter__') and hasattr(u_los_obs[0], '__iter__'))
        ) and not isinstance(u_los_obs, np.ndarray)

        if multi_ifg:
            # Expect lists (or tuples) of per-IFG arrays: u_los_obs[i], X_obs[i], Y_obs[i]
            u_list = [np.asarray(u).flatten() for u in u_los_obs]
            X_list = [np.asarray(x).flatten() for x in X_obs]
            Y_list = [np.asarray(y).flatten() for y in Y_obs]

            if not (len(u_list) == len(X_list) == len(Y_list)):
                raise ValueError("When passing multiple IFGs, u_los_obs/X_obs/Y_obs must be lists of the same length")

            n_ifgs = len(u_list)

            # Normalize variogram params to per-IFG lists if scalars were provided
            def _ensure_list_param(p):
                if p is None:
                    return [None] * n_ifgs
                if isinstance(p, (list, tuple, np.ndarray)):
                    if len(p) != n_ifgs:
                        raise ValueError('Sill/nugget/range_param list length must match number of IFGs')
                    return list(p)
                else:
                    return [p] * n_ifgs

            sill_list = _ensure_list_param(sill)
            nugget_list = _ensure_list_param(nugget)
            range_list = _ensure_list_param(range_param)

            # Estimate per-IFG covariance matrices and build block-diagonal matrix
            C_blocks = []
            for ui, Xi, Yi, s_i, n_i, r_i in zip(u_list, X_list, Y_list, sill_list, nugget_list, range_list):
                Ci, s_est, n_est, r_est = estimate_noise_covariance(Xi, Yi, ui, s_i, n_i, r_i)
                C_blocks.append(Ci)

            # Build full block-diagonal covariance and compute inverse/logdet
            C = block_diag(*C_blocks)
            try:
                C_inv = np.linalg.inv(C)
                C_logdet = np.linalg.slogdet(C)[1]
            except np.linalg.LinAlgError:
                C += np.eye(len(C)) * 1e-8 * np.trace(C) / len(C)
                C_inv = np.linalg.inv(C)
                C_logdet = np.linalg.slogdet(C)[1]

            # Concatenate observation coordinates & values for forward model and likelihood
            u_los_obs = np.concatenate(u_list)
            X_obs = np.concatenate(X_list)
            Y_obs = np.concatenate(Y_list)

            # Build per-point LOS direction arrays (handle per-IFG incidence/heading)
            inc_list = incidence_angle if isinstance(incidence_angle, (list, tuple, np.ndarray)) else [incidence_angle] * n_ifgs
            head_list = heading if isinstance(heading, (list, tuple, np.ndarray)) else [heading] * n_ifgs

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

            # Use array-valued LOS components in subsequent calculations
            los_e = los_e_arr
            los_n = los_n_arr
            los_u = los_u_arr

        else:
            # Single IFG (existing behaviour)
            u_los_obs = np.array(u_los_obs).flatten()
            X_obs = np.array(X_obs).flatten()
            Y_obs = np.array(Y_obs).flatten()

            # Estimate noise covariance matrix
            C, sill, nugget, range_param = estimate_noise_covariance(X_obs, Y_obs, u_los_obs, sill, nugget, range_param)

            # Compute inverse and determinant for likelihood calculation
            try:
                C_inv = np.linalg.inv(C)
                C_logdet = np.linalg.slogdet(C)[1]
            except np.linalg.LinAlgError:
                C += np.eye(len(C)) * 1e-8 * np.trace(C) / len(C)
                C_inv = np.linalg.inv(C)
                C_logdet = np.linalg.slogdet(C)[1]

            # Convert angles to radians for LOS calculation
            inc_rad = np.radians(incidence_angle)
            head_rad = np.radians(heading)

            # Line-of-sight unit vector components (scalars)
            los_e = np.sin(inc_rad) * np.cos(head_rad)
            los_n = -np.sin(inc_rad) * np.sin(head_rad)
            los_u = -np.cos(inc_rad)

        # Define forward models
        def forward_model(params, model_type):
            """
            Generic forward model dispatcher
            """
            if isinstance(model_type,list):
                ue, un, uv = np.zeros_like(u_los_obs), np.zeros_like(u_los_obs), np.zeros_like(u_los_obs)
                for i, model in enumerate(model_type):
                    if model.lower() == 'pcdm':
                        ue_i, un_i, uv_i = pCDM_fast.pCDM(X_obs, Y_obs, params[i]['X0'], params[i]['Y0'], params[i]['depth'],
                                            params[i]['omegaX'], params[i]['omegaY'], params[i]['omegaZ'],
                                            params[i]['DVx'], params[i]['DVy'], params[i]['DVz'], 0.25)
                    elif model.lower() == 'mogi':
                        # Mogi point source model
                        # Expected parameters: X0, Y0, depth, DV
                        # ue, un, uv = mogi_model(X_obs, Y_obs, params[i]['X0'], params[i]['Y0'], 
                        #                        params[i]['depth'], params[i]['DV'])
                        pass # Placeholder to avoid error
                    elif model.lower() == 'yang':
                        # Yang finite spherical source model
                        # Expected parameters: X0, Y0, depth, DV, radius
                        # ue, un, uv = yang_model(X_obs, Y_obs, params[i]['X0'], params[i]['Y0'], 
                        #                        params[i]['depth'], params[i]['DV'], params[i]['radius'])
                        pass  # Placeholder to avoid error
                    elif model.lower() == 'mctigue':
                        # McTigue prolate spheroid model
                        # Expected parameters: X0, Y0, depth, DV, a, c, strike, dip, plunge
                        # ue, un, uv = mctigue_model(X_obs, Y_obs, params[i]['X0'], params[i]['Y0'], 
                        #                           params[i]['depth'], params[i]['DV'], params[i]['a'], 
                        #                           params[i]['c'], params[i]['strike'], params[i]['dip'], 
                        #                           params[i]['plunge'])
                        pass  # Placeholder to avoid error
                    elif model.lower() == 'okada':
                        # okada.clear_numba_cache()
                        # Okada rectangular dislocation model
                        ue_i, un_i, uv_i = okada.disloc3d3(X_obs, Y_obs, params[i]['X0'], params[i]['Y0'],
                                                        params[i]['depth'], params[i]['length'], params[i]['width'], params[i]['slip'], params[i]['opening'],
                                                        params[i]['strike'], params[i]['dip'], params[i]['rake'], 0.25
                                                        )
                    elif model.lower() == 'une':
                        uv_i, ue_i, un_i = UNE_three.model(X_obs, Y_obs, depth=params[i]['depth'], yield_kt=params[i]['yield_kt'],
                              dv_factor=params[i]['dv_factor'],
                              chimney_amp=params[i]['chimney_amp'],
                              chimney_height_fac=10,
                              chimney_peck_k=0.35,
                              compact_amp=params[i]['compact_amp'],
                              anelastic_fac=5,
                              x0=params[i]['X0'], y0=params[i]['Y0'],
                              nu=0.25, mu=30e9 )
                        
                    ue, un, uv = ue + ue_i, un + un_i, uv + uv_i   
                    
                    

            else:
                if model_type.lower() == 'pcdm':
                    ue, un, uv = pCDM_fast.pCDM(X_obs, Y_obs, params['X0'], params['Y0'], params['depth'],
                                        params['omegaX'], params['omegaY'], params['omegaZ'],
                                        params['DVx'], params['DVy'], params['DVz'], 0.25)

                elif model_type.lower() == 'mogi':
                    # Mogi point source model
                    # Expected parameters: X0, Y0, depth, DV
                    # ue, un, uv = mogi_model(X_obs, Y_obs, params['X0'], params['Y0'], 
                    #                        params['depth'], params['DV'])
                    pass # Placeholder to avoid error

                elif model_type.lower() == 'yang':
                    # Yang finite spherical source model
                    # Expected parameters: X0, Y0, depth, DV, radius
                    # ue, un, uv = yang_model(X_obs, Y_obs, params['X0'], params['Y0'], 
                    #                        params['depth'], params['DV'], params['radius'])
                    pass  # Placeholder to avoid error

                elif model_type.lower() == 'mctigue':
                    # McTigue prolate spheroid model
                    # Expected parameters: X0, Y0, depth, DV, a, c, strike, dip, plunge
                    # ue, un, uv = mctigue_model(X_obs, Y_obs, params['X0'], params['Y0'], 
                    #                           params['depth'], params['DV'], params['a'], 
                    #                           params['c'], params['strike'], params['dip'], 
                    #                           params['plunge'])
                    pass  # Placeholder to avoid error

                elif model_type.lower() == 'okada':
                    # okada.clear_numba_cache()
                    # Okada rectangular dislocation model
                    # Expected parameters: X0, Y0, depth, length, width, strike, dip, rake, slip, opening
                    ue, un, uv = okada.disloc3d3(X_obs, Y_obs, params['X0'], params['Y0'],
                                                        params['depth'], params['length'], params['width'], params['slip'], params['opening'],
                                                        params['strike'], params['dip'], params['rake'], 0.25
                                                        )
                elif model_type.lower() == 'une':
                    uv, ue, un = UNE_three.model(X_obs, Y_obs, depth=params['depth'], yield_kt=params['yield_kt'],
                              dv_factor=params['dv_factor'],
                              chimney_amp=params['chimney_amp'],
                              chimney_height_fac=10,
                              chimney_peck_k=0.35,
                              compact_amp=params['compact_amp'],
                              anelastic_fac=5,
                              x0=params['X0'], y0=params['Y0'],
                              nu=0.25, mu=30e9 )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            
            return ue, un, uv

     
        
        # Initialize adaptive proposal tracking (supports single- or multi-model inputs)
        # Normalize model inputs so MCMC operates on a flat parameter dict while
        # forward_model is given structured per-model dicts when required.
        if isinstance(model_type, list):
            model_list = [m.lower() for m in model_type]
        else:
            model_list = [model_type.lower()]

        num_models = len(model_list)
        single_model_mode = (num_models == 1)

        # Ensure initial/prior/proposal/max_step inputs are lists when doing joint inversion
        if initial_params is None:
            raise ValueError("initial_params must be provided (dict for single model or list of dicts for multiple models)")
        initial_params_list = initial_params if isinstance(initial_params, list) else [initial_params]
        priors_list = priors if isinstance(priors, list) else [priors]
        proposal_std_list = proposal_std if isinstance(proposal_std, list) else [proposal_std]
        max_step_sizes_list = max_step_sizes if isinstance(max_step_sizes, list) else [max_step_sizes]

        if not (len(initial_params_list) == len(priors_list) == len(proposal_std_list) == len(max_step_sizes_list) == num_models):
            if num_models == 1:
                pass
            else:
                raise ValueError("When providing multiple models, initial_params/priors/proposal_std/max_step_sizes must each be lists with one entry per model")

        # Build flattened parameter dictionaries. For backward compatibility we keep
        # unprefixed parameter names when there is only a single model.
        flat_initial = {}
        flat_priors = {}
        flat_proposal_std = {}
        flat_max_step_sizes = {}
        param_structure = {}   # label -> list of param names (for unflattening)
        labels = []

        for idx, m in enumerate(model_list):
            params_dict = initial_params_list[idx]
            priors_dict = priors_list[idx]
            prop_std_dict = proposal_std_list[idx]
            max_step_dict = max_step_sizes_list[idx]

            if single_model_mode:
                label = m
            else:
                label = f"{m}_{idx+1}"
            labels.append(label)
            param_structure[label] = list(params_dict.keys())

            for key in params_dict.keys():
                flat_key = key if single_model_mode else f"{label}__{key}"
                flat_initial[flat_key] = params_dict[key]
                flat_priors[flat_key] = priors_dict[key]
                flat_proposal_std[flat_key] = prop_std_dict[key]
                flat_max_step_sizes[flat_key] = max_step_dict[key]

        # Use flattened dicts for the adaptive MCMC bookkeeping
        param_names = list(flat_proposal_std.keys())
        d = len(param_names)   # number of parameters

        # Auto-select adaptive_interval if the caller left it at the default.
        # Rule: each window should contain ~100 proposals per parameter
        # (round-robin means each param gets adaptive_interval / d proposals).
        # We also want ~20 adaptation events inside burn-in (= 20% of n_iterations),
        # so adaptive_interval = burn_in / 20.  Take the larger of the two floors.
        _burn_in_est  = int(n_iterations * 0.2)
        interval_from_params  = max(50 * d, 100)   # ≥100 proposals/param
        interval_from_burnin  = max(_burn_in_est // 20, 1)  # ~20 events in burn-in
        _auto_interval = max(interval_from_params, interval_from_burnin)
        if adaptive_interval == 1000 and _auto_interval != 1000:
            print(f"  [auto] adaptive_interval set to {_auto_interval} "
                  f"(d={d} params, ~100 proposals/param/window, ~20 events in burn-in)")
            adaptive_interval = _auto_interval

        proposal_std_adaptive = flat_proposal_std.copy()
        acceptance_counts = {param: 0 for param in param_names}
        proposal_counts = {param: 0 for param in param_names}
        # Track rejection reasons per parameter (diagnostics + adaptation guard)
        proposal_prior_violations = {param: 0 for param in param_names}
        proposal_forward_errors = {param: 0 for param in param_names}
        # Choose adaptation method (normalize)
        adapt_method_mode = (adapt_method.lower() if isinstance(adapt_method, str) else 'conservative')
        # Round-robin option for proposing parameters (helps ensure every param is seen)
        use_round_robin_updates = True
        rr_index = 0
        rr_cycle = param_names.copy()

        # Helper: convert flat param dict -> structured per-model dict/list for forward_model
        def _unflatten_params(flat_params):
            if single_model_mode:
                return flat_params.copy()
            structured = []
            for label in labels:
                pdict = {}
                for p in param_structure[label]:
                    pdict[p] = flat_params[f"{label}__{p}"]
                structured.append(pdict)
            return structured

        # Prepare display / filename identifiers for single vs joint models
        model_display_name = "+".join(model_list)
        model_id_for_files = "_".join(model_list)

        # Simulated annealing: single-model and joint (multi-model) SA are supported.
        # If use_sa_init=True the code will run SA for single or joint models depending on model_type.

        # Initialize current_params from flattened initial values
        current_params = flat_initial.copy()

        # If using simulated annealing for initialization (single-model only)
        if use_sa_init:
            if not single_model_mode:
                # Joint simulated annealing for all models
                print(f"\nUsing JOINT simulated annealing for models: {model_display_name} ...")
                # Prepare SA step sizes and bounds as lists (double step sizes for SA)
                sa_step_sizes_list = [{k: v * 2.0 for k, v in prop.items()} for prop in proposal_std_list]
                sa_bounds_list = priors_list
                sa_starting = initial_params_list if initial_params_list is not None else None

                best_params_list, best_energy, energy_trace, temp_trace = pCDM_BI_simulated_annealing.simulated_annealing_optimization(
                    u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                    C_inv, C_logdet, los_e, los_n, los_u,
                    SA_iterations=int(n_iterations*2), initial_temp=10.0, cooling_rate=0.95, min_temp=0.01,
                    step_sizes=sa_step_sizes_list, bounds=sa_bounds_list, starting_params=sa_starting,
                    model_type=model_list
                )

                # Integrate SA result into flattened current_params
                if isinstance(best_params_list, list):
                    for idx, pdict in enumerate(best_params_list):
                        label = labels[idx]
                        for pk, pv in pdict.items():
                            current_params[f"{label}__{pk}"] = pv
                print(f"Joint simulated annealing completed for {model_display_name}. Using best parameters as MCMC initial state.")
                pCDM_BI_plotting_funcs.plot_sa_diagnostics(energy_trace, temp_trace)

            else:
                # Single-model SA (existing behaviour)
                print(f"\nUsing simulated annealing for initial {model_list[0]} parameter estimation...")
                sa_step_sizes = {key: val * 2.0 for key, val in proposal_std_list[0].items()}
                sa_bounds = priors_list[0]
                best_params, best_energy, energy_trace, temp_trace = pCDM_BI_simulated_annealing.simulated_annealing_optimization(
                    u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                    C_inv, C_logdet, los_e, los_n, los_u,
                    SA_iterations=int(n_iterations*2), initial_temp=10.0, cooling_rate=0.95, min_temp=0.01,
                    step_sizes=sa_step_sizes, bounds=sa_bounds, starting_params=initial_params_list[0],
                    model_type=model_list[0]
                )
                # Replace flattened initial for single-model case
                if single_model_mode:
                    for k in best_params.keys():
                        current_params[k] = best_params[k]
                print(f"Simulated annealing completed for {model_list[0]}. Using best parameters as MCMC initial state.")
                pCDM_BI_plotting_funcs.plot_sa_diagnostics(energy_trace, temp_trace)
        
        def _log_prior_single_model(pdict, mname):
            """Model-specific prior checks for a single model's parameter dict.
            Returns 0.0 for valid, -np.inf for invalid.
            """
            m = mname.lower()
            if m == 'pcdm':
                return 0.0
            elif m in ['mogi', 'yang']:
                if 'DV' in pdict and abs(pdict['DV']) > 1e10:
                    return -np.inf
                return 0.0
            elif m == 'mctigue':
                if 'a' in pdict and 'c' in pdict:
                    if pdict['a'] <= 0 or pdict['c'] <= 0:
                        return -np.inf
                    if pdict['a'] > 10000 or pdict['c'] > 10000:
                        return -np.inf
                return 0.0
            elif m == 'okada':
                if 'length' in pdict and 'width' in pdict:
                    if pdict['length'] <= 0 or pdict['width'] <= 0:
                        return -np.inf
                if 'depth' in pdict and pdict['depth'] <= 0:
                    return -np.inf
                return 0.0
            elif m == 'une':
                if 'depth' in pdict and pdict['depth'] <= 0:
                    return -np.inf
                if 'yield_kt' in pdict and pdict['yield_kt'] <= 0:
                    return -np.inf
                return 0.0
            else:
                return 0.0

        def log_prior(params):
            """Calculate log prior probability for flattened parameter dict.
            Supports both single-model (unprefixed keys) and multi-model (prefix__param keys).
            """
            # Check flat prior bounds
            for key, (lower, upper) in flat_priors.items():
                val = params.get(key, None)
                if val is None:
                    return -np.inf
                if not (lower <= val <= upper):
                    return -np.inf

            # Model-specific constraints: check each model separately
            if single_model_mode:
                single_pdict = {k: params[k] for k in params.keys()}
                if _log_prior_single_model(single_pdict, model_list[0]) == -np.inf:
                    return -np.inf
            else:
                for idx, label in enumerate(labels):
                    pdict = {p: params[f"{label}__{p}"] for p in param_structure[label]}
                    mname = model_list[idx]
                    if _log_prior_single_model(pdict, mname) == -np.inf:
                        return -np.inf

            return 0.0
        
        def log_likelihood(params):
            """Calculate log likelihood with correlated noise"""
            try:
                    # Forward model (handle flattened multi-model params)
                if single_model_mode:
                    ue, un, uv = forward_model(params, model_list[0])
                else:
                    params_list = []
                    for label in labels:
                        pdict = {p: params[f"{label}__{p}"] for p in param_structure[label]}
                        params_list.append(pdict)
                    ue, un, uv = forward_model(params_list, model_list)
                # print(  f"Forward model computed for parameters: {params}")
                # print(  f"Predicted displacements (first 5): ue={ue[:5]}, un={un[:5]}, uv={uv[:5]}")
                # Convert to line-of-sight
                u_los_pred = -((ue * los_e) + (un * los_n) + (uv * los_u))
                u_los_pred = u_los_pred.flatten()
                # Calculate likelihood with correlated noise
                residuals = u_los_obs - u_los_pred
                # print(residuals)
                rms_value = np.sqrt(np.mean(residuals**2))
                
                # Optimized multivariate Gaussian likelihood using einsum for faster matrix operations
                log_lik = -0.5 * (np.einsum('i,ij,j->', residuals, C_inv, residuals) + C_logdet + len(residuals) * np.log(2 * np.pi))
                
                return log_lik, rms_value
                
            except Exception as e:
                print(f"Forward model error: {e}")
                return -np.inf, np.nan
        
        def log_posterior(params):
            """Calculate log posterior probability"""
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf, np.nan, -np.inf
            log_lik, residual_rms = log_likelihood(params)
            return lp + log_lik, residual_rms, log_lik
        
        def adapt_proposal_scales(acceptance_counts, proposal_counts, proposal_std_adaptive,
                                  target_acceptance, max_step_sizes, verbose=False):
            """
            Adapt each parameter's proposal std to drive acceptance toward target.

            Rule (damped multiplicative update):
                sigma  <-  sigma * exp(gamma * (acc_rate - target))

            gamma = 0.5 damps the update so one window with 0% acceptance only
            shrinks sigma by exp(0.5 * -0.234) ≈ 0.89 (11%), not 21%.
            Over 20 burn-in windows at zero acceptance that is 0.89^20 ≈ 0.10,
            i.e. a 10x shrinkage at most — reasonable rather than catastrophic.

            - acc too HIGH (proposals too small) -> sigma grows.
            - acc too LOW  (proposals too large) -> sigma shrinks.
            - sigma clamped to [10% of initial, max_step_size].
            - Skipped for parameters with fewer than 3 proposals this window.
            """
            GAMMA = 0.5   # learning rate — smaller = gentler per-window change

            for param in param_names:
                n = proposal_counts[param]
                if n < 3:
                    continue   # too few proposals to estimate rate reliably

                acc_rate = acceptance_counts[param] / n
                new_std  = proposal_std_adaptive[param] * np.exp(GAMMA * (acc_rate - target_acceptance))

                # Clamp: floor at 10% of initial (not 1%) to avoid near-zero collapse
                new_std = np.clip(new_std,
                                  flat_proposal_std[param] * 0.10,
                                  max_step_sizes[param])
                proposal_std_adaptive[param] = new_std

                if verbose:
                    print(f"    {param}: acc={acc_rate:.3f}  "
                          f"(target {target_acceptance:.3f})  "
                          f"-> sigma={new_std:.4g}")

            # Reset window counters
            acceptance_counts = {p: 0 for p in param_names}
            proposal_counts   = {p: 0 for p in param_names}
            return acceptance_counts, proposal_counts, proposal_std_adaptive

        # Calculate burn-in point (used for adaptive scaling cutoff)
        if burn_in is None:
            burn_in = int(n_iterations * 0.2)
        else:
            burn_in = max(0, min(int(burn_in), int(n_iterations) - 1))
        
        # Initialize with flattened provided parameters
        # current_params = flat_initial.copy()
        
        # Storage for samples
        samples = {key: [] for key in current_params.keys()}
        log_likelihood_trace = []
        residuals_evolution = []
        # Store proposal std and acceptance rate only at adaptation checkpoints
        # (one entry per adaptation event, not every iteration)
        proposal_std_evolution = {key: [] for key in param_names}
        proposal_std_evolution_iters = []          # iteration number of each checkpoint
        acceptance_rate_evolution = {key: [] for key in param_names}
        
        current_log_post, residual_rms, log_like_current = log_posterior(current_params)
        
        print(f"Starting MCMC sampling with {model_display_name} model and correlated noise...")
        print(f"Model parameters: {list(param_names)}")
        if isinstance(sill, (list, tuple, np.ndarray)):
            print(f"Noise parameters - Sill: {sill}")
            print(f"                Nugget: {nugget}")
            print(f"                Range:  {range_param}")
        else:
            print(f"Noise parameters - Sill: {sill:.6f}, Nugget: {nugget:.6f}, Range: {range_param:.3f}")
        print(f"Target acceptance rate: {target_acceptance:.2f}")
        print(f"Adaptation interval: {adaptive_interval} iterations")
        print(f"Initial parameters:")
        for key, val in current_params.items():
            print(f"  {key}: {val:.6f}")
        print(f"Initial proposal standard deviations (learning rates):")
        for key, val in flat_proposal_std.items():
            print(f"  {key}: {val:.6f}")
        
        # DIAGNOSTIC: Print initial state quality
        print(f"\n*** INITIAL STATE DIAGNOSTICS ***")
        print(f"  Initial log-posterior: {current_log_post:.4f}")
        print(f"  Initial log-likelihood: {log_like_current:.4f}")
        print(f"  Initial RMS residual: {residual_rms:.6f}")
        if not np.isfinite(current_log_post):
            print(f"  ⚠️  WARNING: Initial log-posterior is {current_log_post}!")
            print(f"     This means the starting point is invalid (prior or likelihood issue).")
            print(f"     Check: 1) Initial params within priors, 2) Forward model runs without error.")
        print(f"********************************\n")
        
        # Define autocorrelation function for use during MCMC
        def calculate_autocorr(chain, max_lag=50):
            """Calculate autocorrelation for a chain"""
            if len(chain) < 2:
                return np.array([1.0])
            chain = chain - np.mean(chain)
            c0 = np.dot(chain, chain) / len(chain)
            if c0 == 0:
                return np.array([1.0])
            acf = [1.0]
            for lag in range(1, max_lag):
                if lag >= len(chain):
                    break
                c_lag = np.dot(chain[:-lag], chain[lag:]) / len(chain)
                acf.append(c_lag / c0)
            return np.array(acf)
        
        def calculate_effective_sample_size(chain, max_lag=50):
            """Calculate effective sample size and autocorr time
            
            For multivariate chains (2D array), uses Mahalanobis distance
            For univariate chains (1D array), uses standard autocorrelation
            """
            # Handle multivariate case (all parameters combined)
            if chain.ndim == 2:
                # chain shape: (n_samples, n_params)
                n_samples, n_params = chain.shape
                
                # Center the chain
                chain_centered = chain - np.mean(chain, axis=0)
                
                # Calculate covariance matrix
                cov_matrix = np.cov(chain_centered.T)
                
                # Regularize if singular
                if np.linalg.matrix_rank(cov_matrix) < n_params:
                    cov_matrix += np.eye(n_params) * 1e-6 * np.trace(cov_matrix) / n_params
                
                # Calculate Mahalanobis distances from mean at each step
                try:
                    cov_inv = np.linalg.inv(cov_matrix)
                    mahal_distances = []
                    for i in range(n_samples):
                        diff = chain_centered[i]
                        mahal_dist = np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
                        mahal_distances.append(mahal_dist)
                    mahal_distances = np.array(mahal_distances)
                    
                    # Calculate autocorrelation of Mahalanobis distances
                    md_centered = mahal_distances - np.mean(mahal_distances)
                    c0 = np.dot(md_centered, md_centered) / n_samples
                    if c0 == 0:
                        return n_samples, 1.0
                    
                    tau_int = 1.0
                    for lag in range(1, max_lag):
                        if lag >= n_samples:
                            break
                        c_lag = np.dot(md_centered[:-lag], md_centered[lag:]) / n_samples
                        acf_lag = c_lag / c0
                        tau_int += 2.0 * acf_lag
                        if acf_lag < 0.05:
                            break
                    
                    n_eff = n_samples / tau_int
                    return n_eff, tau_int
                except:
                    return n_samples, 1.0
            
            # Handle univariate case (single parameter)
            else:
                if len(chain) < 2:
                    return len(chain), 1.0
                chain = chain - np.mean(chain)
                c0 = np.dot(chain, chain) / len(chain)
                if c0 == 0:
                    return len(chain), 1.0
                
                tau_int = 1.0
                for lag in range(1, max_lag):
                    if lag >= len(chain):
                        break
                    c_lag = np.dot(chain[:-lag], chain[lag:]) / len(chain)
                    acf_lag = c_lag / c0
                    tau_int += 2.0 * acf_lag
                    if acf_lag < 0.05:
                        break
                
                n_eff = len(chain) / tau_int
                return n_eff, tau_int
        
        for i in range(n_iterations):
            # Propose new parameters
            proposed_params = current_params.copy()
            
            # Select parameter to update (round-robin or random)
            if use_round_robin_updates:
                param_to_update = rr_cycle[rr_index % len(rr_cycle)]
                rr_index += 1
            else:
                param_to_update = np.random.choice(param_names)

            # Propose change for the selected parameter
            proposed_params[param_to_update] += np.random.normal(0, proposal_std_adaptive[param_to_update])

            # Update proposal count
            proposal_counts[param_to_update] += 1
            
            # Calculate acceptance probability (numerically stable)
            proposed_log_post, proposed_residual_rms, log_like_prop = log_posterior(proposed_params)

            # Track why proposals are rejected (prior violation vs forward-model failure)
            if proposed_log_post == -np.inf:
                # If prior rejects it, log_prior will be -inf; otherwise assume forward-model failure
                if not np.isfinite(log_prior(proposed_params)):
                    proposal_prior_violations[param_to_update] += 1
                else:
                    proposal_forward_errors[param_to_update] += 1

            delta = proposed_log_post - current_log_post
            if not np.isfinite(delta):
                alpha = 0.0
            elif delta >= 0:
                alpha = 1.0
            else:
                # safe exp for negative delta
                alpha = float(np.exp(delta))

            # Accept or reject
            if np.random.rand() < alpha:
                current_params = proposed_params
                current_log_post = proposed_log_post
                residual_rms = proposed_residual_rms
                log_like_current = log_like_prop
                acceptance_counts[param_to_update] += 1
            
            # Store samples
            for key in current_params.keys():
                samples[key].append(current_params[key])
            
            residuals_evolution.append(residual_rms)
            log_likelihood_trace.append(log_like_current)
            
            # Adaptive step size adjustment (ONLY during burn-in phase)
            # CRITICAL FIX: After burn-in, proposal scales are frozen to preserve detailed balance
            if (i + 1) % adaptive_interval == 0 and i > 0 and (i + 1) <= burn_in:
                # Calculate current acceptance rates
                current_acceptance_rates = {}
                for param in param_names:
                    if proposal_counts[param] > 0:
                        current_acceptance_rates[param] = acceptance_counts[param] / proposal_counts[param]
                    else:
                        current_acceptance_rates[param] = 0.0
                
                # Store acceptance rates
                for key in param_names:
                    acceptance_rate_evolution[key].append(current_acceptance_rates[key])
                # Record proposal std at this checkpoint
                for key in param_names:
                    proposal_std_evolution[key].append(proposal_std_adaptive[key])
                proposal_std_evolution_iters.append(i + 1)
                
                # Print adaptation info
                verbose_adapt = True  # Print every adaptation event during burn-in
                if verbose_adapt:
                    print(f"\n  Adaptation at iteration {i+1} (burn-in):")
                
                # Adapt proposal scales during burn-in
                acceptance_counts, proposal_counts, proposal_std_adaptive = adapt_proposal_scales(
                    acceptance_counts, proposal_counts, proposal_std_adaptive,
                    target_acceptance, flat_max_step_sizes, verbose=verbose_adapt)

                # Notify when entering stationary phase
                if (i + 1) == burn_in:
                    print(f"\n*** BURN-IN PHASE COMPLETE (iteration {i+1}) ***")
                    print(f"Proposal scales FROZEN to preserve detailed balance.")
                    print(f"Subsequent samples are theoretically valid for Bayesian inference.\n")
            
            elif (i + 1) % adaptive_interval == 0 and (i + 1) > burn_in:
                # After burn-in: calculate acceptance rates for diagnostics, but DON'T adapt
                current_acceptance_rates = {}
                for param in param_names:
                    if proposal_counts[param] > 0:
                        current_acceptance_rates[param] = acceptance_counts[param] / proposal_counts[param]
                    else:
                        current_acceptance_rates[param] = 0.0
                
                # Store acceptance rates (for diagnostics only)
                for key in param_names:
                    acceptance_rate_evolution[key].append(current_acceptance_rates[key])
                # Record frozen proposal std so post-burn-in is visible on plot
                for key in param_names:
                    proposal_std_evolution[key].append(proposal_std_adaptive[key])
                proposal_std_evolution_iters.append(i + 1)
                
                # Reset counters but DO NOT adapt proposal scales
                acceptance_counts = {param: 0 for param in param_names}
                proposal_counts = {param: 0 for param in param_names}
                
                
                # Print diagnostics
                if len(samples[param_names[0]]) > 0 and (i + 1) % int(n_iterations*0.1) == 0:
                    current_samples = {param: samples[param][-min(int(n_iterations*0.1), len(samples[param])):] for param in param_names}
                    current_means = {param: np.mean(vals) for param, vals in current_samples.items()}
                    phase_status = "[BURN-IN]" if (i + 1) <= burn_in else "[POST-BURN-IN]"
                    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {i+1}/{n_iterations} ({model_display_name}) {phase_status} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"  Proposal scales (fixed after burn-in):")
                    for param in param_names:
                        print(f"    {param}: {proposal_std_adaptive[param]:.6f} "
                            f"(acceptance: {current_acceptance_rates[param]:.3f})")

                    print(f"  Current most likely parameters (last {min(int(n_iterations*0.1), len(samples[param_names[0]]))} samples):")
                    for param in param_names:
                        print(f"    {param}: {current_means[param]:.6f}")
                    print(f"  Current RMS: {residual_rms:.6f}")
                    
                    # Calculate and print convergence diagnostics for ALL parameters (joint)
                    if (i + 1) > burn_in:
                        # Create multivariate chain from all parameters
                        post_burnin_samples = np.column_stack([np.array(samples[param][burn_in:]) for param in param_names])
                        if post_burnin_samples.shape[0] > 1:
                            n_eff, tau_int = calculate_effective_sample_size(post_burnin_samples, max_lag=min(50, post_burnin_samples.shape[0]//2))
                            efficiency = (n_eff / post_burnin_samples.shape[0]) * 100 if post_burnin_samples.shape[0] > 0 else 0
                            print(f"  MCMC Efficiency (Joint - All {len(param_names)} parameters):")
                            print(f"    Autocorr time (tau): {tau_int:.2f}")
                            print(f"    Effective samples: {n_eff:.0f} / {post_burnin_samples.shape[0]}")
                            print(f"    Efficiency: {efficiency:.1f}%")
                            if n_eff < 100:
                                print(f"    ⚠️  Low effective sample size - consider more iterations")
                            elif efficiency < 10:
                                print(f"    ⚠️  Low efficiency - parameter correlations affect mixing")
                    else:
                        print(f"  MCMC Efficiency: [Still in burn-in, diagnostics after burn-in]")
                    
                    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        # Extract post-burn-in samples (burn_in calculated at start of MCMC)
        samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}
        
        # MCMC Diagnostics: Calculate autocorrelation and effective sample size
        print(f"\n{'='*80}")
        print(f"MCMC DIAGNOSTICS AND CONVERGENCE ASSESSMENT")
        print(f"{'='*80}")
        print(f"Burn-in period: {burn_in} iterations ({100*burn_in/n_iterations:.1f}% of {n_iterations})")
        print(f"Post-burn-in samples: {len(samples_burned[param_names[0]])}")
        print(f"\n*** CRITICAL FIX APPLIED ***")
        print(f"Proposal scales were FROZEN after burn-in to preserve detailed balance.")
        print(f"This ensures post-burn-in samples satisfy MCMC ergodicity requirements.\n")
        
        # Calculate autocorrelation for first parameter as a diagnostic
        def calculate_autocorr(chain, max_lag=50):
            """Calculate autocorrelation for a chain"""
            chain = chain - np.mean(chain)
            c0 = np.dot(chain, chain) / len(chain)
            acf = [1.0]
            for lag in range(1, max_lag):
                c_lag = np.dot(chain[:-lag], chain[lag:]) / len(chain)
                acf.append(c_lag / c0)
            return np.array(acf)
        
        first_param = param_names[0]
        autocorr = calculate_autocorr(samples_burned[first_param], max_lag=min(50, len(samples_burned[first_param])//2))
        
        # Find integrated autocorrelation time (tau)
        tau_int = 1.0
        for i, ac in enumerate(autocorr):
            if ac < 0.05:  # Stop when autocorr drops below 5%
                tau_int = 1.0 + 2.0 * np.sum(autocorr[1:i])
                break
        
        n_eff = len(samples_burned[first_param]) / tau_int
        print(f"Individual Parameter Autocorrelation ('{first_param}'):")
        print(f"  Integrated autocorrelation time (tau): {tau_int:.2f}")
        print(f"  Effective sample size: {n_eff:.0f} (out of {len(samples_burned[first_param])})")
        print(f"  Sampling efficiency: {n_eff/len(samples_burned[first_param])*100:.1f}%")
        
        # Calculate JOINT chain efficiency across all parameters
        print(f"\nJoint Chain Efficiency (All {len(param_names)} Parameters):")
        joint_samples = np.column_stack([samples_burned[param] for param in param_names])
        n_eff_joint, tau_int_joint = calculate_effective_sample_size(joint_samples, max_lag=min(50, joint_samples.shape[0]//2))
        efficiency_joint = (n_eff_joint / joint_samples.shape[0]) * 100
        print(f"  Integrated autocorrelation time (tau): {tau_int_joint:.2f}")
        print(f"  Effective sample size: {n_eff_joint:.0f} (out of {joint_samples.shape[0]})")
        print(f"  Sampling efficiency: {efficiency_joint:.1f}%")
        
        if n_eff_joint < 100:
            print(f"\n⚠️  WARNING: Low effective sample size ({n_eff_joint:.0f}).")
            print(f"   Consider: increasing n_iterations, adjusting proposal_std, or adding thinning.")
        elif efficiency_joint < 10:
            print(f"\n⚠️  NOTE: Low joint efficiency ({efficiency_joint:.1f}%).")
            print(f"   This suggests parameter correlations affect mixing.")
            print(f"   Individual parameters may still be well-sampled ({n_eff:.1f}%).")
        else:
            print(f"\n✓ Good joint chain efficiency ({efficiency_joint:.1f}%). Samples suitable for multivariate inference.")
        
        print(f"{'='*80}\n")

        # Create DataFrame with samples
        df_samples = pd.DataFrame(samples_burned)

        # Add additional columns
        df_samples['iteration'] = range(burn_in, len(samples[list(samples.keys())[0]]))
        df_samples['log_likelihood'] = log_likelihood_trace[burn_in:]
        df_samples['rms_residual'] = residuals_evolution[burn_in:]

        # Save to CSV
        csv_filename = f"mcmc_samples_{model_id_for_files}_n{n_iterations}_accept{target_acceptance}.csv"
        if figure_folder is not None:
            csv_filename = f"{figure_folder}/{csv_filename}"

        df_samples.to_csv(csv_filename, index=False)
        print(f"MCMC samples saved to: {csv_filename}")

        # Calculate optimal parameters and save summary
        summary_stats = []
        optimal_params = {}
        map_params = {}
        
        for param in samples_burned.keys():
            optimal_params[param] = np.mean(samples_burned[param])
            
            mean_val = np.mean(samples_burned[param])
            std_val = np.std(samples_burned[param])
            q025 = np.percentile(samples_burned[param], 2.5)
            q975 = np.percentile(samples_burned[param], 97.5)
            
            summary_stats.append({
                'parameter': param,
                'mean': mean_val,
                'std': std_val,
                'q025': q025,
                'q975': q975
            })

        # Calculate MAP estimate
        best_idx = np.argmax(log_likelihood_trace[burn_in:])
        for param in samples_burned.keys():
            map_params[param] = samples_burned[param][best_idx]

        print(f"\n{model_display_name} Model - Optimal Parameters (Posterior Mean):")
        print("-" * 50)
        for param, value in optimal_params.items():
            print(f"{param:8s}: {value:8.4f}")

        print(f"\n{model_display_name} Model - MAP (Maximum A Posteriori) Parameters:")
        print("-" * 50)
        for param, value in map_params.items():
            print(f"{param:8s}: {value:8.4f}")

        # Save summary and results
        df_summary = pd.DataFrame(summary_stats)
        summary_filename = f"mcmc_summary_{model_id_for_files}_n{n_iterations}_accept{target_acceptance}.csv"
        if figure_folder is not None:
            summary_filename = f"{figure_folder}/{summary_filename}"

        df_summary.to_csv(summary_filename, index=False)
        print(f"Summary statistics saved to: {summary_filename}")

        # Save detailed output
        output_filename = f"inference_results_{model_id_for_files}_n{n_iterations}_accept{target_acceptance}.txt"
        if figure_folder is not None:
            output_filename = f"{figure_folder}/{output_filename}"

        with open(output_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BAYESIAN INFERENCE RESULTS - {model_display_name} MODEL\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model Type: {model_display_name}\n")
            f.write(f"Model Parameters: {list(param_names)}\n\n")
            
            # Write final acceptance rate
            final_acceptance = sum(acceptance_counts.values()) / max(1, sum(proposal_counts.values()))
            f.write(f"MCMC completed. Final overall acceptance rate: {final_acceptance:.3f}\n\n")
            
            # Write final proposal standard deviations
            f.write("Final proposal standard deviations:\n")
            for param in param_names:
                f.write(f"{param:8s}: {proposal_std_adaptive[param]:.6f}\n")
            f.write("\n")
            
            # Write optimal parameters
            f.write("Optimal Model Parameters (Posterior Mean):\n")
            f.write("-" * 50 + "\n")
            for param, value in optimal_params.items():
                f.write(f"{param:8s}: {value:8.4f}\n")
            f.write("\n")
            
            f.write("Maximum A Posteriori (MAP) Parameters:\n")
            f.write("-" * 50 + "\n")
            for param, value in map_params.items():
                f.write(f"{param:8s}: {value:8.4f}\n")
            f.write("\n")
            
            # Write posterior summary statistics
            f.write("Posterior Summary Statistics:\n")
            f.write("-" * 50 + "\n")
            for param in samples_burned.keys():
                mean_val = np.mean(samples_burned[param])
                std_val = np.std(samples_burned[param])
                q025 = np.percentile(samples_burned[param], 2.5)
                q975 = np.percentile(samples_burned[param], 97.5)
                f.write(f"{param:8s}: {mean_val:8.4f} ± {std_val:6.4f} [{q025:8.4f}, {q975:8.4f}]\n")
            f.write("\n")
            
            # Write final RMS residual
            if len(residuals_evolution) > 0:
                final_rms = residuals_evolution[-1]
                f.write(f"Final RMS residual: {final_rms:.6f}\n")
                
                post_burnin_rms = residuals_evolution[burn_in:]
                if len(post_burnin_rms) > 0:
                    mean_rms = np.mean(post_burnin_rms)
                    std_rms = np.std(post_burnin_rms)
                    f.write(f"Post burn-in RMS: {mean_rms:.6f} ± {std_rms:.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")

        print(f"Inference results saved to: {output_filename}")
        
        return (samples, log_likelihood_trace, residuals_evolution, 
                proposal_std_evolution, acceptance_rate_evolution,
                proposal_std_evolution_iters)

def gen_synthetic_data(true_params, grid_size=100, noise_level=0.5, model_type='pcdm'):
    """
    Generate synthetic data for testing various deformation models.
        
    Parameters:
    -----------
    true_params : dict
        True parameter values for the model. If None, uses default values based on model_type.
    grid_size : int
        Size of the observation grid (grid_size x grid_size points)
    noise_level : float
        Noise level as fraction of signal standard deviation
    model_type : str
        Type of forward model ('pCDM', 'okada', 'mogi', etc.)
        
    Returns:
    --------
    u_los_obs : ndarray
        Observed line-of-sight displacements with noise
    X_flat, Y_flat : ndarray
        Flattened observation coordinates
    incidence_angle : float
        Satellite incidence angle
    heading : float
        Satellite heading
    noise_sill, noise_nugget, noise_range : float
        Noise model parameters
    """
    # Create grid based on grid_size parameter
    x_range = np.linspace(-50000, 50000, grid_size)
    y_range = np.linspace(-50000, 50000, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    print(f"Generated grid with {grid_size}x{grid_size} points.")
    print(f"model_type: {model_type}")

    if true_params is None:
        print(true_params)
        
        if model_type.lower() == 'pcdm':
            # True parameters
            true_params = {
                'X0': 0.5, 'Y0': -0.3, 'depth': 12000,
        'DVx': 5e7, 'DVy': 5e7, 'DVz': 5e7,
        'omegaX': 10.0, 'omegaY': -30.0, 'omegaZ': -10.0
        }
   
        elif model_type.lower() == 'okada':
            # True parameters for Okada model
            true_params = {
                'X0': 0.0, 'Y0': 0.0, 'depth': 4.0e3,
                'length': 10e3, 'width': 4e3,
                'strike': 120, 'dip': 30, 'rake': 10,
                'slip': 3, 'opening': 0.0
            }
        elif model_type.lower() == 'une':
            # True parameters for UNE model
            true_params = {
                'X0': 0.0, 'Y0': 0.0, 'depth': 2000.0,
                'yield_kt': 500.0, 'dv_factor': 0.1,
                'chimney_amp': 0.15, 'compact_amp': 0.05
            }
         
    if model_type.lower() == 'pcdm':
        print(true_params)
             # Generate true displacements
        ue_true, un_true, uv_true = pCDM_fast.pCDM(X_flat, Y_flat, true_params['X0'], true_params['Y0'], 
                                            true_params['depth'], true_params['omegaX'], 
                                            true_params['omegaY'], true_params['omegaZ'],
                                            true_params['DVx'], true_params['DVy'], 
                                            true_params['DVz'], 0.25)
    elif model_type.lower() == 'okarda':
        print(true_params)
        ue_true, un_true, uv_true = okada.disloc3d3(X_flat, Y_flat, xoff=true_params['X0'], yoff=true_params['Y0'], 
                                                depth=true_params['depth'], length=true_params['length'], 
                                                width=true_params['width'], slip=true_params['slip'], opening=true_params['opening'],
                                                strike=true_params['strike'],
                                                dip=true_params['dip'], rake=true_params['rake'], nu=0.25)

    elif model_type.lower() == 'une':
        print(true_params)
        uv_true, ue_true, un_true = UNE_three.model(X_flat, Y_flat,
                                                depth=true_params['depth'], yield_kt=true_params['yield_kt'],
                                                dv_factor=true_params['dv_factor'],
                                                chimney_amp=true_params['chimney_amp'],
                                                chimney_height_fac=10, chimney_peck_k=0.35,
                                                compact_amp=true_params['compact_amp'],
                                                anelastic_fac=5,
                                                x0=true_params['X0'], y0=true_params['Y0'],
                                                nu=0.25, mu=30e9)

            



    # Convert to LOS
    incidence_angle = 39.4
    heading = -169.9
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)
    
    u_los_true = -((ue_true * los_e) + (un_true * los_n) + (uv_true * los_u))
    u_los_true = u_los_true.flatten()
    print(np.max(u_los_true), np.min(u_los_true))
    # Add noise
    noise_std = np.std(u_los_true) * noise_level  # 5% noise
    u_los_obs = u_los_true + np.random.normal(0, noise_std, len(u_los_true))
    
    # Generate spatially correlated noise
    n_obs = len(u_los_obs)
    distances = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            distances[i, j] = np.sqrt((X_flat[i] - X_flat[j])**2 + (Y_flat[i] - Y_flat[j])**2)
    
    # Noise parameters
    noise_sill = (noise_std)**2  # Total variance
    noise_nugget = noise_sill * 0.01  # 1% nugget effect
    noise_range = np.max(distances) / 4.0  # Correlation length
    
    # Construct noise covariance matrix using exponential model
    C_noise = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            if i == j:
                C_noise[i, j] = noise_sill
            else:
                h = distances[i, j]
                C_noise[i, j] = (noise_sill - noise_nugget) * np.exp(-h / noise_range)
    
    # Generate spatially correlated noise
    spatially_correlated_noise = np.random.multivariate_normal(np.zeros(n_obs), C_noise)
    
    # Add spatially correlated noise instead of independent noise
    u_los_obs = u_los_true + spatially_correlated_noise

    # Plot the synthetic data for visualization
    if model_type.lower() == 'pcdm':
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot true LOS displacement
        sc1 = axes[0,0].tricontourf(X_flat, Y_flat, u_los_true, levels=50, cmap='RdBu_r')
        axes[0,0].set_title('True LOS Displacement (mm)')
        axes[0,0].set_xlabel('X (km)')
        axes[0,0].set_ylabel('Y (km)')
        axes[0,0].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[0,0])
        
        # Plot observed LOS displacement (with noise)
        sc2 = axes[0,1].tricontourf(X_flat, Y_flat, u_los_obs, levels=50, cmap='RdBu_r')
        axes[0,1].set_title('Observed LOS Displacement (mm)')
        axes[0,1].set_xlabel('X (km)')
        axes[0,1].set_ylabel('Y (km)')
        axes[0,1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[0,1])
        
        # Plot noise
        sc3 = axes[1,0].tricontourf(X_flat, Y_flat, spatially_correlated_noise, levels=50, cmap='viridis')
        axes[1,0].set_title('Spatially Correlated Noise (mm)')
        axes[1,0].set_xlabel('X (km)')
        axes[1,0].set_ylabel('Y (km)')
        axes[1,0].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[1,0])
        
        # Plot difference (should be just the noise)
        difference = u_los_obs - u_los_true
        sc4 = axes[1,1].tricontourf(X_flat, Y_flat, difference, levels=50, cmap='viridis')
        axes[1,1].set_title('Difference (Obs - True) (mm)')
        axes[1,1].set_xlabel('X (km)')
        axes[1,1].set_ylabel('Y (km)')
        axes[1,1].set_aspect('equal')
        plt.colorbar(sc4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(f'synthetic_data_{model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

    elif model_type.lower() == 'okada':
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot true LOS displacement
        sc1 = axes[0,0].tricontourf(X_flat/1000, Y_flat/1000, u_los_true, levels=50, cmap='RdBu_r')
        axes[0,0].set_title('True LOS Displacement (m)')
        axes[0,0].set_xlabel('X (km)')
        axes[0,0].set_ylabel('Y (km)')
        axes[0,0].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[0,0])
        
        # Plot observed LOS displacement (with noise)
        sc2 = axes[0,1].tricontourf(X_flat/1000, Y_flat/1000, u_los_obs, levels=50, cmap='RdBu_r')
        axes[0,1].set_title('Observed LOS Displacement (m)')
        axes[0,1].set_xlabel('X (km)')
        axes[0,1].set_ylabel('Y (km)')
        axes[0,1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[0,1])
        
        # Plot noise
        sc3 = axes[1,0].tricontourf(X_flat/1000, Y_flat/1000, spatially_correlated_noise, levels=50, cmap='viridis')
        axes[1,0].set_title('Spatially Correlated Noise (m)')
        axes[1,0].set_xlabel('X (km)')
        axes[1,0].set_ylabel('Y (km)')
        axes[1,0].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[1,0])
        
        # Plot difference (should be just the noise)
        difference = u_los_obs - u_los_true
        sc4 = axes[1,1].tricontourf(X_flat/1000, Y_flat/1000, difference, levels=50, cmap='viridis')
        axes[1,1].set_title('Difference (Obs - True) (m)')
        axes[1,1].set_xlabel('X (km)')
        axes[1,1].set_ylabel('Y (km)')
        axes[1,1].set_aspect('equal')
        # Prepare model geometry for plotting
        xcent = float(true_params['X0'])
        ycent = float(true_params['Y0'])
        strike = float(true_params['strike'])
        dip = float(true_params['dip'])
        rake = float(true_params['rake'])
        slip = float(true_params['slip'])
        length = float(true_params['length'])
        centroid_depth = float(true_params['depth'])
        width = float(true_params['width'])
        model = [xcent, ycent, strike, dip, rake, slip, length, centroid_depth, width]
        # end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = okada.fault_for_plotting(model)
        # plt.plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White',figure=fig)
        # plt.scatter(end1x/1000, end1y/1000, color='white',figure=fig)
        # plt.plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White',figure=fig)
     
        plt.colorbar(sc4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(f'synthetic_data_{model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

    print(f"True parameters used for {model_type} model:")
    for key, value in true_params.items():
        print(f"  {key}: {value}")
    print(f"Noise level: {noise_level*100:.1f}%")
    print(f"RMS of true signal: {np.std(u_los_true)*1000:.3f} mm")
    print(f"RMS of noise: {np.std(spatially_correlated_noise)*1000:.3f} mm")

    return u_los_obs, X_flat, Y_flat, incidence_angle, heading, noise_sill, noise_nugget, noise_range

def save_inference_state(samples, log_lik_trace, rms_evolution, proposal_std_evolution, 
                        acceptance_rate_evolution, X_obs, Y_obs, u_los_obs, 
                        incidence_angle, heading, inference_params, figure_folder=None,model_type='pCDM'):
    """
    Save complete inference state to pickle file for later regeneration.
    
    Parameters:
    -----------
    samples : dict
        MCMC samples for each parameter
    log_lik_trace : array
        Log-likelihood trace
    rms_evolution : array
        RMS residual evolution
    proposal_std_evolution : dict
        Evolution of proposal standard deviations
    acceptance_rate_evolution : dict
        Evolution of acceptance rates
    X_obs, Y_obs : array_like
        Observation coordinates
    u_los_obs : array_like
        Observed line-of-sight displacements
    incidence_angle : float
        Satellite incidence angle
    heading : float
        Satellite heading
    inference_params : dict
        All inference parameters (initial params, priors, etc.)
    figure_folder : str, optional
        Folder to save the pickle file
    """
    
    # Create comprehensive state dictionary
    inference_state = {
        # Core results
        'samples': samples,
        'log_likelihood_trace': log_lik_trace,
        'rms_evolution': rms_evolution,
        'proposal_std_evolution': proposal_std_evolution,
        'acceptance_rate_evolution': acceptance_rate_evolution,
        
        # Input data
        'X_obs': X_obs,
        'Y_obs': Y_obs,
        'u_los_obs': u_los_obs,
        'incidence_angle': incidence_angle,
        'heading': heading,
        
        # All inference parameters
        'inference_params': inference_params,
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'n_iterations': len(log_lik_trace),
        'burn_in': inference_params.get('burn_in', int(len(log_lik_trace) * 0.2))
    }
    
    # Generate filename
    n_iterations = len(log_lik_trace)
    target_acceptance = inference_params.get('target_acceptance', 0.23)
    if isinstance(model_type, list):
        model_id = "_".join([m.lower() for m in model_type])
    else:
        model_id = str(model_type).lower()
    pickle_filename = f"bayesian_inference_state_n{n_iterations}_accept{target_acceptance}_{model_id}.pkl"
    
    if figure_folder is not None:
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
        pickle_filename = os.path.join(figure_folder, pickle_filename)
    
    # Save to pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(inference_state, f)
    
    print(f"Complete inference state saved to: {pickle_filename}")
    print(f"File size: {os.path.getsize(pickle_filename) / (1024*1024):.2f} MB")
    
    return pickle_filename

def load_inference_state(pickle_filename):
    """
    Load inference state from pickle file.
    
    Parameters:
    -----------
    pickle_filename : str
        Path to the pickle file
        
    Returns:
    --------
    inference_state : dict
        Complete inference state dictionary
    """
    with open(pickle_filename, 'rb') as f:
        inference_state = pickle.load(f)
    
    print(f"Loaded inference state from: {pickle_filename}")
    print(f"Timestamp: {inference_state.get('timestamp', 'Unknown')}")
    print(f"Number of iterations: {inference_state.get('n_iterations', 'Unknown')}")
    print(f"Burn-in period: {inference_state.get('burn_in', 'Unknown')}")
    
    return inference_state

def regenerate_plots_from_state(pickle_filename, new_figure_folder=None):
    """
    Regenerate all plots from saved inference state.
    
    Parameters:
    -----------
    pickle_filename : str
        Path to the pickle file containing inference state
    new_figure_folder : str, optional
        New folder to save regenerated plots. If None, uses timestamp-based folder.
    """
    # Load the inference state
    inference_state = load_inference_state(pickle_filename)
    
    # Extract data
    samples = inference_state['samples']
    log_lik_trace = inference_state['log_likelihood_trace']
    rms_evolution = inference_state['rms_evolution']
    proposal_std_evolution = inference_state['proposal_std_evolution']
    acceptance_rate_evolution = inference_state['acceptance_rate_evolution']
    X_obs = inference_state['X_obs']
    Y_obs = inference_state['Y_obs']
    u_los_obs = inference_state['u_los_obs']
    incidence_angle = inference_state['incidence_angle']
    heading = inference_state['heading']
    inference_params = inference_state['inference_params']
    burn_in = inference_state['burn_in']
    
    # Create figure folder if not specified
    if new_figure_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_figure_folder = f"regenerated_plots_{timestamp}"
    
    if not os.path.exists(new_figure_folder):
        os.makedirs(new_figure_folder)
    
    print(f"Regenerating plots in folder: {new_figure_folder}")
    
    # Regenerate plots using the plotting function
    pCDM_BI_plotting_funcs.plot_inference_results(
        samples,
        log_lik_trace,
        rms_evolution,
        burn_in=burn_in,
        X_obs=X_obs,
        Y_obs=Y_obs,
        u_los_obs=u_los_obs,
        incidence_angle=incidence_angle,
        heading=heading,
        figure_folder=new_figure_folder,
        proposal_std_evolution=proposal_std_evolution,
        acceptance_rate_evolution=acceptance_rate_evolution,
        adaptive_interval=inference_params.get('adaptive_interval', 1000),
        target_acceptance=inference_params.get('target_acceptance', 0.23),
        model_type=inference_params.get('model_type', 'pCDM')
    )

   
    print(f"All plots regenerated in: {new_figure_folder}")

def run_baysian_inference(u_los_obs, X_obs, Y_obs, incidence_angle, heading, 
                         n_iterations=10000, sill=None, nugget=None, range_param=None,
                         initial_params=None, priors=None, proposal_std=None, 
                         max_step_sizes=None, adaptive_interval=1000, 
                         target_acceptance=0.23, adapt_method='conservative', figure_folder=None,use_sa_init=False,model_type='pCDM',u_los_obs_already_los_in_m=None,
                         burn_in=None):
    """
    Run Bayesian inference and plot results.
    
    Parameters:
    -----------
    u_los_obs : array_like
        Observed line-of-sight displacements
    X_obs, Y_obs : array_like
        Observation coordinates
    incidence_angle : float
        Satellite incidence angle in degrees
    heading : float
        Satellite heading in degrees
    n_iterations : int
        Number of MCMC iterations
    sill : float, optional
        Variogram sill parameter. If None, estimated from data.
    nugget : float, optional
        Variogram nugget parameter. If None, estimated as 1% of sill.
    range_param : float, optional
        Variogram range parameter. If None, estimated as 1/3 of max distance.
    initial_params : dict, optional
        Initial parameter values. If None, uses default values.
    priors : dict, optional
        Prior bounds for each parameter. If None, uses default bounds.
    proposal_std : dict, optional
        Initial proposal standard deviations (learning rates). If None, uses defaults.
    max_step_sizes : dict, optional
        Maximum allowed step sizes for adaptive scaling. If None, uses defaults.
    adaptive_interval : int
        Number of iterations between adaptation steps
    target_acceptance : float
        Target acceptance rate
    adapt_method : str, optional
        Adaptation routine to use during burn-in ('conservative' or 'standard').
    figure_folder : str, optional
        Folder to save figures
    """
    # Create figure folder if it doesn't exist
    if figure_folder is not None and not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    
 
    
    # convert from Phase to Displacement (m)
    # Support single IFG or list-of-IFGs (leave unchanged when user signals data are already LOS in metres)
    conv_factor = 0.0555/(4*np.pi)
    if u_los_obs_already_los_in_m is not None:
        # caller indicated values are already LOS in metres — do not convert
        pass
    else:
        if isinstance(u_los_obs, (list, tuple)):
            u_los_obs = [ -np.asarray(u).astype(float) * conv_factor for u in u_los_obs ]
        else:
            u_los_obs = -np.asarray(u_los_obs).astype(float) * conv_factor

    # Estimate noise parameters if not provided (supports single or list-of-IFGs)
    if sill is None:
        if isinstance(u_los_obs, (list, tuple)):
            sill = [np.var(np.asarray(u)) for u in u_los_obs]
        else:
            sill = np.var(u_los_obs)

    if nugget is None:
        if isinstance(sill, (list, tuple, np.ndarray)):
            nugget = [s * 0.01 for s in sill]
        else:
            nugget = sill * 0.01

    if range_param is None:
        if isinstance(X_obs, (list, tuple)):
            range_param = []
            for Xi, Yi in zip(X_obs, Y_obs):
                Xi_arr = np.asarray(Xi).flatten()
                Yi_arr = np.asarray(Yi).flatten()
                n_obs = len(Xi_arr)
                max_dist = 0.0
                for ii in range(n_obs):
                    for jj in range(ii+1, n_obs):
                        dist = np.hypot(Xi_arr[ii] - Xi_arr[jj], Yi_arr[ii] - Yi_arr[jj])
                        if dist > max_dist:
                            max_dist = dist
                range_param.append(max_dist / 3.0)
        else:
            X_arr = np.asarray(X_obs).flatten()
            Y_arr = np.asarray(Y_obs).flatten()
            n_obs = len(X_arr)
            max_dist = 0.0
            for i in range(n_obs):
                for j in range(i+1, n_obs):
                    dist = np.hypot(X_arr[i] - X_arr[j], Y_arr[i] - Y_arr[j])
                    if dist > max_dist:
                        max_dist = dist
            range_param = max_dist / 3.0
    
    print("=" * 80)
    print("BAYESIAN INFERENCE CONFIGURATION")
    print("=" * 80)
    
    print(f"\nNoise Model Parameters:")
    if isinstance(sill, (list, tuple, np.ndarray)):
        print(f"  Sill:         {sill}")
        print(f"  Nugget:       {nugget}")
        print(f"  Range:        {range_param}")
    else:
        print(f"  Sill:         {sill:.6f}")
        print(f"  Nugget:       {nugget:.6f}")
        print(f"  Range:        {range_param:.3f}")
    
    # Print initial parameters/prior/proposal in a way that supports single- or multi-model inputs
    if isinstance(model_type, list):
        for idx, m in enumerate(model_type):
            print(f"\nModel [{idx+1}] - {m} - Initial Parameters:")
            for k, v in (initial_params[idx] if isinstance(initial_params, list) else initial_params).items():
                print(f"  {k:10s}: {v:8.4f}")
            print(f"\nModel [{idx+1}] - {m} - Prior Bounds:")
            for k, (lower, upper) in (priors[idx] if isinstance(priors, list) else priors).items():
                print(f"  {k:10s}: [{lower:8.4f}, {upper:8.4f}]")
            print(f"\nModel [{idx+1}] - {m} - Initial Proposal Std:")
            for k, v in (proposal_std[idx] if isinstance(proposal_std, list) else proposal_std).items():
                print(f"  {k:10s}: {v:8.6f}")
            print(f"\nModel [{idx+1}] - {m} - Max Step Sizes:")
            for k, v in (max_step_sizes[idx] if isinstance(max_step_sizes, list) else max_step_sizes).items():
                print(f"  {k:10s}: {v:8.4f}")
    else:
        print(f"\nInitial Parameters:")
        for key, val in initial_params.items():
            print(f"  {key:10s}: {val:8.4f}")
        print(f"\nPrior Bounds:")
        for key, (lower, upper) in priors.items():
            print(f"  {key:10s}: [{lower:8.4f}, {upper:8.4f}]")
        print(f"\nInitial Proposal Standard Deviations (Learning Rates):")
        for key, val in proposal_std.items():
            print(f"  {key:10s}: {val:8.6f}")
        print(f"\nMaximum Step Sizes:")
        for key, val in max_step_sizes.items():
            print(f"  {key:10s}: {val:8.4f}")
    
    print(f"\nAdaptive MCMC Settings:")
    print(f"  Adaptation interval:   {adaptive_interval:5d} iterations")
    print(f"  Target acceptance:     {target_acceptance:.3f}")
    print(f"  Total iterations:      {n_iterations:5d}")
    
    print("=" * 80)

    # Resolve burn_in here so downstream functions all see the same value
    _burn_in = int(n_iterations * 0.2) if burn_in is None else max(0, min(int(burn_in), int(n_iterations) - 1))

    # Store all inference parameters for saving
    inference_params = {
        'n_iterations': n_iterations,
        'sill': sill,
        'nugget': nugget,
        'range_param': range_param,
        'initial_params': initial_params,
        'priors': priors,
        'proposal_std': proposal_std,
        'max_step_sizes': max_step_sizes,
        'adaptive_interval': adaptive_interval,
        'target_acceptance': target_acceptance,
        'adapt_method': adapt_method,
        'use_sa_init': use_sa_init,
        'model_type': model_type,
        'burn_in': _burn_in
    }

    # Run inference with custom parameters
    (samples, log_lik_trace, rms_evolution,
     proposal_std_evolution, acceptance_rate_evolution,
     proposal_std_evolution_iters) = bayesian_inference_pCDM_with_noise(
        u_los_obs, X_obs, Y_obs, incidence_angle, heading, 
        n_iterations=int(n_iterations),
        sill=sill, nugget=nugget, range_param=range_param,
        adapt_method=adapt_method,
        initial_params=initial_params,
        priors=priors,
        proposal_std=proposal_std,
        max_step_sizes=max_step_sizes,
        adaptive_interval=adaptive_interval,
        target_acceptance=target_acceptance,
        use_sa_init=use_sa_init,
        figure_folder=figure_folder,model_type=model_type,
        burn_in=_burn_in)    
    # Save complete inference state to pickle
    pickle_filename = save_inference_state(
        samples=samples,
        log_lik_trace=log_lik_trace,
        rms_evolution=rms_evolution,
        proposal_std_evolution=proposal_std_evolution,
        acceptance_rate_evolution=acceptance_rate_evolution,
        X_obs=X_obs,
        Y_obs=Y_obs,
        u_los_obs=u_los_obs,
        incidence_angle=incidence_angle,
        heading=heading,
        inference_params=inference_params,
        figure_folder=figure_folder,
        model_type=model_type
    )
    
    # Plot results
    pCDM_BI_plotting_funcs.plot_inference_results(samples, log_lik_trace, rms_evolution, burn_in=_burn_in, 
                          X_obs=X_obs, Y_obs=Y_obs, u_los_obs=u_los_obs, 
                          incidence_angle=incidence_angle, heading=heading, figure_folder=figure_folder,
                          proposal_std_evolution=proposal_std_evolution,
                          proposal_std_evolution_iters=proposal_std_evolution_iters,
                          acceptance_rate_evolution=acceptance_rate_evolution,
                          adaptive_interval=adaptive_interval, target_acceptance=target_acceptance, model_type=model_type)
    
    return samples, log_lik_trace, rms_evolution, pickle_filename

def synthetic_test_okada():
    # Default parameters
    # gc.collect()
    default_initial = {
        'X0': 0,
        'Y0': 0,
        'depth': 5000,
        'length': 10000,
        'width': 8000,
        'strike': 100,
        'dip': 30,
        'rake': 90,
        'slip': 2,
        'opening': 0
    }
    
    default_priors = {
        'X0': (-15000,15000),
        'Y0': (-15000, 15000),
        'depth': (100, 35000),
        'length': (1000, 20000),
        'width': (1000, 20000),
        'strike': (0, 360),
        'dip': (0, 90),
        'rake': (-180, 180),
        'slip': (-10, 10),
        'opening': (0, 0)
    }
    
    default_learning_rates = {
        'X0': 10000,
        'Y0': 10000,
        'depth': 1000,
        'length': 1000,
        'width': 1000,
        'strike': 10,
        'dip': 10,
        'rake': 10,
        'slip': 0.1,
        'opening': 0.0
    }

    max_step_sizes = {
            'X0': 100000.0,
            'Y0': 100000.0,
            'depth': 10000.0,
            'length': 5000.0,
            'width': 5000.0,
            'strike': 20,
            'dip': 20,
            'rake': 20,
            'slip': 2,
            'opening': 0
        }
    
    # Generate synthetic data
    u_los_obs, X_obs, Y_obs, incidence_angle, heading, noise_sill, noise_nugget, noise_range = gen_synthetic_data(
        None, grid_size=40, noise_level=0.5, model_type='okada')
    
    print(f"Generated synthetic data shapes:")
    print(f"  u_los_obs: {u_los_obs.shape}")
    print(f"  X_obs: {X_obs.shape}")
    print(f"  Y_obs: {Y_obs.shape}")
    print(f"  incidence_angle: {incidence_angle}")
    print(f"  heading: {heading}")
    print(f"  noise parameters: sill={noise_sill:.6f}, nugget={noise_nugget:.6f}, range={noise_range:.3f}")
    # okada.clear_numba_cache()
    # Run Bayesian inference with synthetic data and default settings
    samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
        u_los_obs=u_los_obs,
        X_obs=X_obs,
        Y_obs=Y_obs,
        incidence_angle=incidence_angle,
        heading=heading,
        n_iterations=int(1e5),
        sill=noise_sill,
        nugget=noise_nugget,
        range_param=noise_range,
        initial_params=default_initial,
        priors=default_priors,
        proposal_std=default_learning_rates,
        max_step_sizes=max_step_sizes,
        adaptive_interval=1000,
        target_acceptance=0.23,
        figure_folder="figure_test_synth_okada",
        use_sa_init=True, model_type='okada',u_los_obs_already_los_in_m=True)
    



def synthetic_test_pcdm():
    # Default parameters
    default_initial = {
        'X0': 0,
        'Y0': 0,
        'depth': 5000,
        'DVx': 7e7,
        'DVy': 7e7,
        'DVz': 7e7,
        'omegaX': 0,
        'omegaY': 0,
        'omegaZ': 0
    }
    
    default_priors = {
        'X0': (-15000,15000),
        'Y0': (-15000, 15000),
        'depth': (100, 35000),
        'DVx': (1e2, 1e9),
        'DVy': (1e2, 1e9),
        'DVz': (1e2, 1e9),
        'omegaX': (-45, 45),
        'omegaY': (-45, 45),
        'omegaZ': (-45, 45)
    }
    
    default_learning_rates = {
        'X0': 100,
        'Y0': 100,
        'depth': 100,
        'DVx': 1e4,
        'DVy': 1e4,
        'DVz': 1e4,
        'omegaX': 1,
        'omegaY': 1,
        'omegaZ': 1
    }

    max_step_sizes = {
            'X0': 100000.0,
            'Y0': 100000.0,
            'depth': 10000.0,
            'DVx': 1e7,
            'DVy': 1e7,
            'DVz': 1e7,
            'omegaX': 20,
            'omegaY': 20,
            'omegaZ': 20
        }
    
    # Generate synthetic data
    u_los_obs, X_obs, Y_obs, incidence_angle, heading, noise_sill, noise_nugget, noise_range = gen_synthetic_data(
        true_params=None, grid_size=50, noise_level=0.05,)
    
    print(f"Generated synthetic data shapes:")
    print(f"  u_los_obs: {u_los_obs.shape}")
    print(f"  X_obs: {X_obs.shape}")
    print(f"  Y_obs: {Y_obs.shape}")
    print(f"  incidence_angle: {incidence_angle}")
    print(f"  heading: {heading}")
    print(f"  noise parameters: sill={noise_sill:.6f}, nugget={noise_nugget:.6f}, range={noise_range:.3f}")

    # Run Bayesian inference with synthetic data and default settings
    samples, log_lik_trace, rms_evolution,pickle_filename = run_baysian_inference(
        u_los_obs=u_los_obs,
        X_obs=X_obs,
        Y_obs=Y_obs,
        incidence_angle=incidence_angle,
        heading=heading,
        n_iterations=int(1e4),
        sill=noise_sill,
        nugget=noise_nugget,
        range_param=noise_range,
        initial_params=default_initial,
        priors=default_priors,
        proposal_std=default_learning_rates,
        max_step_sizes=max_step_sizes,
        adaptive_interval=1000,
        target_acceptance=0.23,
        figure_folder="figure_test_synth",
        use_sa_init=True, model_type='pCDM',u_los_obs_already_los_in_m=True)


def synthetic_test_pcdm_okada():
    """Synthetic test and joint inversion for pCDM + Okada.

    - Generates a stacked signal (pcdm + okada) on a regular grid
    - Adds spatially-correlated noise
    - Runs joint MCMC inversion with model_type=['pCDM','okada']
    """
    # True parameters for pCDM component
    true_pcdm = {
        'X0': 0.0, 'Y0': 0.0, 'depth': 8000.0,
        'DVx': 4e7, 'DVy': 2e7, 'DVz': 6e7,
        'omegaX': 5.0, 'omegaY': -10.0, 'omegaZ': 2.0
    }

    # True parameters for Okada component
    true_okada = {
        'X0': 1500.0, 'Y0': -1200.0, 'depth': 3000.0,
        'length': 12000.0, 'width': 6000.0,
        'strike': 110.0, 'dip': 30.0, 'rake': 90.0,
        'slip': 1.5, 'opening': 0.0
    }

    # Grid
    grid_size = 40
    x_range = np.linspace(-20000, 20000, grid_size)
    y_range = np.linspace(-20000, 20000, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Forward-model each component
    ue_p, un_p, uv_p = pCDM_fast.pCDM(X_flat, Y_flat,
                                      true_pcdm['X0'], true_pcdm['Y0'], true_pcdm['depth'],
                                      true_pcdm['omegaX'], true_pcdm['omegaY'], true_pcdm['omegaZ'],
                                      true_pcdm['DVx'], true_pcdm['DVy'], true_pcdm['DVz'], 0.25)

    ue_o, un_o, uv_o = okada.disloc3d3(X_flat, Y_flat,
                                      true_okada['X0'], true_okada['Y0'], true_okada['depth'],
                                      true_okada['length'], true_okada['width'], true_okada['slip'], true_okada['opening'],
                                      true_okada['strike'], true_okada['dip'], true_okada['rake'], 0.25)

    # Sum contributions
    ue_true = ue_p + ue_o
    un_true = un_p + un_o
    uv_true = uv_p + uv_o

    # LOS geometry (use same as gen_synthetic_data)
    incidence_angle = 39.4
    heading = -169.9
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)

    u_los_true = -((ue_true * los_e) + (un_true * los_n) + (uv_true * los_u))
    u_los_true = u_los_true.flatten()

    # Add spatially correlated noise
    noise_level = 0.3
    noise_std = np.std(u_los_true) * noise_level
    n_obs = len(u_los_true)
    distances = np.sqrt((X_flat.reshape(-1,1) - X_flat.reshape(1,-1))**2 + (Y_flat.reshape(-1,1) - Y_flat.reshape(1,-1))**2)
    noise_sill = (noise_std)**2
    noise_nugget = noise_sill * 0.01
    noise_range = np.max(distances) / 4.0
    C_noise = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            if i == j:
                C_noise[i,j] = noise_sill
            else:
                C_noise[i,j] = (noise_sill - noise_nugget) * np.exp(-distances[i,j] / noise_range)
    spatially_correlated_noise = np.random.multivariate_normal(np.zeros(n_obs), C_noise)

    u_los_obs = u_los_true + spatially_correlated_noise

    # Plot synthetic observed field (quick check)
    plt.figure(figsize=(6,5))
    plt.tricontourf(X_flat, Y_flat, u_los_obs, levels=40, cmap='RdBu_r')
    plt.colorbar(label='LOS displacement (m)')
    plt.title('Synthetic observed LOS (pcdm + okada + noise)')
    plt.savefig('synthetic_data_pcdm_okada.png', dpi=200)
    plt.show()
    # plt.close()

    print('Synthetic joint dataset created')
    print(f'  points: {n_obs}, noise std (m): {noise_std:.4e}')

    # Set up MCMC defaults for both models
    pcdm_initial = {
        'X0': 0.0, 'Y0': 0.0, 'depth': 6000.0,
        'DVx': 3e7, 'DVy': 3e7, 'DVz': 3e7,
        'omegaX': 0.0, 'omegaY': 0.0, 'omegaZ': 0.0
    }
    pcdm_priors = {
        'X0': (-20000,20000), 'Y0': (-20000,20000), 'depth': (100,40000),
        'DVx': (1e4, 1e9), 'DVy': (1e4, 1e9), 'DVz': (1e4, 1e9),
        'omegaX': (-45,45), 'omegaY': (-45,45), 'omegaZ': (-45,45)
    }
    pcdm_prop = {
        'X0': 500.0, 'Y0': 500.0, 'depth': 500.0,
        'DVx': 1e5, 'DVy': 1e5, 'DVz': 1e5,
        'omegaX': 1.0, 'omegaY': 1.0, 'omegaZ': 1.0
    }
    pcdm_max = {
        'X0': 5000.0, 'Y0': 5000.0, 'depth': 20000.0,
        'DVx': 1e8, 'DVy': 1e8, 'DVz': 1e8,
        'omegaX': 30.0, 'omegaY': 30.0, 'omegaZ': 30.0
    }

    okada_initial = {
        'X0': 1500.0, 'Y0': -1200.0, 'depth': 3000.0,
        'length': 10000.0, 'width': 5000.0, 'strike': 110.0, 'dip': 30.0, 'rake': 90.0, 'slip': 1.0, 'opening': 0.0
    }
    okada_priors = {
        'X0': (-30000,30000), 'Y0': (-30000,30000), 'depth': (100,20000),
        'length': (1000,30000), 'width': (1000,20000), 'strike': (0,360), 'dip': (0,90), 'rake': (-180,180), 'slip': (-10,10), 'opening': (-100,100)
    }
    okada_prop = {
        'X0': 500.0, 'Y0': 500.0, 'depth': 500.0,
        'length': 500.0, 'width': 500.0, 'strike': 1.0, 'dip': 1.0, 'rake': 1.0, 'slip': 0.1, 'opening': 0.5
    }
    okada_max = {
        'X0': 50000.0, 'Y0': 50000.0, 'depth': 20000.0,
        'length': 50000.0, 'width': 30000.0, 'strike': 360, 'dip': 90, 'rake': 180, 'slip': 10, 'opening': 100
    }

    # Run joint Bayesian inference (pcdm + okada)
    samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
        u_los_obs=u_los_obs,
        X_obs=X_flat,
        Y_obs=Y_flat,
        incidence_angle=incidence_angle,
        heading=heading,
        n_iterations=int(1e4),
        sill=noise_sill,
        nugget=noise_nugget,
        range_param=noise_range,
        initial_params=[pcdm_initial, okada_initial],
        priors=[pcdm_priors, okada_priors],
        proposal_std=[pcdm_prop, okada_prop],
        max_step_sizes=[pcdm_max, okada_max],
        adaptive_interval=1000,
        target_acceptance=0.23,
        figure_folder='figure_test_joint_pcdm_okada',
        use_sa_init=True,
        model_type=['pCDM','okada'],
        u_los_obs_already_los_in_m=True)

    return samples, log_lik_trace, rms_evolution, pickle_filename


def synthetic_test_multi_ifg(n_ifgs=2, grid_size=50, noise_level=0.30, n_iterations=1e4, use_sa_init=True):
    """Synthetic test that verifies multi-IFG (Option A) handling.

    - Builds a single true displacement field (pCDM) on a regular grid
    - Creates `n_ifgs` LOS observations by changing incidence/heading
    - Adds independent spatially-correlated noise per IFG
    - Runs a short MCMC using concatenated IFGs + block-diagonal covariance
    The function is intentionally short/fast so it can be used as a smoke test.
    """
    # True pCDM parameters (compact test case)
    true_pcdm = {
        'X0': 0.0, 'Y0': 0.0, 'depth': 8000.0,
        'DVx': 3e7, 'DVy': 2e7, 'DVz': 4e7,
        'omegaX': 5.0, 'omegaY': -10.0, 'omegaZ': 2.0
    }

    # Create observation grid (small for quick test)
    x_range = np.linspace(-25000, 25000, grid_size)
    y_range = np.linspace(-25000, 25000, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_obs = len(X_flat)

    # True displacement field (pCDM)
    ue_true, un_true, uv_true = pCDM_fast.pCDM(X_flat, Y_flat,
                                              true_pcdm['X0'], true_pcdm['Y0'], true_pcdm['depth'],
                                              true_pcdm['omegaX'], true_pcdm['omegaY'], true_pcdm['omegaZ'],
                                              true_pcdm['DVx'], true_pcdm['DVy'], true_pcdm['DVz'], 0.25)

    # Define simple LOS geometries for multiple IFGs (can be expanded)
    if n_ifgs == 1:
        inc_list = [39.4]
        head_list = [-169.9]
    else:
        inc_list = [39.4, 35.0] + [39.4] * max(0, n_ifgs-2)
        head_list = [-169.9, -10.0] + [-169.9] * max(0, n_ifgs-2)
        inc_list = inc_list[:n_ifgs]
        head_list = head_list[:n_ifgs]

    for inc, head in zip(inc_list, head_list):
        incr, headr = np.radians(inc), np.radians(head)
        print(inc, head, np.sin(incr)*np.cos(headr), -np.sin(incr)*np.sin(headr), -np.cos(incr))

    # Per-IFG noisy LOS observations
    u_list = []
    sill_list = []
    nugget_list = []
    range_list = []

    # Precompute distances for variogram range estimate
    coords = np.column_stack((X_flat, Y_flat))
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
    max_dist = np.max(dists)

    for inc_a, head_a in zip(inc_list, head_list):
        inc_rad = np.radians(inc_a)
        head_rad = np.radians(head_a)
        los_e = np.sin(inc_rad) * np.cos(head_rad)
        los_n = -np.sin(inc_rad) * np.sin(head_rad)
        los_u = -np.cos(inc_rad)

        # LOS projection of the same true field
        u_los_true = -((ue_true * los_e) + (un_true * los_n) + (uv_true * los_u))

        # Noise model (spatially correlated)
        noise_std = np.std(u_los_true) * noise_level
        noise_sill = noise_std**2
        noise_nugget = noise_sill * 0.01
        noise_range = max_dist / 4.0

        # Build covariance and draw noise
        C_noise, _, _, _ = estimate_noise_covariance(X_flat, Y_flat, u_los_true, sill=noise_sill, nugget=noise_nugget, range_param=noise_range)
        try:
            noise = np.random.multivariate_normal(np.zeros(n_obs), C_noise)
        except np.linalg.LinAlgError:
            # fall back to white noise if covariance numerically unstable
            noise = np.random.normal(0, noise_std, size=n_obs)

        u_obs = u_los_true + noise
        u_list.append(u_obs)
        sill_list.append(noise_sill)
        nugget_list.append(noise_nugget)
        range_list.append(noise_range)

    print(f"Synthetic multi-IFG: {n_ifgs} IFGs, grid {grid_size}×{grid_size}, total points per IFG: {n_obs}")

    # Quick plotting of the first IFG (visual sanity check) — save instead of interactive show
    plt.figure(figsize=(6,4))
    plt.tricontourf(X_flat, Y_flat, u_list[0], levels=30, cmap='RdBu_r')
    plt.colorbar(label='LOS (m)')
    plt.title('Synthetic IFG #1 (example)')
    plt.tight_layout()
    os.makedirs('figure_test_multi_ifg', exist_ok=True)
    plt.savefig('figure_test_multi_ifg/synthetic_IFG1.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Prepare MCMC priors / initial / proposal for single pCDM model (fast test)
    pcdm_initial = {
        'X0': 0.0, 'Y0': 0.0, 'depth': 6000.0,
        'DVx': 2e7, 'DVy': 2e7, 'DVz': 2e7,
        'omegaX': 0.0, 'omegaY': 0.0, 'omegaZ': 0.0
    }
    pcdm_priors = {
        'X0': (-20000,20000), 'Y0': (-20000,20000), 'depth': (100,40000),
        'DVx': (1e4, 1e9), 'DVy': (1e4, 1e9), 'DVz': (1e4, 1e9),
        'omegaX': (-45,45), 'omegaY': (-45,45), 'omegaZ': (-45,45)
    }
    pcdm_prop = {
        'X0': 500.0, 'Y0': 500.0, 'depth': 500.0,
        'DVx': 1e5, 'DVy': 1e5, 'DVz': 1e5,
        'omegaX': 1.0, 'omegaY': 1.0, 'omegaZ': 1.0
    }
    pcdm_max = {k: max(1.0, abs(v)*100.0) for k, v in pcdm_prop.items()}

    # Run the short MCMC (multi-IFG inputs are lists)
    samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
        u_los_obs=u_list,
        X_obs=[X_flat]*n_ifgs,
        Y_obs=[Y_flat]*n_ifgs,
        incidence_angle=inc_list,
        heading=head_list,
        n_iterations=int(n_iterations),
        sill=sill_list,
        nugget=nugget_list,
        range_param=range_list,
        initial_params=pcdm_initial,
        priors=pcdm_priors,
        proposal_std=pcdm_prop,
        max_step_sizes=pcdm_max,
        adaptive_interval=int(n_iterations//100),
        target_acceptance=0.23,
        figure_folder='figure_test_multi_ifg',
        use_sa_init=use_sa_init,
        model_type='pCDM',
        adapt_method='standard',  # use robust adaptation for this test
        u_los_obs_already_los_in_m=True
    )

    # Print simple recovery diagnostics
    recovered = {p: np.mean(samples[p]) for p in samples.keys()}
    print('\nMulti-IFG synthetic test - Recovery (posterior mean):')
    for k, v in recovered.items():
        true_v = true_pcdm.get(k, None)
        if true_v is not None:
            print(f"  {k:8s}: mean={v:.4e}, true={true_v:.4e}, err={(v-true_v):+.4e}")
        else:
            print(f"  {k:8s}: mean={v:.4e}")

    return samples, log_lik_trace, rms_evolution, pickle_filename


if __name__ == "__main__":
    # synthetic_test_okada()
    # synthetic_test_pcdm()
    # synthetic_test_pcdm_okada()
    # synthetic_test_multi_ifg()  # uncomment to run a quick multi-IFG smoke test
    # Example with custom parameters
    custom_initial = {
        'X0': 0,
        'Y0': 0,
        'depth': 620 ,
        'DVx': -1e5,
        'DVy': -1e5,
        'DVz': -1e5,
        'omegaX': 0,
        'omegaY': 0,
        'omegaZ': 0
    }
    #Input Priors here
    custom_priors = {
        'X0': (-1500,1500),
        'Y0': (-1500, 1500),
        'depth': (100, 35000),
        'DVx': (-1e8, -1e1),
        'DVy': (-1e8, -1e1),
        'DVz': (-1e8, -1e1),
        'omegaX': (-45, 45),
        'omegaY': (-45, 45),
        'omegaZ': (-45, 45)
    }
    # Intial Learning Rates Here
    custom_learning_rates = {
        'X0': 10,
        'Y0': 10,
        'depth': 10,
        'DVx': 1e2 ,
        'DVy': 1e2,
        'DVz': 1e2,
        'omegaX': 1,
        'omegaY': 1,
        'omegaZ': 1
    }
    # Max Step Sizes Here
    max_step_sizes = {
            'X0': 1000.0,
            'Y0': 1000.0,
            'depth': 1000.0,
            'DVx': 1e4,
            'DVy': 1e4,
            'DVz': 1e4,
            'omegaX': 30,
            'omegaY': 30,
            'omegaZ': 30
        }
    
    ##################################################################

    # ##### Values to Edit for Okada ######

    custom_initial_o = {
        'X0': 0,
        'Y0': 0,
        'depth': 5000,
        'length': 10000,
        'width': 8000,
        'strike': 0,
        'dip': 30,
        'rake': 90,
        'slip': 1.0,
        'opening': 0
    }
    #Input Priors here
    custom_priors_o = {
        'X0': (-1500,1500),
        'Y0': (-1500, 1500),
        'depth': (100, 35000),
        'length': (1000, 50000),
        'width': (1000, 30000),
        'strike': (0, 360),
        'dip': (0, 90),
        'rake': (-180, 180),
        'slip': (-10.0, 10.0),
        'opening': (0,0)
    }
    # Intial Learning Rates Here
    custom_learning_rates_o = {
        'X0': 100,
        'Y0': 100,
        'depth': 100,
        'length': 100,
        'width': 100,
        'strike': 1,
        'dip': 1,
        'rake': 1,
        'slip': 0.1,
        'opening': 0.0
    }
    # Max Step Sizes Here
    max_step_sizes_o = {
            'X0': 100000.0,
            'Y0': 100000.0,
            'depth': 10000.0,
            'length': 10000.0,
            'width': 8000.0,
            'strike': 20,
            'dip': 20,
            'rake': 20,
            'slip': 2.0,
            'opening': 0.0
        }
    
    ##################################################################




  
    

    custom_initial_nt = {
        'X0': 0,
        'Y0': 0,
        'depth': 620 ,
        'yield_kt': 200,
        'dv_factor': 0.1,
        'chimney_amp': 0.15,
        'compact_amp': 0.05,
    }

    custom_priors_nt = {
        'X0': (-1500,1500),
        'Y0': (-1500, 1500),
        'depth': (100, 35000),
        'yield_kt': (50, 500),
        'dv_factor': (-1, 1.0),
        'chimney_amp': (0.01, 1.0),
        'compact_amp': (0.01, 1.0),
    }

    custom_learning_rates_nt = {
        'X0': 10,
        'Y0': 10,
        'depth': 10,
        'yield_kt': 10,
        'dv_factor': 0.01,
        'chimney_amp': 0.01,
        'compact_amp': 0.01,
    }

    max_step_sizes_nt = {
            'X0': 1000.0,
            'Y0': 1000.0,
            'depth': 1000.0,
            'yield_kt': 50,
            'dv_factor': 0.1,
            'chimney_amp': 0.1,
            'compact_amp': 0.1,
        }






    ############################## EDIT IN YOUR OWN DATA HERE ##############################
    # # Load data from .npy file
  
    data = np.load('./divider/19920424_19930305.diff.unw.geo_downsampled.npy', allow_pickle=True)
    data_dict = data.item()
    nugget = 1.000000e-10
    sill = 2.388520e-09
    range_param = 3635.4

    # #divider 
    # data = np.load('./divider/19920424_19930305.diff.unw.geo_downsampled.npy', allow_pickle=True)
    # data_dict = data.item()
    # nugget = 2.237931e-05
    # sill=2.868079e-05
    # range_param = 3676.1


    # nugget =7.475808e-06
    # sill=5.775924e-05
    # range_param = 44960.6
    number_of_iterations = int(12e6)
    model_type=['une','okada']  # Change to 'pCDM' or 'okada' as needed case insensitive
    # model_type = ['une']
    use_simulated_annealing_for_first_guess = True
    # referencePoint = [ 37.27246,-116.35976] # add manually if you would like # junction 
    referencePoint = [37.02068,-115.98791] # divider 

    # use this if step01 was used to generate data 
    # referencePoint = [data_dict['center_lat'], data_dict['center_lon']]
    print(f"Using reference point: {referencePoint}")
    ###################################################################

    # gen_synthetic_data(custom_initial, grid_size=50, noise_level=0.05, model_type=model_type)

    # Example (joint inversion) - **commented out** small-run usage:
    # pcdm_init = custom_initial
    # okada_init = {
    #     'X0': 0,'Y0':0,'depth':5000,'length':10000,'width':8000,'strike':100,'dip':30,'rake':90,'slip':1,'opening':0
    # }
    # run_baysian_inference(
    #     u_los_obs=u_los_obs, X_obs=X_obs, Y_obs=Y_obs,
    #     incidence_angle=incidence_angle, heading=heading,
    #     n_iterations=200,  # short smoke test
    #     initial_params=[pcdm_init, okada_init],
    #     priors=[custom_priors, default_priors],
    #     proposal_std=[custom_learning_rates, default_learning_rates],
    #     max_step_sizes=[max_step_sizes, max_step_sizes],
    #     model_type=['pCDM','okada'],
    #     figure_folder='figure_test_joint_smoke', use_sa_init=False)

    # Extract data from the loaded object
    print(data_dict.keys())
    u_los_obs = np.array(data_dict['Phase'].T).flatten() 
   
    u_los_obs = -u_los_obs/0.0555*4*np.pi  # Convert displacement to phase
    Lon = np.array(data_dict['Lon']).flatten()
    Lat = np.array(data_dict['Lat']).flatten()
   
    X_obs, Y_obs = convert_lat_long_2_xy(Lat, Lon, referencePoint[0], referencePoint[1])
    print(f"Converted Lon/Lat to X/Y with reference point {referencePoint}")
    print(f"X_obs range: {X_obs.min():.3f} to {X_obs.max():.3f}")
    print(f"Y_obs range: {Y_obs.min():.3f} to {Y_obs.max():.3f}")
    print(len(X_obs), len(Y_obs), len(u_los_obs))
    incidence_angle = np.array(data_dict['Inc']).flatten() * (180/np.pi)  # Convert to deg
    heading = np.array(data_dict['Heading']).flatten() * (180/np.pi)  # Convert to deg

    incidence_angle = np.full_like(np.array(data_dict['Inc']).flatten(), 20.5800)
    heading = np.full_like(np.array(data_dict['Heading']).flatten(), 192.47)

    print(f"Loaded data shapes:")
    print(f"  u_los_obs: {u_los_obs.shape}")
    print(f"  X_obs: {X_obs.shape}")
    print(f"  Y_obs: {Y_obs.shape}")
    print(f"  incidence_angle: {incidence_angle}")
    print(f"  heading: {heading}")
  

    # Run Bayesian inference with synthetic data and custom settings
    samples, log_lik_trace, rms_evolution,pickle_filename = run_baysian_inference(
        u_los_obs=u_los_obs, 
        X_obs=X_obs, 
        Y_obs=Y_obs, 
        incidence_angle=incidence_angle, 
        heading=heading,
        n_iterations=number_of_iterations,
        sill=sill,
        nugget=nugget,
        range_param=range_param,
        initial_params=[custom_initial_nt, custom_initial_o],
        priors=[custom_priors_nt, custom_priors_o],
        proposal_std=[custom_learning_rates_nt, custom_learning_rates_o],
        max_step_sizes=[max_step_sizes_nt, max_step_sizes_o],
        adaptive_interval=number_of_iterations//100,  # auto-tuned inside if left at default 1000
        target_acceptance=0.23,
        figure_folder="Test_Results",
        use_sa_init=use_simulated_annealing_for_first_guess,
        model_type=model_type,  # Change to 'pCDM' or 'okada' as needed
        burn_in=int(number_of_iterations*0.3)

    )
    
  
# # #   #### Reload old inference state and regenerate plots example ####
#     pickle_file = "figure_test_junction_1e6/bayesian_inference_state_n1000000_accept0.23_pCDM.pkl"
#     regenerate_plots_from_state(pickle_file, new_figure_folder="regenerated_plots_test")