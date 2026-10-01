import sys
sys.dont_write_bytecode = True        # avoid stale .pyc on network filesystems

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.interpolate import griddata
import pCDM_model as pCDM_fast

import matplotlib.pyplot as plt
import os 
import pandas as pd
import llh2local as llh
import local2llh as l2llh
import scipy.io as sio
import misc_plotting_functions_multimodel as pCDM_BI_plotting_funcs
import misc_simulated_annealing_multimodel as pCDM_BI_simulated_annealing
import okada_model as okada
import pickle
from datetime import datetime
import gc
import UNE_three_component_fast as UNE_three
from model_registry import MODEL_REGISTRY, get_param_names, forward_from_registry

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
                                            adaptive_interval=1000, target_acceptance=0.234,
                                            adapt_method='conservative',
                                            initial_params=None, priors=None, max_step_sizes=None,
                                            use_sa_init=True, figure_folder=None, model_type='pCDM',
                                            burn_in=None, log_space_params=None,
                                            fit_ramp=False):
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

            # Estimate per-IFG covariance matrices; keep blocks separate to avoid
            # computing zero off-diagonal elements in the full block-diagonal matrix.
            C_inv_blocks = []
            n_per_ifg = []
            C_logdet = 0.0
            for ui, Xi, Yi, s_i, n_i, r_i in zip(u_list, X_list, Y_list, sill_list, nugget_list, range_list):
                Ci, s_est, n_est, r_est = estimate_noise_covariance(Xi, Yi, ui, s_i, n_i, r_i)
                ni = len(Ci)
                Ci += np.eye(ni) * 1e-8 * np.trace(Ci) / ni
                C_inv_blocks.append(np.linalg.inv(Ci))
                C_logdet += np.linalg.slogdet(Ci)[1]
                n_per_ifg.append(ni)
            # Alias so single-IFG log_likelihood path can still use C_inv
            C_inv = C_inv_blocks[0] if len(C_inv_blocks) == 1 else None

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

            # Per-IFG coordinate arrays needed for ramp evaluation
            _ramp_X_list = X_list
            _ramp_Y_list = Y_list
            _n_ramps = n_ifgs

        else:
            # Single IFG (existing behaviour)
            u_los_obs = np.array(u_los_obs).flatten()
            X_obs = np.array(X_obs).flatten()
            Y_obs = np.array(Y_obs).flatten()

            # Estimate noise covariance matrix
            C, sill, nugget, range_param = estimate_noise_covariance(X_obs, Y_obs, u_los_obs, sill, nugget, range_param)

            # Compute inverse and determinant for likelihood calculation.
            # Always regularise: near-singular matrices don't raise LinAlgError
            # but produce numerically garbage inverses.
            C += np.eye(len(C)) * 1e-8 * np.trace(C) / len(C)
            C_inv = np.linalg.inv(C)
            C_logdet = np.linalg.slogdet(C)[1]
            C_inv_blocks = [C_inv]
            n_per_ifg = [len(C_inv)]

            # Per-IFG coordinate arrays needed for ramp evaluation (single IFG = one block)
            _ramp_X_list = [X_obs]
            _ramp_Y_list = [Y_obs]
            _n_ramps = 1

            # Convert angles to radians for LOS calculation
            inc_rad = np.radians(incidence_angle)
            head_rad = np.radians(heading)

            # Line-of-sight unit vector components (scalars)
            los_e = np.sin(inc_rad) * np.cos(head_rad)
            los_n = -np.sin(inc_rad) * np.sin(head_rad)
            los_u = -np.cos(inc_rad)

        # Define forward models
        # For single-model mode, _cached_fwd_single is the direct function reference
        # to avoid isinstance/dict-lookup overhead every iteration.
        _cached_fwd_single = None  # set after model_list is ready

        def forward_model(params, model_type):
            """
            Generic forward model dispatcher.
            For single-model (the common hot path), uses _cached_fwd_single directly.
            For joint inversion (list of models), sums contributions.
            """
            if isinstance(model_type, list):
                ue = np.zeros_like(u_los_obs)
                un = np.zeros_like(u_los_obs)
                uv = np.zeros_like(u_los_obs)
                for i, model in enumerate(model_type):
                    ue_i, un_i, uv_i = _fwd_cache[model](
                        X_obs, Y_obs, params[i])
                    ue, un, uv = ue + ue_i, un + un_i, uv + uv_i
            else:
                ue, un, uv = _cached_fwd_single(
                    X_obs, Y_obs, params)
            return ue, un, uv

     
        
        # Initialize adaptive proposal tracking (supports single- or multi-model inputs)
        # Normalize model inputs so MCMC operates on a flat parameter dict while
        # forward_model is given structured per-model dicts when required.
        if isinstance(model_type, list):
            model_list = [m.lower() for m in model_type]
        else:
            model_list = [model_type.lower()]

        # Cache forward AND prior function references once (avoids dict lookup per iteration)
        _fwd_cache = {m: MODEL_REGISTRY[m]['forward'] for m in model_list}
        _prior_cache = {m: MODEL_REGISTRY[m]['prior'] for m in model_list
                        if m in MODEL_REGISTRY}

        num_models = len(model_list)
        single_model_mode = (num_models == 1)

        # Assign the cached single-model forward function now that _fwd_cache is ready
        if single_model_mode:
            _cached_fwd_single = _fwd_cache[model_list[0]]

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
                # Use .get() with fallback to handle missing keys in proposal_std_dict and max_step_dict
                flat_proposal_std[flat_key] = prop_std_dict.get(key, 0.01)  # Default 0.01 if missing
                flat_max_step_sizes[flat_key] = max_step_dict.get(key, 0.5)  # Default 0.5 if missing

        # ── Ramp / offset nuisance parameters ──────────────────────────────────
        # fit_ramp: False → no ramp; 'offset' → constant only; True/'linear' → a*X + b*Y + c
        # Each IFG gets its own ramp so that orbital/atmospheric long-wavelength
        # signals can be absorbed independently per acquisition pair.
        _fit_linear_ramp = fit_ramp in (True, 'linear')
        _fit_any_ramp    = bool(fit_ramp)
        if _fit_any_ramp:
            _u_std = max(float(np.std(u_los_obs)), 1e-9)
            _x_ptp = max(float(np.ptp(X_obs)), 1.0)
            _y_ptp = max(float(np.ptp(Y_obs)), 1.0)
            _grad_range = _u_std * 10.0 / max(_x_ptp, _y_ptp)  # max plausible gradient
            for _rj in range(_n_ramps):
                _sfx = f"_{_rj}" if _n_ramps > 1 else ""
                flat_initial[f"ramp_c{_sfx}"]        = 0.0
                flat_priors[f"ramp_c{_sfx}"]         = (-_u_std * 5, _u_std * 5)
                flat_proposal_std[f"ramp_c{_sfx}"]   = _u_std * 0.01
                flat_max_step_sizes[f"ramp_c{_sfx}"] = _u_std * 0.5
                if _fit_linear_ramp:
                    for _ax in ('a', 'b'):
                        flat_initial[f"ramp_{_ax}{_sfx}"]        = 0.0
                        flat_priors[f"ramp_{_ax}{_sfx}"]         = (-_grad_range, _grad_range)
                        flat_proposal_std[f"ramp_{_ax}{_sfx}"]   = _grad_range * 0.01
                        flat_max_step_sizes[f"ramp_{_ax}{_sfx}"] = _grad_range * 0.5

        # Use flattened dicts for the adaptive MCMC bookkeeping.
        # IMPORTANT: only sample truly free parameters (non-zero prior span,
        # non-zero proposal std, non-zero max step). Fixed parameters remain
        # in current_params/samples but are excluded from AM scaling so they
        # cannot artificially shrink global proposal covariance.
        param_names = []
        fixed_param_names = []
        for p in flat_proposal_std.keys():
            lo, hi = flat_priors[p]
            prior_span = hi - lo
            if (prior_span <= 0.0) or (flat_proposal_std[p] <= 0.0) or (flat_max_step_sizes[p] <= 0.0):
                fixed_param_names.append(p)
            else:
                param_names.append(p)

        d = len(param_names)   # number of sampled parameters
        if d <= 0:
            raise ValueError("No free parameters to sample: all parameters appear fixed by priors/proposal/max_step settings.")
        param_idx = {p: j for j, p in enumerate(param_names)}

        # --- Log-space sampling setup ---
        # Parameters in log_space_set are sampled as log(θ) internally.
        # Priors must be strictly positive. SA still runs in linear space;
        # transformation happens after SA (see below).
        log_space_set = set()
        if log_space_params:
            for p in log_space_params:
                if p not in flat_priors:
                    print(f"  Warning: '{p}' in log_space_params not found in sampled parameters — skipping.")
                    continue
                lo, hi = flat_priors[p]
                if lo <= 0:
                    raise ValueError(
                        f"Cannot sample '{p}' in log space: lower prior bound ({lo}) must be > 0. "
                        f"Restrict the prior to positive values or remove '{p}' from log_space_params.")
                log_space_set.add(p)
        if log_space_set:
            print(f"  Log-space sampling enabled for: {sorted(log_space_set)}")

        # ---- GBIS sampler — Bagnardi & Hooper (2018) ----
        # Proposal: each parameter perturbed independently by a uniform draw
        #   θ_i* = θ_i + step_i * U(-1, 1)
        # Step sizes are adapted via a periodic sensitivity test that perturbs
        # each parameter one at a time and adjusts step_i to target ~23%
        # overall acceptance (77% rejection).  No cross-parameter covariance
        # is maintained.  This matches the GBIS MATLAB implementation exactly.

        # Per-parameter step sizes initialised from user-supplied proposal_std
        step_sizes = {p: float(flat_proposal_std[p]) for p in param_names}

        # Running prob_target — starts at 0.5^(1/d) and is updated multiplicatively
        # each sensitivity checkpoint: probTarget *= rejectionRatio / 0.77
        # This matches GBIS MATLAB (GBISrun.m line 197).
        # The first checkpoint skips the rejection-ratio update (iKeepSave == 0 guard).
        prob_target       = 0.5 ** (1.0 / max(d, 1))
        _sens_test_count  = 0           # how many sensitivity tests have fired

        # Sensitivity test state
        in_sensitivity_test = False
        sens_param_idx      = 0                 # which parameter is being tested
        prob_sens           = np.zeros(d)       # recorded single-step acceptance probs
        i_keep_at_last_sens = 0                 # iKeep snapshot at last sensitivity test
        i_reject_at_last_sens = 0              # iReject snapshot at last sensitivity test

        # Accept/reject counters (note: sensitivity-test iterations are not counted)
        n_accepted_window = 0
        n_proposed_window = 0
        n_accepted_total  = 0
        n_proposed_total  = 0
        n_accepted_burn   = 0

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

        # --- Apply log-space transformation (after SA, which runs in linear space) ---
        # current_params now holds linear values from SA or flat_initial.
        # Transform the internal representation and rebuild C_proposal.
        if log_space_set:
            for p in log_space_set:
                lo, hi = flat_priors[p]   # still linear here
                init_val = current_params[p]
                if init_val <= 0:
                    raise ValueError(
                        f"Log-space parameter '{p}' has non-positive initial value {init_val:.6g}. "
                        f"Check initial_params or SA output.")
                flat_priors[p] = (np.log(lo), np.log(hi))
                flat_proposal_std[p] = flat_proposal_std[p] / init_val
                flat_max_step_sizes[p] = flat_max_step_sizes[p] / init_val
                current_params[p] = np.log(init_val)
            # Rebuild GBIS step sizes with the updated (log-scale) proposal_std
            for p in param_names:
                step_sizes[p] = float(flat_proposal_std[p])

        def _to_model_params(internal_params):
            """Convert internal (possibly log-space) params to physical model params."""
            if not log_space_set:
                return internal_params
            model_p = internal_params.copy()
            for p in log_space_set:
                if p in model_p:
                    model_p[p] = np.exp(model_p[p])
            return model_p

        def _log_prior_single_model(pdict, mname):
            """Model-specific prior checks via cached prior functions."""
            fn = _prior_cache.get(mname)
            return fn(pdict) if fn is not None else 0.0

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
                if _log_prior_single_model(params, model_list[0]) == -np.inf:
                    return -np.inf
            else:
                for idx, label in enumerate(labels):
                    pdict = {p: params[f"{label}__{p}"] for p in param_structure[label]}
                    mname = model_list[idx]
                    if _log_prior_single_model(pdict, mname) == -np.inf:
                        return -np.inf

            # Jacobian correction for log-space parameters.
            # Sampling ψ = log(θ) with a flat prior on θ requires adding
            # log|dθ/dψ| = log(θ) = ψ for each log-space param so that the
            # MH ratio targets the correct flat-in-linear-space posterior.
            if log_space_set:
                return sum(params[p] for p in log_space_set if p in params)
            return 0.0
        
        # Pre-compute constant normalisation term for the Gaussian log-likelihood
        _ll_const = C_logdet + len(u_los_obs) * np.log(2 * np.pi)

        def log_likelihood(params):
            """Calculate log likelihood with correlated noise"""
            try:
                # Transform log-space parameters back to physical values for the forward model
                params = _to_model_params(params)
                # Forward model (handle flattened multi-model params)
                if single_model_mode:
                    ue, un, uv = _cached_fwd_single(X_obs, Y_obs, params)
                else:
                    params_list = []
                    for label in labels:
                        pdict = {p: params[f"{label}__{p}"] for p in param_structure[label]}
                        params_list.append(pdict)
                    ue, un, uv = forward_model(params_list, model_list)
                # Convert to line-of-sight
                u_los_pred = -((ue * los_e) + (un * los_n) + (uv * los_u))
                u_los_pred = u_los_pred.ravel()
                # Apply per-IFG ramp/offset correction (if enabled)
                if _fit_any_ramp:
                    _rs = 0
                    for _rj, (_Xj, _Yj, _nj) in enumerate(zip(_ramp_X_list, _ramp_Y_list, n_per_ifg)):
                        _sfx = f"_{_rj}" if _n_ramps > 1 else ""
                        _ramp = np.full(_nj, params[f"ramp_c{_sfx}"])
                        if _fit_linear_ramp:
                            _ramp = _ramp + params[f"ramp_a{_sfx}"] * _Xj + params[f"ramp_b{_sfx}"] * _Yj
                        u_los_pred[_rs:_rs + _nj] += _ramp
                        _rs += _nj
                # Calculate likelihood with correlated noise
                residuals = u_los_obs - u_los_pred
                rms_value = np.sqrt(np.mean(residuals**2))
                
                # Per-block quadratic form: avoids zero off-diagonal work in block-diagonal C_inv
                quad = 0.0
                start = 0
                for _Ci, _ni in zip(C_inv_blocks, n_per_ifg):
                    _ri = residuals[start:start + _ni]
                    quad += np.dot(_ri, _Ci @ _ri)
                    start += _ni
                log_lik = -0.5 * (quad + _ll_const)
                
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
        
        # Calculate burn-in point (used for adaptive scaling cutoff)
        if burn_in is None:
            burn_in = int(n_iterations * 0.2)
        else:
            burn_in = max(0, min(int(burn_in), int(n_iterations) - 1))

        # GBIS sensitivity schedule: dense early, sparse late (matches GBISrun.m line 282)
        #   [1:100:10000, 11000:1000:30000, 40000:10000:nRuns]
        sens_schedule = set(
            list(range(1, min(10001, n_iterations + 1), 100)) +
            list(range(11000, min(30001, n_iterations + 1), 1000)) +
            list(range(40000, n_iterations + 1, 10000))
        )

        print(f"  GBIS sampler (Bagnardi & Hooper 2018): diagonal uniform proposal, d={d}")
        print(f"  Step sizes adapted via GBIS sensitivity schedule ({len(sens_schedule)} checkpoints).")
        print(f"  Target rejection rate: 77%  (= 23% acceptance)")
        print(f"  Burn-in period: {burn_in} iterations.")
        
        # Initialize with flattened provided parameters
        # current_params = flat_initial.copy()
        
        # Storage for samples
        samples = {key: [] for key in current_params.keys()}
        log_likelihood_trace = []
        log_posterior_trace  = []
        residuals_evolution = []
        # Store proposal std and acceptance rate only at adaptation checkpoints
        # (one entry per adaptation event, not every iteration)
        proposal_std_evolution = {key: [] for key in param_names}
        proposal_std_evolution_iters = []          # iteration number of each checkpoint
        acceptance_rate_evolution = {key: [] for key in param_names}
        
        current_log_post, residual_rms, log_like_current = log_posterior(current_params)
        
        print(f"Starting MCMC sampling with {model_display_name} model and correlated noise...")
        print(f"Sampled parameters ({len(param_names)}): {list(param_names)}")
        if len(fixed_param_names) > 0:
            print(f"Fixed parameters ({len(fixed_param_names)}): {list(fixed_param_names)}")
        if isinstance(sill, (list, tuple, np.ndarray)):
            print(f"Noise parameters - Sill: {sill}")
            print(f"                Nugget: {nugget}")
            print(f"                Range:  {range_param}")
        else:
            print(f"Noise parameters - Sill: {sill:.6f}, Nugget: {nugget:.6f}, Range: {range_param:.3f}")
        print(f"Target rejection rate: 77.0% (target acceptance: 23.0%)")
        print(f"Sensitivity schedule: {len(sens_schedule)} checkpoints (GBIS-style)")
        print(f"Initial parameters:")
        for key, val in current_params.items():
            display_val = np.exp(val) if key in log_space_set else val
            suffix = " (log-sampled)" if key in log_space_set else ""
            print(f"  {key}: {display_val:.6f}{suffix}")
        print(f"Initial proposal standard deviations (learning rates):")
        for key, val in flat_proposal_std.items():
            print(f"  {key}: {val:.6f}")
        
        # DIAGNOSTIC: Print initial state quality
        print(f"\n*** INITIAL STATE DIAGNOSTICS ***")
        print(f"  Initial log-posterior: {current_log_post:.4f}")
        print(f"  Initial log-likelihood: {log_like_current:.4f}")
        print(f"  Initial RMS residual: {residual_rms:.6f}")
        if not np.isfinite(current_log_post):
            print(f"  ⚠️  FATAL: Initial log-posterior is {current_log_post}!")
            print(f"     Check: 1) Initial params within priors, 2) Forward model runs without error.")
            print(f"********************************\n")
            raise ValueError(
                f"Initial log-posterior is {current_log_post}. "
                f"Fix the starting parameters or priors before running MCMC — "
                f"a non-finite initial state means every proposal will be accepted or rejected "
                f"regardless of quality, making the entire chain meaningless.")
        print(f"********************************\n")

        # ---- Skip Hessian-based initialisation ----
        # Learning rates are set only from user-provided proposal_std values.
        print("Using user-provided proposal standard deviations (learning rates).\n")

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
        
        def _ess_1d(x, max_lag=None):
            """Effective sample size for a 1-D chain using Geyer's initial
            positive sequence estimator (Geyer 1992).  Robust against
            premature truncation and negative autocorrelation oscillations.
            Returns (n_eff, tau_int).

            max_lag : int or None
                Cap on the ACF summation lag.  None (default) uses n//2 for
                the most accurate estimate.  Pass a smaller value (e.g. 2000)
                for fast mid-run checks — be aware this underestimates tau
                when the true tau exceeds max_lag/2.
            """
            n = len(x)
            if n < 4:
                return float(n), 1.0
            x = np.asarray(x, dtype=float)
            x = x - x.mean()
            c0 = np.dot(x, x) / n
            if c0 == 0:
                return float(n), 1.0

            _max_lag = n // 2 if max_lag is None else min(max_lag, n // 2)
            acf = np.empty(_max_lag + 1)
            acf[0] = 1.0
            for lag in range(1, _max_lag + 1):
                acf[lag] = np.dot(x[:-lag], x[lag:]) / (n * c0)

            # Geyer's initial positive sequence: sum consecutive pairs
            # (acf[2k] + acf[2k+1]) and stop at the first non-positive pair.
            tau_int = acf[0]  # = 1.0
            k = 0
            while 2 * k + 2 <= _max_lag:
                pair_sum = acf[2 * k + 1] + acf[2 * k + 2]
                if pair_sum <= 0:
                    break
                tau_int += 2.0 * pair_sum
                k += 1
            # Ensure tau >= 1 (can't have n_eff > n)
            tau_int = max(tau_int, 1.0)
            return n / tau_int, tau_int

        def calculate_effective_sample_size(chain, max_lag=None):
            """Effective sample size for a chain.

            For a 2-D array (n_samples × n_params), computes ESS per parameter
            and returns the **minimum** ESS and its corresponding tau.
            This is the standard approach used by ArviZ / Stan / PyMC.

            For a 1-D array, returns the ESS of that single parameter.

            max_lag : int or None
                Passed to _ess_1d.  None uses n//2 (accurate).  A smaller
                value (e.g. 2000) speeds up mid-run checks at the cost of
                underestimating tau when true tau >> max_lag/2.
            """
            chain = np.asarray(chain, dtype=float)
            if chain.ndim == 2:
                n_samples, n_params = chain.shape
                ess_per_param = np.empty(n_params)
                tau_per_param = np.empty(n_params)
                for j in range(n_params):
                    ess_per_param[j], tau_per_param[j] = _ess_1d(chain[:, j], max_lag=max_lag)
                worst = np.argmin(ess_per_param)
                return ess_per_param[worst], tau_per_param[worst]
            else:
                return _ess_1d(chain, max_lag=max_lag)
        
        print(f"Starting MCMC loop ({n_iterations:,} iterations). "
              f"First progress output at sensitivity checkpoint (iter 1), "
              f"then every 10%.", flush=True)

        for i in range(n_iterations):
            # ---- GBIS sampler: independent uniform proposal per parameter ----
            # Each parameter is perturbed by step_i * U(-1, 1) independently.
            # No cross-parameter covariance; step sizes adapted via sensitivity test.
            proposed_params = current_params.copy()
            for p in param_names:
                lo, hi = flat_priors[p]
                step_val = step_sizes[p] * (np.random.rand() - 0.5) * 2.0
                val = current_params[p] + step_val
                # Reflection off bounds (preserves proposal symmetry)
                for _ in range(4):
                    if val > hi:
                        val = 2.0 * hi - val
                    if val < lo:
                        val = 2.0 * lo - val
                    if lo <= val <= hi:
                        break
                proposed_params[p] = float(np.clip(val, lo, hi))

            n_proposed_window += 1
            n_proposed_total  += 1

            # MH acceptance (GBIS: P = -resExp/2, equivalent to our log-posterior)
            proposed_log_post, proposed_residual_rms, log_like_prop = log_posterior(proposed_params)

            delta_lp = proposed_log_post - current_log_post
            if not np.isfinite(delta_lp):
                accepted = False
            elif delta_lp >= 0:
                accepted = True
            else:
                accepted = np.random.rand() < float(np.exp(delta_lp))

            if accepted:
                current_params   = proposed_params
                current_log_post = proposed_log_post
                residual_rms     = proposed_residual_rms
                log_like_current = log_like_prop
                n_accepted_window += 1
                n_accepted_total  += 1

            # Store samples in linear (physical) space even for log-space params
            for key in current_params.keys():
                val = current_params[key]
                samples[key].append(np.exp(val) if key in log_space_set else val)
            residuals_evolution.append(residual_rms)
            log_likelihood_trace.append(log_like_current)
            log_posterior_trace.append(current_log_post)

            # Burn-in completion message
            if i < burn_in and (i + 1) == burn_in:
                n_accepted_burn = n_accepted_total
                burn_rej_rate = 100.0 * (burn_in - n_accepted_burn) / max(burn_in, 1)
                print(f"\n{'='*80}")
                print(f"*** BURN-IN PHASE COMPLETE (iteration {i+1}) ***")
                print(f"{'='*80}")
                print(f"Burn-in rejection rate: {burn_rej_rate:.1f}%  (target ~77.0%)")
                print(f"Current step sizes:")
                for pname in param_names:
                    print(f"  {pname:20s}: {step_sizes[pname]:.4g}")
                print()

            # ---- GBIS sensitivity test: adapt step sizes per GBIS schedule ----
            is_sens_point = (i + 1) in sens_schedule

            if is_sens_point:
                # Update running prob_target from rejection rate in this window.
                # Skip on the very first test (matches MATLAB iKeepSave==0 guard).
                if _sens_test_count > 0 and n_proposed_window > 0:
                    rej_ratio = 1.0 - n_accepted_window / n_proposed_window
                    prob_target = float(np.clip(
                        prob_target * (rej_ratio / 0.77), 1e-6, 0.9999))
                prob_target_local = prob_target
                _sens_test_count += 1

                # Perturb each parameter individually to measure local curvature
                for j_s in range(d):
                    p_name = param_names[j_s]
                    lo, hi = flat_priors[p_name]
                    sign = 1.0 if np.random.randn() >= 0.0 else -1.0
                    val_s = current_params[p_name] + sign * step_sizes[p_name] * 0.5
                    # GBIS MATLAB line 379-381: if step exceeds upper bound,
                    # subtract the full step (i.e. try the opposite direction)
                    if val_s > hi:
                        val_s = val_s - step_sizes[p_name]
                    val_s = float(np.clip(val_s, lo, hi))
                    test_params = current_params.copy()
                    test_params[p_name] = val_s
                    test_log_post, _, _ = log_posterior(test_params)
                    dP = test_log_post - current_log_post
                    # Store the raw Metropolis ratio (may exceed 1 if test point is better)
                    # then invert values > 1 to symmetrise — matches GBIS MATLAB line 201:
                    #   probSens(probSens > 1) = 1./probSens(probSens > 1)
                    ratio = float(np.exp(dP)) if np.isfinite(dP) else 0.0
                    prob_sens[j_s] = 1.0 / ratio if ratio > 1.0 else ratio

                # Adjust step sizes towards prob_target_local
                for j_s in range(d):
                    p_name = param_names[j_s]
                    p_diff = prob_target_local - prob_sens[j_s]
                    if p_diff > 0:
                        # Sensitivity test accepted too readily → step too small for curvature
                        # (counter-intuitive: high sens-acceptance means step spans steep region)
                        step_sizes[p_name] *= np.exp(-p_diff / prob_target_local * 2.0)
                    else:
                        step_sizes[p_name] *= np.exp(-p_diff / (1.0 - prob_target_local) * 2.0)
                    prior_span = flat_priors[p_name][1] - flat_priors[p_name][0]
                    step_sizes[p_name] = float(np.clip(step_sizes[p_name], 1e-12, prior_span))

                # Store per-checkpoint diagnostics for plotting
                win_acc_rate = n_accepted_window / max(n_proposed_window, 1)
                for jj, key in enumerate(param_names):
                    acceptance_rate_evolution[key].append(win_acc_rate)
                    proposal_std_evolution[key].append(step_sizes[key])
                proposal_std_evolution_iters.append(i + 1)

                # Progress report: every checkpoint for first 1k iters, then every 10%
                phase_str = "BURN-IN" if i < burn_in else "POST-BURN-IN"
                _report_interval = int(max(n_iterations * 0.1, 1))
                if (i + 1) <= 1000 or (i + 1) % _report_interval == 0:
                    rej_rate_pct = 100.0 * (1.0 - win_acc_rate)
                    n_window = min(int(n_iterations * 0.1), len(samples[param_names[0]]))
                    current_means = {p: np.mean(samples[p][-n_window:]) for p in param_names}
                    print(f"~~~~~ Iteration {i+1}/{n_iterations} ({model_display_name}) [{phase_str}] ~~~~~", flush=True)
                    print(f"  Window rejection rate: {rej_rate_pct:.1f}%  (target 77.0%)")
                    print(f"  Current step sizes:")
                    for pname in param_names:
                        print(f"    {pname}: {step_sizes[pname]:.4g}")
                    print(f"  Current parameter means (last {n_window} samples):")
                    for param in param_names:
                        print(f"    {param}: {current_means[param]:.6f}")
                    print(f"  Current RMS: {residual_rms:.6f}")
                    print("~" * 70)

                # Reset window counters
                n_accepted_window = 0
                n_proposed_window = 0
        
        # Extract post-burn-in samples (burn_in calculated at start of MCMC)
        samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}
        
        # ---- GBIS-STYLE REJECTION RATE DIAGNOSTICS ----
        print(f"\n{'='*80}")
        print(f"GBIS SAMPLER DIAGNOSTICS (Bagnardi & Hooper 2018)")
        print(f"{'='*80}")
        overall_acc_rate = n_accepted_total / max(n_proposed_total, 1)
        post_burn_n = n_iterations - burn_in
        post_burn_acc = n_accepted_total - n_accepted_burn
        post_burn_acc_rate = post_burn_acc / max(post_burn_n, 1)
        overall_rej_rate  = 1.0 - overall_acc_rate
        post_burn_rej_rate = 1.0 - post_burn_acc_rate
        print(f"Burn-in period:          {burn_in} iterations")
        print(f"Post-burn-in iterations: {post_burn_n}")
        print(f"Overall rejection rate:       {overall_rej_rate*100:.1f}%  "
              f"(acceptance {overall_acc_rate*100:.1f}%)")
        print(f"Post-burn-in rejection rate:  {post_burn_rej_rate*100:.1f}%  "
              f"(acceptance {post_burn_acc_rate*100:.1f}%)  [target ~77%]")
        if post_burn_rej_rate < 0.50:
            print("  ⚠️  Low rejection rate (<50%): steps may be too small — chain under-explores.")
        elif post_burn_rej_rate > 0.95:
            print("  ⚠️  Very high rejection rate (>95%): steps may be too large — chain barely moves.")
        else:
            print("  ✓ Rejection rate within acceptable range.")
        print(f"\nFinal step sizes:")
        for pname in param_names:
            print(f"  {pname:20s}: {step_sizes[pname]:.4g}")
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

        # Calculate MAP estimate using log-posterior (not log-likelihood) so that
        # the Jacobian correction from log-space sampling is accounted for.
        best_idx = np.argmax(log_posterior_trace[burn_in:])
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
            final_acceptance = n_accepted_total / max(1, n_proposed_total)
            f.write(f"MCMC completed. Final overall acceptance rate: {final_acceptance:.3f}\n\n")

            # Write final GBIS step sizes
            f.write(f"Sampler: GBIS diagonal uniform proposal (Bagnardi & Hooper 2018)\n")
            f.write(f"Post-burn-in rejection rate: {post_burn_rej_rate*100:.1f}%  (target 77%)\n")
            f.write("Final step sizes:\n")
            for param in param_names:
                f.write(f"{param:8s}: {step_sizes[param]:.6g}\n")
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

def save_synthetic_as_gbis_mat(u_los_obs, X_flat, Y_flat, incidence_angle, heading,
                                out_path, reference_point=(50.63366157664304, 29.7546416200275),
                                wavelength=0.056):
    """
    Save synthetic LOS displacement data (as produced by gen_synthetic_data) in the
    standard GBIS .mat input format — Phase, Lat, Lon, Inc, Heading, each an (N,1)
    array — so it can be loaded directly by MATLAB GBIS for a side-by-side
    comparison against the pyGBIS synthetic-test inversion.

    X_flat/Y_flat are local Cartesian coordinates in metres, referenced to
    `reference_point` (lon0, lat0) in decimal degrees — matches the convention
    used by llh2local.py/local2llh.py elsewhere in this file (origin is lon,lat;
    local2llh.py itself expects its xy input in km, hence the /1000 below).

    Phase is recovered from u_los_obs by inverting the same phase -> LOS metres
    convention used throughout this script (u_los = -phase * wavelength / (4*pi)):
        phase = -u_los_obs / conv_factor
    `wavelength` defaults to 0.056 m to match both `run_baysian_inference`'s
    default and the wavelength recorded in GBIS .inp files (e.g.
    us6000e2k3_NP1.inp uses insar{}.wavelength = 0.056), for a fair pyGBIS vs
    MATLAB GBIS comparison.
    """
    lon0, lat0 = reference_point
    xy_km = np.array([np.asarray(X_flat).flatten(), np.asarray(Y_flat).flatten()], dtype=float) / 1000.0
    llh_out = l2llh.local2llh(xy_km, np.array([lon0, lat0], dtype=float))
    lon = llh_out[0, :]
    lat = llh_out[1, :]

    conv_factor = wavelength / (4 * np.pi)
    phase = -np.asarray(u_los_obs).flatten() / conv_factor

    n = len(phase)
    inc_arr = np.full(n, incidence_angle, dtype=float)
    head_arr = np.full(n, heading, dtype=float)

    sio.savemat(out_path, {
        'Phase':   phase.reshape(-1, 1),
        'Lat':     lat.reshape(-1, 1),
        'Lon':     lon.reshape(-1, 1),
        'Inc':     inc_arr.reshape(-1, 1),
        'Heading': head_arr.reshape(-1, 1),
    })
    print(f"Synthetic data saved in GBIS .mat format: {out_path}  "
          f"(N={n}, reference=[lon={lon0}, lat={lat0}], wavelength={wavelength})")


def gen_synthetic_data(true_params, grid_size=100, noise_level=0.5, model_type='pcdm',
                        save_gbis_mat=False, gbis_mat_path=None,
                        reference_point=(50.63366157664304, 29.7546416200275),
                        gbis_wavelength=0.056):
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
    save_gbis_mat : bool
        If True, also save the noisy synthetic data as a GBIS-readable .mat file
        (via save_synthetic_as_gbis_mat) so the same synthetic test can be run
        through MATLAB GBIS for comparison.
    gbis_mat_path : str, optional
        Output path for the .mat file. Defaults to 'synthetic_data_{model_type}_GBIS.mat'.
    reference_point : (lon0, lat0), optional
        Local Cartesian origin (decimal degrees) used to convert X_flat/Y_flat back
        to Lon/Lat for the .mat file.
    gbis_wavelength : float, optional
        Radar wavelength (m) used for the LOS metres -> phase (radians) conversion
        when writing the .mat file.

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
        mkey = model_type.lower()
        if mkey not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_type}'. "
                             f"Registered: {list(MODEL_REGISTRY.keys())}")
        true_params = MODEL_REGISTRY[mkey]['default_params'].copy()

    mkey = model_type.lower()
    if mkey not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_type}'. "
                         f"Registered: {list(MODEL_REGISTRY.keys())}")
    print(true_params)
    ue_true, un_true, uv_true = MODEL_REGISTRY[mkey]['forward'](
        X_flat, Y_flat, true_params)



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

    # Plot the synthetic data for visualization (generic for all models)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sc1 = axes[0,0].tricontourf(X_flat, Y_flat, u_los_true, levels=50, cmap='RdBu_r')
    axes[0,0].set_title(f'True LOS Displacement ({model_type})')
    axes[0,0].set_xlabel('X'); axes[0,0].set_ylabel('Y'); axes[0,0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0,0])

    sc2 = axes[0,1].tricontourf(X_flat, Y_flat, u_los_obs, levels=50, cmap='RdBu_r')
    axes[0,1].set_title('Observed LOS Displacement')
    axes[0,1].set_xlabel('X'); axes[0,1].set_ylabel('Y'); axes[0,1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[0,1])

    sc3 = axes[1,0].tricontourf(X_flat, Y_flat, spatially_correlated_noise, levels=50, cmap='viridis')
    axes[1,0].set_title('Spatially Correlated Noise')
    axes[1,0].set_xlabel('X'); axes[1,0].set_ylabel('Y'); axes[1,0].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[1,0])

    difference = u_los_obs - u_los_true
    sc4 = axes[1,1].tricontourf(X_flat, Y_flat, difference, levels=50, cmap='viridis')
    axes[1,1].set_title('Difference (Obs - True)')
    axes[1,1].set_xlabel('X'); axes[1,1].set_ylabel('Y'); axes[1,1].set_aspect('equal')
    plt.colorbar(sc4, ax=axes[1,1])

    plt.tight_layout()
    plt.savefig(f'synthetic_data_{model_type}.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"True parameters used for {model_type} model:")
    for key, value in true_params.items():
        print(f"  {key}: {value}")
    print(f"Noise level: {noise_level*100:.1f}%")
    print(f"RMS of true signal: {np.std(u_los_true)*1000:.3f} mm")
    print(f"RMS of noise: {np.std(spatially_correlated_noise)*1000:.3f} mm")

    if save_gbis_mat:
        _path = gbis_mat_path or f'synthetic_data_{model_type}_GBIS.mat'
        save_synthetic_as_gbis_mat(u_los_obs, X_flat, Y_flat, incidence_angle, heading,
                                    _path, reference_point=reference_point,
                                    wavelength=gbis_wavelength)

    return u_los_obs, X_flat, Y_flat, incidence_angle, heading, noise_sill, noise_nugget, noise_range

def save_inference_state(samples, log_lik_trace, rms_evolution, proposal_std_evolution,
                        acceptance_rate_evolution, X_obs, Y_obs, u_los_obs,
                        incidence_angle, heading, inference_params, figure_folder=None,model_type='pCDM',
                        reference_point=None):
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
        
        # Extract priors from inference_params if available
        'priors': inference_params.get('priors') if isinstance(inference_params, dict) else None,
        
        # Common coordinate reference [lat, lon] used for all IFGs
        'reference_point': reference_point,

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

def regenerate_plots_from_state(pickle_filename, new_figure_folder=None, full_res_npy_paths=None):
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
        model_type=inference_params.get('model_type', 'pCDM'),
        priors=inference_state.get('priors'),
        full_res_npy_paths=full_res_npy_paths,
    )

    print(f"All plots regenerated in: {new_figure_folder}")

def run_baysian_inference(u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                         n_iterations=10000, sill=None, nugget=None, range_param=None,
                         initial_params=None, priors=None, proposal_std=None,
                         max_step_sizes=None, adaptive_interval=1000,
                         target_acceptance=0.23, adapt_method='conservative', figure_folder=None, use_sa_init=False, model_type='pCDM', u_los_obs_already_los_in_m=None,
                         burn_in=None, log_space_params=None, full_res_npy_paths=None, ifg_dates=None,
                         fit_ramp=False, reference_point=None, noise_rms=None, wavelength=0.056,
                         sill_in_m2=False):
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
    fit_ramp : str, optional
        Whether to fit a linear ramp to the data ('True' or 'False')
    wavelength : float, optional
        Radar wavelength (m) used for the phase (radians) -> LOS metres conversion
        u_los = -phase * wavelength / (4*pi). Defaults to 0.056 to match the
        wavelength recorded in GBIS .inp files (e.g. insar{}.wavelength = 0.056
        in us6000e2k3_NP1.inp) for apples-to-apples comparison with MATLAB GBIS.
    """
    # Create figure folder if it doesn't exist
    if figure_folder is not None and not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    
 
    
    # convert from Phase (radians) to LOS displacement (metres)
    # Support single IFG or list-of-IFGs (leave unchanged when user signals data are already LOS in metres)
    conv_factor = wavelength/(4*np.pi)   # rad → m  (default 0.056 m, matching GBIS .inp convention)
    _phase_converted = False
    if u_los_obs_already_los_in_m is not None:
        # caller indicated values are already LOS in metres — do not convert
        pass
    else:
        _phase_converted = True
        if isinstance(u_los_obs, (list, tuple)):
            u_los_obs = [ -np.asarray(u).astype(float) * conv_factor for u in u_los_obs ]
        else:
            u_los_obs = -np.asarray(u_los_obs).astype(float) * conv_factor

    # If we converted phase→metres, sill/nugget supplied by the user are in rad²
    # (from step02, which fits the variogram to the raw phase).  Scale them to m².
    if _phase_converted and not sill_in_m2:
        conv_sq = conv_factor ** 2
        if sill is not None:
            if isinstance(sill, (list, tuple)):
                sill   = [s * conv_sq for s in sill]
                if nugget is not None:
                    nugget = [n * conv_sq for n in nugget]
            else:
                sill   = float(sill)   * conv_sq
                if nugget is not None:
                    nugget = float(nugget) * conv_sq

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
        figure_folder=figure_folder, model_type=model_type,
        burn_in=_burn_in,
        log_space_params=log_space_params,
        fit_ramp=fit_ramp)
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
        model_type=model_type,
        reference_point=reference_point,
    )
    
    # Plot results
    pCDM_BI_plotting_funcs.plot_inference_results(samples, log_lik_trace, rms_evolution, burn_in=_burn_in,
                          X_obs=X_obs, Y_obs=Y_obs, u_los_obs=u_los_obs,
                          incidence_angle=incidence_angle, heading=heading, figure_folder=figure_folder,
                          proposal_std_evolution=proposal_std_evolution,
                          proposal_std_evolution_iters=proposal_std_evolution_iters,
                          acceptance_rate_evolution=acceptance_rate_evolution,
                          adaptive_interval=adaptive_interval, target_acceptance=target_acceptance, model_type=model_type, priors=priors,
                          full_res_npy_paths=full_res_npy_paths, ifg_dates=ifg_dates,
                          reference_point=reference_point, noise_rms=noise_rms)
    
    return samples, log_lik_trace, rms_evolution, pickle_filename

# Per-model MCMC defaults used by synthetic_test().
# Keys: initial params, prior bounds, proposal std, max step sizes,
# and preferred grid_size / noise_level for a realistic test signal.
_SYNTH_DEFAULTS = {
    'pcdm': {
        'grid_size': 50, 'noise_level': 0.05,
        'initial':  {'X0': 0, 'Y0': 0, 'depth': 5000,
                     'DVx': 7e7, 'DVy': 7e7, 'DVz': 7e7,
                     'omegaX': 0, 'omegaY': 0, 'omegaZ': 0},
        'priors':   {'X0': (-15000, 15000), 'Y0': (-15000, 15000), 'depth': (100, 35000),
                     'DVx': (1e2, 1e9), 'DVy': (1e2, 1e9), 'DVz': (1e2, 1e9),
                     'omegaX': (-45, 45), 'omegaY': (-45, 45), 'omegaZ': (-45, 45)},
        'proposal': {'X0': 100, 'Y0': 100, 'depth': 100,
                     'DVx': 1e4, 'DVy': 1e4, 'DVz': 1e4,
                     'omegaX': 1, 'omegaY': 1, 'omegaZ': 1},
        'max_step': {'X0': 1e5, 'Y0': 1e5, 'depth': 1e4,
                     'DVx': 1e7, 'DVy': 1e7, 'DVz': 1e7,
                     'omegaX': 20, 'omegaY': 20, 'omegaZ': 20},
    },
    'okada': {
        'grid_size': 40, 'noise_level': 0.5,
        'initial':  {'X0': 0, 'Y0': 0, 'depth': 5000,
                     'length': 10000, 'width': 8000,
                     'strike': 100, 'dip': 30, 'rake': 90, 'slip': 2, 'opening': 0},
        'priors':   {'X0': (-15000, 15000), 'Y0': (-15000, 15000), 'depth': (100, 35000),
                     'length': (1000, 20000), 'width': (1000, 20000),
                     'strike': (0, 360), 'dip': (0, 90), 'rake': (-180, 180),
                     'slip': (-10, 10), 'opening': (0, 0)},
        'proposal': {'X0': 10000, 'Y0': 10000, 'depth': 1000,
                     'length': 1000, 'width': 1000,
                     'strike': 10, 'dip': 10, 'rake': 10, 'slip': 0.1, 'opening': 0.0},
        'max_step': {'X0': 1e5, 'Y0': 1e5, 'depth': 1e4,
                     'length': 5000, 'width': 5000,
                     'strike': 20, 'dip': 20, 'rake': 20, 'slip': 2, 'opening': 0},
    },
    'une': {
        'grid_size': 100, 'noise_level': 0.5,
        'initial':  {'X0': 0, 'Y0': 0, 'depth': 1000,
                     'yield_kt': 150, 'dv_factor': 0.1,
                     'chimney_amp': 0.15, 'compact_amp': 0.05},
        'priors':   {'X0': (-15000, 15000), 'Y0': (-15000, 15000), 'depth': (100, 35000),
                     'yield_kt': (50, 500), 'dv_factor': (-1, 1.0),
                     'chimney_amp': (0.01, 1.0), 'compact_amp': (0.01, 1.0)},
        'proposal': {'X0': 100, 'Y0': 100, 'depth': 100,
                     'yield_kt': 10, 'dv_factor': 0.1,
                     'chimney_amp': 0.1, 'compact_amp': 0.1},
        'max_step': {'X0': 1e5, 'Y0': 1e5, 'depth': 1e4,
                     'yield_kt': 100, 'dv_factor': 0.2,
                     'chimney_amp': 0.2, 'compact_amp': 0.2},
    },
    'mogi': {
        'grid_size': 50, 'noise_level': 0.1,
        'initial':  {'X0': 0, 'Y0': 0, 'depth': 5000, 'DV': 1e6},
        'priors':   {'X0': (-15000, 15000), 'Y0': (-15000, 15000),
                     'depth': (100, 35000), 'DV': (-1e9, 1e9)},
        'proposal': {'X0': 500, 'Y0': 500, 'depth': 500, 'DV': 1e4},
        'max_step': {'X0': 1e5, 'Y0': 1e5, 'depth': 1e4, 'DV': 1e8},
    },
    'mctigue': {
        'grid_size': 50, 'noise_level': 0.1,
        'initial':  {'X0': 0, 'Y0': 0, 'depth': 5000, 'DV': 1e6, 'a': 1000, 'c': 1000},
        'priors':   {'X0': (-15000, 15000), 'Y0': (-15000, 15000),
                     'depth': (100, 35000), 'DV': (-1e9, 1e9),
                     'a': (1, 10000), 'c': (1, 10000)},
        'proposal': {'X0': 500, 'Y0': 500, 'depth': 500, 'DV': 1e4, 'a': 100, 'c': 100},
        'max_step': {'X0': 1e5, 'Y0': 1e5, 'depth': 1e4, 'DV': 1e8, 'a': 5000, 'c': 5000},
    },
}


def synthetic_test(
    model_type,
    initial_params=None,
    priors=None,
    proposal_std=None,
    max_step_sizes=None,
    grid_size=None,
    noise_level=None,
    n_iterations=int(1e4),
    use_sa_init=True,
    figure_folder=None,
):
    """Run a synthetic recovery test for any registered model or joint combination.

    Parameters
    ----------
    model_type : str or list of str
        Registered model name(s), e.g. 'pcdm', ['pcdm', 'okada'].
        For a list the forward displacements are summed before LOS projection.
    initial_params : dict or list of dicts, optional
        Starting parameters for MCMC. None uses built-in defaults from
        _SYNTH_DEFAULTS (falling back to MODEL_REGISTRY default_params).
    priors : dict or list of dicts, optional
        Prior bounds {param: (lo, hi)}. None uses _SYNTH_DEFAULTS.
    proposal_std : dict or list of dicts, optional
        Initial MCMC step sizes. None uses _SYNTH_DEFAULTS.
    max_step_sizes : dict or list of dicts, optional
        Maximum allowed step per iteration. None uses _SYNTH_DEFAULTS.
    grid_size : int, optional
        Square grid side length (total points = grid_size**2). Defaults to the
        first model's entry in _SYNTH_DEFAULTS (or 50 if not listed).
    noise_level : float, optional
        Noise standard deviation as a fraction of signal std. Defaults to the
        first model's entry in _SYNTH_DEFAULTS (or 0.3 if not listed).
    n_iterations : int
        Number of MCMC iterations.
    use_sa_init : bool
        Run simulated annealing before MCMC.
    figure_folder : str, optional
        Output directory. Auto-generated from model names if None.

    Returns
    -------
    samples, log_lik_trace, rms_evolution, pickle_filename
    """
    if isinstance(model_type, str):
        model_list = [model_type.lower()]
    else:
        model_list = [m.lower() for m in model_type]

    for m in model_list:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{m}'. Registered: {list(MODEL_REGISTRY.keys())}")

    def _defaults(m):
        d = _SYNTH_DEFAULTS.get(m)
        if d is not None:
            return d
        dp = MODEL_REGISTRY[m].get('default_params', {})
        prop = {k: max(abs(v) * 0.05, 1e-6) for k, v in dp.items()}
        return {
            'grid_size': 50, 'noise_level': 0.3,
            'initial':  dp.copy(),
            'priors':   {k: (min(v * 0.01, -abs(v) * 10), abs(v) * 10) for k, v in dp.items()},
            'proposal': prop,
            'max_step': {k: abs(v) * 5 for k, v in dp.items()},
        }

    first_def = _defaults(model_list[0])
    gs     = grid_size   if grid_size   is not None else first_def['grid_size']
    nl     = noise_level if noise_level is not None else first_def['noise_level']
    folder = figure_folder or f"figure_synth_{'_'.join(model_list)}"

    x_range = np.linspace(-50000, 50000, gs)
    y_range = np.linspace(-50000, 50000, gs)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    X_flat = X_mesh.flatten()
    Y_flat = Y_mesh.flatten()
    n_obs  = len(X_flat)

    ue_total = np.zeros(n_obs)
    un_total = np.zeros(n_obs)
    uv_total = np.zeros(n_obs)
    for m in model_list:
        true_p = MODEL_REGISTRY[m]['default_params'].copy()
        ue_m, un_m, uv_m = MODEL_REGISTRY[m]['forward'](X_flat, Y_flat, true_p)
        ue_total += ue_m
        un_total += un_m
        uv_total += uv_m
        print(f"  {m} true params: " + ", ".join(f"{k}={v}" for k, v in true_p.items()))

    incidence_angle = 39.4
    heading         = -169.9
    inc_rad  = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)
    u_los_true = -((ue_total * los_e) + (un_total * los_n) + (uv_total * los_u)).flatten()

    noise_std    = max(np.std(u_los_true) * nl, 1e-4)
    noise_sill   = noise_std ** 2
    noise_nugget = noise_sill * 0.01
    noise_range  = np.sqrt((x_range[-1] - x_range[0]) ** 2 + (y_range[-1] - y_range[0]) ** 2) / 4.0
    C_noise, _, _, _ = estimate_noise_covariance(
        X_flat, Y_flat, u_los_true,
        sill=noise_sill, nugget=noise_nugget, range_param=noise_range)
    try:
        noise = np.random.multivariate_normal(np.zeros(n_obs), C_noise)
    except np.linalg.LinAlgError:
        noise = np.random.normal(0, noise_std, n_obs)
    u_los_obs = u_los_true + noise
    noise_rms = float(np.sqrt(np.mean(noise**2)))

    print(f"\nSynthetic test  --  model(s): {'+'.join(model_list)}")
    print(f"  Grid: {gs}x{gs} = {n_obs} points")
    print(f"  Signal RMS: {np.std(u_los_true)*1e3:.3f} mm  |  "
          f"Noise std: {noise_std*1e3:.3f} mm  (SNR {np.std(u_los_true)/noise_std:.1f})")
    print(f"  Noise: sill={noise_sill:.3e}, nugget={noise_nugget:.3e}, range={noise_range:.0f} m")

    os.makedirs(folder, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.tricontourf(X_flat, Y_flat, u_los_true, levels=50, cmap='RdBu_r')
    ax1.set_title(f"True LOS ({'+'.join(model_list)})")
    ax1.set_aspect('equal')
    ax2.tricontourf(X_flat, Y_flat, u_los_obs, levels=50, cmap='RdBu_r')
    ax2.set_title("Observed LOS (+ noise)")
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"{folder}/synthetic_data.png", dpi=200, bbox_inches='tight')
    plt.close()

    def _resolve(arg, key):
        if arg is not None:
            return arg if isinstance(arg, list) else [arg]
        return [_defaults(m)[key].copy() for m in model_list]

    init_list  = _resolve(initial_params, 'initial')
    prior_list = _resolve(priors,         'priors')
    prop_list  = _resolve(proposal_std,   'proposal')
    step_list  = _resolve(max_step_sizes, 'max_step')

    if len(model_list) == 1:
        init_list  = init_list[0]
        prior_list = prior_list[0]
        prop_list  = prop_list[0]
        step_list  = step_list[0]

    samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
        u_los_obs=u_los_obs,
        X_obs=X_flat,
        Y_obs=Y_flat,
        incidence_angle=incidence_angle,
        heading=heading,
        n_iterations=n_iterations,
        sill=noise_sill,
        nugget=noise_nugget,
        range_param=noise_range,
        initial_params=init_list,
        priors=prior_list,
        proposal_std=prop_list,
        max_step_sizes=step_list,
        adaptive_interval=max(100, n_iterations // 100),
        target_acceptance=0.23,
        figure_folder=folder,
        use_sa_init=use_sa_init,
        model_type=model_type,
        u_los_obs_already_los_in_m=True,
        noise_rms=noise_rms,
    )

    print(f"\n{'='*70}")
    print(f"Synthetic test complete -- {'+'.join(model_list)}")
    print(f"  Results saved to: {folder}/")
    print(f"{'='*70}")
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
    noise_list = []
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
        noise_list.append(noise)
        sill_list.append(noise_sill)
        nugget_list.append(noise_nugget)
        range_list.append(noise_range)

    noise_rms = float(np.sqrt(np.mean(np.concatenate(noise_list)**2)))

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
        u_los_obs_already_los_in_m=True,
        noise_rms=noise_rms,
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



def synthetic_test_okada_2ifg(n_iterations=int(1e5), noise_level=5e-2, figure_folder='figure_synth_okada_2ifg_new_noise',
                               save_gbis_mat=True, gbis_mat_dir=None,
                               reference_point=(50.63366157664304, 29.7546416200275),
                               gbis_wavelength=0.056):
    """
    Synthetic Okada two-interferogram recovery test.

    Two IFGs are simulated with different satellite viewing geometries (ascending
    and descending) over the same fault.  Known parameters are used to generate
    synthetic LOS data, Gaussian noise is added, and the inversion is run.
    The printed recovery table lets you check whether the sampler finds the truth.

    save_gbis_mat : bool
        If True (default), also save both synthetic IFGs as GBIS-readable .mat
        files (via save_synthetic_as_gbis_mat) so the same synthetic test can be
        run through MATLAB GBIS for comparison.
    gbis_mat_dir : str, optional
        Directory (relative to the current working directory unless absolute)
        to save the .mat files into. Defaults to 'synthetic_okada_2ifg_GBIS_mat'.
    reference_point : (lon0, lat0), optional
        Local Cartesian origin (decimal degrees) used to convert X_obs/Y_obs
        back to Lon/Lat for the .mat files — arbitrary for a synthetic test.
    gbis_wavelength : float, optional
        Radar wavelength (m) used for the LOS metres -> phase (radians)
        conversion when writing the .mat files.
    """
    # --- True fault parameters ---
    true_params = {
        'X0':     0.0,       # m east  of reference
        'Y0':     0.0,       # m north of reference
        'depth':  3000.0,    # m
        'length': 8000.0,    # m
        'width':  6000.0,    # m
        'strike': 45.0,      # degrees
        'dip':    60.0,      # degrees
        'rake':   90.0,      # degrees (pure thrust)
        'slip':   0.5,       # m
        'opening': 0.0,
    }

    # --- Two viewing geometries ---
    # IFG 1: Descending pass (inc ~33.9°, heading ~-169.7°)
    inc1,  head1  = 33.9, -169.7
    # IFG 2: ascending pass (inc ~33.9°, heading ~-10.4°)
    inc2,  head2  = 33.9,  -10.4

    # --- Observation grid ---
    grid = np.linspace(-35000, 35000, 50)   # 30×30 = 900 points per IFG
    Xg, Yg = np.meshgrid(grid, grid)
    X_obs = Xg.flatten()
    Y_obs = Yg.flatten()

    # --- Build LOS vectors ---
    def los_vec(inc_deg, head_deg):
        inc_r  = np.radians(inc_deg)
        head_r = np.radians(head_deg)
        le = np.sin(inc_r) * np.cos(head_r)
        ln = -np.sin(inc_r) * np.sin(head_r)
        lu = -np.cos(inc_r)
        return le, ln, lu

    # --- Forward model via MODEL_REGISTRY ---
    fwd = MODEL_REGISTRY['okada']['forward']
    ue, un, uv = fwd(X_obs, Y_obs, true_params)

    le1, ln1, lu1 = los_vec(inc1, head1)
    le2, ln2, lu2 = los_vec(inc2, head2)

    los1 = -(ue * le1 + un * ln1 + uv * lu1)
    los2 = -(ue * le2 + un * ln2 + uv * lu2)

    np.random.seed(42)

    # --- Noise covariance parameters (in metres²) ---
    sill        = 1e-3
    nugget      = sill * 0.01
    range_param = 15000.0   # 10 km correlation length

    # --- Spatially correlated atmospheric noise via exponential covariance ---
    n_obs = len(X_obs)
    H = np.sqrt((X_obs[:, None] - X_obs[None, :])**2 +
                (Y_obs[:, None] - Y_obs[None, :])**2)
    C_noise = (sill - nugget) * np.exp(-H / range_param)
    np.fill_diagonal(C_noise, sill)
    noise1 = np.random.multivariate_normal(np.zeros(n_obs), C_noise)
    noise2 = np.random.multivariate_normal(np.zeros(n_obs), C_noise)
    los1 += noise1
    los2 += noise2

    noise_rms = float(np.sqrt(np.mean(np.concatenate([noise1, noise2])**2)))
    print(f'  Noise RMS (both IFGs combined): {noise_rms*100:.3f} cm  ({noise_rms*1000:.2f} mm)')

    # --- Linear ramp + offset (different per IFG, simulating orbital error) ---
    # Coords normalised to [-1, 1] so ramp coefficients are in metres
    x_n = X_obs / 30000.0
    y_n = Y_obs / 30000.0
    los1 += 0.003 + 0.008 * x_n - 0.005 * y_n   # ~1.3 cm peak-to-peak ramp
    los2 += -0.005 - 0.004 * x_n + 0.009 * y_n  # ~1.3 cm peak-to-peak, different axis

    if save_gbis_mat:
        _mat_dir = gbis_mat_dir or 'synthetic_okada_2ifg_GBIS_mat'
        os.makedirs(_mat_dir, exist_ok=True)
        save_synthetic_as_gbis_mat(los1, X_obs, Y_obs, inc1, head1,
                                    os.path.join(_mat_dir, 'ifg1_synthetic_GBIS.mat'),
                                    reference_point=reference_point, wavelength=gbis_wavelength)
        save_synthetic_as_gbis_mat(los2, X_obs, Y_obs, inc2, head2,
                                    os.path.join(_mat_dir, 'ifg2_synthetic_GBIS.mat'),
                                    reference_point=reference_point, wavelength=gbis_wavelength)

    X_obs1, Y_obs1 = X_obs, Y_obs
    X_obs2, Y_obs2 = X_obs, Y_obs

    # --- Priors centred loosely on truth ---
    priors = {
        'X0':     (-5000, 5000),
        'Y0':     (-5000, 5000),
        'depth':  (500, 8000),
        'length': (1000, 20000),
        'width':  (5000, 10000),
        'strike': (0, 90),
        'dip':    (30, 89),
        'rake':   (60, 120),
        'slip':   (0.01, 2.0),
        'opening': (0, 0),
    }
    initial = {
        'X0':     1000.0,       # m east  of reference
        'Y0':     1000.0,       # m north of reference
        'depth':  6000.0,    # m
        'length': 15000.0,    # m
        'width':  5500.0,    # m
        'strike': 80.0,      # degrees
        'dip':    35.0,      # degrees
        'rake':   62.0,      # degrees (pure thrust)
        'slip':   1,       # m
        'opening': 0.0,
    }

    step_sizes = {
        'X0': 200, 'Y0': 200, 'depth': 200, 'length': 500, 'width': 300,
        'strike': 2, 'dip': 2, 'rake': 2, 'slip': 0.05, 'opening': 0,
    }
    max_steps = {
        'X0': 5000, 'Y0': 5000, 'depth': 5000, 'length': 15000, 'width': 10000,
        'strike': 90, 'dip': 60, 'rake': 60, 'slip': 3, 'opening': 0,
    }

    print('\n' + '='*70)
    print('SYNTHETIC TEST: Okada 2-IFG (ascending + descending)')
    print('True parameters:')
    for k, v in true_params.items():
        print(f'  {k:10s}: {v}')
    print('='*70 + '\n')

    samples, log_lik_trace, rms_evolution, pkl = run_baysian_inference(
        u_los_obs=[los1, los2],
        X_obs=[X_obs1, X_obs2],
        Y_obs=[Y_obs1, Y_obs2],
        incidence_angle=[inc1, inc2],
        heading=[head1, head2],
        n_iterations=n_iterations,
        sill=[sill, sill],
        nugget=[nugget, nugget],
        range_param=[range_param, range_param],
        initial_params=[initial],
        priors=[priors],
        proposal_std=[step_sizes],
        max_step_sizes=[max_steps],
        adaptive_interval=n_iterations // 100,
        target_acceptance=0.23,
        figure_folder=figure_folder,
        use_sa_init=True,
        model_type=['okada'],
        burn_in=int(n_iterations * 0.3),
        u_los_obs_already_los_in_m=True,   # data already in metres
        fit_ramp=True,
        noise_rms=noise_rms,
        sill_in_m2=True,
        # GBIS_output_clean_comparison.py's build_step03_dataframe() expects the pickle's
        # reference_point as [lat, lon] (matching the other real-data blocks in this file);
        # this function's own `reference_point` param is (lon0, lat0) -- convert here so the
        # saved pickle actually carries a usable reference point (previously this was never
        # passed at all, so it defaulted to None and the GBIS/py-GBIS comparison plots
        # silently skipped the py-GBIS side).
        reference_point=[reference_point[1], reference_point[0]],
    )

    print('\nRecovery (posterior mean vs truth):')
    print(f'  {"param":10s}  {"true":>12s}  {"mean":>12s}  {"error":>12s}')
    for p in true_params:
        if p == 'opening':
            continue
        mean_val = float(np.mean(samples[p]))
        true_val = true_params[p]
        print(f'  {p:10s}  {true_val:12.4f}  {mean_val:12.4f}  {mean_val - true_val:+12.4f}')

    return samples


if __name__ == "__main__":
    # synthetic_test('okada', n_iterations=int(1e5))   # single-IFG Okada smoke test
    # synthetic_test_okada_2ifg()                       # 2-IFG recovery test (uncomment to run)
    # synthetic_test('pcdm')
    # synthetic_test(['pcdm', 'okada'])          # joint inversion
    # synthetic_test_multi_ifg()                 # multi-IFG smoke test
    # Example with custom parameters
    custom_initial = {
        'X0': 0,
        'Y0': 0,
        'depth': 620 ,
        'DVx': -1e5,
        'DVy': -1e5,
        'DVz': -1e5,
        'omegaX': 10,
        'omegaY': 10,
        'omegaZ': -10
    }
    #Input Priors here
    custom_priors = {
        'X0': (-1500,1500),
        'Y0': (-1500, 1500),
        'depth': (100, 1500),
        'DVx': (-1e8, -1e1),
        'DVy': (-1e8, -1),
        'DVz': (-1e8, -1e1),
        'omegaX': (0, 90),
        'omegaY': (-45, 45),
        'omegaZ': (-45, 25)
    }
    # Intial Learning Rates Here
    custom_learning_rates = {
        'X0': 10,
        'Y0': 10,
        'depth': 10,
        'DVx': 1e3,
        'DVy': 1e3,
        'DVz': 1e3,
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

    custom_initial_two = {
        'X0': 0,
        'Y0': 0,
        'depth': 620 ,
        'DVx': 1e5,
        'DVy': 1e5,
        'DVz': 1e5,
        'omegaX': 20,
        'omegaY': -20,
        'omegaZ': -20
    }
    #Input Priors here
    custom_priors_two = {
        'X0': (-1500,1500),
        'Y0': (-1500, 1500),
        'depth': (100, 35000),
        'DVx': (1e1, 1e8),
        'DVy': (1e1, 1e8),
        'DVz': (1e1, 1e9),
        'omegaX': (0, 90),
        'omegaY': (-90, 0),
        'omegaZ': (-90, 0)
    }
    # Intial Learning Rates Here
    custom_learning_rates_two = {
        'X0': 10,
        'Y0': 10,
        'depth': 10,
        'DVx': 1e3,
        'DVy': 1e3,
        'DVz': 1e3,
        'omegaX': 1,
        'omegaY': 1,
        'omegaZ': 1
    }
    # Max Step Sizes Here
    max_step_sizes_two = {
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
        'depth': 6000,
        'length': 7432,
        'width': 7432,
        'strike': 308,
        'dip': 28,
        'rake': 80,
        'slip': 1.0,
        'opening': 0
    }
    #Input Priors here
    custom_priors_o = {
        'X0': (-25000,25000),
        'Y0': (-25000, 25000),
        'depth': (100, 15000),
        'length': (500, 25000),
        'width': (1000, 12500),
        'strike': (270, 360),
        'dip': (0, 45),
        'rake': (-180, 180),
        'slip': (0, 3.0),
        'opening': (0,0)
    }
    # Intial Learning Rates Here
    custom_learning_rates_o = {
        'X0': 1000,
        'Y0': 1000,
        'depth': 1000,
        'length': 1000,
        'width': 1000,
        'strike': 20,
        'dip': 20,
        'rake': 20,
        'slip': 0.5,
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




  

    custom_initial_mogi = {
        'X0': 0,
        'Y0': 0,
        'depth': 620,
        'DV': 1e4
    }

    custom_priors_mogi = {
        'X0': (-1500,1500),
        'Y0': (-1500, 1500),
        'depth': (100, 5000),
        'DV': (1e2, 1e8)
    }

    custom_learning_rates_mogi = {
        'X0': 10,
        'Y0': 10,
        'depth': 10,
        'DV': 1e4
    }

    max_step_sizes_mogi = {
            'X0': 1000.0,
            'Y0': 1000.0,
            'depth': 1000.0,
            'DV': 1e8
    }
 

    # ====================================================================
    # DATA LOADING — edit the three variables below, then uncomment one
    # inversion block further down
    # ====================================================================

    # Path to .npy file produced by step01
    datapath = '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/20210402_20210520.geo.unw_processed.npy'

    # Variogram parameters — paste values printed by step02
    sill        = 1.93e1   # total variance (rad²)
    nugget      = 4.54e-19    # nugget (rad²)
    range_param = 111939  # correlation length (m)

    # Run settings
    number_of_iterations                    = int(1e6)
    use_simulated_annealing_for_first_guess = True

    # ====================================================================
    # DATA LOADING — no edits needed below for standard GBIS .npy format
    # ====================================================================
    data_dict = np.load(datapath, allow_pickle=True).item()

    # Phase is unwrapped InSAR phase in radians.
    # run_baysian_inference converts to LOS metres internally:
    #   u_los = -phase * wavelength / (4π)
    # Do NOT manually convert here.
    u_los_obs = np.array(data_dict['Phase']).flatten()
    Lon       = np.array(data_dict['Lon']).flatten()
    Lat       = np.array(data_dict['Lat']).flatten()

    # Inc and Heading are in degrees — pass directly, no unit conversion needed.
    incidence_angle = np.array(data_dict['Inc']).flatten()
    heading         = np.array(data_dict['Heading']).flatten()

    # Reference point taken from the file's centre pixel (set by step01).
    referencePoint = [float(data_dict['center_lat']), float(data_dict['center_lon'])]
    referencePoint = [29.753,50.678]
    X_obs, Y_obs = convert_lat_long_2_xy(Lat, Lon, referencePoint[0], referencePoint[1])
    print(f"Reference point: {referencePoint}")
    print(f"X range: {X_obs.min():.0f} – {X_obs.max():.0f} m   "
          f"Y range: {Y_obs.min():.0f} – {Y_obs.max():.0f} m   "
          f"N={len(u_los_obs)}")

    # ====================================================================
    # SINGLE-IFG INVERSION
    # Uncomment exactly one block below (or write your own using the same
    # pattern — pass lists of length 1 for initial_params / priors / etc.)
    # ====================================================================

    # --- Mogi ---
    # model_type = ['mogi']
    # samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
    #     u_los_obs=u_los_obs, X_obs=X_obs, Y_obs=Y_obs,
    #     incidence_angle=incidence_angle, heading=heading,
    #     n_iterations=number_of_iterations,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial_mogi],
    #     priors=[custom_priors_mogi],
    #     proposal_std=[custom_learning_rates_mogi],
    #     max_step_sizes=[max_step_sizes_mogi],
    #     adaptive_interval=number_of_iterations // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_mogi',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type,
    #     burn_in=int(number_of_iterations * 0.3),
    # )

    # --- pCDM ---
    # model_type = ['pCDM']
    # samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
    #     u_los_obs=u_los_obs, X_obs=X_obs, Y_obs=Y_obs,
    #     incidence_angle=incidence_angle, heading=heading,
    #     n_iterations=number_of_iterations,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial],
    #     priors=[custom_priors],
    #     proposal_std=[custom_learning_rates],
    #     max_step_sizes=[max_step_sizes],
    #     adaptive_interval=number_of_iterations // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_pcdm',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type,
    #     burn_in=int(number_of_iterations * 0.3),
    # )

    # --- Okada ---
    # model_type = ['okada']
    # samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
    #     u_los_obs=u_los_obs, X_obs=X_obs, Y_obs=Y_obs,
    #     incidence_angle=incidence_angle, heading=heading,
    #     n_iterations=number_of_iterations,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial_o],
    #     priors=[custom_priors_o],
    #     proposal_std=[custom_learning_rates_o],
    #     max_step_sizes=[max_step_sizes_o],
    #     adaptive_interval=number_of_iterations // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_okada',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type,
    #     burn_in=int(number_of_iterations * 0.3),
    # )

    # --- Joint multi-model single IFG (e.g. Mogi + Okada) ---
    # model_type = ['mogi', 'okada']
    # samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
    #     u_los_obs=u_los_obs, X_obs=X_obs, Y_obs=Y_obs,
    #     incidence_angle=incidence_angle, heading=heading,
    #     n_iterations=number_of_iterations,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial_mogi, custom_initial_o],
    #     priors=[custom_priors_mogi, custom_priors_o],
    #     proposal_std=[custom_learning_rates_mogi, custom_learning_rates_o],
    #     max_step_sizes=[max_step_sizes_mogi, max_step_sizes_o],
    #     adaptive_interval=number_of_iterations // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_mogi_okada',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type,
    #     burn_in=int(number_of_iterations * 0.3),
    # )

    # ====================================================================
    # MULTI-IFG INVERSION
    # Load each IFG the same way as above; pass lists of the per-IFG
    # arrays and variogram parameters to run_baysian_inference.
    # The covariance matrix is assembled as block-diagonal automatically.
    # ====================================================================
    datapath2      = '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/20210410_20210422.geo.unw_processed.npy'
    data_dict2     = np.load(datapath2, allow_pickle=True).item()
    u_los_obs2     = np.array(data_dict2['Phase']).flatten()
    Lon2, Lat2     = np.array(data_dict2['Lon']).flatten(), np.array(data_dict2['Lat']).flatten()
    inc2           = np.array(data_dict2['Inc']).flatten()
    head2          = np.array(data_dict2['Heading']).flatten()
    # Use IFG1's referencePoint as the common origin for all IFGs so that X0,Y0 is
    # consistent across the joint inversion (each file's own centre can differ by km).
    X_obs2, Y_obs2 = convert_lat_long_2_xy(Lat2, Lon2, referencePoint[0], referencePoint[1])
    sill2, nugget2, range_param2 = 2.977872e+00,1.063133e-14, 58345.3/3  # from step02 for IFG 2


    ######################################################### IFG 3 ########################################################

    datapath3      = '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/Dsc_20210329_20210422.geo.unw_processed.npy'
    data_dict3     = np.load(datapath3, allow_pickle=True).item()
    u_los_obs3     = np.array(data_dict3['Phase']).flatten()
    Lon3, Lat3     = np.array(data_dict3['Lon']).flatten(), np.array(data_dict3['Lat']).flatten()
    inc3           = np.array(data_dict3['Inc']).flatten()
    head3          = np.array(data_dict3['Heading']).flatten()
    X_obs3, Y_obs3 = convert_lat_long_2_xy(Lat3, Lon3, referencePoint[0], referencePoint[1])
    sill3, nugget3, range_param3 = 1.135832e+00,4.782810e-15, 47311.8/3
  # from step02 for IFG 2
    ######################################################## IFG 4 ########################################################

    datapath4      = '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/Asc_20210414_20210520.geo.unw_processed.npy'
    data_dict4     = np.load(datapath4, allow_pickle=True).item()
    u_los_obs4     = np.array(data_dict4['Phase']).flatten()
    Lon4, Lat4     = np.array(data_dict4['Lon']).flatten(), np.array(data_dict4['Lat']).flatten()
    inc4           = np.array(data_dict4['Inc']).flatten()
    head4          = np.array(data_dict4['Heading']).flatten()
    X_obs4, Y_obs4 = convert_lat_long_2_xy(Lat4, Lon4, referencePoint[0], referencePoint[1])
    sill4, nugget4, range_param4 = 2.766344e+01,9.750962e-19, 62039.9/3

    # ====================================================================
    # RELOAD SAVED STATE AND REGENERATE PLOTS
    # ====================================================================
    # pickle_file = 'Results_mogi/bayesian_inference_state_....pkl'
    # regenerate_plots_from_state(pickle_file, new_figure_folder='regenerated_plots')

    model_type = ['okada' ]  # joint inversion
    # Full-resolution .npy files for each IFG (or None to skip).
    # Set the path for whichever IFG you want displayed at full resolution
    # in the top row of Model_Comparison.png.
    # Index 0 → ifg_1 (20210402_20210520), Index 1 → ifg_2 (20210410_20210422).
    _full_res_npy_paths = [
        '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/20210402_20210520.geo.unw_processed_clipped_full_resolution.npy',  # ifg_1
        '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/20210410_20210422.geo.unw_processed_clipped_full_resolution.npy',  # ifg_2 — set a path here if a full-resolution file exists
        '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/Dsc_20210329_20210422.geo.unw_processed_clipped_full_resolution.npy',
        '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/Asc_20210414_20210520.geo.unw_processed_clipped_full_resolution.npy'
    ]

#     samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
#     u_los_obs=[u_los_obs, u_los_obs2, u_los_obs3, u_los_obs4],
#     X_obs=[X_obs, X_obs2, X_obs3, X_obs4],
#     Y_obs=[Y_obs, Y_obs2, Y_obs3, Y_obs4],
#     incidence_angle=[incidence_angle, inc2, inc3, inc4],
#     heading=[heading, head2, head3, head4],
#     n_iterations=number_of_iterations,
#     sill=[sill, sill2, sill3, sill4],
#     nugget=[nugget, nugget2, nugget3, nugget4],
#     range_param=[range_param, range_param2, range_param3, range_param4],
#     initial_params=[custom_initial_o],
#     priors=[custom_priors_o],
#     proposal_std=[custom_learning_rates_o],
#     max_step_sizes=[max_step_sizes_o],
#     adaptive_interval=number_of_iterations // 100,
#     target_acceptance=0.23,
#     figure_folder='Results_okada_2ifg_1e6_again',
#     use_sa_init=use_simulated_annealing_for_first_guess,
#     model_type=model_type,
#     burn_in=int(number_of_iterations * 0.3),
#     full_res_npy_paths=_full_res_npy_paths,
#     fit_ramp='offset',
#     reference_point=referencePoint,
# )

  
    # ====================================================================
    referencePoint_e2k3 = [29.7546416200275, 50.63366157664304]  # geo.referencePoint from us6000e2k3_NP1.inp (lat, lon)

    custom_initial_e2k3_o = {
        'X0': 2020.0058, 'Y0': 2585.4895, 'depth': 9744.5563, 'length': 7432, 'width': 7432,
        'strike': 308, 'dip': 28, 'rake': 99.72, 'slip': 0.379043, 'opening': 0
    }
    custom_priors_e2k3_o = {
        'X0': (-10393.994, 14434.006), 'Y0': (-9828.510, 14999.490), 'depth': (2544.556, 17744.556),
        'length': (3716, 37161), 'width': (1114, 22296), 'strike': (168, 328),
        'dip': (5,58.41), 'rake': (-180, 180), 'slip': (0, 7.383742), 'opening': (0, 0)
    }
    custom_learning_rates_e2k3_o = {
        'X0': 100, 'Y0': 100, 'depth': 500, 'length': 100, 'width': 100,
        'strike': 1, 'dip': 1, 'rake': 1.511591, 'slip': 0.01, 'opening': 0
    }
    
    _e2k3_dir = '/uolstore/Research/a/a285/homes/ee18jwc/code/py-GBIS/pyGBIS_converted'
    _e2k3_ifgs = [
        # (npy filename, sill [m²], range [m], nugget [m²]) -- from us6000e2k3_NP1.inp, already in m²
        ('GEOC_137D_05960_131313_floatml_clipped_signal_masked_QAed_20210417_20210511.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 0.00011110153481611631, 81269.45085299318, 1.6178879457358917e-05),
        ('GEOC_035D_05978_131209_floatml_GACOS_Corrected_clipped_signal_masked_QAed_20210410_20210422.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 1.938607495675568e-05, 42152.07588180337, 4.887593708277281e-08),
        ('GEOC_137D_05960_131313_floatml_clipped_signal_masked_QAed_20210417_20210523.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 4.183159921684887e-05, 36760.12211572188, 4.176442585761907e-06),
        ('GEOC_101A_05977_060913_floatml_GACOS_Corrected_clipped_signal_masked_QAed_20210414_20210520.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 0.0003652472020324005, 65501.7540753663, 4.005498249489375e-05),
        ('GEOC_101A_05977_060913_floatml_GACOS_Corrected_clipped_signal_masked_QAed_20210414_20210508.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 0.0006908417166250139, 48573.2026638629, 6.52086959230951e-05),
        ('GEOC_101A_05977_060913_floatml_GACOS_Corrected_clipped_signal_masked_QAed_20210402_20210508.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 0.0003577606345341974, 33381.66282043795, 3.207697037302865e-05),
        ('GEOC_137D_05960_131313_floatml_clipped_signal_masked_QAed_20210405_20210511.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 1.1212465783785728e-05, 39001.2099057197, 2.183671261319572e-06),
        ('GEOC_035D_05978_131209_floatml_GACOS_Corrected_clipped_signal_masked_QAed_20210329_20210422.ds_unw_Lon_Lat_Inc_Heading.GBIS.npy', 2.8585501864699194e-05, 34613.46629159857, 1.6672004707714093e-06),
    ]
    
    u_los_list, X_list, Y_list, inc_list, head_list = [], [], [], [], []
    sill_list, nugget_list, range_list = [], [], []
    for _fname, _sill_i, _range_i, _nugget_i in _e2k3_ifgs:
        _dd = np.load(os.path.join(_e2k3_dir, _fname), allow_pickle=True).item()
        _lat_i = np.array(_dd['Lat']).flatten()
        _lon_i = np.array(_dd['Lon']).flatten()
        _Xi, _Yi = convert_lat_long_2_xy(_lat_i, _lon_i, referencePoint_e2k3[0], referencePoint_e2k3[1])
        u_los_list.append(np.array(_dd['Phase']).flatten())
        X_list.append(_Xi)
        Y_list.append(_Yi)
        inc_list.append(np.array(_dd['Inc']).flatten())
        head_list.append(np.array(_dd['Heading']).flatten())
        sill_list.append(_sill_i)
        range_list.append(_range_i)
        nugget_list.append(_nugget_i)
    
    # model_type = ['okada']  # joint inversion — swap/add model names as needed
    # samples, log_lik_trace, rms_evolution, pickle_filename = run_baysian_inference(
    #     u_los_obs=u_los_list, X_obs=X_list, Y_obs=Y_list,
    #     incidence_angle=inc_list, heading=head_list,
    #     n_iterations=number_of_iterations,
    #     sill=sill_list, nugget=nugget_list, range_param=range_list,
    #     initial_params=[custom_initial_e2k3_o],
    #     priors=[custom_priors_e2k3_o],
    #     proposal_std=[custom_learning_rates_e2k3_o],
    #     max_step_sizes=[max_step_sizes_o],
    #     adaptive_interval=number_of_iterations // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_okada_us6000e2k3_8ifg',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type,
    #     burn_in=int(number_of_iterations * 0.3),
    #     fit_ramp='offset',
    #     reference_point=referencePoint_e2k3,
    #     sill_in_m2=True,  # sill/nugget values in _e2k3_ifgs are already in m²
    # )

    # ====================================================================
    # DIVIDER — Mogi + Okada joint single-IFG inversion.
    # Priors are kept deliberately wide (horizontal bounds derived from the
    # data's own footprint, DV allowed to be positive or negative, dip/strike/
    # rake left essentially unconstrained) since the source type/geometry for
    # this IFG isn't known ahead of time.
    # ====================================================================
    # divider
    data = np.load('./divider/19920424_19930305.diff.unw.geo_downsampled.npy', allow_pickle=True)
    data_dict = data.item()
    nugget = 2.283109e-05
    sill = 4.099456e-05
    range_param = 7098.4

    # Other step02 sill/nugget/range estimates tried for this same IFG (kept for
    # reference in case the values above turn out not to fit the noise well):
    # nugget = 2.237931e-05
    # sill   = 2.868079e-05
    # range_param = 3676.1
    #
    # nugget = 7.475808e-06
    # sill   = 5.775924e-05
    # range_param = 44960.6

    number_of_iterations_divider = int(1e6)  # "runs for 1e6 iterations" — overrides the
                                              # 3e5 above, which was left over from a
                                              # shorter test run
    model_type_divider = ['PCDM']  # joint inversion — swap/add model names as needed
    referencePoint_divider = [37.02068, -115.98791]  # divider (lat, lon)

 
    u_los_obs_divider = np.array(data_dict['Phase'].T).flatten()
    Lon_divider = np.array(data_dict['Lon']).flatten()
    Lat_divider = np.array(data_dict['Lat']).flatten()
    incidence_angle_divider = np.full(Lon_divider.shape, 20.5800)
    heading_divider = np.full(Lon_divider.shape, 192.47)

    X_obs_divider, Y_obs_divider = convert_lat_long_2_xy(
        Lat_divider, Lon_divider, referencePoint_divider[0], referencePoint_divider[1])
    print(f"[divider] Reference point: {referencePoint_divider}")
    print(f"[divider] X range: {X_obs_divider.min():.0f} - {X_obs_divider.max():.0f} m   "
          f"Y range: {Y_obs_divider.min():.0f} - {Y_obs_divider.max():.0f} m   "
          f"N={len(u_los_obs_divider)}")

    # Wide priors: horizontal bounds come from the data's own footprint (padded by 50%)
    # rather than a hand-picked number, since the true source location isn't known yet.
    _extent_divider = max(X_obs_divider.max() - X_obs_divider.min(),
                          Y_obs_divider.max() - Y_obs_divider.min())
    _xy_bound_divider = 0.05 * _extent_divider

    custom_initial_mogi_divider = {
        'X0': 0, 'Y0': 0, 'depth': 500, 'DV': 1e4,
    }
    custom_priors_mogi_divider = {
        'X0': (-1000, 1000),
        'Y0': (-1000, 1000),
        'depth': (50, 1000),
        'DV': (-1e5, 1e5),   # sign (inflation vs. deflation) is not assumed
    }
    custom_learning_rates_mogi_divider = {
        'X0': 0.02 * _xy_bound_divider, 'Y0': 0.02 * _xy_bound_divider,
        'depth': 500, 'DV': 5e3,
    }
    max_step_sizes_mogi_divider = {
        'X0': _xy_bound_divider, 'Y0': _xy_bound_divider, 'depth': 20000, 'DV': 1e8,
    }

    custom_initial_okada_divider = {
        'X0': 0, 'Y0': 0, 'depth': 500, 'length': 500, 'width': 500,
        'strike': 180, 'dip': 45, 'rake': -90, 'slip': 0.5, 'opening': 0,
    }
    custom_priors_okada_divider = {
        'X0': (-2000, 2000),
        'Y0': (-2000, 2000),
        'depth': (50, 2000), 'length': (50, 30000), 'width': (50, 20000),
        'strike': (150, 225), 'dip': (30, 60), 'rake': (-100, 30),
        'slip': (0, 1.0), 'opening': (0, 0),
    }
    custom_learning_rates_okada_divider = {
        'X0': 0.02 * _xy_bound_divider, 'Y0': 0.02 * _xy_bound_divider, 'depth': 500,
        'length': 500, 'width': 500, 'strike': 20, 'dip': 10, 'rake': 20,
        'slip': 0.2, 'opening': 0.0,
    }
    max_step_sizes_okada_divider = {
        'X0': _xy_bound_divider, 'Y0': _xy_bound_divider, 'depth': 20000,
        'length': 30000, 'width': 20000, 'strike': 360, 'dip': 90, 'rake': 360,
        'slip': 5.0, 'opening': 0.0,
    }
    model_type_divider = ['okada','mogi' ]  # joint inversion
    samples_divider, log_lik_trace_divider, rms_evolution_divider, pickle_filename_divider = run_baysian_inference(
        u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
        incidence_angle=incidence_angle_divider, heading=heading_divider,
        n_iterations=number_of_iterations_divider,
        sill=sill, nugget=nugget, range_param=range_param,
        initial_params=[custom_initial_okada_divider,custom_initial_mogi_divider],
        priors=[custom_priors_okada_divider,custom_priors_mogi_divider],
        proposal_std=[custom_learning_rates_okada_divider,custom_learning_rates_mogi_divider],
        max_step_sizes=[max_step_sizes_okada_divider,max_step_sizes_mogi_divider],
        adaptive_interval=number_of_iterations_divider // 100,
        target_acceptance=0.23,
        figure_folder='Results_divider_okada_mogi_current',
        use_sa_init=use_simulated_annealing_for_first_guess,
        model_type=model_type_divider,
        burn_in=int(number_of_iterations_divider * 0.3),
        reference_point=referencePoint_divider,
        u_los_obs_already_los_in_m=True, 
        fit_ramp='linear'
    )

    model_type_divider = ['okada']  # joint inversion
    samples_divider, log_lik_trace_divider, rms_evolution_divider, pickle_filename_divider = run_baysian_inference(
        u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
        incidence_angle=incidence_angle_divider, heading=heading_divider,
        n_iterations=number_of_iterations_divider,
        sill=sill, nugget=nugget, range_param=range_param,
        initial_params=[custom_initial_okada_divider],
        priors=[custom_priors_okada_divider],
        proposal_std=[custom_learning_rates_okada_divider],
        max_step_sizes=[max_step_sizes_okada_divider],
        adaptive_interval=number_of_iterations_divider // 100,
        target_acceptance=0.23,
        figure_folder='Results_divider_okada_current',
        use_sa_init=use_simulated_annealing_for_first_guess,
        model_type=model_type_divider,
        burn_in=int(number_of_iterations_divider * 0.3),
        reference_point=referencePoint_divider,
        u_los_obs_already_los_in_m=True, 
        fit_ramp='linear'
    )

    model_type_divider = ['pCDM']  # joint inversion
    samples_divider, log_lik_trace_divider, rms_evolution_divider, pickle_filename_divider = run_baysian_inference(
            u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
            incidence_angle=incidence_angle_divider, heading=heading_divider,
            n_iterations=number_of_iterations_divider,
            sill=sill, nugget=nugget, range_param=range_param,
            initial_params=[custom_initial],
            priors=[custom_priors],
            proposal_std=[custom_learning_rates],
            max_step_sizes=[max_step_sizes],
            adaptive_interval=number_of_iterations_divider // 100,
            target_acceptance=0.23,
            figure_folder='Results_divider_pCDM_current',
            use_sa_init=use_simulated_annealing_for_first_guess,
            model_type=model_type_divider,
            burn_in=int(number_of_iterations_divider * 0.3),
            reference_point=referencePoint_divider,
            u_los_obs_already_los_in_m=True, 
            fit_ramp='linear'
        )


       
    model_type_divider = ['une_knothe']
    custom_initial_une_knothe = {
       'X0': 0.0, 'Y0': 0.0,
       'depth': 500.0, 'cavity_radius_m': 21,
       'dv_factor': 0.05,
       'chimney_volume_fraction': 0.05,
       'chimney_x': 0.0, 'chimney_y': 0.0,
       'chimney_sigma_ratio': 1.73, 'chimney_rotation_deg': -21.6,
       'knothe_influence_angle_deg': 30.0,
       }
   
    custom_priors_une_knothe = {
    'X0': (-1500, 1500),
    'Y0': (-1500, 1500),
    'depth': (100, 1000),
    'cavity_radius_m': (10, 100),
    'dv_factor': (0.01, 1.0),
    'chimney_volume_fraction': (0.01, 1),
    'chimney_x': (-500, 500),
    'chimney_y': (-500, 500),
    'chimney_sigma_ratio': (0.5, 4),
    'chimney_rotation_deg': (-90, 90),
    'knothe_influence_angle_deg': (5, 80)
    }

    custom_learning_rates_une_knothe = {
    'X0': 100,
    'Y0': 100,
    'depth': 100,
    'cavity_radius_m': 10,
    'dv_factor': 0.01,
    'chimney_volume_fraction': 0.01,
    'chimney_x': 10.0,
    'chimney_y': 10.0,
    'chimney_sigma_ratio': 0.1,
    'chimney_rotation_deg': 1.0,
    'knothe_influence_angle_deg': 1.0
    }

    max_step_sizes_une_knothe = {
    'X0': 1000.0,
    'Y0': 1000.0,
    'depth': 1000.0,
    'cavity_radius_m': 50,
    'dv_factor': 0.1,
    'chimney_volume_fraction': 0.1,
    'chimney_x': 100.0,
    'chimney_y': 100.0,
    'chimney_sigma_ratio': 0.5,
    'chimney_rotation_deg': 20.0,
    'knothe_influence_angle_deg': 10.0
    }
   

    # samples_divider, log_lik_trace_divider, rms_evolution_divider, pickle_filename_divider = run_baysian_inference(
    #         u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
    #         incidence_angle=incidence_angle_divider, heading=heading_divider,
    #         n_iterations=number_of_iterations_divider,
    #         sill=sill, nugget=nugget, range_param=range_param,
    #         initial_params=[custom_initial_une_knothe],
    #         priors=[custom_priors_une_knothe],
    #         proposal_std=[custom_learning_rates_une_knothe],
    #         max_step_sizes=[max_step_sizes_une_knothe],
    #         adaptive_interval=number_of_iterations_divider // 100,
    #         target_acceptance=0.23,
    #         figure_folder='Results_divider_une_knothe',
    #         use_sa_init=use_simulated_annealing_for_first_guess,
    #         model_type=model_type_divider,
    #         burn_in=int(number_of_iterations_divider * 0.3),
    #         reference_point=referencePoint_divider,
    #         u_los_obs_already_los_in_m=True,
    #         fit_ramp='linear'
    #     )

    # ====================================================================
    # DIVIDER — UNE_forward_CDM_collapse (two-stage McTigue uplift + elastic
    # closing-column collapse). Reuses the divider data/sill/nugget/range
    # loaded above. Wide priors: source geometry/type is not known ahead of
    # time for this dataset (same philosophy as the mogi+okada block above).
    #
    # collapse_model is fixed at 'column' (the physical option) and
    # chimney_anchor left at the module default ('cavity') inside the
    # model_registry.py wrapper -- both are categorical choices, not
    # continuous parameters to explore via MCMC. height_fac is likewise
    # fixed (not independent of phi_bulk -- see CollapseBudget in
    # UNE_forward_CDM_collapse.py); phi_bulk is the free parameter instead.
    #
    # NOTE: the collapse column sums ~864 elastic point-source elements per
    # forward-model call; UNE_forward_CDM_collapse.py's closing_column() was
    # changed to call a Numba-compiled kernel (_closing_column_core) instead
    # of a plain Python loop over those elements, since the loop version
    # measured ~113 ms/call here (~2-3 days for 1e6 iterations) vs ~2 ms/call
    # compiled (~1 hour for 1e6) -- verified numerically identical to the
    # loop version to floating-point precision first.
    # ====================================================================
    model_type_divider = ['une_cdm_collapse']

    # axis_ratio/rotation_deg are no longer sampled -- the column is held circular
    # (fixed at axis_ratio=1, rotation_deg=0 inside model_registry.py's wrapper).
    # Both were unidentifiable from this single-LOS interferogram: rotation_deg came
    # back at 0.8 +/- 52.0 deg, spanning essentially its whole (-90,90) prior, because
    # rotating a circle is a no-op and axis_ratio's posterior included near-circular
    # values. Separating an ellipticity from its azimuth needs a second look geometry.
    custom_initial_une_cdm = {
        'X0': 0.0, 'Y0': 0.0, 'depth': 500.0, 'r_c': 21.0,
        'dv_factor': 0.05, 'phi_bulk': 0.225,
        'chimney_x': 0.0, 'chimney_y': 0.0, 'drift_exponent': 1.0,
    }
    custom_priors_une_cdm = {
        'X0': (-_xy_bound_divider, _xy_bound_divider),
        'Y0': (-_xy_bound_divider, _xy_bound_divider),
        'depth': (100, 2000), 'r_c': (5, 150),
        'dv_factor': (0.01, 1.0), 'phi_bulk': (0.01, 0.2424),
        'chimney_x': (-300, 300), 'chimney_y': (-300, 300),
        'drift_exponent': (0.01, 5.0),
    }
    custom_learning_rates_une_cdm = {
        'X0': 100, 'Y0': 100, 'depth': 50, 'r_c': 5,
        'dv_factor': 0.02, 'phi_bulk': 0.02,
        'chimney_x': 20, 'chimney_y': 20, 'drift_exponent': 0.1,
    }
    max_step_sizes_une_cdm = {
        'X0': _xy_bound_divider, 'Y0': _xy_bound_divider, 'depth': 2000, 'r_c': 150,
        'dv_factor': 1.0, 'phi_bulk': 0.45,
        'chimney_x': 500, 'chimney_y': 500, 'drift_exponent': 3.0,
    }

    # samples_divider, log_lik_trace_divider, rms_evolution_divider, pickle_filename_divider = run_baysian_inference(
    #     u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
    #     incidence_angle=incidence_angle_divider, heading=heading_divider,
    #     n_iterations=number_of_iterations_divider,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial_une_cdm],
    #     priors=[custom_priors_une_cdm],
    #     proposal_std=[custom_learning_rates_une_cdm],
    #     max_step_sizes=[max_step_sizes_une_cdm],
    #     adaptive_interval=number_of_iterations_divider // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_divider_une_cdm_collapse',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type_divider,
    #     burn_in=int(number_of_iterations_divider * 0.3),
    #     reference_point=referencePoint_divider,
    #     u_los_obs_already_los_in_m=True,
    #     fit_ramp='linear',
    # )

    


    # ====================================================================
    # DIVIDER — UNE_forward_CDM_stack (stacked compound dislocation models).
    # Same two physical stages and the same nine free parameters as the
    # une_cdm_collapse block above, so the two runs are directly comparable;
    # what changes is the elastic source. Every layer is a CDM (Nikkhoo et
    # al. 2017, GJI 208, 877-894) instead of a McTigue sphere plus a cloud of
    # 864 Mogi points: layer 0 is the co-explosive uplift CDM at the working
    # point, layers 1..N are closing CDMs stacked face to face up the chimney
    # column. Divider needs that uplift layer because its interferogram
    # brackets the shot; Foxall (2000, UCRL-JC-138986) modelled JUNCTION with
    # three stacked closing layers alone, since that interferogram starts a
    # month after the event.
    #
    # Fixed inside model_registry.py's wrapper, not sampled: n_collapse
    # (UNE_CDM_STACK_N_COLLAPSE -- an integer layer count cannot be explored
    # by a Gaussian random walk; 1 gives the two-layer stack, 3 gives
    # Foxall's geometry), closure='vertical', height_fac (not independent of
    # phi_bulk), and axis_ratio/rotation_deg (held circular for the same
    # identifiability reason as in the une_cdm_collapse run).
    #
    # The wrapper uses volume_convention='mogi', so dv_factor and phi_bulk
    # mean the same cavity volume changes as in the une_cdm_collapse run and
    # the priors below carry across unchanged. The surface bowl VOLUME then
    # matches that model to ~0.1%, but the bowl SHAPE does not and should
    # not: a horizontal closing crack concentrates subsidence about 3x more
    # sharply than an isotropic point source of the same volume (peak u_z
    # 3/(2 pi d^2) per unit potency versus 1/(2 pi d^2)). Expect the fitted
    # phi_bulk/r_c to shift accordingly rather than reproducing the Mogi MAP.
    #
    # Cost: ~1.5 ms per forward call at 3000 points for the two-layer stack
    # (~25 min per 1e6 iterations), ~2.8 ms for four layers.
    # ====================================================================
    # Parameterised by the two stage VOLUMES ('une_cdm_stack_vol'), not by
    # dv_factor/phi_bulk. The dimensionless form is still registered as
    # 'une_cdm_stack' if a like-for-like comparison with the une_cdm_collapse
    # run is wanted, but dv_factor is degenerate with r_c -- layer 0's surface
    # field depends only on their product dv_factor * V_cav(r_c), to within a
    # ~2% finite-source correction over r_c = 15-40 m -- so sampling both just
    # buys a banana. r_c stays in the volume parameterisation because there it
    # is pure geometry: it sets H_c = 5.5 r_c and the column radius, and at
    # FIXED volumes it still swings the collapse peak from -56 to -209 mm over
    # r_c = 15-60 m by moving the column's depth extent.
    #
    # The bulking budget becomes a prior instead of a hard tie (see
    # _prior_une_cdm_stack_vol): the residual void cannot exceed the cavity
    # void, and the implied phi_bulk must stay above ~0.05, which caps
    # dV_collapse at 0.794 * V_cav.
    model_type_divider = ['une_cdm_stack_vol']

    # --- structural choices for this run --------------------------------
    # n_collapse (how many CDMs the chimney column is cut into) cannot be
    # sampled -- it is an integer -- so it and the other structural constants
    # live as module constants on model_registry and are set HERE, so this
    # block records the whole configuration of the run rather than leaving
    # half of it in another file.
    #
    # n_collapse = 16: each collapse layer is a horizontal closing crack (a
    # rectangular tensile dislocation of zero dip), whose field does not
    # depend on a thickness it does not have.  The cracks are EQUALLY SPACED,
    # one at the centre of each of n equal depth intervals of the column, so
    # the stack approximates closure distributed continuously over depth with
    # an error falling as n^-2.  Against a 1024-crack reference, n = 16 is
    # within 0.02 mm in LOS for a typical geometry (d = 400 m, r_c = 21 m) and
    # within 0.2 mm at the most demanding contained geometry (d / r_c = 8.6,
    # where the top cracks come closest to the surface) -- both far below the
    # noise.  n = 8 would be 0.07 / 0.8 mm at half the cost (~2.5 vs ~5 ms per
    # forward call over 5184 points).
    import model_registry as _model_registry
    _model_registry.UNE_CDM_STACK_N_COLLAPSE = 16
    #
    # loss_exponent -- WHERE IN THE COLUMN THE VOID CLOSES.  The closing
    # volume is spread over the chimney as w(zeta) ~ zeta ** p, zeta running
    # 0 at the cavity roof to 1 at the chimney tip:
    #     0     closure spread uniformly over the whole column;
    #     1     linear ramp, none at the roof rising to a maximum at the tip;
    #     >1    closure concentrated in the uppermost part of the column;
    #     >>1   effectively a single closing crack at the arrest horizon.
    # Physically it encodes how far the residual void migrates before caving
    # stalls: rubble low in the column has already bulked and compacted, so
    # void accumulates upward.  Observationally it sets the depth of the
    # closure centroid, hence the WIDTH of the bowl; the amplitude is
    # absorbed by dV_collapse.
    #
    # FIXED at 0 -- closure spread EQUALLY across the chimney -- not sampled.  Note what the sampled `depth` actually is:
    # the WORKING POINT depth (the uplift CDM sits there, and the column
    # hangs below it, from depth - r_c up to depth - 5.5 r_c).  So `depth`
    # and this exponent shift the closure centroid in the same way, and with
    # both stage volumes free they trade along a ridge flat to ~0.2 mm on a
    # 100 mm signal (p = 5 at depth 338 m is matched by p = 1 at 324 m).
    # Sampling it gave a marginal that simply climbed to its prior bound --
    # prior volume along the ridge, not information.  The identifiable
    # quantity is the closure centroid depth, which
    # report_chimney_geometry() prints from the posterior.  Change this
    # constant to test a more top-heavy column as a sensitivity check, but
    # do not sample it alongside depth.
    _model_registry.UNE_CDM_STACK_LOSS = 0.0   # uniform closure across the chimney

    custom_initial_une_cdm_stack = {
        'X0': 0.0, 'Y0': 0.0, 'depth': 500.0, 'r_c': 21.0,
        'dV_uplift': 5.0e3, 'dV_collapse': 4.0e3,
        'aspect_y': 1.0, 'aspect_z': 1.0,
        'source_omegaX': 0.0, 'source_omegaY': 0.0, 'source_omegaZ': 0.0,
        'chimney_x': 0.0, 'chimney_y': 0.0,
    }
    custom_priors_une_cdm_stack = {
        'X0': (-_xy_bound_divider, _xy_bound_divider),
        'Y0': (-_xy_bound_divider, _xy_bound_divider),
        'depth': (100, 2000), 'r_c': (5, 100),
        'dV_uplift': (0, 5.0e4), 'dV_collapse': (0, 5.0e4),
        # NOTE: phi_bulk is no longer sampled.  The chimney height is fixed
        # instead, at eta = 5.5 cavity radii (model_registry.
        # UNE_CDM_STACK_HEIGHT_FAC), and the bulking porosity is reported as
        # a derived product.  The pair is not identifiable: phi_bulk acts
        # only through H_c = eta r_c, which moves the top of the column,
        # while `depth` moves the whole column, and sweeping phi_bulk over
        # 0.15-0.40 then refitting `depth` alone leaves 0.01-0.06 mm rms on a
        # 7.6 mm signal.  Sampling it gave a marginal that railed at its
        # upper bound -- and, via simulated annealing, an initial state that
        # violated the budget and aborted the run.
        # Shape of the co-explosive CDM: aspect_y = ay/ax, aspect_z = az/ax,
        # with r_c setting its size as the radius of the sphere of equal
        # volume.  Both 1 makes the source equant, hence isotropic, hence
        # identical to a Mogi point source -- so fixing them would assume the
        # explosion was spherically symmetric.  A flattened cavity and damage
        # zone is likely where the rock is layered.  At fixed volume the shape
        # changes the uplift signal substantially (peak 5 to 37 mm across this
        # range), and with the depth pinned by the collapse stage it is
        # separable from volume at the 4-7 mm level.
        'aspect_y': (0.2, 10.0), 'aspect_z': (0.01, 5.0),
        # Orientation of the co-explosive CDM: the two tilts of its axes away
        # from vertical.  The CDM's third rotation, the azimuth about the
        # vertical, is carried for completeness but should be expected to go
        # unresolved: at this depth it moves the field by 0.01 mm for an
        # axisymmetric source and 0.56 mm even for a triaxial one, against
        # 6-10 mm for either tilt.  All three act only to the extent the
        # source is aspherical (a sphere is invariant under rotation), so
        # expect them to be unresolved if aspect_y and aspect_z both return
        # near 1.  A flat marginal on source_omegaZ is the expected outcome,
        # not a failure -- but check it against aspect_y before reading
        # anything into it.
        'source_omegaX': (-90.0, 90.0), 'source_omegaY': (-90.0, 90.0),
        'source_omegaZ': (-90.0, 90.0),
        'chimney_x': (-500, 500), 'chimney_y': (-500, 500),
    }
    custom_learning_rates_une_cdm_stack = {
        'X0': 100, 'Y0': 100, 'depth': 50, 'r_c': 5,
        'dV_uplift': 2.0e2, 'dV_collapse': 2.0e2,
        'aspect_y': 0.1, 'aspect_z': 0.1,
        'source_omegaX': 5.0, 'source_omegaY': 5.0, 'source_omegaZ': 5.0,
        'chimney_x': 20, 'chimney_y': 20,
    }
    max_step_sizes_une_cdm_stack = {
        'X0': _xy_bound_divider, 'Y0': _xy_bound_divider, 'depth': 2000, 'r_c': 100,
        'dV_uplift': 5.0e4, 'dV_collapse': 5.0e4,
        'aspect_y': 4.8, 'aspect_z': 4.8,
        'source_omegaX': 180.0, 'source_omegaY': 180.0,
        'source_omegaZ': 180.0,
        'chimney_x': 500, 'chimney_y': 500,
    }

    # samples_divider, log_lik_trace_divider, rms_evolution_divider, pickle_filename_divider = run_baysian_inference(
    #     u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
    #     incidence_angle=incidence_angle_divider, heading=heading_divider,
    #     n_iterations=number_of_iterations_divider,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial_une_cdm_stack],
    #     priors=[custom_priors_une_cdm_stack],
    #     proposal_std=[custom_learning_rates_une_cdm_stack],
    #     max_step_sizes=[max_step_sizes_une_cdm_stack],
    #     adaptive_interval=number_of_iterations_divider // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_divider_une_cdm_stack_vol',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type_divider,
    #     burn_in=int(number_of_iterations_divider * 0.3),
    #     reference_point=referencePoint_divider,
    #     u_los_obs_already_los_in_m=True,
    #     fit_ramp='linear',
    # )

    # The chimney axis tilt is a DERIVED quantity here: drift_exponent is
    # fixed at 1 and no prior constrains the offset between the uplift source
    # and the collapse column, so whatever tilt the data choose is a result,
    # not an assumption.  Print it for the record -- a caving column
    # propagates essentially vertically, so a large tilt is a statement about
    # the model, and is worth quoting either way.
    # import UNE_forward_CDM_stack as _une_stack
    # _une_stack.report_chimney_geometry(
    #     samples_divider, burn_in=int(number_of_iterations_divider * 0.3))

    # ====================================================================
    # DIVIDER — the same model with an ISOTROPIC co-explosive source
    # (UNE_forward_CDM_stack_mogi: Mogi point source + the same CDM column).
    #
    # This is the run to QUOTE; the 14-parameter run above is the evidence
    # that quoting it costs nothing.  Five parameters are dropped -- the two
    # shape ratios and the three rotations of the uplift CDM -- and the
    # argument for dropping them is identifiability, not convenience:
    #
    #   Drive the full CDM to a strongly triaxial shape, then refit ONLY
    #   dV_uplift and depth with an equant source.  Across shapes from 5:1
    #   prolate to 1000:1 flattened the residual is 0.05-0.30 mm rms
    #   (max 1.5 mm), on an uplift stage that is itself only ~8 mm
    #   peak-to-peak.  A sphere of the right volume at the right depth
    #   reproduces any of these sources to well below the noise.
    #
    #   The refit also shows what the shape parameters are really doing:
    #   they are a re-parameterisation of volume and depth.  A source
    #   flattened to az/ax = 0.2 is read by the equant fit as 1.77x the
    #   volume, 133 m deeper.  Sampling them therefore buys correlated
    #   marginals on three quantities where the data constrain two.
    #
    #   The azimuth is weaker still: 0.03 mm at 30 deg, against 0.36 and
    #   0.30 mm for the two tilts.  At a depth of 24 cavity radii the source
    #   is a point, and a point has no azimuth.
    #
    # Regenerate those numbers at the posterior mode of the run above rather
    # than at the assumed working point -- that is the honest version of the
    # test, and it is one call:
    #
    #   import UNE_forward_CDM_stack_mogi as _mogi_stack
    #   _mogi_stack.aspect_sensitivity(x=X_obs_divider, y=Y_obs_divider,
    #                                  incidence_deg=..., heading_deg=...,
    #                                  depth=<MAP depth>, r_c=<MAP r_c>,
    #                                  uplift_dv=<MAP dV_uplift>,
    #                                  collapse_dv=<MAP dV_collapse>,
    #                                  phi_bulk=<MAP phi_bulk>,
    #                                  derive_height=True, n_collapse=8)
    #   _mogi_stack.rotation_sensitivity(...)      # same keywords
    #   _mogi_stack.plot_aspect_sensitivity(..., noise_mm=<data noise>)
    #
    # Everything else is held identical to the run above so the comparison
    # measures the five parameters and nothing else: the same collapse column
    # (byte-identical -- UNE_forward_CDM_stack_mogi calls the parent's
    # build_stack), the same n_collapse and loss_exponent, the same priors on
    # the nine shared parameters, the same initial values, the same iteration
    # count and the same data.
    #
    # The co-explosive strength is sampled as dV_eff, Mogi's effective volume
    # change -- a strength, not the volume of the cavity -- so the uplift is
    # the textbook (1 - nu) dV_eff / (pi R^3) (dx, dy, d).  The CDM run above
    # samples a potency instead, dV_uplift = 1.8 dV_eff at nu = 0.25, so its
    # initial value, prior, step and maximum step are converted by 1/1.8
    # below to keep the two runs equivalent.
    #
    # One intended difference: r_c no longer touches the uplift field, since
    # a point source has no size.  It now enters only through the chimney --
    # the cavity volume in the bulking budget, the column radius 1.2 r_c and
    # the column's depth extent -- which is a cleaner separation than the
    # 0.3% finite-source handle it lost.
    # ====================================================================
    model_type_divider_mogi = ['une_mogi_stack_vol']
    custom_initial_une_mogi_stack = {
        'X0': 0.0, 'Y0': 0.0, 'depth': 500.0, 'r_c': 21.0,
        'dV_eff': 5.0e3, 'dV_collapse': 4.0e3,
        'chimney_x': 0.0, 'chimney_y': 0.0,
    }
    custom_priors_une_mogi_stack = {
        'X0': (-_xy_bound_divider, _xy_bound_divider),
        'Y0': (-_xy_bound_divider, _xy_bound_divider),
        'depth': (100, 2000), 'r_c': (5, 100),
        'dV_eff': (0, 5.0e4), 'dV_collapse': (0, 5.0e4),   
        'chimney_x': (-750, 750), 'chimney_y': (-750, 750),
    }
    custom_learning_rates_une_mogi_stack = {
        'X0': 100, 'Y0': 100, 'depth': 50, 'r_c': 5,
        'dV_eff': 2.0e2 , 'dV_collapse': 2.0e2,
        'chimney_x': 20, 'chimney_y': 20,
    }
    max_step_sizes_une_mogi_stack = {
        'X0': _xy_bound_divider, 'Y0': _xy_bound_divider, 'depth': 2000, 'r_c': 100,
        'dV_eff': 5.0e4 , 'dV_collapse': 5.0e4,
        'chimney_x': 750, 'chimney_y': 750,
    }
   

    # (samples_divider_mogi, log_lik_trace_divider_mogi,
    #  rms_evolution_divider_mogi, pickle_filename_divider_mogi) = run_baysian_inference(
    #     u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
    #     incidence_angle=incidence_angle_divider, heading=heading_divider,
    #     n_iterations=number_of_iterations_divider,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial_une_mogi_stack],
    #     priors=[custom_priors_une_mogi_stack],
    #     proposal_std=[custom_learning_rates_une_mogi_stack],
    #     max_step_sizes=[max_step_sizes_une_mogi_stack],
    #     adaptive_interval=number_of_iterations_divider // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_divider_une_mogi_stack_vol',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type_divider_mogi,
    #     burn_in=int(number_of_iterations_divider * 0.3),
    #     reference_point=referencePoint_divider,
    #     u_los_obs_already_los_in_m=True,
    #     fit_ramp='linear',
    # )
    # import UNE_forward_CDM_stack as _une_stack
    # _une_stack.report_chimney_geometry(
    #     samples_divider_mogi,
    #     burn_in=int(number_of_iterations_divider * 0.3))

    # Derived products of this run, printed from the posterior: the compaction
    # of the rubble column, dphi = dV_collapse / (pi beta^2 eta r_c^3) -- the
    # pore space it lost as it settled -- and the tilt of the chimney axis from
    # vertical, arctan(offset / H_col).  Neither is sampled: both follow from
    # the fitted parameters, so they are results, not inputs.  The report also
    # states dphi as a fraction of the pore space the bulked column holds,
    # 4/(3 beta^2 eta) = 0.168, and flags a tilt beyond 30 degrees, which is
    # not a caving geometry.
 

    # ====================================================================
    # DIVIDER — two-point-source model (UNE_forward_CDM_stack_mogi_edited).
    #
    # An isotropic point source for the co-explosive stage at (X0, Y0, depth)
    # and a point tensile crack -- the zero-size limit of the horizontal
    # closing crack -- for the collapse stage at
    # (chimney_x, chimney_y, chimney_depth).  It replaces the stacked
    # closing-crack chimney of the une_mogi_stack_vol run above: the cavity
    # radius r_c and the chimney constants (eta, beta, p, n) no longer enter
    # the forward model.
    #
    # Why.  In the chimney model `depth` and `r_c` jointly place the column
    # and trade off almost exactly, so r_c was set by its lower prior bound
    # (the misfit rose monotonically from r_c = 5 m) and dragged `depth` with
    # it (corr +0.62).  A single point crack with fitted depth and volume
    # reproduces the 64-crack column's LOS field to within 2.2% of peak
    # for r_c = 10-60 m, so nothing the data can see is lost, and the depth the
    # data actually constrain -- the collapse centroid -- is sampled directly
    # as `chimney_depth`.  On this interferogram its profile likelihood has
    # an interior minimum: Delta-chi2 = 1 at about -60 / +75 m.
    #
    # What to read off the posterior:
    #   depth          depth of the co-explosive point source (the working
    #                  point).  The weaker signal, so the wider marginal.
    #   chimney_depth  depth of the collapse centroid.  The best-constrained
    #                  depth in the model.
    #   r_c            NOT sampled.  report_two_point_geometry() prints
    #                  r_c = (depth - chimney_depth) / 3.25, the separation that
    #                  a chimney of 5.5 cavity radii closing uniformly along
    #                  its length would produce.  Defined only
    #                  where the collapse point is shallower than the
    #                  working point.
    #
    # The two depths are deliberately NOT ordered by the prior.  The
    # unconstrained best fit to this interferogram puts chimney_depth ~90 m
    # BELOW depth (formally a negative r_c), and forcing the physical order
    # costs Delta-chi2 = 2 at equal depths and 7 at a 91 m separation.  An
    # ordering prior would hide that inside a bound, as r_c's lower bound did
    # in the chimney model.  Without it, the report states what fraction of
    # the posterior is unphysical.
    #
    # Every model-specific prior here is a positivity bound already implied by
    # the boxes below, so simulated annealing cannot hand MCMC a starting state
    # the model prior rejects -- the cause of the "Initial log-posterior is
    # -inf" abort in the chimney runs.
    # ====================================================================
    # import UNE_forward_CDM_stack_mogi_edited as _two_point
    # model_type_divider_two_point = ['une_two_point']

    # # dV_eff is Mogi's effective volume change of the co-explosive source: a
    # # strength, not a cavity volume.
    # custom_initial_two_point = {
    #     'X0': 0.0, 'Y0': 0.0, 'depth': 400.0, 'dV_eff': 8.0e3,
    #     'chimney_x': 0.0, 'chimney_y': 0.0, 'chimney_depth': 400.0,
    #     'dV_collapse': 9.0e3,
    # }
    # custom_priors_two_point = {
    #     'X0': (-_xy_bound_divider, _xy_bound_divider),
    #     'Y0': (-_xy_bound_divider, _xy_bound_divider),
    #     'depth': (100, 2000), 'dV_eff': (0, 3.0e4),
    #     'chimney_x': (-500, 500), 'chimney_y': (-500, 500),
    #     'chimney_depth': (100, 2000), 'dV_collapse': (0, 5.0e4),
    # }
    # custom_learning_rates_two_point = {
    #     'X0': 20, 'Y0': 20, 'depth': 20, 'dV_eff': 5.0e2,
    #     'chimney_x': 20, 'chimney_y': 20, 'chimney_depth': 20,
    #     'dV_collapse': 5.0e2,
    # }
    # max_step_sizes_two_point = {
    #     'X0': _xy_bound_divider, 'Y0': _xy_bound_divider,
    #     'depth': 1000, 'dV_eff': 1.5e4,
    #     'chimney_x': 500, 'chimney_y': 500,
    #     'chimney_depth': 1000, 'dV_collapse': 2.5e4,
    # }

    # # (samples_divider_two_point, log_lik_trace_divider_two_point,
    # #  rms_evolution_divider_two_point, pickle_filename_divider_two_point) = run_baysian_inference(
    # #     u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
    # #     incidence_angle=incidence_angle_divider, heading=heading_divider,
    # #     n_iterations=number_of_iterations_divider,
    # #     sill=sill, nugget=nugget, range_param=range_param,
    # #     initial_params=[custom_initial_two_point],
    # #     priors=[custom_priors_two_point],
    # #     proposal_std=[custom_learning_rates_two_point],
    # #     max_step_sizes=[max_step_sizes_two_point],
    # #     adaptive_interval=number_of_iterations_divider // 100,
    # #     target_acceptance=0.23,
    # #     figure_folder='Results_divider_une_two_point',
    # #     use_sa_init=use_simulated_annealing_for_first_guess,
    # #     model_type=model_type_divider_two_point,
    # #     burn_in=int(number_of_iterations_divider * 0.3),
    # #     reference_point=referencePoint_divider,
    # #     u_los_obs_already_los_in_m=True,
    # #     fit_ramp='linear',
    # # )

    # # subsidence_factor = a = V_trough / V_cav, taken from UNE subsidence
    # # observations (not coal mining).  With it the report derives r_c from
    # # dV_collapse, the working-point depth as chimney_depth + 4 r_c, and the
    # # uplift depth factor depth / working-point depth.  Leave None to report
    # # the two apparent depths only.
    # _two_point.report_two_point_geometry(
    #     samples_divider_two_point,
    #     burn_in=int(number_of_iterations_divider * 0.3),
    #     subsidence_factor=None)

    # (Commented out: phi_b is a DERIVED quantity in the model being reported,
    # not a fixed input.  The 'une_mogi_stack_budget' registry entry remains
    # available if you want to fix phi_b and eta and derive dV_collapse from
    # r_c instead -- uncomment from here.)
    # # ====================================================================
    # # DIVIDER — phi_b and eta BOTH fixed at literature values.
    # # The budget then closes: dV_collapse = (1 - 0.75 phi_b eta) V_cav(r_c) is
    # # derived, not sampled, so r_c is set by the collapse amplitude (well
    # # determined) instead of by chimney length (barely determined).  Seven
    # # free parameters.
    # #
    # # Caveat: r_c scales as f^(-1/3), f = 1 - 0.75 phi_b eta, and f -> 0 at
    # # phi_b = 4/(3 eta) = 0.2424.  For the same signal, phi_b = 0.15/0.20/0.24
    # # imply r_c = 18/23/61 m and yields of 2/5/81 kt, so the fitted r_c
    # # restates the assumed phi_b as much as it measures anything.  Change
    # # _model_registry.UNE_CDM_STACK_PHI_BULK to test the sensitivity.
    # # ====================================================================
    # _model_registry.UNE_CDM_STACK_PHI_BULK = 0.20
    # model_type_divider_budget = ['une_mogi_stack_budget']

    # custom_initial_budget = {
    #     'X0': 0.0, 'Y0': 0.0, 'depth': 450.0, 'r_c': 23.0, 'dV_eff': 5.0e3,
    #     'chimney_x': 0.0, 'chimney_y': 0.0,
    # }
    # custom_priors_budget = {
    #     'X0': (-_xy_bound_divider, _xy_bound_divider),
    #     'Y0': (-_xy_bound_divider, _xy_bound_divider),
    #     'depth': (100, 2000), 'r_c': (5, 100), 'dV_eff': (0, 3.0e4),
    #     'chimney_x': (-500, 500), 'chimney_y': (-500, 500),
    # }
    # custom_learning_rates_budget = {
    #     'X0': 20, 'Y0': 20, 'depth': 20, 'r_c': 2, 'dV_eff': 5.0e2,
    #     'chimney_x': 20, 'chimney_y': 20,
    # }
    # max_step_sizes_budget = {
    #     'X0': _xy_bound_divider, 'Y0': _xy_bound_divider,
    #     'depth': 1000, 'r_c': 50, 'dV_eff': 1.5e4,
    #     'chimney_x': 500, 'chimney_y': 500,
    # }

    # (samples_divider_budget, log_lik_trace_divider_budget,
    #  rms_evolution_divider_budget, pickle_filename_divider_budget) = run_baysian_inference(
    #     u_los_obs=u_los_obs_divider, X_obs=X_obs_divider, Y_obs=Y_obs_divider,
    #     incidence_angle=incidence_angle_divider, heading=heading_divider,
    #     n_iterations=number_of_iterations_divider,
    #     sill=sill, nugget=nugget, range_param=range_param,
    #     initial_params=[custom_initial_budget],
    #     priors=[custom_priors_budget],
    #     proposal_std=[custom_learning_rates_budget],
    #     max_step_sizes=[max_step_sizes_budget],
    #     adaptive_interval=number_of_iterations_divider // 100,
    #     target_acceptance=0.23,
    #     figure_folder='Results_divider_une_mogi_stack_budget',
    #     use_sa_init=use_simulated_annealing_for_first_guess,
    #     model_type=model_type_divider_budget,
    #     burn_in=int(number_of_iterations_divider * 0.3),
    #     reference_point=referencePoint_divider,
    #     u_los_obs_already_los_in_m=True,
    #     fit_ramp='linear',
    # )

    # # dV_collapse is derived here, so add it to the chains before reporting.
    # import numpy as _np_b
    # import UNE_forward_CDM_stack as _une_stack_b
    # _f_b = 1.0 - 0.75 * _model_registry.UNE_CDM_STACK_PHI_BULK * \
    #     _model_registry.UNE_CDM_STACK_HEIGHT_FAC
    # _s_b = dict(samples_divider_budget)
    # _s_b['dV_collapse'] = _f_b * _une_stack_b.cavity_volume(
    #     _np_b.asarray(_s_b['r_c'], float))
    # _une_stack_b.report_chimney_geometry(
    #     _s_b, burn_in=int(number_of_iterations_divider * 0.3))
