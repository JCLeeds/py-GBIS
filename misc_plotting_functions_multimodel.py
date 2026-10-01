import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.interpolate import griddata
from model_registry import MODEL_REGISTRY, get_param_names, forward_from_registry

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import llh2local as llh
import local2llh as l2llh

def plot_sa_diagnostics(energy_trace, temperature_trace, figure_folder=None):
    """
    Plot simulated annealing diagnostics.
    
    Parameters:
    -----------
    energy_trace : list
        Energy evolution during SA
    temperature_trace : list
        Temperature evolution during SA
    figure_folder : str, optional
        Folder to save figures
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    iterations = np.arange(len(energy_trace))
    
    # Plot energy evolution
    ax1.plot(iterations, energy_trace, 'b-', alpha=0.7, linewidth=1,label='Energy (Negative Log Likelihood)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy (Negative Log Likelihood)')
    ax1.set_title('Simulated Annealing: Energy Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Add running minimum
    running_min = np.minimum.accumulate(energy_trace)
    ax1.plot(iterations, running_min, 'r-', linewidth=2, alpha=0.8, label='Running minimum')
    ax1.legend()
    
    # Plot temperature evolution
    ax2.semilogy(iterations, temperature_trace, 'r-', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature (log scale)')
    ax2.set_title('Simulated Annealing: Temperature Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/SA_diagnostics.png", dpi=300, bbox_inches='tight')




def plot_adaptation_diagnostics(proposal_std_evolution, acceptance_rate_evolution,
                               adaptive_interval, target_acceptance=0.23, figure_folder=None,
                               model_type='pCDM', proposal_std_iters=None):
    """
    Plot diagnostics for adaptive MCMC including proposal scale evolution
    and acceptance rate evolution.
    """
    from matplotlib.ticker import MaxNLocator
    param_names = list(proposal_std_evolution.keys())
    n_params = len(param_names)

    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    n_checkpoints = len(proposal_std_evolution[param_names[0]])
    if proposal_std_iters is not None and len(proposal_std_iters) == n_checkpoints:
        std_x = np.array(proposal_std_iters)
    else:
        std_x = np.arange(1, n_checkpoints + 1) * adaptive_interval

    # --- Proposal scale evolution ---
    fig_height = max(6, 2.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, fig_height))
    axes = np.array(axes).reshape(n_rows, n_cols)

    _LINE_C = '#2166AC'
    for i, param in enumerate(param_names):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        vals = np.array(proposal_std_evolution[param])
        ax.plot(std_x, vals, color=_LINE_C, linewidth=0.8, marker='o',
                markersize=2, alpha=0.8)

        if len(vals) > 1:
            diffs = np.abs(np.diff(vals))
            frozen_idx = np.argmax(diffs < 1e-12 * (vals[:-1] + 1e-30))
            if 0 < frozen_idx < len(std_x) - 1:
                ax.axvline(std_x[frozen_idx], color='#D6604D', linestyle='--',
                           linewidth=1.0, alpha=0.8, label='Burn-in end')
                ax.legend(fontsize=6.5, framealpha=0.7, loc='upper right')

        display = param if '__' not in param else param.replace('__', ':')
        ax.set_title(display, fontsize=8, fontweight='bold', pad=3)
        ax.set_ylabel('Proposal std', fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        if row == n_rows - 1 or (i + n_cols) >= n_params:
            ax.set_xlabel('Iteration', fontsize=7)

    for j in range(n_params, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Proposal scale evolution (one point per adaptation event)',
                 fontsize=9, y=1.01)
    plt.tight_layout(pad=1.2)
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/proposal_scale_evolution.png", dpi=300,
                    bbox_inches='tight')
    plt.close(fig)

    # --- Acceptance rate evolution ---
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(12, fig_height))
    axes2 = np.array(axes2).reshape(n_rows, n_cols)

    for i, param in enumerate(param_names):
        row, col = divmod(i, n_cols)
        ax = axes2[row, col]
        acc_vals = acceptance_rate_evolution[param]
        if len(acc_vals) > 0:
            acc_x = std_x[:len(acc_vals)]
            ax.plot(acc_x, acc_vals, color='#4DAC26', linewidth=0.8, marker='o',
                    markersize=2, alpha=0.8)
            ax.axhline(target_acceptance, color='#D6604D', linestyle='--',
                       linewidth=1.0, alpha=0.9,
                       label=f'Target ({target_acceptance:.0%})')

        display = param if '__' not in param else param.replace('__', ':')
        ax.set_title(display, fontsize=8, fontweight='bold', pad=3)
        ax.set_ylabel('Acceptance rate', fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        if row == n_rows - 1 or (i + n_cols) >= n_params:
            ax.set_xlabel('Iteration', fontsize=7)
        if i == 0:
            ax.legend(fontsize=6.5, framealpha=0.7, loc='upper right')

    for j in range(n_params, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes2[row, col].set_visible(False)

    fig2.suptitle('Per-parameter acceptance rate (one point per adaptation event)',
                  fontsize=9, y=1.01)
    plt.tight_layout(pad=1.2)
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/acceptance_rate_evolution.png", dpi=300,
                    bbox_inches='tight')
    plt.close(fig2)
    
   


def plot_inference_results(samples, log_likelihood_trace, rms_evolution, burn_in=2000,
                          u_los_obs=None, X_obs=None, Y_obs=None,
                          incidence_angle=None, heading=None, figure_folder=None,
                          proposal_std_evolution=None, acceptance_rate_evolution=None,
                          proposal_std_evolution_iters=None,
                          adaptive_interval=None, target_acceptance=0.23, model_type='pCDM',
                          priors=None, full_res_npy_paths=None, ifg_dates=None,
                          reference_point=None, noise_rms=None):
    """
    Plot MCMC results including trace plots, posterior distributions,
    and comparison between initial and optimal models.
    """
    # Normalise priors to a flat {param: (lo, hi)} dict so sub-functions work
    # regardless of whether the caller passed a dict or a list-of-dicts.
    flat_priors_plot = {}
    if priors is not None:
        if isinstance(priors, list):
            for p_dict in priors:
                if isinstance(p_dict, dict):
                    flat_priors_plot.update(p_dict)
        elif isinstance(priors, dict):
            flat_priors_plot = priors
    flat_priors_plot = flat_priors_plot if flat_priors_plot else None

    # Remove burn-in samples
    samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}
    log_lik_burned = np.array(log_likelihood_trace[burn_in:])
    
    # Plot log-likelihood trace
    plt.figure(figsize=(12, 8))
    plt.plot(log_likelihood_trace)
    plt.axvline(burn_in, color='r', linestyle='--', alpha=0.7, label='Burn-in')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood Trace')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/log_likelihood_trace.png", dpi=300)
    
    # Plot parameter posterior histograms
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    all_params = get_param_names(model_type, samples)
    all_params = [p for p in all_params if not p.startswith('ramp_')]
    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig_height = max(5, 2.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, fig_height))
    axes = np.array(axes).reshape(n_rows, n_cols)

    _HIST_C = '#4472C4'
    _MEAN_C = '#C00000'
    _MAP_C  = '#70AD47'

    best_idx = np.argmax(log_likelihood_trace[burn_in:])

    def _sfmt(v):
        """3-4 significant figures, no unnecessary trailing zeros."""
        return f'{v:.4g}'

    for i, param in enumerate(all_params):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        data = samples_burned[param]
        ax.hist(data, bins=60, density=True,
                color=_HIST_C, alpha=0.75, linewidth=0, edgecolor='none')

        mean_val = np.mean(data)
        std_val  = np.std(data)
        map_val  = data[best_idx]

        ax.axvline(mean_val, color=_MEAN_C, linestyle='--', linewidth=1.2, zorder=5)
        ax.axvline(map_val,  color=_MAP_C,  linestyle=':',  linewidth=1.5, zorder=5)

        display = param if '__' not in param else param.replace('__', ':')
        ax.set_title(display, fontsize=8.5, fontweight='bold', pad=3)
        ax.set_xlabel(display, fontsize=7.5)
        ax.set_ylabel('Density' if col == 0 else '', fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

        ax.text(0.97, 0.97,
                f'$\\mu$ = {_sfmt(mean_val)}\n$\\sigma$ = {_sfmt(std_val)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=_MEAN_C, linestyle='--', linewidth=1.2, label='Posterior mean'),
        Line2D([0], [0], color=_MAP_C,  linestyle=':',  linewidth=1.5, label='MAP estimate'),
    ]
    fig.legend(handles=legend_handles, loc='lower right',
               bbox_to_anchor=(1.0, 0.0), fontsize=7.5, framealpha=0.9)

    for j in range(n_params, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout(pad=1.2)
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/MCMC_traces_posteriors.png", dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    
 
    # Plot adaptation diagnostics if available
    if (proposal_std_evolution is not None and acceptance_rate_evolution is not None 
        and adaptive_interval is not None):
        plot_adaptation_diagnostics(proposal_std_evolution, acceptance_rate_evolution,
                                   adaptive_interval, target_acceptance, figure_folder,
                                   model_type=model_type,
                                   proposal_std_iters=proposal_std_evolution_iters)
    
    # Plot other diagnostics if observation data is provided
    if all(x is not None for x in [u_los_obs, X_obs, Y_obs, incidence_angle, heading]):
        plot_model_comparison(samples, u_los_obs, X_obs, Y_obs,
                             incidence_angle, heading, log_likelihood_trace, burn_in, figure_folder=figure_folder, model_type=model_type,
                             full_res_npy_paths=full_res_npy_paths, ifg_dates=ifg_dates,
                             reference_point=reference_point, noise_rms=noise_rms)
        plot_rms_evolution(rms_evolution, figure_folder=figure_folder,model_type=model_type)
        plot_parameter_convergence(samples, burn_in, figure_folder=figure_folder, model_type=model_type, priors=flat_priors_plot)
        plot_model_components(samples, u_los_obs, X_obs, Y_obs,
                              incidence_angle, heading, log_likelihood_trace,
                              burn_in=burn_in, figure_folder=figure_folder, model_type=model_type)

    # 2-D parameter trade-off corner plot
    plot_corner(samples, burn_in=burn_in, figure_folder=figure_folder, model_type=model_type, priors=flat_priors_plot)

    # Combined write-up figure (corner + data fit + multi-model components)
    if all(x is not None for x in [u_los_obs, X_obs, Y_obs, incidence_angle, heading]):
        try:
            plot_report_figure(samples, u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                               log_likelihood_trace, burn_in=burn_in, figure_folder=figure_folder,
                               model_type=model_type, ifg_dates=ifg_dates)
        except Exception as exc:
            print(f"  Warning: plot_report_figure failed: {exc}")

    # Print summary statistics
    print("\nPosterior Summary Statistics:")
    print("-" * 50)
    for param in samples_burned.keys():
        mean_val = np.mean(samples_burned[param])
        std_val = np.std(samples_burned[param])
        q025 = np.percentile(samples_burned[param], 2.5)
        q975 = np.percentile(samples_burned[param], 97.5)
        print(f"{param:8s}: {mean_val:8.4f} ± {std_val:6.4f} [{q025:8.4f}, {q975:8.4f}]")

def plot_rms_evolution(rms_evolution, figure_folder=None,model_type='pCDM'):
    """
    Plot RMS residual as a function of MCMC iteration.
    """

    # Plot RMS evolution
    plt.figure(figsize=(12, 8))
    
    # Top subplot: RMS evolution
    # plt.subplot(2, 1, 1)
    plt.plot(rms_evolution, 'b-', alpha=0.7, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('RMS Residual')
    plt.title('RMS Residual Evolution')
    plt.grid(True, alpha=0.3)
    
    # Add running mean
    window_size = min(500, len(rms_evolution) // 10)
    if window_size > 1:
        running_mean = np.convolve(rms_evolution, np.ones(window_size)/window_size, mode='valid')
        # Create x-axis that matches the length of running_mean
        x_running = np.arange(window_size//2, window_size//2 + len(running_mean))
        plt.plot(x_running, running_mean, 
                'r-', linewidth=2, label=f'Running mean ({window_size} iterations)')
        plt.legend()

    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/RMS_evolution.png", dpi=300)
    
    
    plt.tight_layout()
    # plt.show()
   
    return rms_evolution

def plot_parameter_convergence(samples, burn_in=2000, figure_folder=None, model_type='pCDM', priors=None):
    """
    Plot the convergence of each parameter over MCMC iterations.
    
    Parameters:
    -----------
    samples : dict
        MCMC samples for each parameter
    burn_in : int
        Number of burn-in iterations to mark on plot
    figure_folder : str, optional
        Folder to save figures
    """

    # Determine parameter list.
    # - If this is a multi-model run (model_type is list or samples use prefixed keys) use the exact sample keys.
    # - Otherwise fall back to the standard single-model parameter lists for nicer ordering.
    all_params = get_param_names(model_type, samples)
    all_params = [p for p in all_params if not p.startswith('ramp_')]

    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig_height = max(6, 2.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, fig_height))
    axes = np.array(axes).reshape(n_rows, n_cols)

    _TRACE_C   = '#2166AC'
    _RUNNING_C = '#F4A582'
    _BURNIN_C  = '#D6604D'

    def _sfmt(v):
        return f'{v:.4g}'

    legend_drawn = False
    for i, param in enumerate(all_params):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        iterations = np.arange(len(samples[param]))
        ax.plot(iterations, samples[param],
                color=_TRACE_C, linewidth=0.25, alpha=0.6, rasterized=True)

        if burn_in < len(samples[param]):
            ax.axvline(burn_in, color=_BURNIN_C, linestyle='--',
                       linewidth=1.0, alpha=0.9)

        window_size = min(500, len(samples[param]) // 10)
        if window_size > 1:
            running_mean = np.convolve(
                samples[param], np.ones(window_size) / window_size, mode='valid')
            x_running = np.arange(window_size // 2,
                                  window_size // 2 + len(running_mean))
            ax.plot(x_running, running_mean,
                    color=_RUNNING_C, linewidth=1.4, alpha=0.95)

        if priors is not None:
            pkey = param if param in priors else (
                param.split('__')[1] if '__' in param else None)
            if pkey and pkey in priors:
                lo, hi = priors[pkey]
                ax.set_ylim(lo, hi)

        if len(samples[param]) > burn_in:
            final_mean = np.mean(samples[param][burn_in:])
            final_std  = np.std(samples[param][burn_in:])
            ax.text(0.97, 0.97,
                    f'{_sfmt(final_mean)} ± {_sfmt(final_std)}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))

        display_name = param if '__' not in param else param.replace('__', ':')
        ax.set_title(display_name, fontsize=8.5, fontweight='bold', pad=3)
        ax.set_ylabel(display_name, fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        if row == n_rows - 1 or (i + n_cols) >= n_params:
            ax.set_xlabel('Iteration', fontsize=7.5)

    # Single shared legend in figure
    legend_handles = [
        Line2D([0], [0], color=_TRACE_C,   linewidth=1.0, label='MCMC chain'),
        Line2D([0], [0], color=_RUNNING_C, linewidth=1.4, label='Running mean'),
        Line2D([0], [0], color=_BURNIN_C,  linewidth=1.0, linestyle='--', label='Burn-in end'),
    ]
    fig.legend(handles=legend_handles, loc='lower right',
               bbox_to_anchor=(1.0, 0.0), fontsize=7.5, framealpha=0.9)

    for j in range(n_params, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout(pad=1.2)
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/parameter_convergence.png", dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    
    # Calculate and print convergence diagnostics
    print("\nConvergence Diagnostics (post burn-in):")
    print("-" * 60)
    for param in all_params:
        if len(samples[param]) > burn_in:
            post_burnin = np.array(samples[param][burn_in:])
            
            # Split into first and second half for comparison
            mid_point = len(post_burnin) // 2
            first_half_mean = np.mean(post_burnin[:mid_point])
            second_half_mean = np.mean(post_burnin[mid_point:])
            
            # Simple convergence metric: difference between halves
            convergence_diff = abs(first_half_mean - second_half_mean)
            overall_std = np.std(post_burnin)
            
            print(f"{param:8s}: 1st half mean={first_half_mean:8.4f}, "
                    f"2nd half mean={second_half_mean:8.4f}, "
                    f"diff={convergence_diff:8.4f} ({convergence_diff/overall_std:.2f}σ)")
            


def plot_model_comparison(samples, u_los_obs, X_obs, Y_obs,
                         incidence_angle, heading, log_likelihood_trace, burn_in=2000, figure_folder=None, model_type='pCDM',
                         full_res_npy_paths=None, ifg_dates=None, _ifg_idx=None, reference_point=None, noise_rms=None):
    """
    Plot comparison between initial model, optimal model, and residuals.
    """
    # Get initial parameters (first sample)
    initial_params = {key: val[0] for key, val in samples.items()}
    
    # Get optimal parameters (mean of post-burn-in samples)
    samples_burned = {key: np.array(val[burn_in:]) for key, val in samples.items()}
    optimal_params = {key: np.mean(vals) for key, vals in samples_burned.items()}
    # Also calculate the MAP (maximum a posteriori) estimate using log likelihood
    # Find the sample with highest likelihood
    best_idx = np.argmax(log_likelihood_trace[burn_in:])
    map_params = {key: samples_burned[key][best_idx] for key in samples_burned.keys()}

    print(f"\nMaximum Likelihood Parameters:")
    for param, value in map_params.items():
        print(f"  {param:10s}: {value:8.4f}")

    # Use MAP parameters as optimal parameters for model calculation
    optimal_params = map_params
    # Multi-IFG support (Option A): if observation inputs are lists of interferograms,
    # call the single-IFG plotting routine once per IFG and save figures separately.
    if isinstance(u_los_obs, (list, tuple, np.ndarray)) and len(u_los_obs) > 0 and hasattr(u_los_obs[0], '__iter__'):
        n_ifgs = len(u_los_obs)
        inc_list  = incidence_angle if isinstance(incidence_angle, (list, tuple, np.ndarray)) else [incidence_angle] * n_ifgs
        head_list = heading if isinstance(heading, (list, tuple, np.ndarray)) else [heading] * n_ifgs
        dates_list = list(ifg_dates) if ifg_dates is not None else [None] * n_ifgs
        for j in range(n_ifgs):
            _date_j = dates_list[j] if j < len(dates_list) else None
            sub_folder = None
            if figure_folder is not None:
                folder_name = _date_j if _date_j else f"ifg_{j+1}"
                sub_folder = os.path.join(figure_folder, folder_name)
                os.makedirs(sub_folder, exist_ok=True)
            _fr_j = None
            if full_res_npy_paths is not None:
                if isinstance(full_res_npy_paths, (list, tuple)):
                    _fr_j = full_res_npy_paths[j] if j < len(full_res_npy_paths) else None
                else:
                    _fr_j = full_res_npy_paths
            try:
                plot_model_comparison(samples, u_los_obs[j], X_obs[j], Y_obs[j], inc_list[j], head_list[j],
                                      log_likelihood_trace, burn_in, figure_folder=sub_folder,
                                      model_type=model_type, full_res_npy_paths=_fr_j, ifg_dates=_date_j,
                                      _ifg_idx=j, reference_point=reference_point, noise_rms=noise_rms)
            except Exception as exc:
                print(f"  Warning: plot_model_comparison failed for IFG {j+1} ({_date_j}): {exc}")
        return

    # Convert angles to radians for LOS calculation (single IFG)
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    
    # Line-of-sight unit vector components
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)

    # Load full-resolution data for top-row panels (optional)
    X_full = Y_full = u_los_full = u_los_opt_full = res_opt_full = None
    los_e_full = los_n_full = los_u_full = None
    if isinstance(full_res_npy_paths, str):
        try:
            fr_data = np.load(full_res_npy_paths, allow_pickle=True).item()
            fr_lat = np.array(fr_data['Lat']).flatten()
            fr_lon = np.array(fr_data['Lon']).flatten()
            # Use the common reference point if provided; otherwise fall back to the file's centre
            if reference_point is not None:
                ref_lat = float(reference_point[0])
                ref_lon = float(reference_point[1])
            else:
                ref_lon = float(fr_data['center_lon'])
                ref_lat = float(fr_data['center_lat'])
            ll_fr = np.array([fr_lon, fr_lat], dtype=float)
            xy_fr = llh.llh2local(ll_fr, np.array([ref_lon, ref_lat], dtype=float))
            u_full_raw = np.array(fr_data['Phase']).flatten()
            valid = np.isfinite(u_full_raw)
            X_full = xy_fr[0, :][valid]
            Y_full = xy_fr[1, :][valid]
            # Convert radians → LOS metres (Sentinel-1 C-band: λ = 55.5 mm)
            _conv = 0.0555 / (4 * np.pi)
            u_los_full = -u_full_raw[valid] * _conv
            # Per-pixel LOS unit vectors from the full-res Inc/Heading arrays
            inc_f_rad  = np.radians(np.array(fr_data['Inc']).flatten()[valid])
            head_f_rad = np.radians(np.array(fr_data['Heading']).flatten()[valid])
            los_e_full = np.sin(inc_f_rad) * np.cos(head_f_rad)
            los_n_full = -np.sin(inc_f_rad) * np.sin(head_f_rad)
            los_u_full = -np.cos(inc_f_rad)
            print(f"  Full-resolution data loaded: {len(u_los_full):,} points")
        except Exception as exc:
            print(f"  Warning: could not load full-resolution data ({exc})")

    # Calculate initial and optimal models. Support single-model OR multi-model (prefixed sample keys).
    def _sum_models_from_sample_dict(sample_dict):
        """Sum contributions from one or more models using keys present in sample_dict.
        If keys are prefixed (label__param), detect labels and sum per-label contributions.
        If keys are unprefixed assume a single model and use model_type string.
        """
        # Detect prefixed multi-model keys
        if any('__' in k for k in sample_dict.keys()):
            labels = sorted({k.split('__')[0] for k in sample_dict.keys()})
            ue_sum = np.zeros_like(X_obs, dtype=float)
            un_sum = np.zeros_like(X_obs, dtype=float)
            uv_sum = np.zeros_like(X_obs, dtype=float)
            for label in labels:
                model_name = label.split('_')[0]
                params_here = {k.split('__')[1]: sample_dict[k] for k in sample_dict.keys() if k.startswith(label + '__')}
                try:
                    ue_i, un_i, uv_i = forward_from_registry(model_name, X_obs, Y_obs, params_here)
                except ValueError:
                    ue_i = np.zeros_like(X_obs)
                    un_i = np.zeros_like(X_obs)
                    uv_i = np.zeros_like(X_obs)
                ue_sum += np.asarray(ue_i, dtype=float)
                un_sum += np.asarray(un_i, dtype=float)
                uv_sum += np.asarray(uv_i, dtype=float)
            return ue_sum, un_sum, uv_sum
        else:
            m = model_type.lower() if isinstance(model_type, str) else list(model_type)[0].lower()
            return forward_from_registry(m, X_obs, Y_obs, sample_dict)

    # Initial model
    ue_init, un_init, uv_init = _sum_models_from_sample_dict(initial_params)
    u_los_init = -(ue_init * los_e + un_init * los_n + uv_init * los_u)

    # Optimal model (MAP)
    ue_opt, un_opt, uv_opt = _sum_models_from_sample_dict(optimal_params)
    u_los_opt = -(ue_opt * los_e + un_opt * los_n + uv_opt * los_u)
    
    # ── Ramp/offset correction ──────────────────────────────────────────────
    # Detect ramp parameters in the sample dict (named ramp_a/b/c or ramp_a_j/b_j/c_j)
    _ramp_keys = [k for k in optimal_params if k.startswith('ramp_')]
    _has_ramp = bool(_ramp_keys)
    _fit_linear = any('ramp_a' in k for k in _ramp_keys)

    def _compute_ramp(p, X, Y):
        """Evaluate ramp at coordinates X, Y using params p (single-IFG; no suffix)."""
        r = p.get('ramp_c', 0.0) * np.ones(len(X))
        if _fit_linear:
            r = r + p.get('ramp_a', 0.0) * X + p.get('ramp_b', 0.0) * Y
        return r

    def _compute_ramp_suffixed(p, X, Y, sfx):
        """Evaluate ramp with per-IFG suffix (multi-IFG)."""
        r = p.get(f'ramp_c{sfx}', 0.0) * np.ones(len(X))
        if _fit_linear:
            r = r + p.get(f'ramp_a{sfx}', 0.0) * X + p.get(f'ramp_b{sfx}', 0.0) * Y
        return r

    if _has_ramp:
        # Determine if multi-IFG suffix ('ramp_c_0') or plain ('ramp_c')
        _multi_sfx = any(k[-2:].lstrip('_').isdigit() for k in _ramp_keys if k.startswith('ramp_c'))
        if _multi_sfx:
            # Use _ifg_idx to select the correct IFG's ramp params (default 0)
            _ramp_sfx = f'_{_ifg_idx}' if _ifg_idx is not None else '_0'
            ramp_opt_obs  = _compute_ramp_suffixed(optimal_params,  X_obs, Y_obs, _ramp_sfx)
            ramp_init_obs = _compute_ramp_suffixed(initial_params,  X_obs, Y_obs, _ramp_sfx)
        else:
            ramp_opt_obs  = _compute_ramp(optimal_params,  X_obs, Y_obs)
            ramp_init_obs = _compute_ramp(initial_params,  X_obs, Y_obs)
        u_los_opt_with_ramp  = u_los_opt  + ramp_opt_obs
        u_los_init_with_ramp = u_los_init + ramp_init_obs
    else:
        u_los_opt_with_ramp  = u_los_opt
        u_los_init_with_ramp = u_los_init
        ramp_opt_obs = np.zeros_like(u_los_obs)

    # Calculate residuals
    residual_init = u_los_obs - u_los_init_with_ramp
    residual_opt  = u_los_obs - u_los_opt_with_ramp

    # Evaluate MAP model at full-resolution points (for top-row panels)
    if X_full is not None:
        def _eval_at(X, Y, sample_dict):
            if any('__' in k for k in sample_dict.keys()):
                labels = sorted({k.split('__')[0] for k in sample_dict.keys()})
                ue_s = np.zeros(len(X), dtype=float)
                un_s = np.zeros(len(X), dtype=float)
                uv_s = np.zeros(len(X), dtype=float)
                for label in labels:
                    model_name = label.split('_')[0]
                    params_here = {k.split('__')[1]: sample_dict[k] for k in sample_dict.keys() if k.startswith(label + '__')}
                    try:
                        ue_i, un_i, uv_i = forward_from_registry(model_name, X, Y, params_here)
                        ue_s += np.asarray(ue_i, dtype=float)
                        un_s += np.asarray(un_i, dtype=float)
                        uv_s += np.asarray(uv_i, dtype=float)
                    except ValueError:
                        pass
                return ue_s, un_s, uv_s
            else:
                m = model_type.lower() if isinstance(model_type, str) else list(model_type)[0].lower()
                return forward_from_registry(m, X, Y, sample_dict)

        ue_f, un_f, uv_f = _eval_at(X_full, Y_full, optimal_params)
        u_los_opt_full = -(ue_f * los_e_full + un_f * los_n_full + uv_f * los_u_full)
        # Add ramp to full-res optimal model so it matches the observed data
        if _has_ramp:
            if _multi_sfx:
                _ramp_sfx = f'_{_ifg_idx}' if _ifg_idx is not None else '_0'
                ramp_full = _compute_ramp_suffixed(optimal_params, X_full, Y_full, _ramp_sfx)
            else:
                ramp_full = _compute_ramp(optimal_params, X_full, Y_full)
            u_los_opt_full = u_los_opt_full + ramp_full
        res_opt_full = u_los_full - u_los_opt_full

    # Create regular grid for interpolation if data is scattered
    if len(np.unique(X_obs)) > 1 and len(np.unique(Y_obs)) > 1:
        # Create regular grid
        x_min, x_max = np.min(X_obs), np.max(X_obs)
        y_min, y_max = np.min(Y_obs), np.max(Y_obs)
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate data to regular grid
        
        u_obs_grid      = griddata((X_obs, Y_obs), u_los_obs,              (Xi, Yi), method='cubic')
        u_init_grid     = griddata((X_obs, Y_obs), u_los_init_with_ramp,  (Xi, Yi), method='cubic')
        u_opt_grid      = griddata((X_obs, Y_obs), u_los_opt_with_ramp,   (Xi, Yi), method='cubic')
        res_init_grid   = griddata((X_obs, Y_obs), residual_init,          (Xi, Yi), method='cubic')
        res_opt_grid    = griddata((X_obs, Y_obs), residual_opt,           (Xi, Yi), method='cubic')
        ramp_opt_grid   = griddata((X_obs, Y_obs), ramp_opt_obs,           (Xi, Yi), method='cubic') if _has_ramp else None

        X_plot, Y_plot = Xi, Yi
        u_obs_plot      = u_obs_grid
        u_init_plot     = u_init_grid
        u_opt_plot      = u_opt_grid
        res_init_plot   = res_init_grid
        res_opt_plot    = res_opt_grid
        X_plot_scatter, Y_plot_scatter = X_obs, Y_obs
        u_obs_plot_scatter  = u_los_obs
        u_init_plot_scatter = u_los_init_with_ramp
        u_opt_plot_scatter  = u_los_opt_with_ramp
        res_init_plot_scatter = residual_init
        res_opt_plot_scatter  = residual_opt
    else:
        # Assume data is already on regular grid
        try:
            grid_shape = (int(np.sqrt(len(X_obs))), int(np.sqrt(len(X_obs))))
            X_plot = X_obs.reshape(grid_shape)
            Y_plot = Y_obs.reshape(grid_shape)
            u_obs_plot    = u_los_obs.reshape(grid_shape)
            u_init_plot   = u_los_init_with_ramp.reshape(grid_shape)
            u_opt_plot    = u_los_opt_with_ramp.reshape(grid_shape)
            res_init_plot = residual_init.reshape(grid_shape)
            res_opt_plot  = residual_opt.reshape(grid_shape)
            ramp_opt_grid = ramp_opt_obs.reshape(grid_shape) if _has_ramp else None

            X_plot_scatter, Y_plot_scatter = X_obs, Y_obs
            u_obs_plot_scatter    = u_los_obs
            u_init_plot_scatter   = u_los_init_with_ramp
            u_opt_plot_scatter    = u_los_opt_with_ramp
            res_init_plot_scatter = residual_init
            res_opt_plot_scatter  = residual_opt

        except:
            print("Could not reshape data for plotting. Using scatter plots instead.")
            X_plot, Y_plot = X_obs, Y_obs
            u_obs_plot    = u_los_obs
            u_init_plot   = u_los_init_with_ramp
            u_opt_plot    = u_los_opt_with_ramp
            res_init_plot = residual_init
            res_opt_plot  = residual_opt
            ramp_opt_grid = None
    
    # Create comparison plot — 3 rows when a ramp was fitted, 2 otherwise
    _n_rows = 3 if _has_ramp else 2
    fig, axes = plt.subplots(_n_rows, 3, figsize=(15, 5 * _n_rows))
    axes = np.array(axes).reshape(_n_rows, 3)
    
    # Determine common color scale for observed and optimal
    vmin = np.nanmin([u_obs_plot, u_opt_plot])
    vmax = np.nanmax([u_obs_plot, u_opt_plot])
    # Center color scale on zero
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin = -vmax_abs
    vmax = vmax_abs
    
    # Residual color scale
    res_vmax = np.nanmax(np.abs(res_opt_plot))
    res_vmin = -res_vmax
    
    if X_plot.ndim == 2:
        if X_full is not None:
            # Full-resolution scatter for top row
            _top_vmax = max(abs(np.nanmin(u_los_full)), abs(np.nanmax(u_los_full)),
                            abs(np.nanmin(u_los_opt_full)), abs(np.nanmax(u_los_opt_full)))
            _s = max(0.05, min(2.0, 5e4 / len(X_full)))
            im1 = axes[0, 0].scatter(X_full, Y_full, c=u_los_full, cmap='RdBu_r',
                                     vmin=-_top_vmax, vmax=_top_vmax, s=_s, rasterized=True)
            axes[0, 0].set_title('Observed Data (Full Resolution)')
            im2 = axes[0, 1].scatter(X_full, Y_full, c=u_los_opt_full, cmap='RdBu_r',
                                     vmin=-_top_vmax, vmax=_top_vmax, s=_s, rasterized=True)
            axes[0, 1].set_title('Optimal Model (Full Resolution)')
            im3 = axes[0, 2].scatter(X_full, Y_full, c=res_opt_full, cmap='RdBu_r',
                                     vmin=-_top_vmax, vmax=_top_vmax, s=_s, rasterized=True)
            axes[0, 2].set_title('Optimal Residual (Full Resolution)')
            _top_row_im = im1  # used for the shared bottom colorbar, see below
        else:
            # Contour plots from gridded interpolation
            im1 = axes[0, 0].contourf(X_plot, Y_plot, u_obs_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title('Observed Data')
            im2 = axes[0, 1].contourf(X_plot, Y_plot, u_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0, 1].set_title('Optimal Model')
            im3 = axes[0, 2].contourf(X_plot, Y_plot, res_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0, 2].set_title('Optimal Residual')
            _top_row_im = im1  # used for the shared bottom colorbar, see below
    
        # Scatter plots
        im1 = axes[1,0].scatter(X_plot_scatter, Y_plot_scatter, c=u_obs_plot_scatter, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1,0].set_title('Observed Data')
        
        im2 = axes[1,1].scatter(X_plot_scatter, Y_plot_scatter, c=u_opt_plot_scatter, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1,1].set_title('Optimal Model')
        
        # Bottom row: Residual
        im3 = axes[1,2].scatter(X_plot_scatter, Y_plot_scatter, c=res_opt_plot_scatter, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1,2].set_title('Optimal Residual')
    else:
        # Scatter plots
        im1 = axes[0,0].scatter(X_plot, Y_plot, c=u_obs_plot, cmap='RdBu_r')
        axes[0,0].set_title('Observed Data')
        
        im2 = axes[0,1].scatter(X_plot, Y_plot, c=u_opt_plot, cmap='RdBu_r')
        axes[0,1].set_title('Optimal Model')
        
        # Bottom row: Residual
        im3 = axes[0,2].scatter(X_plot, Y_plot, c=res_opt_plot, cmap='RdBu_r')
        axes[0,2].set_title('Optimal Residual')
        _top_row_im = im1  # used for the shared bottom colorbar, see below
   
    

    # ── Ramp row (row 2) — only when ramp parameters were estimated ─────────
    if _has_ramp and _n_rows == 3:
        _ramp_abs = np.nanmax(np.abs(ramp_opt_obs)) if ramp_opt_obs is not None else 1.0
        _ramp_vmax = _ramp_abs if _ramp_abs > 0 else 1.0

        # Col 0: ramp field on observation grid
        if ramp_opt_grid is not None and X_plot.ndim == 2:
            _rim1 = axes[2, 0].contourf(X_plot, Y_plot, ramp_opt_grid, levels=20,
                                         cmap='RdBu_r', vmin=-_ramp_vmax, vmax=_ramp_vmax)
            plt.colorbar(_rim1, ax=axes[2, 0], fraction=0.046, pad=0.04).set_label('m')
        else:
            _rim1 = axes[2, 0].scatter(X_plot_scatter, Y_plot_scatter, c=ramp_opt_obs,
                                        cmap='RdBu_r', vmin=-_ramp_vmax, vmax=_ramp_vmax)
            plt.colorbar(_rim1, ax=axes[2, 0], fraction=0.046, pad=0.04).set_label('m')
        axes[2, 0].set_title('Ramp / Offset (MAP)')
        axes[2, 0].set_xlabel('X (m)')
        axes[2, 0].set_ylabel('Y (m)')

        # Col 1: ramp-corrected observed data (observed minus ramp)
        _u_deramped = u_los_obs - ramp_opt_obs
        _dr_vmax = max(abs(np.nanmin(_u_deramped)), abs(np.nanmax(_u_deramped)))
        _rim2 = axes[2, 1].scatter(X_plot_scatter, Y_plot_scatter, c=_u_deramped,
                                    cmap='RdBu_r', vmin=-_dr_vmax, vmax=_dr_vmax)
        plt.colorbar(_rim2, ax=axes[2, 1], fraction=0.046, pad=0.04).set_label('m')
        axes[2, 1].set_title('Ramp-Corrected Observed')
        axes[2, 1].set_xlabel('X (m)')

        # Col 2: source model only (no ramp)
        _sm_vmax = max(abs(np.nanmin(u_los_opt)), abs(np.nanmax(u_los_opt)))
        _rim3 = axes[2, 2].scatter(X_plot_scatter, Y_plot_scatter, c=u_los_opt,
                                    cmap='RdBu_r', vmin=-_sm_vmax, vmax=_sm_vmax)
        plt.colorbar(_rim3, ax=axes[2, 2], fraction=0.046, pad=0.04).set_label('m')
        axes[2, 2].set_title('Source Model Only (no ramp)')
        axes[2, 2].set_xlabel('X (m)')

    # Axis labels for rows 0 and 1
    for _r in range(min(2, _n_rows)):
        axes[_r, 0].set_ylabel('Y (m)')
        for _c in range(3):
            axes[_r, _c].set_xlabel('X (m)')

    # ── Fault trace overlay (Okada models only) ─────────────────────────────
    _mt = model_type if isinstance(model_type, str) else (list(model_type)[0] if model_type else '')
    if str(_mt).lower() == 'okada' and 'X0' in optimal_params:
        def _okada_corners_m(p):
            sr = np.radians(p['strike']); dr = np.radians(p['dip'])
            ss, cs, cd = np.sin(sr), np.cos(sr), np.cos(dr)
            s_e, s_n = ss, cs        # along-strike unit vector (E, N)
            d_e, d_n = cs, -ss       # up-dip horizontal unit vector
            hw = (p['width'] / 2) * cd  # horizontal half-width projection
            L  =  p['length'] / 2
            x0, y0 = p['X0'], p['Y0']
            X_tc = x0 - d_e * hw;  Y_tc = y0 - d_n * hw
            X_bc = x0 + d_e * hw;  Y_bc = y0 + d_n * hw
            cx = np.array([X_tc - s_e*L, X_tc + s_e*L, X_bc + s_e*L, X_bc - s_e*L])
            cy = np.array([Y_tc - s_n*L, Y_tc + s_n*L, Y_bc + s_n*L, Y_bc - s_n*L])
            return cx, cy

        _cx, _cy = _okada_corners_m(optimal_params)
        _poly_x  = np.append(_cx, _cx[0])
        _poly_y  = np.append(_cy, _cy[0])
        # top (shallowest) edge = indices 0–1
        _top_x, _top_y = _cx[:2], _cy[:2]

        for _r in range(axes.shape[0]):
            for _c in range(axes.shape[1]):
                _ax = axes[_r, _c]
                if not _ax.get_visible():
                    continue
                _ax.plot(_poly_x, _poly_y, '-', color='black', lw=1.5, zorder=10)
                _ax.plot(_top_x, _top_y, '-', color='black', lw=3.0, zorder=11,
                         label='Fault top edge')
                _ax.plot(optimal_params['X0'], optimal_params['Y0'],
                         '+', color='black', ms=8, mew=2, zorder=12)

    if noise_rms is not None:
        fig.suptitle(f'Noise RMS: {noise_rms*100:.2f} cm', fontsize=11, y=1.001)

    # Reserve a fixed strip at the bottom of the *whole* figure for a shared
    # colorbar BEFORE calling tight_layout, via its `rect` argument -- this
    # makes tight_layout fit every row (2 or 3) into the space above the
    # strip, so the colorbar (placed inside the strip afterwards) can never
    # overlap row content regardless of how many rows the figure has. This
    # replaces two earlier attempts (a hardcoded-position axes that assumed a
    # fixed row count, then a per-row colorbar squeezed between rows) that
    # both ended up overlapping panels for some row counts.
    _cbar_strip = 0.10
    plt.tight_layout(pad=1.2, h_pad=2.0, rect=[0, _cbar_strip, 1, 1])
    _cbar_ax = fig.add_axes([0.15, _cbar_strip * 0.25, 0.7, _cbar_strip * 0.35])
    fig.colorbar(_top_row_im, cax=_cbar_ax, orientation='horizontal', label='Displacement (m)')

    if figure_folder is not None:
        _date_str = f"_{ifg_dates}" if isinstance(ifg_dates, str) and ifg_dates else ""
        plt.savefig(f"{figure_folder}/Model_Comparison{_date_str}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_corner_unused(samples, burn_in=0, figure_folder=None, model_type='pCDM',
                       nbins=40, smooth=True, figsize_per_param=3.0, priors=None):
    """
    Corner plot showing parameter trade-offs via seaborn PairGrid.

    Lower triangle – 2-D density (viridis) showing how parameters trade off.
    Diagonal       – marginal histogram with mean (red) and mode (green) lines.
    Upper triangle – Pearson correlation coefficient.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde
    from matplotlib.ticker import MaxNLocator

    TICK_SIZE  = 11
    LABEL_SIZE = 13

    all_params = get_param_names(model_type, samples)
    all_params = [p for p in all_params if p in samples]
    n = len(all_params)
    if n < 2:
        print('plot_corner: need at least 2 parameters — skipping.')
        return

    n_total   = len(samples[all_params[0]])
    burn_safe = max(0, min(int(burn_in), n_total - 2))
    print(f'  plot_corner: {n_total - burn_safe:,} post-burn-in samples.')

    chains = np.column_stack(
        [np.asarray(samples[p][burn_safe:], dtype=float) for p in all_params]
    )

    labels       = [p.replace('__', '\n') for p in all_params]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # Axis ranges — always add 3 % padding so data never clips the edge
    if priors is not None:
        lo = np.zeros(n)
        hi = np.zeros(n)
        for i, param in enumerate(all_params):
            key = param if param in priors else param.split('__')[1] if '__' in param else None
            if key and key in priors:
                lo[i], hi[i] = priors[key]
            else:
                lo[i] = np.percentile(chains[:, i], 1)
                hi[i] = np.percentile(chains[:, i], 99)
        span = np.where((hi - lo) > 0, hi - lo, np.abs(hi) * 0.1 + 1e-9)
        lo -= 0.03 * span
        hi += 0.03 * span
    else:
        lo   = np.percentile(chains, 1,  axis=0)
        hi   = np.percentile(chains, 99, axis=0)
        span = np.where((hi - lo) > 0, hi - lo, np.abs(hi) * 0.1 + 1e-9)
        lo  -= 0.05 * span
        hi  += 0.05 * span

    # Track the last contourf artist for the colorbar
    _density_artist = [None]

    def _plot_2d_density(x, y, **kwargs):
        ax     = plt.gca()
        ix, iy = label_to_idx[x.name], label_to_idx[y.name]

        H, _, _ = np.histogram2d(x.values, y.values, bins=nbins,
                                  range=[[lo[ix], hi[ix]], [lo[iy], hi[iy]]])
        H = H.T

        if H.max() == 0:
            ax.set_visible(False)
            return

        if smooth:
            H = gaussian_filter(H.astype(float), sigma=max(0.5, nbins / 40.0))

        xi = np.linspace(lo[ix], hi[ix], H.shape[1])
        yi = np.linspace(lo[iy], hi[iy], H.shape[0])
        cf = ax.contourf(xi, yi, np.sqrt(H), levels=25, cmap='viridis', extend='both')
        _density_artist[0] = cf
        ax.set_xlim(lo[ix], hi[ix])
        ax.set_ylim(lo[iy], hi[iy])

    def _plot_diagonal(x, **kwargs):
        ax = plt.gca()
        ix = label_to_idx[x.name]
        x_data = x.values

        if np.std(x_data) == 0:
            ax.set_visible(False)
            return

        counts, edges = np.histogram(x_data, bins=nbins,
                                     range=(lo[ix], hi[ix]), density=True)
        ax.bar(edges[:-1], counts, width=np.diff(edges),
               color='#4B9CD3', alpha=0.65, align='edge', linewidth=0)

        kde_mode = None
        if hi[ix] > lo[ix] and np.std(x_data) > 0:
            try:
                kde_fn = gaussian_kde(x_data, bw_method='scott')
                x_kde  = np.linspace(lo[ix], hi[ix], 400)
                y_kde  = kde_fn(x_kde)
                ax.plot(x_kde, y_kde, color='white', lw=1.8, zorder=4)
                kde_mode = x_kde[np.argmax(y_kde)]
            except Exception:
                pass

        ax.axvline(np.mean(x_data), color='#FF6B6B', lw=2.0, ls='-',
                   zorder=5, label='mean')
        if kde_mode is not None:
            ax.axvline(kde_mode, color='#2ECC71', lw=2.0, ls='--', zorder=5,
                       label='mode')

        ax.set_xlim(lo[ix], hi[ix])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_visible(False)

        # Legend only on the first diagonal panel
        if ix == 0:
            ax.legend(fontsize=8, loc='upper right',
                      framealpha=0.8, handlelength=1.4)

    def _plot_upper_corr(*_, **__):
        plt.gca().set_visible(False)

    df = pd.DataFrame(chains, columns=labels)
    g  = sns.PairGrid(df, height=figsize_per_param, aspect=1,
                      despine=False, layout_pad=0.5)

    g.map_lower(_plot_2d_density)
    g.map_diag(_plot_diagonal)
    g.map_upper(_plot_upper_corr)

    # ── Tick / label polish ───────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            ax = g.axes[i, j]
            if not ax.get_visible():
                continue

            ax.tick_params(labelsize=TICK_SIZE, length=4, width=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

            show_xlabel = (i == n - 1)
            ax.tick_params(axis='x', labelbottom=show_xlabel)
            if show_xlabel:
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(45)
                    lbl.set_ha('right')

            # Left column off-diagonal: show y-axis ticks and labels
            if j == 0 and i > 0:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
                ax.tick_params(axis='y', labelleft=True, labelsize=TICK_SIZE)
            elif i != j:
                ax.tick_params(axis='y', labelleft=False)

    # ── Axis (parameter) labels along the diagonal ───────────────────────
    for i, lbl in enumerate(labels):
        diag_ax = g.axes[i, i]
        diag_ax.set_title(lbl, fontsize=LABEL_SIZE, pad=4,
                          fontweight='semibold', color='white',
                          bbox=dict(boxstyle='round,pad=0.25',
                                    facecolor='#2c3e50', alpha=0.85,
                                    linewidth=0))

    # ── Shared density colorbar (lower-triangle scale) ────────────────────
    g.figure.subplots_adjust(hspace=0.12, wspace=0.12, bottom=0.10)
    if _density_artist[0] is not None:
        cbar_ax = g.figure.add_axes([0.12, 0.02, 0.45, 0.018])
        cb = g.figure.colorbar(_density_artist[0], cax=cbar_ax,
                               orientation='horizontal')
        cb.set_label('∝ √(density)', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    g.figure.suptitle(
        f'Parameter Trade-off Corner Plot   (n = {chains.shape[0]:,} post-burn-in)',
        fontsize=13, y=1.005, fontweight='bold')

    if figure_folder is not None:
        g.figure.savefig(f"{figure_folder}/corner_plot.png",
                         dpi=200, bbox_inches='tight')
        print(f'  Saved: {figure_folder}/corner_plot.png')

    # plt.show()


def plot_corner(samples, burn_in=0, figure_folder=None, model_type='pCDM',
                nbins=40, smooth=True, figsize_per_param=2.5, priors=None):
    """
    Corner plot: lower triangle 2-D density + diagonal marginal histograms.
    Near-zero-variance parameters are automatically excluded.
    """
    fig = _draw_corner(None, samples, burn_in=burn_in, model_type=model_type,
                       nbins=nbins, smooth=smooth, figsize_per_param=figsize_per_param)
    if fig is None:
        return
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=1.5, h_pad=0.8, w_pad=0.8)

    if figure_folder is not None:
        fig.savefig(f"{figure_folder}/corner_plot.png", dpi=200, bbox_inches='tight')
        print(f'  Saved: {figure_folder}/corner_plot.png')


def _draw_corner(fig, samples, burn_in=0, model_type='pCDM', nbins=40, smooth=True,
                 figsize_per_param=2.5, title=None, title_size=None, max_kde_points=200000):
    """
    Draw the corner plot onto `fig` (a Figure or SubFigure). If `fig` is None a
    new figure is created. Returns the figure, or None if there are < 2 active
    parameters.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde
    from matplotlib.ticker import MaxNLocator

    # ── Data preparation ───────────────────────────────────────────────────
    all_params = get_param_names(model_type, samples)
    all_params = [p for p in all_params if p in samples and not p.startswith('ramp_')]

    n_total   = len(samples[all_params[0]])
    burn_safe = max(0, min(int(burn_in), n_total - 2))

    chains_all = np.column_stack(
        [np.asarray(samples[p][burn_safe:], dtype=float) for p in all_params]
    )

    # Exclude parameters that didn't mix (relative std below threshold)
    stds  = chains_all.std(axis=0)
    means = np.abs(chains_all.mean(axis=0))
    active = stds / np.where(means > 0, means, 1.0) > 1e-4

    params = [p for p, a in zip(all_params, active) if a]
    chains = chains_all[:, active]
    n      = len(params)

    if n < 2:
        print('plot_corner: fewer than 2 variable parameters — skipping.')
        return

    print(f'  plot_corner: {chains.shape[0]:,} post-burn-in samples, {n} active parameters.')

    labels = [p.replace('__', '\n') for p in params]

    # ── Axis ranges — always data-driven, with generous padding ────────────
    # Use wide percentiles (0.5/99.5) so the tails are visible, then add
    # extra whitespace so you can judge whether the chain has hit an edge.
    lo = np.percentile(chains, 0.5,  axis=0)
    hi = np.percentile(chains, 99.5, axis=0)

    span = np.where((hi - lo) > 0, hi - lo, np.abs(hi) * 0.1 + 1e-9)
    lo  -= 0.3 * span
    hi  += 0.3 * span

    # ── Figure ──────────────────────────────────────────────────────────────
    LABEL_SIZE = max(8, min(12, int(130 / n)))
    TICK_SIZE  = max(7, min(10, int(110 / n)))

    if fig is None:
        fig = plt.figure(figsize=(figsize_per_param * n, figsize_per_param * n))
    axes = fig.subplots(n, n, squeeze=False)

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            if col > row:
                ax.set_visible(False)
                continue

            is_diag   = (row == col)
            is_bottom = (row == n - 1)

            # ── Diagonal: marginal histogram ──────────────────────────────
            if is_diag:
                x_data = chains[:, col]
                counts, edges = np.histogram(x_data, bins=nbins,
                                             range=(lo[col], hi[col]))
                ax.bar(edges[:-1], counts, width=np.diff(edges),
                       color='#4B9CD3', alpha=0.75, align='edge', linewidth=0)

                kde_mode = None
                try:
                    # KDE cost scales with sample count; a thinned chain gives the same curve
                    step = max(1, len(x_data) // max_kde_points)
                    kde_fn = gaussian_kde(x_data[::step], bw_method='scott')
                    x_kde  = np.linspace(lo[col], hi[col], 400)
                    y_kde  = kde_fn(x_kde) * len(x_data) * np.diff(edges)[0]
                    ax.plot(x_kde, y_kde, color='#1a5276', lw=1.5, zorder=4)
                    kde_mode = x_kde[np.argmax(y_kde)]
                except Exception:
                    pass

                if kde_mode is not None:
                    ax.axvline(kde_mode, color='red', lw=1.5, ls='--', zorder=5)

                ax.set_xlim(lo[col], hi[col])
                ax.set_ylim(0, counts.max() * 1.15)
                ax.set_title(labels[col], fontsize=LABEL_SIZE,
                             fontweight='bold', pad=4)

                if col == 0:
                    ax.set_ylabel('Frequency', fontsize=LABEL_SIZE - 1)
                    ax.tick_params(axis='y', labelsize=TICK_SIZE)
                else:
                    ax.tick_params(axis='y', labelleft=False)

            # ── Lower triangle: 2-D density ───────────────────────────────
            else:
                x_data = chains[:, col]
                y_data = chains[:, row]

                H, _, _ = np.histogram2d(
                    x_data, y_data, bins=nbins,
                    range=[[lo[col], hi[col]], [lo[row], hi[row]]]
                )
                H = H.T
                if smooth:
                    H = gaussian_filter(H.astype(float), sigma=max(0.5, nbins / 40.0))

                xi = np.linspace(lo[col], hi[col], H.shape[1])
                yi = np.linspace(lo[row], hi[row], H.shape[0])
                ax.contourf(xi, yi, np.sqrt(H), levels=25, cmap='viridis', extend='both')

                ax.set_xlim(lo[col], hi[col])
                ax.set_ylim(lo[row], hi[row])

                if col == 0:
                    ax.set_ylabel(labels[row], fontsize=LABEL_SIZE - 1)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
                    ax.tick_params(axis='y', labelsize=TICK_SIZE)
                else:
                    ax.tick_params(axis='y', labelleft=False)

            # ── X-axis ────────────────────────────────────────────────────
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

            if is_bottom:
                ax.set_xlabel(labels[col], fontsize=LABEL_SIZE - 1)
                ax.tick_params(axis='x', labelsize=TICK_SIZE, labelbottom=True)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                         rotation_mode='anchor')
            else:
                ax.tick_params(axis='x', labelbottom=False)

    # ── Title and layout ─────────────────────────────────────────────────────
    if title is None:
        title = f'Parameter Posterior Corner Plot   (n = {chains.shape[0]:,} post-burn-in)'
    fig.suptitle(title, fontsize=title_size or max(10, LABEL_SIZE + 1), fontweight='bold')
    return fig


def plot_model_components(samples, u_los_obs, X_obs, Y_obs,
                          incidence_angle, heading, log_likelihood_trace,
                          burn_in=0, figure_folder=None, model_type='pCDM'):
    """
    Plot each model's MAP LOS contribution in its own panel, with geometric
    annotations, plus a final combined panel.

    - Okada panels   : fault-plane surface projection drawn as a rectangle;
                       thick line = up-dip (shallowest) edge; arrow points up-dip.
    - UNE panels     : circle + crosshair at the cavity centre (X0, Y0).
    - pCDM panels    : star marker at the pressure-source centre.
    - Combined panel : sum of all contributions, all annotations overlaid.

    Works for single-model OR multi-model (prefixed sample keys) runs.
    For multi-IFG observations the first interferogram is used.
    """
    # ------------------------------------------------------------------ #
    # 0. Flatten multi-IFG inputs to a single IFG                        #
    # ------------------------------------------------------------------ #
    def _first_ifg(x):
        if isinstance(x, (list, tuple)) and len(x) > 0 and hasattr(x[0], '__iter__'):
            return np.asarray(x[0])
        return np.asarray(x)

    u_los_use = _first_ifg(u_los_obs)
    X_use     = _first_ifg(X_obs)
    Y_use     = _first_ifg(Y_obs)
    inc_use   = float(np.mean(_first_ifg(incidence_angle)))
    head_use  = float(np.mean(_first_ifg(heading)))

    # ------------------------------------------------------------------ #
    # 1. MAP parameter set (post-burn-in best)                           #
    # ------------------------------------------------------------------ #
    samples_burned = {k: np.array(v[burn_in:]) for k, v in samples.items()}
    best_idx   = np.argmax(log_likelihood_trace[burn_in:])
    map_params = {k: samples_burned[k][best_idx] for k in samples_burned}

    # ------------------------------------------------------------------ #
    # 2. LOS unit vector                                                  #
    # ------------------------------------------------------------------ #
    inc_rad  = np.radians(inc_use)
    head_rad = np.radians(head_use)
    los_e =  np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)

    def _to_los(ue, un, uv):
        return -(np.asarray(ue, float) * los_e +
                 np.asarray(un, float) * los_n +
                 np.asarray(uv, float) * los_u)

    # ------------------------------------------------------------------ #
    # 3. Per-label forward models                                         #
    # ------------------------------------------------------------------ #
    is_multi = any('__' in k for k in map_params)
    if is_multi:
        model_labels = sorted({k.split('__')[0] for k in map_params})
    else:
        ml = model_type if isinstance(model_type, str) else list(model_type)[0]
        model_labels = [ml.lower()]

    contributions = {}   # label -> 1-D LOS array
    annot_info    = {}   # label -> dict for annotation

    for label in model_labels:
        if is_multi:
            model_name = label.split('_')[0].lower()
            p = {k.split('__')[1]: map_params[k]
                 for k in map_params if k.startswith(label + '__')}
        else:
            model_name = label
            p = map_params

        if model_name in MODEL_REGISTRY:
            ue, un, uv = forward_from_registry(model_name, X_use, Y_use, p)
            contributions[label] = _to_los(ue, un, uv)
            # Store all model params so annotation helpers (_draw_okada etc.) can access them
            annot_info[label] = {**p, 'type': model_name}
        else:
            contributions[label] = np.zeros_like(X_use, float)
            annot_info[label] = {'type': 'unknown', 'X0': 0.0, 'Y0': 0.0}

    combined = sum(contributions.values())

    # ------------------------------------------------------------------ #
    # 4. Shared colour scale (98th-pct of absolute values)               #
    # ------------------------------------------------------------------ #
    all_vals = np.concatenate(list(contributions.values()) + [combined])
    vmax_abs = np.nanpercentile(np.abs(all_vals[np.isfinite(all_vals)]), 98)
    vmax_abs = vmax_abs if vmax_abs > 0 else 1.0
    CMAP = 'RdBu_r'

    # ------------------------------------------------------------------ #
    # 5. Layout                                                           #
    # ------------------------------------------------------------------ #
    n_models = len(model_labels)
    n_panels = n_models + 1          # individual + combined
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5 * n_panels, 5.5),
                             constrained_layout=False)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(bottom=0.18, wspace=0.35)

    # ------------------------------------------------------------------ #
    # 6. Helpers                                                          #
    # ------------------------------------------------------------------ #
    def _scatter(ax, data, title):
        # Interpolate scattered points onto a regular grid for contourf
        x_min, x_max = X_use.min(), X_use.max()
        y_min, y_max = Y_use.min(), Y_use.max()
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((X_use, Y_use), data, (Xi, Yi), method='linear')
        cf = ax.contourf(Xi, Yi, Zi, levels=30, cmap=CMAP,
                         vmin=-vmax_abs, vmax=vmax_abs)
        ax.contour(Xi, Yi, Zi, levels=10, colors='k',
                   linewidths=0.3, alpha=0.25)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='datalim')
        return cf

    def _draw_okada(ax, info, color='black'):
        """Draw fault-plane surface projection; thick line = shallowest edge."""
        sk  = np.radians(info['strike'])
        dp  = np.radians(info['dip'])
        L   = info['length'] / 2.0
        # Horizontal projection of half-width up-dip
        hw  = info['width'] * np.cos(dp)
        x0, y0 = info['X0'], info['Y0']
        # Along-strike unit vector (E,N plane;  strike = azimuth from N)
        sx, sy =  np.sin(sk),  np.cos(sk)
        # Up-dip horizontal unit vector (perpendicular to strike, toward updip)
        ux, uy = -np.cos(sk), np.sin(sk)
        # Four corners (ref point assumed = along-strike centre of fault)
        TL = np.array([x0 + L*sx + hw*ux, y0 + L*sy + hw*uy])
        TR = np.array([x0 - L*sx + hw*ux, y0 - L*sy + hw*uy])
        BL = np.array([x0 + L*sx - hw*ux, y0 + L*sy - hw*uy])
        BR = np.array([x0 - L*sx - hw*ux, y0 - L*sy - hw*uy])
        # Fault outline
        poly_x = [TL[0], TR[0], BR[0], BL[0], TL[0]]
        poly_y = [TL[1], TR[1], BR[1], BL[1], TL[1]]
        ax.plot(poly_x, poly_y, '-', color=color, lw=1.2,
                label='Fault plane (proj.)', zorder=5)
        # Shallowest (top) edge – thick
        ax.plot([TL[0], TR[0]], [TL[1], TR[1]], '-', color=color, lw=3.5, zorder=6)
        # Arrow from centre toward up-dip direction
        cx_top = (TL[0] + TR[0]) / 2.0
        cy_top = (TL[1] + TR[1]) / 2.0
        ax.annotate('', xy=(cx_top, cy_top), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                    zorder=7)
        # Centre mark
        ax.plot(x0, y0, '+', color=color, ms=8, mew=2, zorder=7)

    def _draw_une(ax, info, color='black'):
        """Draw circle + crosshair at cavity centre."""
        x0, y0 = info['X0'], info['Y0']
        r = 0.025 * (X_use.max() - X_use.min())
        circle = plt.Circle((x0, y0), r, color=color, fill=False,
                             lw=1.8, label='Cavity centre', zorder=5)
        ax.add_patch(circle)
        ax.plot(x0, y0, '+', color=color, ms=10, mew=2.5, zorder=6)
        ax.annotate(f'  depth={info["depth"]:.0f} m',
                    xy=(x0, y0), fontsize=7, color=color, zorder=6,
                    xytext=(x0 + 1.6*r, y0 + 1.6*r),
                    arrowprops=dict(arrowstyle='-', color='grey', lw=0.8))

    def _draw_pcdm(ax, info, color='black'):
        ax.plot(info['X0'], info['Y0'], '*', color=color,
                ms=11, mew=1.5, label='Source centre', zorder=5)

    ANNOTATION_COLORS = ['black', 'yellow', 'lime', 'magenta']

    def _annotate(ax, label, color='black'):
        info = annot_info[label]
        if info['type'] == 'okada':
            _draw_okada(ax, info, color)
        elif info['type'] == 'une':
            _draw_une(ax, info, color)
        elif info['type'] == 'pcdm':
            _draw_pcdm(ax, info, color)

    # ------------------------------------------------------------------ #
    # 7. Individual model panels                                          #
    # ------------------------------------------------------------------ #
    sc_ref = None
    for i, label in enumerate(model_labels):
        ax   = axes[i]
        pretty = label.replace('_', ' ').title()
        sc_ref = _scatter(ax, contributions[label], f'{pretty}  LOS')
        _annotate(ax, label, color='black')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

    # ------------------------------------------------------------------ #
    # 8. Combined panel                                                   #
    # ------------------------------------------------------------------ #
    ax_comb = axes[-1]
    sc_ref  = _scatter(ax_comb, combined, 'Combined LOS')
    for j, label in enumerate(model_labels):
        col = ANNOTATION_COLORS[j % len(ANNOTATION_COLORS)]
        _annotate(ax_comb, label, color=col)
    ax_comb.legend(fontsize=6, loc='upper right', framealpha=0.7)

    # ------------------------------------------------------------------ #
    # 9. Shared colour bar                                                #
    # ------------------------------------------------------------------ #
    cbar_ax = fig.add_axes([0.1, 0.07, 0.8, 0.03])
    fig.colorbar(sc_ref, cax=cbar_ax, orientation='horizontal',
                 label='LOS displacement (m)')

    fig.suptitle('Model Component LOS Predictions  (MAP estimate)', fontsize=11)

    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/model_components_LOS.png",
                    dpi=200, bbox_inches='tight')
        print(f'  Saved: {figure_folder}/model_components_LOS.png')

    # plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Report figure: corner plot + data fit + (multi-model) component panels
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_SCALE = {'m': 1.0, 'cm': 1e2, 'mm': 1e3}


def _split_map_models(map_params, model_type):
    """Return [(label, model_name, params)] for single or prefixed multi-model dicts."""
    if any('__' in k for k in map_params):
        labels = sorted({k.split('__')[0] for k in map_params if '__' in k})
        return [(lab, lab.split('_')[0].lower(),
                 {k.split('__')[1]: v for k, v in map_params.items() if k.startswith(lab + '__')})
                for lab in labels]
    m = model_type if isinstance(model_type, str) else list(model_type)[0]
    return [(m.lower(), m.lower(), map_params)]


def _ramp_for_ifg(p, X, Y, j, n_ifgs):
    """MAP ramp/offset at (X, Y) for IFG j; handles 'ramp_c' and 'ramp_c_{j}' naming."""
    sfx = f'_{j}' if f'ramp_c_{j}' in p else ''
    if not sfx and n_ifgs > 1 and 'ramp_c' not in p:
        return np.zeros(len(X))
    return (p.get(f'ramp_c{sfx}', 0.0)
            + p.get(f'ramp_a{sfx}', 0.0) * X
            + p.get(f'ramp_b{sfx}', 0.0) * Y) * np.ones(len(X))


def _pixel_grid(x, y, max_cells=400):
    """Nearest-neighbour index grid for (possibly scattered) points, like GMT xyz2grd.

    Returns (idx, valid, extent): idx maps each cell to its nearest data point and
    valid masks cells further than ~one sample spacing from any point.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x, y]))
    spacing = np.median(tree.query(np.column_stack([x, y]), k=2)[0][:, 1])
    spacing = max(spacing, max(np.ptp(x), np.ptp(y)) / max_cells)
    xi = np.arange(x.min(), x.max() + spacing, spacing)
    yi = np.arange(y.min(), y.max() + spacing, spacing)
    Xi, Yi = np.meshgrid(xi, yi)
    d, idx = tree.query(np.column_stack([Xi.ravel(), Yi.ravel()]))
    extent = (xi[0] - spacing / 2, xi[-1] + spacing / 2, yi[0] - spacing / 2, yi[-1] + spacing / 2)
    return idx.reshape(Xi.shape), (d <= 1.5 * spacing).reshape(Xi.shape), extent


def _draw_source_marker(ax, name, p, color='black', km=True):
    """Overlay MAP source geometry (Okada outline, or a centre marker otherwise)."""
    s = 1e-3 if km else 1.0
    x0, y0 = p.get('X0'), p.get('Y0')
    if x0 is None or y0 is None:
        return
    if name == 'okada' and all(k in p for k in ('strike', 'dip', 'length', 'width')):
        # (X0, Y0) is the fault centroid (see okada_model.disloc3d3); the fault dips
        # to the right of strike, so the shallow edge sits up-dip of the centroid.
        sk, dp = np.radians(p['strike']), np.radians(p['dip'])
        se, sn = np.sin(sk), np.cos(sk)           # along-strike
        de, dn = np.cos(sk), -np.sin(sk)          # horizontal down-dip
        hw, L = 0.5 * p['width'] * np.cos(dp), 0.5 * p['length']
        top = np.array([[x0 - de*hw - se*L, y0 - dn*hw - sn*L],
                        [x0 - de*hw + se*L, y0 - dn*hw + sn*L]])
        bot = np.array([[x0 + de*hw + se*L, y0 + dn*hw + sn*L],
                        [x0 + de*hw - se*L, y0 + dn*hw - sn*L]])
        poly = np.vstack([top, bot, top[:1]]) * s
        ax.plot(poly[:, 0], poly[:, 1], '-', color=color, lw=1.0, zorder=10)
        ax.plot(top[:, 0] * s, top[:, 1] * s, '-', color=color, lw=2.8, zorder=11,
                solid_capstyle='butt')
        ax.plot(x0 * s, y0 * s, '+', color=color, ms=7, mew=1.5, zorder=12)
    else:
        marker = '*' if name == 'pcdm' else 'o' if name.startswith('une') else 'X'
        ax.plot(x0 * s, y0 * s, marker, mfc='none' if marker == 'o' else color,
                mec=color, ms=9, mew=1.5, zorder=12)


def plot_report_figure(samples, u_los_obs, X_obs, Y_obs, incidence_angle, heading,
                       log_likelihood_trace, burn_in=0, figure_folder=None,
                       model_type='pCDM', ifg_dates=None, include_components='auto', include_corner=True,
                       units='mm', show_sources=True, wrapped=False, wavelength=0.0555,
                       fig_width=14.0, filename='report_figure.png', dpi=300):
    """
    Single summary figure for write-ups, stacking:

      (a) posterior corner plot (ramp parameters excluded, as in plot_corner),
      (b) data fit: observed | MAP model (+ramp) | residual, one row per IFG,
      (c) model components (multi-model runs only): each source's MAP LOS
          contribution plus their sum, on a shared colour scale.

    Parameters
    ----------
    include_components : 'auto' | bool
        'auto' draws row (c) only when more than one source model was fitted.
    units : 'm' | 'cm' | 'mm'
        Displacement units for the colour bars. Map axes are always in km.
    show_sources : bool
        Overlay MAP source locations / Okada fault outlines on the map panels.
    wrapped : bool
        Add a re-wrapped (mod wavelength/2) row under each data-fit row, styled
        after GBIS_output_clean.plot_mod_los_res.
    """
    import matplotlib.colors as mcolors
    from matplotlib.ticker import MaxNLocator

    scale = _UNIT_SCALE[units]
    try:
        from cmcrameri import cm as _cmc
        cmap, cmap_wrap = _cmc.vik, _cmc.romaO
    except ImportError:
        cmap, cmap_wrap = 'RdBu_r', 'twilight'
    wrap_len = wavelength / 2

    # ── MAP parameters ───────────────────────────────────────────────────
    ll = np.asarray(log_likelihood_trace[burn_in:])
    best = int(np.argmax(ll))
    map_params = {k: float(np.asarray(v[burn_in:])[best]) for k, v in samples.items()}
    models = _split_map_models(map_params, model_type)

    # ── Normalise observations to per-IFG lists ─────────────────────────
    multi_ifg = (isinstance(u_los_obs, (list, tuple))
                 or (isinstance(u_los_obs, np.ndarray) and u_los_obs.dtype == object)) \
        and len(u_los_obs) > 0 and hasattr(u_los_obs[0], '__iter__')
    if multi_ifg:
        n_ifgs = len(u_los_obs)

        def _per_ifg(a):
            if isinstance(a, (list, tuple)) or (isinstance(a, np.ndarray) and a.dtype == object):
                return [np.asarray(a[j], float) for j in range(n_ifgs)]
            return [np.asarray(a, float)] * n_ifgs
        U, XS, YS = _per_ifg(u_los_obs), _per_ifg(X_obs), _per_ifg(Y_obs)
        INC, HEAD = _per_ifg(incidence_angle), _per_ifg(heading)
    else:
        n_ifgs = 1
        U, XS, YS = [np.asarray(u_los_obs, float)], [np.asarray(X_obs, float)], [np.asarray(Y_obs, float)]
        INC, HEAD = [np.asarray(incidence_angle, float)], [np.asarray(heading, float)]
    if isinstance(ifg_dates, str):
        ifg_dates = [ifg_dates]
    dates = list(ifg_dates) if ifg_dates is not None else [None] * n_ifgs

    def _los(ue, un, uv, inc, head):
        inc_r, head_r = np.radians(inc), np.radians(head)
        return -(np.asarray(ue, float) * np.sin(inc_r) * np.cos(head_r)
                 - np.asarray(un, float) * np.sin(inc_r) * np.sin(head_r)
                 - np.asarray(uv, float) * np.cos(inc_r))

    # ── Forward models per IFG ───────────────────────────────────────────
    fits = []
    for j in range(n_ifgs):
        X, Y = XS[j], YS[j]
        comps = {}
        for label, name, p in models:
            try:
                comps[label] = _los(*forward_from_registry(name, X, Y, p), INC[j], HEAD[j])
            except ValueError as exc:
                print(f'  plot_report_figure: skipping {label} ({exc})')
                comps[label] = np.zeros_like(X)
        source = sum(comps.values())
        model = source + _ramp_for_ifg(map_params, X, Y, j, n_ifgs)
        fits.append(dict(X=X, Y=Y, grid=_pixel_grid(X, Y), obs=U[j],
                         model=model, res=U[j] - model, comps=comps, source=source))

    if include_components == 'auto':
        include_components = len(models) > 1

    # ── Layout ────────────────────────────────────────────────────────────
    corner_params = [p for p in get_param_names(model_type, samples)
                     if p in samples and not p.startswith('ramp_')]
    n_corner = max(2, len(corner_params))
    aspect = np.ptp(fits[0]['Y']) / max(np.ptp(fits[0]['X']), 1e-9)

    def _row_h(ncols):
        # panel height from width & data aspect, plus room for titles/labels
        return (fig_width * 0.8 / ncols) * aspect + 1.0

    heights = [fig_width * min(1.3, 0.1 * n_corner + 0.1)] if include_corner else []
    heights += [_row_h(3) * (2 if wrapped else 1)] * n_ifgs
    if include_components:
        heights.append(_row_h(len(models) + 1))

    fig = plt.figure(figsize=(fig_width, sum(heights)), layout='constrained')
    subfigs = np.atleast_1d(fig.subfigures(len(heights), 1, height_ratios=heights, hspace=0.02))

    letters = iter('abcdefghijklmnopqrstuvwxyz')
    n_post = len(ll)

    def _block_title(sf, text):
        # block label only; `text` is kept at the call sites as a description
        sf.suptitle(f'({next(letters)})', x=0.0, ha='left',
                    fontsize=13, fontweight='bold')

    # (a) corner
    off = 1 if include_corner else 0
    if include_corner:
        _draw_corner(subfigs[0], samples, burn_in=burn_in, model_type=model_type,
                     figsize_per_param=fig_width / n_corner, title='')
        _block_title(subfigs[0], f'Posterior distributions  (n = {n_post:,} post-burn-in samples; '
                                 f'red dashed = marginal mode)')

    def _style_map(ax, title, ylabel=True):
        ax.set_title(title, fontsize=16)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (km)', fontsize=9)
        if ylabel:
            ax.set_ylabel('Northing (km)', fontsize=9)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

    def _tri_plot(ax, f, values, norm, cm=None):
        idx, valid, ext = f['grid']
        img = np.where(valid, np.asarray(values, float)[idx], np.nan)
        ext_km = [e * 1e-3 for e in ext]
        im = ax.imshow(img, origin='lower', extent=ext_km, cmap=cm or cmap, norm=norm,
                       interpolation='nearest', rasterized=True)
        ax.set_xlim(ext_km[:2])
        ax.set_ylim(ext_km[2:])
        return im

    def _sym_norm(*arrays, pct=99.5):
        v = np.concatenate([np.abs(a[np.isfinite(a)]) for a in arrays])
        vmax = np.percentile(v, pct) if v.size else 1.0
        return mcolors.Normalize(-(vmax or 1.0), vmax or 1.0)

    def _overlay(ax, colors=None):
        if not show_sources:
            return
        for k, (label, name, p) in enumerate(models):
            _draw_source_marker(ax, name, p, color=(colors[k] if colors else 'black'))

    # (b) data fit rows
    # Same layout as plot_mod_los_res: Data | Model | Residual, all on the data's
    # colour range, with the re-wrapped versions underneath.
    keys = ('obs', 'model', 'res')
    for j, f in enumerate(fits):
        sf = subfigs[off + j]
        axes = np.atleast_2d(sf.subplots(2 if wrapped else 1, 3, sharex=True, sharey=True))
        vmax = np.nanmax(np.abs(f['obs'])) * scale
        norm = mcolors.Normalize(-vmax, vmax)
        for c, (ax, key, title) in enumerate(zip(axes[0], keys, ('Data', 'Model', 'Residual'))):
            im = _tri_plot(ax, f, f[key] * scale, norm)
            _style_map(ax, title, ylabel=(c == 0))
            _overlay(ax)
        rms = np.sqrt(np.nanmean(f['res'] ** 2)) * scale
        axes[0, 2].text(0.02, 0.02, f'RMS = {rms:.2f} {units}', transform=axes[0, 2].transAxes,
                        fontsize=9, va='bottom', bbox=dict(fc='white', ec='none', alpha=0.8))
        sf.colorbar(im, ax=axes[0], shrink=0.9, extend='both', pad=0.01,
                    label=f'LOS displacement ({units})')
        if wrapped:
            wnorm = mcolors.Normalize(0, wrap_len * scale)
            for c, (ax, key) in enumerate(zip(axes[1], keys)):
                im = _tri_plot(ax, f, np.mod(f[key], wrap_len) * scale, wnorm, cm=cmap_wrap)
                _style_map(ax, '', ylabel=(c == 0))
                _overlay(ax)
            sf.colorbar(im, ax=axes[1], shrink=0.9, pad=0.01,
                        label=f'Wrapped LOS ({units}, mod λ/2)')
        tag = dates[j] if j < len(dates) and dates[j] else (f'IFG {j + 1}' if n_ifgs > 1 else '')
        _block_title(sf, 'Data fit' + (f' — {tag}' if tag else '')
                     + ('  (model includes fitted ramp)' if any(k.startswith('ramp_') for k in map_params) else ''))

    # (c) components (first IFG geometry)
    if include_components:
        f = fits[0]
        sf = subfigs[-1]
        axes = np.atleast_1d(sf.subplots(1, len(models) + 1, sharex=True, sharey=True))
        # Same scale as the data panels (b), unclipped, so the sources can be
        # compared with each other and with the data at a glance.
        vmax = np.nanmax(np.abs(f['obs'])) * scale
        norm = mcolors.Normalize(-vmax, vmax)
        colors = ['black', 'darkorange', 'green', 'purple']
        for k, (label, name, p) in enumerate(models):
            im = _tri_plot(axes[k], f, f['comps'][label] * scale, norm)
            _style_map(axes[k], label.replace('_', ' ').title(), ylabel=(k == 0))
            if show_sources:
                _draw_source_marker(axes[k], name, p, color='black')
        im = _tri_plot(axes[-1], f, f['source'] * scale, norm)
        _style_map(axes[-1], 'Sum of sources (no ramp)', ylabel=False)
        _overlay(axes[-1], colors=colors[:len(models)] if len(models) <= len(colors) else None)
        sf.colorbar(im, ax=axes, shrink=0.9, pad=0.01,
                    label=f'LOS displacement ({units})')
        _block_title(sf, 'MAP source contributions'
                     + (f' — {dates[0]}' if n_ifgs > 1 and dates[0] else (' — IFG 1' if n_ifgs > 1 else '')))

    if figure_folder is not None:
        out = os.path.join(figure_folder, filename)
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)


def test_corner_plot():
    """
    Test function to visualize corner plot with synthetic MCMC samples.
    Generates correlated posterior samples and displays the corner plot.
    """
    print("Generating synthetic MCMC samples for corner plot test...")
    
    # Define parameter names and priors (Mogi-like model)
    param_names = ['X0', 'Y0', 'depth', 'DV']
    n_params = len(param_names)
    n_samples = 5000
    
    # Define priors
    priors = {
        'X0': (-1500, 1500),
        'Y0': (-1500, 1500),
        'depth': (100, 5000),
        'DV': (-1e8, -1e4)
    }
    
    # Create synthetic posterior with correlations
    # True values (around which posterior is centered)
    true_params = np.array([100, 50, 800, -5e7])
    
    # Covariance matrix with some correlations
    # Depth and DV are correlated, X0 and Y0 slightly correlated
    cov = np.array([
        [50000,     5000,    0,      0],      # X0
        [5000,      50000,   0,      0],      # Y0
        [0,         0,       300000, 1e8],    # depth
        [0,         0,       1e8,    5e15]    # DV
    ])
    
    # Generate samples from multivariate normal
    samples_array = np.random.multivariate_normal(true_params, cov, n_samples)
    
    # Convert to prior bounds to make it more realistic (clip to priors)
    for i, param in enumerate(param_names):
        lo, hi = priors[param]
        samples_array[:, i] = np.clip(samples_array[:, i], lo, hi)
    
    # Create samples dict
    samples = {param: samples_array[:, i].tolist() for i, param in enumerate(param_names)}
    
    # Generate test figure folder
    test_folder = "test_corner_plot"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    print(f"Sample statistics:")
    for i, param in enumerate(param_names):
        mean = np.mean(samples_array[:, i])
        std = np.std(samples_array[:, i])
        print(f"  {param}: mean={mean:.2e}, std={std:.2e}, "
              f"prior=[{priors[param][0]:.2e}, {priors[param][1]:.2e}]")
    
    # Plot corner with priors
    print(f"\nGenerating corner plot with priors...")
    plot_corner(samples, burn_in=0, figure_folder=test_folder, model_type='pCDM',
                nbins=40, smooth=True, figsize_per_param=2.0, priors=priors)
    
    # Also generate without priors for comparison
    print(f"Generating corner plot without priors (for comparison)...")
    test_folder_no_priors = "test_corner_plot_no_priors"
    if not os.path.exists(test_folder_no_priors):
        os.makedirs(test_folder_no_priors)
    plot_corner(samples, burn_in=0, figure_folder=test_folder_no_priors, model_type='pCDM',
                nbins=40, smooth=True, figsize_per_param=2.0, priors=None)
    
    print(f"\nTest plots saved to:")
    print(f"  {test_folder}/corner_plot.png (with prior bounds)")
    print(f"  {test_folder_no_priors}/corner_plot.png (without prior bounds)")
    print("\nYou can open these files to see the corner plot visualization.")


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_corner_plot()


