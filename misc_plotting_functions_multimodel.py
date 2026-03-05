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
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/SA_diagnostics.png", dpi=300, bbox_inches='tight')




def plot_adaptation_diagnostics(proposal_std_evolution, acceptance_rate_evolution, 
                               adaptive_interval, target_acceptance=0.23, figure_folder=None,
                               model_type='pCDM', proposal_std_iters=None):
    """
    Plot diagnostics for adaptive MCMC including proposal scale evolution 
    and acceptance rate evolution.
    
    Parameters:
    -----------
    proposal_std_evolution : dict
        Evolution of proposal standard deviations for each parameter
    acceptance_rate_evolution : dict
        Evolution of acceptance rates for each parameter
    adaptive_interval : int
        Interval between adaptations
    target_acceptance : float
        Target acceptance rate
    figure_folder : str, optional
        Folder to save figures
    proposal_std_iters : list, optional
        Actual iteration numbers for each proposal_std checkpoint
    """
    param_names = list(proposal_std_evolution.keys())
    n_params = len(param_names)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Build x-axis: use real iteration numbers if provided, else checkpoint index * adaptive_interval
    n_checkpoints = len(proposal_std_evolution[param_names[0]])
    if proposal_std_iters is not None and len(proposal_std_iters) == n_checkpoints:
        std_x = np.array(proposal_std_iters)
    else:
        std_x = np.arange(1, n_checkpoints + 1) * adaptive_interval
    
    # Plot proposal standard deviation evolution
    fig_height = max(8, 2.5 * n_rows)
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(param_names):
        ax = plt.subplot(n_rows, n_cols, i+1)
        vals = np.array(proposal_std_evolution[param])
        plt.plot(std_x, vals, 'b-o', alpha=0.7, markersize=3)
        # Mark where burn-in ends (values freeze after that)
        if len(vals) > 1:
            # Find first index where value stops changing
            diffs = np.abs(np.diff(vals))
            frozen_idx = np.argmax(diffs < 1e-12 * (vals[:-1] + 1e-30))
            if frozen_idx > 0 and frozen_idx < len(std_x) - 1:
                plt.axvline(std_x[frozen_idx], color='r', linestyle='--',
                            alpha=0.6, label='burn-in end')
                ax.legend(fontsize=7)
        plt.xlabel('Iteration')
        plt.ylabel(f'Proposal std')
        plt.title(f'{param}')
        plt.grid(True, alpha=0.3)

    plt.suptitle('Proposal scale evolution (one point per adaptation event)', y=1.01)
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/proposal_scale_evolution.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    # Plot acceptance rate evolution
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(param_names):
        plt.subplot(n_rows, n_cols, i+1)
        acc_vals = acceptance_rate_evolution[param]
        if len(acc_vals) > 0:
            # x-axis: acceptance rates are stored at same checkpoints as std
            acc_x = std_x[:len(acc_vals)]
            plt.plot(acc_x, acc_vals, 'go-', alpha=0.7, markersize=4)
            plt.axhline(target_acceptance, color='r', linestyle='--', alpha=0.8, 
                       label=f'Target ({target_acceptance:.2f})')
        
        plt.xlabel('Iteration')
        plt.ylabel('Acceptance rate')
        plt.title(f'{param}')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        if i == 0:
            plt.legend()

    plt.suptitle('Per-parameter acceptance rate (one point per adaptation event)', y=1.01)
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/acceptance_rate_evolution.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
   


def plot_inference_results(samples, log_likelihood_trace, rms_evolution, burn_in=2000, 
                          u_los_obs=None, X_obs=None, Y_obs=None, 
                          incidence_angle=None, heading=None, figure_folder=None,
                          proposal_std_evolution=None, acceptance_rate_evolution=None,
                          proposal_std_evolution_iters=None,
                          adaptive_interval=None, target_acceptance=0.23,model_type='pCDM'):
    """
    Plot MCMC results including trace plots, posterior distributions,
    and comparison between initial and optimal models.
    """
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
    
    # Plot parameter traces and histograms
    # If samples contain prefixed (multi-model) keys or model_type is a list, use the keys directly
    if isinstance(model_type, list) or any('__' in k for k in samples.keys()):
        all_params = list(samples.keys())
    else:
        # Define parameters based on single-model model_type
        if model_type.lower() == 'pcdm':
            all_params = ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz', 'omegaX', 'omegaY', 'omegaZ']
        elif model_type.lower() == 'mogi':
            all_params = ['X0', 'Y0', 'depth', 'dV']
        elif model_type.lower() == 'okada':
            all_params = ['X0', 'Y0', 'depth', 'length', 'width', 'strike', 'dip', 'rake', 'slip','opening']
        elif model_type.lower() == 'une':
            all_params = ['X0', 'Y0', 'depth', 'yield_kt', 'dv_factor', 'chimney_amp', 'compact_amp']
        else:
            # Fallback: use all available parameters from samples
            all_params = list(samples.keys())
    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig_height = max(8, 2.5 * n_rows)
    plt.figure(figsize=(12, fig_height))
    
    for i, param in enumerate(all_params):
        plt.subplot(n_rows, n_cols, i+1)
        plt.hist(samples_burned[param], bins=50, density=True, alpha=0.7, color='skyblue')
        plt.xlabel(param)
        plt.ylabel('Density')
        plt.title(f'{param} Posterior')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(samples_burned[param])
        std_val = np.std(samples_burned[param])
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8)
        # Also add maximum likelihood line
        # Find the sample with highest likelihood after burn-in
        burn_in_offset = burn_in
        best_idx = np.argmax(log_likelihood_trace[burn_in_offset:])
        map_value = samples_burned[param][best_idx]
        plt.axvline(map_value, color='green', linestyle=':', alpha=0.8, linewidth=2)
        plt.text(0.02, 0.98, f'μ={mean_val:.4f}\nσ={std_val:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/MCMC_traces_posteriors.png", dpi=300)
    
    
    plt.tight_layout()
    plt.show()
    
 
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
                             incidence_angle, heading, log_likelihood_trace, burn_in, figure_folder=figure_folder,model_type=model_type)
        plot_rms_evolution(rms_evolution, figure_folder=figure_folder,model_type=model_type)
        plot_parameter_convergence(samples, burn_in, figure_folder=figure_folder,model_type=model_type)
        plot_model_components(samples, u_los_obs, X_obs, Y_obs,
                              incidence_angle, heading, log_likelihood_trace,
                              burn_in=burn_in, figure_folder=figure_folder, model_type=model_type)

    # 2-D parameter trade-off corner plot
    plot_corner(samples, burn_in=burn_in, figure_folder=figure_folder, model_type=model_type)

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
    plt.show()
   
    return rms_evolution

def plot_parameter_convergence(samples, burn_in=2000, figure_folder=None, model_type='pCDM'):
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
    if isinstance(model_type, list) or any('__' in k for k in samples.keys()):
        all_params = list(samples.keys())
    else:
        if model_type.lower() == 'pcdm':
            all_params = ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz', 'omegaX', 'omegaY', 'omegaZ']
        elif model_type.lower() == 'mogi':
            all_params = ['X0', 'Y0', 'depth', 'dV']
        elif model_type.lower() == 'okada':
            all_params = ['X0', 'Y0', 'depth', 'length', 'width', 'strike', 'dip', 'rake', 'slip','opening']
        elif model_type.lower() == 'une':
            all_params = ['X0', 'Y0', 'depth', 'yield_kt', 'dv_factor', 'chimney_amp', 'compact_amp']
        else:
            # Fallback: use all available parameters from samples
            all_params = list(samples.keys())
    
    # Calculate grid dimensions
    n_params = len(all_params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Create figure
    fig_height = max(8, 2.5 * n_rows)
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(all_params):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Plot trace
        iterations = np.arange(len(samples[param]))
        plt.plot(iterations, samples[param], alpha=0.7, color='blue', linewidth=0.5)
        
        # Mark burn-in period
        if burn_in < len(samples[param]):
            plt.axvline(burn_in, color='red', linestyle='--', alpha=0.8, 
                        label='Burn-in end' if i == 0 else "")
        
        # Calculate and plot running mean for convergence assessment
        window_size = min(500, len(samples[param]) // 10)
        if window_size > 1:
            running_mean = np.convolve(samples[param], np.ones(window_size)/window_size, mode='valid')
            x_running = np.arange(window_size//2, window_size//2 + len(running_mean))
            plt.plot(x_running, running_mean, 'orange', linewidth=2, alpha=0.8)
        
        plt.xlabel('Iteration')
        # If keys are prefixed (joint-model) show a prettier label like 'pcdm_1:X0'
        display_name = param if '__' not in param else param.replace('__', ':')
        plt.ylabel(display_name)
        plt.title(f'{display_name} Convergence')
        plt.grid(True, alpha=0.3)
        
        # Add final value text
        if len(samples[param]) > burn_in:
            final_mean = np.mean(samples[param][burn_in:])
            final_std = np.std(samples[param][burn_in:])
            plt.text(0.02, 0.98, f'Final: {final_mean:.4f}±{final_std:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend only to first subplot to avoid clutter
    if burn_in < len(samples[all_params[0]]):
        plt.subplot(n_rows, n_cols, 1)
        plt.legend()
    
  
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/parameter_convergence.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
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
                         incidence_angle, heading, log_likelihood_trace, burn_in=2000, figure_folder=None,model_type='pCDM'):
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
        inc_list = incidence_angle if isinstance(incidence_angle, (list, tuple, np.ndarray)) else [incidence_angle] * n_ifgs
        head_list = heading if isinstance(heading, (list, tuple, np.ndarray)) else [heading] * n_ifgs
        for j in range(n_ifgs):
            sub_folder = None
            if figure_folder is not None:
                sub_folder = os.path.join(figure_folder, f"ifg_{j+1}")
                os.makedirs(sub_folder, exist_ok=True)
            # Recursive call for single-IFG plotting (delegates to same function)
            plot_model_comparison(samples, u_los_obs[j], X_obs[j], Y_obs[j], inc_list[j], head_list[j], log_likelihood_trace, burn_in, figure_folder=sub_folder, model_type=model_type)
        return

    # Convert angles to radians for LOS calculation (single IFG)
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    
    # Line-of-sight unit vector components
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)
    
    # Calculate initial and optimal models. Support single-model OR multi-model (prefixed sample keys).
    def _sum_models_from_sample_dict(sample_dict):
        """Sum contributions from one or more models using keys present in sample_dict.
        If keys are prefixed (label__param), detect labels and sum per-label contributions.
        If keys are unprefixed assume a single model and use model_type string.
        """
        # Detect prefixed multi-model keys
        if any('__' in k for k in sample_dict.keys()):
            # Collect unique labels
            labels = sorted({k.split('__')[0] for k in sample_dict.keys()})
            ue_sum = np.zeros_like(X_obs, dtype=float)
            un_sum = np.zeros_like(X_obs, dtype=float)
            uv_sum = np.zeros_like(X_obs, dtype=float)
            for label in labels:
                # Derive model name from label (e.g. 'pcdm_1' -> 'pcdm')
                model_name = label.split('_')[0]
                # Collect params for this label
                params_here = {k.split('__')[1]: sample_dict[k] for k in sample_dict.keys() if k.startswith(label + '__')}
                # Choose forward model
                if 'pcdm' in model_name:
                    ue_i, un_i, uv_i = pCDM_fast.pCDM(X_obs, Y_obs,
                                                     params_here.get('X0'), params_here.get('Y0'), params_here.get('depth'),
                                                     params_here.get('omegaX', 0), params_here.get('omegaY', 0), params_here.get('omegaZ', 0),
                                                     params_here.get('DVx', 0), params_here.get('DVy', 0), params_here.get('DVz', 0), 0.25)
                elif 'okada' in model_name:
                    ue_i, un_i, uv_i = okada_fast.disloc3d3(X_obs, Y_obs,
                                                            xoff=params_here.get('X0'), yoff=params_here.get('Y0'),
                                                            depth=params_here.get('depth'), length=params_here.get('length'),
                                                            width=params_here.get('width'), slip=params_here.get('slip'), opening=params_here.get('opening'),
                                                            strike=params_here.get('strike'), dip=params_here.get('dip'), rake=params_here.get('rake'), nu=0.25)
                elif 'une' in model_name:
                    uv_i, ue_i, un_i = UNE_three.model(X_obs, Y_obs,
                                                       depth=params_here.get('depth'), yield_kt=params_here.get('yield_kt'),
                                                       dv_factor=params_here.get('dv_factor', 0.1),
                                                       chimney_amp=params_here.get('chimney_amp', 0.15),
                                                       chimney_height_fac=10, chimney_peck_k=0.35,
                                                       compact_amp=params_here.get('compact_amp', 0.05),
                                                       anelastic_fac=5,
                                                       x0=params_here.get('X0', 0), y0=params_here.get('Y0', 0),
                                                       nu=0.25, mu=30e9)
                else:
                    # Unsupported model name -> skip contribution
                    ue_i = np.zeros_like(X_obs)
                    un_i = np.zeros_like(X_obs)
                    uv_i = np.zeros_like(X_obs)
                ue_sum += np.asarray(ue_i, dtype=float)
                un_sum += np.asarray(un_i, dtype=float)
                uv_sum += np.asarray(uv_i, dtype=float)
            return ue_sum, un_sum, uv_sum
        else:
            # Single-model: infer model_type from provided model_type string
            m = model_type.lower() if isinstance(model_type, str) else list(model_type)[0].lower()
            if m == 'pcdm':
                ue, un, uv = pCDM_fast.pCDM(X_obs, Y_obs,
                                            sample_dict['X0'], sample_dict['Y0'], sample_dict['depth'],
                                            sample_dict.get('omegaX', 0), sample_dict.get('omegaY', 0), sample_dict.get('omegaZ', 0),
                                            sample_dict.get('DVx', 0), sample_dict.get('DVy', 0), sample_dict.get('DVz', 0), 0.25)
            elif m == 'okada':
                ue, un, uv = okada_fast.disloc3d3(X_obs, Y_obs, xoff=sample_dict['X0'], yoff=sample_dict['Y0'],
                                                 depth=sample_dict['depth'], length=sample_dict.get('length'),
                                                 width=sample_dict.get('width'), slip=sample_dict.get('slip'), opening=sample_dict.get('opening'),
                                                 strike=sample_dict.get('strike'), dip=sample_dict.get('dip'), rake=sample_dict.get('rake'), nu=0.25)
            elif m == 'une':
                uv, ue, un = UNE_three.model(X_obs, Y_obs,
                                             depth=sample_dict['depth'], yield_kt=sample_dict['yield_kt'],
                                             dv_factor=sample_dict.get('dv_factor', 0.1),
                                             chimney_amp=sample_dict.get('chimney_amp', 0.15),
                                             chimney_height_fac=10, chimney_peck_k=0.35,
                                             compact_amp=sample_dict.get('compact_amp', 0.05),
                                             anelastic_fac=5,
                                             x0=sample_dict.get('X0', 0), y0=sample_dict.get('Y0', 0),
                                             nu=0.25, mu=30e9)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            return ue, un, uv

    # Initial model
    ue_init, un_init, uv_init = _sum_models_from_sample_dict(initial_params)
    u_los_init = -(ue_init * los_e + un_init * los_n + uv_init * los_u)

    # Optimal model (MAP)
    ue_opt, un_opt, uv_opt = _sum_models_from_sample_dict(optimal_params)
    u_los_opt = -(ue_opt * los_e + un_opt * los_n + uv_opt * los_u)
    
    # Calculate residuals
    residual_init = u_los_obs - u_los_init
    residual_opt = u_los_obs - u_los_opt
    
    # Create regular grid for interpolation if data is scattered
    if len(np.unique(X_obs)) > 1 and len(np.unique(Y_obs)) > 1:
        # Create regular grid
        x_min, x_max = np.min(X_obs), np.max(X_obs)
        y_min, y_max = np.min(Y_obs), np.max(Y_obs)
        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate data to regular grid
        
        u_obs_grid = griddata((X_obs, Y_obs), u_los_obs, (Xi, Yi), method='cubic')
        u_init_grid = griddata((X_obs, Y_obs), u_los_init, (Xi, Yi), method='cubic')
        u_opt_grid = griddata((X_obs, Y_obs), u_los_opt, (Xi, Yi), method='cubic')
        res_init_grid = griddata((X_obs, Y_obs), residual_init, (Xi, Yi), method='cubic')
        res_opt_grid = griddata((X_obs, Y_obs), residual_opt, (Xi, Yi), method='cubic')
        
        X_plot, Y_plot = Xi, Yi
        u_obs_plot = u_obs_grid
        u_init_plot = u_init_grid
        u_opt_plot = u_opt_grid
        res_init_plot = res_init_grid
        res_opt_plot = res_opt_grid
        X_plot_scatter, Y_plot_scatter = X_obs, Y_obs
        u_obs_plot_scatter = u_los_obs
        u_init_plot_scatter = u_los_init
        u_opt_plot_scatter = u_los_opt
        res_init_plot_scatter = residual_init
        res_opt_plot_scatter = residual_opt
    else:
        # Assume data is already on regular grid
        try:
            grid_shape = (int(np.sqrt(len(X_obs))), int(np.sqrt(len(X_obs))))
            X_plot = X_obs.reshape(grid_shape)
            Y_plot = Y_obs.reshape(grid_shape)
            u_obs_plot = u_los_obs.reshape(grid_shape)
            u_init_plot = u_los_init.reshape(grid_shape)
            u_opt_plot = u_los_opt.reshape(grid_shape)
            res_init_plot = residual_init.reshape(grid_shape)
            res_opt_plot = residual_opt.reshape(grid_shape)

            X_plot_scatter, Y_plot_scatter = X_obs, Y_obs
            u_obs_plot_scatter = u_los_obs
            u_init_plot_scatter = u_los_init
            u_opt_plot_scatter = u_los_opt
            res_init_plot_scatter = residual_init
            res_opt_plot_scatter = residual_opt
            print('MADE IT THIS FAR')
            
        except:
            print("Could not reshape data for plotting. Using scatter plots instead.")
            X_plot, Y_plot = X_obs, Y_obs
            u_obs_plot = u_los_obs
            u_init_plot = u_los_init
            u_opt_plot = u_los_opt
            res_init_plot = residual_init
            res_opt_plot = residual_opt
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # for ax in axes:
    #     ax.set_aspect('equal')
    
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
        # Contour plots with consistent color scale
        im1 = axes[0,0].contourf(X_plot, Y_plot, u_obs_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0,0].set_title('Observed Data')
        
        im2 = axes[0,1].contourf(X_plot, Y_plot, u_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0,1].set_title('Optimal Model')
        
        # Use the same color scale for residuals as the data for direct comparison
        im3 = axes[0,2].contourf(X_plot, Y_plot, res_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0,2].set_title('Optimal Residual')
        
        # Add a single colorbar across the bottom representing all three plots
        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        plt.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Displacement (m)')
    
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
   
    

    # RMS comparison in the bottom right subplot
    rms_init = np.sqrt(np.nanmean(residual_init**2))
    rms_opt = np.sqrt(np.nanmean(residual_opt**2))
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/Model_Comparison.png", dpi=300)


def plot_corner(samples, burn_in=0, figure_folder=None, model_type='pCDM',
                nbins=40, smooth=True, figsize_per_param=2.0):
    """
    Corner / pair plot: shows the marginal 1-D posterior on the diagonal and
    the 2-D joint posterior (parameter trade-off) in every off-diagonal cell.

    Lower triangle  – filled 2-D histogram (log-density) showing trade-offs.
    Diagonal        – 1-D histogram (marginal posterior) with mean ± 1σ lines.
    Upper triangle  – hidden (blank) for a clean layout.

    Parameters
    ----------
    samples : dict
        Raw MCMC sample dict  {param_name: list/array}.
    burn_in : int
        How many leading samples to discard as burn-in (applied strictly).
    figure_folder : str or None
        Directory in which to save corner_plot.png.  Skipped when None.
    model_type : str or list
        Used to determine parameter ordering.
    nbins : int
        Number of bins along each axis for the 2-D histograms.
    smooth : bool
        If True, apply a light Gaussian smoothing to 2-D histograms.
    figsize_per_param : float
        Inches allocated per parameter axis.
    """
    from scipy.ndimage import gaussian_filter

    # ------------------------------------------------------------------ #
    # 1. Parameter list                                                   #
    # ------------------------------------------------------------------ #
    if isinstance(model_type, list) or any('__' in k for k in samples.keys()):
        all_params = list(samples.keys())
    else:
        _m = model_type.lower() if isinstance(model_type, str) else ''
        if _m == 'pcdm':
            all_params = ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz', 'omegaX', 'omegaY', 'omegaZ']
        elif _m == 'mogi':
            all_params = ['X0', 'Y0', 'depth', 'dV']
        elif _m == 'okada':
            all_params = ['X0', 'Y0', 'depth', 'length', 'width', 'strike', 'dip', 'rake', 'slip', 'opening']
        elif _m == 'une':
            all_params = ['X0', 'Y0', 'depth', 'yield_kt', 'dv_factor', 'chimney_amp', 'compact_amp']
        else:
            all_params = list(samples.keys())
    all_params = [p for p in all_params if p in samples]

    n = len(all_params)
    if n < 2:
        print('plot_corner: need at least 2 parameters — skipping.')
        return

    # ------------------------------------------------------------------ #
    # 2. Slice strictly to post-burn-in samples                          #
    # ------------------------------------------------------------------ #
    n_total    = len(samples[all_params[0]])
    burn_safe  = max(0, min(int(burn_in), n_total - 2))
    n_post     = n_total - burn_safe
    print(f'  plot_corner: using {n_post:,} post-burn-in samples '
          f'(discarded first {burn_safe:,}).')

    chains = np.column_stack(
        [np.asarray(samples[p][burn_safe:], dtype=float) for p in all_params]
    )   # shape (n_post, n_params)

    # Prettier labels for multi-model prefixed keys  e.g. 'pcdm_1__X0' -> 'pcdm_1\nX0'
    labels = [p.replace('__', '\n') for p in all_params]

    # ------------------------------------------------------------------ #
    # 3. Axis ranges: 1–99th percentile of POST-BURN-IN chains + margin  #
    # ------------------------------------------------------------------ #
    lo = np.percentile(chains, 1,  axis=0)
    hi = np.percentile(chains, 99, axis=0)
    span   = hi - lo
    # Protect against degenerate (constant) parameters
    span   = np.where(span > 0, span, np.abs(hi) * 0.1 + 1e-9)
    margin = 0.05 * span
    lo     = lo - margin
    hi     = hi + margin

    # ------------------------------------------------------------------ #
    # 4. Build figure                                                     #
    # ------------------------------------------------------------------ #
    fig_size = max(6, figsize_per_param * n)
    fig, axes = plt.subplots(n, n, figsize=(fig_size, fig_size))
    # axes is always 2-D even for n==2
    if n == 1:
        axes = np.array([[axes]])

    CMAP_2D  = 'viridis'
    COL_HIST = '#4C72B0'
    COL_MEAN = '#E74C3C'
    COL_STD  = '#E74C3C'

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]

            x_data = chains[:, col]   # horizontal = column param
            y_data = chains[:, row]   # vertical   = row    param

            if row == col:
                # ---- Diagonal: marginal 1-D histogram ---------------
                ax.hist(x_data, bins=nbins, color=COL_HIST, alpha=0.75,
                        density=True, range=(lo[col], hi[col]))
                mean_v = np.mean(x_data)
                std_v  = np.std(x_data)
                ax.axvline(mean_v,         color=COL_MEAN, lw=1.5, ls='-')
                ax.axvline(mean_v - std_v, color=COL_STD,  lw=1.0, ls='--')
                ax.axvline(mean_v + std_v, color=COL_STD,  lw=1.0, ls='--')
                ax.set_xlim(lo[col], hi[col])
                ax.yaxis.set_visible(False)

            elif row > col:
                # ---- Lower triangle: 2-D joint density --------------
                H, xedges, yedges = np.histogram2d(
                    x_data, y_data, bins=nbins,
                    range=[[lo[col], hi[col]], [lo[row], hi[row]]])
                H = H.T   # shape (ny, nx)

                if smooth and H.max() > 0:
                    H = gaussian_filter(H.astype(float), sigma=1.0)

                H_log = np.where(H > 0, np.log1p(H), np.nan)
                ax.imshow(H_log, origin='lower',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          aspect='auto', cmap=CMAP_2D, interpolation='nearest')

                # ~1σ and ~2σ contour overlays
                try:
                    valid = H_log[np.isfinite(H_log)]
                    if len(valid) > 1:
                        lvls = sorted(set(np.nanpercentile(valid, p) for p in (39, 86)))
                        ax.contour(H_log, levels=lvls,
                                   colors='white', linewidths=0.8, alpha=0.6,
                                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                   origin='lower')
                except Exception:
                    pass

                ax.set_xlim(lo[col], hi[col])
                ax.set_ylim(lo[row], hi[row])

            else:
                # ---- Upper triangle: blank --------------------------
                ax.set_visible(False)
                continue   # skip all tick / label logic below

            # ---- Tick label visibility (edge cells only) ------------
            # X ticks: only on the bottom row
            if row < n - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(labels[col], fontsize=8)

            # Y ticks: only on the left column, and only for off-diagonal
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=8)
            else:
                ax.tick_params(axis='y', labelleft=False)

            ax.tick_params(labelsize=6)

    fig.suptitle(
        f'Posterior Corner Plot  (post burn-in,  n = {chains.shape[0]:,})',
        fontsize=11)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95)

    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/corner_plot.png", dpi=200, bbox_inches='tight')
        print(f'  Saved: {figure_folder}/corner_plot.png')

    plt.show()


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

        if 'pcdm' in model_name:
            ue, un, uv = pCDM_fast.pCDM(
                X_use, Y_use,
                p['X0'], p['Y0'], p['depth'],
                p.get('omegaX', 0), p.get('omegaY', 0), p.get('omegaZ', 0),
                p.get('DVx', 0), p.get('DVy', 0), p.get('DVz', 0), 0.25)
            contributions[label] = _to_los(ue, un, uv)
            annot_info[label] = {'type': 'pcdm',
                                 'X0': p['X0'], 'Y0': p['Y0'], 'depth': p['depth']}

        elif 'okada' in model_name:
            ue, un, uv = okada_fast.disloc3d3(
                X_use, Y_use,
                xoff=p['X0'], yoff=p['Y0'], depth=p['depth'],
                length=p.get('length', 10000), width=p.get('width', 8000),
                slip=p.get('slip', 0), opening=p.get('opening', 0),
                strike=p.get('strike', 0), dip=p.get('dip', 45),
                rake=p.get('rake', 90), nu=0.25)
            contributions[label] = _to_los(ue, un, uv)
            annot_info[label] = {'type': 'okada',
                                 'X0': p['X0'], 'Y0': p['Y0'], 'depth': p['depth'],
                                 'length': p.get('length', 10000),
                                 'width':  p.get('width', 8000),
                                 'strike': p.get('strike', 0),
                                 'dip':    p.get('dip', 45)}

        elif 'une' in model_name:
            uv, ue, un = UNE_three.model(
                X_use, Y_use,
                depth=p['depth'], yield_kt=p['yield_kt'],
                dv_factor=p.get('dv_factor', 0.1),
                chimney_amp=p.get('chimney_amp', 0.15),
                chimney_height_fac=10, chimney_peck_k=0.35,
                compact_amp=p.get('compact_amp', 0.05),
                anelastic_fac=5,
                x0=p.get('X0', 0), y0=p.get('Y0', 0),
                nu=0.25, mu=30e9)
            contributions[label] = _to_los(ue, un, uv)
            annot_info[label] = {'type': 'une',
                                 'X0': p.get('X0', 0), 'Y0': p.get('Y0', 0),
                                 'depth': p['depth']}
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

    plt.show()

