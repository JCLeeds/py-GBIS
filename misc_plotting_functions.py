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
                               adaptive_interval, target_acceptance=0.23, figure_folder=None,model_type='pCDM'):
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
    """
    param_names = list(proposal_std_evolution.keys())
    n_params = len(param_names)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Plot proposal standard deviation evolution
    fig_height = max(8, 2.5 * n_rows)
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(param_names):
        plt.subplot(n_rows, n_cols, i+1)
        iterations = np.arange(len(proposal_std_evolution[param]))
        plt.plot(iterations, proposal_std_evolution[param], color='b', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel(f'{param} Proposal Std')
        plt.title(f'{param} Proposal Scale Evolution')
        plt.grid(True, alpha=0.3)
        
        # Mark adaptation points
        adaptation_points = np.arange(adaptive_interval, len(iterations), adaptive_interval)
        for ap in adaptation_points:
            if ap < len(iterations):
                plt.axvline(ap, color='r', linestyle='--', alpha=0.1)
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/proposal_scale_evolution.png", dpi=300, bbox_inches='tight')
    
    # Plot acceptance rate evolution
    plt.figure(figsize=(15, fig_height))
    
    for i, param in enumerate(param_names):
        plt.subplot(n_rows, n_cols, i+1)
        if len(acceptance_rate_evolution[param]) > 0:
            adaptation_iterations = np.arange(adaptive_interval, 
                                            adaptive_interval * (len(acceptance_rate_evolution[param]) + 1), 
                                            adaptive_interval)
            plt.plot(adaptation_iterations[:len(acceptance_rate_evolution[param])], 
                    acceptance_rate_evolution[param], 'go-', alpha=0.7, markersize=4)
            plt.axhline(target_acceptance, color='r', linestyle='--', alpha=0.8, 
                       label=f'Target ({target_acceptance:.2f})')
        
        plt.xlabel('Iteration')
        plt.ylabel(f'{param} Acceptance Rate')
        plt.title(f'{param} Acceptance Rate Evolution')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        if i == 0:  # Add legend only to first subplot
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/acceptance_rate_evolution.png", dpi=300, bbox_inches='tight')




def plot_inference_results(samples, log_likelihood_trace, rms_evolution, burn_in=2000, 
                          u_los_obs=None, X_obs=None, Y_obs=None, 
                          incidence_angle=None, heading=None, figure_folder=None,
                          proposal_std_evolution=None, acceptance_rate_evolution=None,
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
    # Define parameters based on model type
    if model_type.lower() == 'pcdm':
        all_params = ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz', 'omegaX', 'omegaY', 'omegaZ']
    elif model_type.lower() == 'mogi':
        all_params = ['X0', 'Y0', 'depth', 'dV']
    elif model_type.lower() == 'okada':
        all_params = ['X0', 'Y0', 'depth', 'length', 'width', 'strike', 'dip', 'rake', 'slip','opening']
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
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/MCMC_traces_posteriors.png", dpi=300)
    
    # Plot adaptation diagnostics if available
    if (proposal_std_evolution is not None and acceptance_rate_evolution is not None 
        and adaptive_interval is not None):
        plot_adaptation_diagnostics(proposal_std_evolution, acceptance_rate_evolution,
                                   adaptive_interval, target_acceptance, figure_folder,model_type=model_type)
    
    # Plot other diagnostics if observation data is provided
    if all(x is not None for x in [u_los_obs, X_obs, Y_obs, incidence_angle, heading]):
        plot_model_comparison(samples, u_los_obs, X_obs, Y_obs, 
                             incidence_angle, heading, log_likelihood_trace, burn_in, figure_folder=figure_folder,model_type=model_type)
        plot_rms_evolution(rms_evolution, figure_folder=figure_folder,model_type=model_type)
        plot_parameter_convergence(samples, burn_in, figure_folder=figure_folder,model_type=model_type)
    
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
    
    plt.tight_layout()
    plt.show()
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/RMS_evolution.png", dpi=300)
    
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

    # Define parameters based on model type
    if model_type.lower() == 'pcdm':
        all_params = ['X0', 'Y0', 'depth', 'DVx', 'DVy', 'DVz', 'omegaX', 'omegaY', 'omegaZ']
    elif model_type.lower() == 'mogi':
        all_params = ['X0', 'Y0', 'depth', 'dV']
    elif model_type.lower() == 'okada':
        all_params = ['X0', 'Y0', 'depth', 'length', 'width', 'strike', 'dip', 'rake', 'slip','opening']
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
        plt.ylabel(param)
        plt.title(f'{param} Convergence')
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
    
    plt.tight_layout()
    plt.show()
    
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/parameter_convergence.png", dpi=300, bbox_inches='tight')
    
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
    # Convert angles to radians for LOS calculation
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)
    
    # Line-of-sight unit vector components
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    los_u = -np.cos(inc_rad)
    
    # Calculate initial model based on model_type
    if model_type.lower() == 'pcdm':
        ue_init, un_init, uv_init = pCDM_fast.pCDM(X_obs, Y_obs, initial_params['X0'], initial_params['Y0'], 
                                         initial_params['depth'], initial_params['omegaX'], 
                                         initial_params['omegaY'], initial_params['omegaZ'],
                                         initial_params['DVx'], initial_params['DVy'], 
                                         initial_params['DVz'], 0.25)
    elif model_type.lower() == 'mogi':
        # # Example for Mogi model - adjust parameters as needed
        # ue_init, un_init, uv_init = calculate_mogi_displacement(X_obs, Y_obs, 
        #                                                       initial_params['X0'], initial_params['Y0'], 
        #                                                       initial_params['depth'], initial_params['dV'])
        pass

    elif model_type.lower() == 'okada':
        # Example for Okada model - adjust parameters as needed
        ue_init, un_init, uv_init =  okada_fast.disloc3d3(X_obs, Y_obs, xoff=initial_params['X0'], yoff=initial_params['Y0'], 
                                                depth=initial_params['depth'], length=initial_params['length'], 
                                                width=initial_params['width'], slip=initial_params['slip'], opening=initial_params['opening'],
                                                strike=initial_params['strike'],
                                                dip=initial_params['dip'], rake=initial_params['rake'], nu=0.25)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    u_los_init = -(ue_init * los_e + un_init * los_n + uv_init * los_u)
    
    # Calculate optimal model based on model_type
    if model_type.lower() == 'pcdm':
        ue_opt, un_opt, uv_opt = pCDM_fast.pCDM(X_obs, Y_obs, optimal_params['X0'], optimal_params['Y0'], 
                                      optimal_params['depth'], optimal_params['omegaX'], 
                                      optimal_params['omegaY'], optimal_params['omegaZ'],
                                      optimal_params['DVx'], optimal_params['DVy'], 
                                      optimal_params['DVz'], 0.25)
    elif model_type.lower() == 'mogi':
        # ue_opt, un_opt, uv_opt = calculate_mogi_displacement(X_obs, Y_obs, 
        #                                                    optimal_params['X0'], optimal_params['Y0'], 
        #                                                    optimal_params['depth'], optimal_params['dV'])
        pass
    elif model_type.lower() == 'okada':
        ue_opt, un_opt, uv_opt = okada_fast.disloc3d3(X_obs, Y_obs, xoff=optimal_params['X0'], yoff=optimal_params['Y0'], 
                                      depth=optimal_params['depth'], length=optimal_params['length'], 
                                      width=optimal_params['width'], slip=optimal_params['slip'], opening=optimal_params['opening'],
                                      strike=optimal_params['strike'],
                                      dip=optimal_params['dip'], rake=optimal_params['rake'], nu=0.25)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
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
        except:
            print("Could not reshape data for plotting. Using scatter plots instead.")
            X_plot, Y_plot = X_obs, Y_obs
            u_obs_plot = u_los_obs
            u_init_plot = u_los_init
            u_opt_plot = u_los_opt
            res_init_plot = residual_init
            res_opt_plot = residual_opt
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
        im1 = axes[0].contourf(X_plot, Y_plot, u_obs_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title('Observed Data')
        
        im2 = axes[1].contourf(X_plot, Y_plot, u_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title('Optimal Model')
        
        # Use the same color scale for residuals as the data for direct comparison
        im3 = axes[2].contourf(X_plot, Y_plot, res_opt_plot, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2].set_title('Optimal Residual')
        
        # Add a single colorbar across the bottom representing all three plots
        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        plt.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Displacement (m)')
    else:
        # Scatter plots
        im1 = axes[0].scatter(X_plot, Y_plot, c=u_obs_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title('Observed Data')
        
        im2 = axes[1].scatter(X_plot, Y_plot, c=u_opt_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title('Optimal Model')
        
        # Bottom row: Residual
        im3 = axes[2].scatter(X_plot, Y_plot, c=res_opt_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2].set_title('Optimal Residual')
    
   
    

    # RMS comparison in the bottom right subplot
    rms_init = np.sqrt(np.nanmean(residual_init**2))
    rms_opt = np.sqrt(np.nanmean(residual_opt**2))
    if figure_folder is not None:
        plt.savefig(f"{figure_folder}/Model_Comparison.png", dpi=300)

  
