
"""
Interactive variogram calculator for InSAR displacement data with polygon selection.
This function loads InSAR phase data, converts coordinates to local projection,
and provides an interactive interface for selecting regions of interest using
polygon selection. It calculates and fits exponential variogram models to the
selected data, allowing users to estimate geostatistical parameters (nugget,
sill, and range) for different spatial regions.
Parameters:
-----------
datapath : str
    Path to the .npy file containing InSAR data dictionary with keys:
    'Phase', 'Lon', 'Lat', 'Inc', 'Heading'
referencePoint : list or array-like
    Reference point [latitude, longitude] for coordinate conversion to local
    projection system
Returns:
--------
None
    Function displays interactive matplotlib plots and prints fitted parameters
Features:
---------
- Interactive polygon selection for spatial subsetting
- Memory-efficient variogram calculation with data sampling
- Exponential variogram model fitting
- Real-time parameter estimation and visualization
- Right-click to reset selection
Usage:
------
1. Left-click to draw polygon vertices on the scatter plot
2. Complete polygon to select/mask data points
3. Right-click anywhere to reset selection
4. Variogram parameters are automatically calculated and displayed
Notes:
------
- Large datasets are automatically sampled for visualization performance
- Phase data is converted to displacement using standard InSAR scaling
- Coordinates are projected to local Cartesian system for distance calculations
Author: John Condon
Date: Nov 2025
"""

import numpy as np
import llh2local as llh
import local2llh as lllh
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
from sklearn.neighbors import NearestNeighbors


def convert_lat_long_2_xy(lat, lon, lat0, lon0):
    ll = [lon.flatten(), lat.flatten()]
    ll = np.array(ll, dtype=float)
    xy = llh.llh2local(ll, np.array([lon0, lat0], dtype=float))
    x = xy[0,:].reshape(lat.shape)
    y = xy[1,:].reshape(lat.shape)
    return xy


def calculate_variogram_memory_efficient(x, y, values, max_lag=None, n_lags=50, max_points=5000, chunk_size=1000):
    """Memory efficient variogram calculation using sampling and chunking"""
    n_points = len(x)
    
    # Sample data if too large
    if n_points > max_points:
        print(f"Sampling {max_points} points from {n_points} for variogram calculation...")
        indices = np.random.choice(n_points, max_points, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
        values_sample = values[indices]
    else:
        x_sample = x
        y_sample = y
        values_sample = values
    
    n_sample = len(x_sample)
    coords = np.column_stack([x_sample, y_sample])
    
    if max_lag is None:
        # Estimate max_lag without computing all distances
        sample_indices = np.random.choice(n_sample, min(1000, n_sample), replace=False)
        sample_coords = coords[sample_indices]
        sample_distances = distance.pdist(sample_coords)
        max_lag = np.percentile(sample_distances, 80)  # Use 80th percentile instead of max/2
    
    lag_bins = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
    variogram = np.zeros(n_lags)
    counts = np.zeros(n_lags)
    
    # Use NearestNeighbors for efficient distance queries
    nbrs = NearestNeighbors(radius=max_lag).fit(coords)
    
    # Process in chunks to manage memory
    for start_idx in range(0, n_sample, chunk_size):
        end_idx = min(start_idx + chunk_size, n_sample)
        chunk_coords = coords[start_idx:end_idx]
        chunk_values = values_sample[start_idx:end_idx]
        
        # Find neighbors within max_lag for this chunk
        for i, (coord, value) in enumerate(zip(chunk_coords, chunk_values)):
            distances, indices = nbrs.radius_neighbors([coord], radius=max_lag)
            distances = distances[0]
            neighbor_indices = indices[0]
            
            # Skip self-distance (distance = 0)
            mask = distances > 0
            distances = distances[mask]
            neighbor_indices = neighbor_indices[mask]
            
            if len(distances) == 0:
                continue
                
            neighbor_values = values_sample[neighbor_indices]
            value_diffs = (value - neighbor_values) ** 2
            # Filter out NaN values
            nan_mask = ~(np.isnan(value_diffs) | np.isnan(value) | np.isnan(neighbor_values))
            value_diffs = value_diffs[nan_mask]
            distances = distances[nan_mask]
            
            # Bin the distances
            bin_indices = np.digitize(distances, lag_bins) - 1
            valid_bins = (bin_indices >= 0) & (bin_indices < n_lags)
            # print(valid_bins)
            for j in range(n_lags):
                mask_bin = (bin_indices == j) & valid_bins
                # print(mask.sum())
                if np.sum(mask_bin) > 0:
                    variogram[j] += np.sum(value_diffs[mask_bin])
                    counts[j] += np.sum(mask_bin)
        
        print(f"Processed chunk {start_idx//chunk_size + 1}/{(n_sample-1)//chunk_size + 1}")
    
    # Normalize by counts
    mask_valid = counts > 0
    variogram[mask_valid] = 0.5 * variogram[mask_valid] / counts[mask_valid]
    print(lag_centers)
    print(variogram)
    return lag_centers, variogram


def covairance_calculator(datapath, referencePoint):

    def on_mouse_click(event):
        """Handle mouse click events"""
        if event.button == 3:  # Right mouse button
            reset_mask()


    def reset_mask():
        """Reset mask to include all points"""
        nonlocal mask_vis
        mask_vis = np.zeros(len(X_vis), dtype=bool)  # Changed from ones to zeros
        update_plots()
        print("Mask reset - all data points selected")

    def onselect(verts):
        """Callback for polygon selection"""
        nonlocal mask_vis
        path = Path(verts)
        mask_vis = path.contains_points(np.column_stack([X_vis, Y_vis]))  # Inverted logic
        update_plots()

    
    def exponential_variogram(h, nugget, sill, range_param):
        """Exponential variogram model"""
        return nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_param))

    def fit_variogram(lag_centers, variogram):
        """Fit exponential variogram model"""
        # Remove zeros from variogram
        valid_idx = variogram > 0
        if np.sum(valid_idx) < 3:
            print('fit_variogram: not enough valid points after removing zeros')
            return None, None
        
        lag_valid = lag_centers[valid_idx]
        var_valid = variogram[valid_idx]
        
        try:
            # Initial parameter estimates
            nugget_init = var_valid[0] if len(var_valid) > 0 else 0
            sill_init = np.max(var_valid)
            range_init = lag_valid[np.argmax(var_valid > 0.95 * sill_init)] if np.any(var_valid > 0.95 * sill_init) else np.max(lag_valid) / 3
            
            popt, _ = curve_fit(exponential_variogram, lag_valid, var_valid, 
                            p0=[nugget_init, sill_init, range_init],
                            bounds=([0, nugget_init, 0], [sill_init, np.inf, np.inf]),
                            maxfev=5000)
            print(popt)
            return popt, lag_valid
        except:
            return None, None
        


        

    def update_plots():
        """Update plots with current mask"""
        nonlocal polygon_selector
        
        # Clear previous plots
        fig.clear()
        
        # Create subplots
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        
        # Plot masked data - now mask_vis=True means excluded, mask_vis=False means selected
        scatter = ax1.scatter(X_vis[~mask_vis], Y_vis[~mask_vis], c=u_los_vis[~mask_vis], s=1, cmap='RdBu_r', vmin=np.min(u_los_vis), vmax=np.max(u_los_vis))
        ax1.scatter(X_vis[mask_vis], Y_vis[mask_vis], c='gray', s=0.5, alpha=0.3)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Data (Selected: {np.sum(~mask_vis)} points)\nRight-click to reset')  # Fixed count
        plt.colorbar(scatter, ax=ax1, label='Phase')
        
        # Calculate and plot variogram for masked data
        n_selected = np.sum(~mask_vis)
        if n_selected > 10:
            print(f"Calculating variogram for {n_selected} selected points (this may take a moment for large datasets)...")
            
            # Map visualization mask back to full dataset
            full_mask = np.zeros(len(X_obs), dtype=bool)
            selected_vis_indices = vis_to_full_map[~mask_vis]
            full_mask[selected_vis_indices] = True
            
            n_full_selected = np.sum(full_mask)
            print(f"Full dataset selected points: {n_full_selected}")
            
            if n_full_selected > 1:
                try:
                    lag_centers, variogram = calculate_variogram_memory_efficient(
                        X_obs[full_mask], Y_obs[full_mask], u_los_obs[full_mask]
                    )
                    
                    if len(lag_centers) > 0 and len(variogram) > 0:
                        ax2.scatter(lag_centers, variogram, alpha=0.7, label='Empirical')
                        
                        # Fit variogram
                        params, lag_valid = fit_variogram(lag_centers, variogram)
                        
                        if params is not None:
                            fitted_nugget, fitted_sill, fitted_range = params
                            
                            # Plot fitted curve
                            h_fit = np.linspace(0, np.max(lag_centers), 100)
                            var_fit = exponential_variogram(h_fit, fitted_nugget, fitted_sill, fitted_range)
                            ax2.plot(h_fit, var_fit, 'r-', label=f'Fitted\nNugget: {fitted_nugget:.2e}\nSill: {fitted_sill:.2e}\nRange: {fitted_range:.0f}')
                            
                            print(f"Fitted parameters for {n_full_selected} points:")
                            print(f"  Nugget: {fitted_nugget:.6e}")
                            print(f"  Sill: {fitted_sill:.6e}")
                            print(f"  Range: {fitted_range:.1f}")
                        else:
                            print("Could not fit variogram model to the data.")
                    else:
                        print("No valid variogram calculated.")
                        ax2.text(0.5, 0.5, 'No valid variogram\ncalculated', 
                                transform=ax2.transAxes, ha='center', va='center',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception as e:
                    print(f"Error calculating variogram: {e}")
                    ax2.text(0.5, 0.5, f'Error calculating\nvariogram:\n{str(e)[:50]}...', 
                            transform=ax2.transAxes, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            else:
                print("Not enough points selected for variogram calculation.")
                ax2.text(0.5, 0.5, 'Not enough points\nselected', 
                        transform=ax2.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            print("Not enough points selected for variogram calculation.")
            ax2.text(0.5, 0.5, 'Select more points\nfor variogram', 
                    transform=ax2.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Semivariance')
        ax2.set_title('Variogram')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Recreate polygon selector
        polygon_selector = PolygonSelector(ax1, onselect, useblit=True)
        
        # Connect mouse click event
        fig.canvas.mpl_connect('button_press_event', on_mouse_click)
        
        plt.tight_layout()
        plt.draw()


    # Load data from .npy file
    print("Loading data...")
    data = np.load(datapath, allow_pickle=True)
    
    # Extract data from the loaded object
    data_dict = data.item()
    # print(data_dict.keys())
    u_los_obs = np.array(data_dict['Phase']).flatten()
    u_los_obs = -u_los_obs*(0.0555/(4*np.pi))
    
    Lon = np.array(data_dict['Lon'])
    Lat = np.array(data_dict['Lat'])
    
    print(f"Original data size: {len(u_los_obs)} points")
    
    # Convert coordinates
    print("Converting coordinates...")
    X_obs, Y_obs = convert_lat_long_2_xy(Lat, Lon, referencePoint[0], referencePoint[1])
    print(f"Converted Lon/Lat to X/Y with reference point {referencePoint}")
    print(f"X_obs range: {X_obs.min():.3f} to {X_obs.max():.3f}")
    print(f"Y_obs range: {Y_obs.min():.3f} to {Y_obs.max():.3f}")
    
    incidence_angle = np.array(data_dict['Inc'])
    heading = np.array(data_dict['Heading'])
    
    print(f"Loaded data shapes:")
    print(f"  u_los_obs: {u_los_obs.shape}")
    print(f"  X_obs: {X_obs.shape}")
    print(f"  Y_obs: {Y_obs.shape}")
    print(f"  incidence_angle: {incidence_angle.shape}")
    print(f"  heading: {heading.shape}")
    
    # Sample data for visualization if too large
    vis_sample_size = 100000
    if len(X_obs) > vis_sample_size:
        print(f"Sampling {vis_sample_size} points for visualization...")
        vis_indices = np.random.choice(len(X_obs), vis_sample_size, replace=False)
        X_vis = X_obs[vis_indices]
        Y_vis = Y_obs[vis_indices]
        u_los_vis = u_los_obs[vis_indices]
        vis_to_full_map = vis_indices
    else:
        X_vis = X_obs
        Y_vis = Y_obs
        u_los_vis = u_los_obs
        vis_to_full_map = np.arange(len(X_obs))
    
    mask_vis = np.zeros(len(X_vis), dtype=bool)  # Changed from ones to zeros - False means selected
    polygon_selector = None
    fig = None

    # ... rest of the functions remain the same ...

    # Create initial plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Initial scatter plot - show all data initially
    scatter = ax1.scatter(X_vis, Y_vis, c=u_los_vis, s=1, cmap='RdBu_r')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Select polygon to mask data\nRight-click to reset')
    plt.colorbar(scatter, ax=ax1, label='Displacement LOS (m) (positive towards satellite)')

    # Initialize polygon selector
    polygon_selector = PolygonSelector(ax1, onselect, useblit=True)

    # Connect mouse click event
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Calculate initial variogram
    print("Calculating initial variogram...")
    lag_centers, variogram = calculate_variogram_memory_efficient(X_obs, Y_obs, u_los_obs)
    ax2.scatter(lag_centers, variogram, alpha=0.7, label='Empirical (All data)')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Semivariance')
    ax2.set_title('Variogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Instructions:")
    print("1. Click points on the left plot to create a polygon")
    print("2. Complete the polygon to mask the data")
    print("3. Right-click anywhere to reset and draw a new polygon")
    print("4. Variogram will be calculated and fitted for selected data")




    # Create initial plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Initial scatter plot
    scatter = ax1.scatter(X_vis, Y_vis, c=u_los_vis, s=1, cmap='RdBu_r')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Select polygon to mask data\nRight-click to reset')
    plt.colorbar(scatter, ax=ax1, label='Phase')

    # Initialize polygon selector
    polygon_selector = PolygonSelector(ax1, onselect, useblit=True)

    # Connect mouse click event
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Calculate initial variogram
    print("Calculating initial variogram...")
    lag_centers, variogram = calculate_variogram_memory_efficient(X_obs, Y_obs, u_los_obs)
    ax2.scatter(lag_centers, variogram, alpha=0.7, label='Empirical (All data)')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Semivariance')
    ax2.set_title('Variogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Instructions:")
    print("1. Click points on the left plot to create a polygon")
    print("2. Complete the polygon to mask the data")
    print("3. Right-click anywhere to reset and draw a new polygon")
    print("4. Variogram will be calculated and fitted for selected data")


if __name__ == "__main__":
    datapath = '20230108_20230201.geo.unw_processed_clipped_full_resolution.npy'
    data = np.load(datapath, allow_pickle=True)
    data_dict = data.item()
    print(data_dict.keys())
    referencePoint = [float(data_dict['center_lat']),float(data_dict['center_lon'])]
    print(referencePoint)
    # sill = 6.1951e-05
    # range_param = 15737.072
    # nugget = 3.311e-06
    # referencePoint = [-67.83951712, -21.77505660]
    ###################################################################
    covairance_calculator(datapath,referencePoint)