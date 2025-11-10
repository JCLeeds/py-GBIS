import numpy as np
from numba import jit, prange
import math

@jit(nopython=True, cache=True, fastmath=True)
def _calc_rotation_matrix(omegaX_rad, omegaY_rad, omegaZ_rad):
    """Fast rotation matrix calculation using Numba"""
    # Pre-calculate trig functions
    cos_x, sin_x = math.cos(omegaX_rad), math.sin(omegaX_rad)
    cos_y, sin_y = math.cos(omegaY_rad), math.sin(omegaY_rad)
    cos_z, sin_z = math.cos(omegaZ_rad), math.sin(omegaZ_rad)
    
    # Combined rotation matrix R = Rz @ Ry @ Rx (computed directly)
    R = np.zeros((3, 3))
    R[0, 0] = cos_y * cos_z
    R[0, 1] = cos_z * sin_x * sin_y + cos_x * sin_z
    R[0, 2] = -cos_x * cos_z * sin_y + sin_x * sin_z
    R[1, 0] = -cos_y * sin_z
    R[1, 1] = cos_x * cos_z - sin_x * sin_y * sin_z
    R[1, 2] = cos_z * sin_x + cos_x * sin_y * sin_z
    R[2, 0] = sin_y
    R[2, 1] = -cos_y * sin_x
    R[2, 2] = cos_x * cos_y
    
    return R

@jit(nopython=True, cache=True, fastmath=True)
def _calc_strike_dip(R_col):
    """Fast strike/dip calculation"""
    Vstrike_x = -R_col[1]
    Vstrike_y = R_col[0]
    norm_sq = Vstrike_x * Vstrike_x + Vstrike_y * Vstrike_y
    
    if norm_sq > 1e-12:
        norm = math.sqrt(norm_sq)
        Vstrike_x /= norm
        Vstrike_y /= norm
        strike = math.degrees(math.atan2(Vstrike_x, Vstrike_y))
    else:
        strike = 0.0
    
    dip = math.degrees(math.acos(abs(R_col[2])))
    return strike, dip

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _PTDdispSurf_vectorized(X, Y, X0, Y0, depth, strike, dip, DV, nu):
    """Vectorized PTD displacement calculation using Numba"""
    n = X.size
    ue = np.zeros(n)
    un = np.zeros(n)
    uv = np.zeros(n)
    
    if abs(DV) < 1e-15:  # Early return for zero potency
        return ue, un, uv
    
    # Pre-calculate constants
    beta_rad = math.radians(strike - 90)
    cos_beta, sin_beta = math.cos(beta_rad), math.sin(beta_rad)
    dip_rad = math.radians(dip)
    sin_dip, cos_dip = math.sin(dip_rad), math.cos(dip_rad)
    sin_dip_sq = sin_dip * sin_dip
    
    # Constants
    one_minus_2nu = 1.0 - 2.0 * nu
    dv_over_2pi = DV / (2.0 * math.pi)
    
    # Vectorized computation
    for i in prange(n):
        # Translate coordinates
        x_trans = X[i] - X0
        y_trans = Y[i] - Y0
        
        # Rotate coordinates
        x = cos_beta * x_trans - sin_beta * y_trans
        y = sin_beta * x_trans + cos_beta * y_trans
        
        # Calculate distances
        x_sq = x * x
        y_sq = y * y
        depth_sq = depth * depth
        r_sq = x_sq + y_sq + depth_sq
        r = math.sqrt(r_sq)
        r_cubed = r_sq * r
        r_fifth = r_cubed * r_sq
        
        r_plus_d = r + depth
        r_plus_d_sq = r_plus_d * r_plus_d
        r_plus_d_cubed = r_plus_d_sq * r_plus_d
        
        # Calculate q
        q = y * sin_dip - depth * cos_dip
        q_sq = q * q
        
        # Calculate I terms
        inv_r_rpd = 1.0 / (r * r_plus_d_sq)
        I1 = one_minus_2nu * y * (inv_r_rpd - x_sq * (3*r + depth) / (r_cubed * r_plus_d_cubed))
        I2 = one_minus_2nu * x * (inv_r_rpd - y_sq * (3*r + depth) / (r_cubed * r_plus_d_cubed))
        I3 = one_minus_2nu * x / r_cubed - I2
        I5 = one_minus_2nu * (1.0 / (r * r_plus_d) - x_sq * (2*r + depth) / (r_cubed * r_plus_d_sq))
        
        # Calculate displacements in rotated coordinates
        common_factor = 3.0 * q_sq / r_fifth
        ue_rot = dv_over_2pi * (common_factor * x - I3 * sin_dip_sq)
        un_rot = dv_over_2pi * (common_factor * y - I1 * sin_dip_sq)
        uv_calc = dv_over_2pi * (common_factor * depth - I5 * sin_dip_sq)
        
        # Rotate back to original coordinates
        ue[i] = cos_beta * ue_rot + sin_beta * un_rot
        un[i] = -sin_beta * ue_rot + cos_beta * un_rot
        uv[i] = uv_calc
    
    return ue, un, uv

def pCDM(X, Y, X0, Y0, depth, omegaX, omegaY, omegaZ, DVx, DVy, DVz, nu):
    """
    Optimized pCDM function with multiple performance enhancements
    """
    # Input validation (minimal)
    DVsign = np.array([np.sign(DVx), np.sign(DVy), np.sign(DVz)])
    if np.any(DVsign > 0) and np.any(DVsign < 0):
        raise ValueError('Input error: DVx, DVy and DVz must have the same sign!')
    
    # Ensure arrays are contiguous and flattened
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64).ravel())
    Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float64).ravel())
    
    # Early exit if all potencies are zero
    if abs(DVx) < 1e-15 and abs(DVy) < 1e-15 and abs(DVz) < 1e-15:
        return np.zeros_like(X), np.zeros_like(Y), np.zeros_like(X)
    
    # Convert angles to radians once
    omegaX_rad = math.radians(omegaX)
    omegaY_rad = math.radians(omegaY)
    omegaZ_rad = math.radians(omegaZ)
    
    # Calculate rotation matrix
    R = _calc_rotation_matrix(omegaX_rad, omegaY_rad, omegaZ_rad)
    
    # Calculate strikes and dips
    strike1, dip1 = _calc_strike_dip(R[:, 0])
    strike2, dip2 = _calc_strike_dip(R[:, 1])
    strike3, dip3 = _calc_strike_dip(R[:, 2])
    
    # Calculate contributions from each PTD
    ue1, un1, uv1 = _PTDdispSurf_vectorized(X, Y, X0, Y0, depth, strike1, dip1, DVx, nu)
    ue2, un2, uv2 = _PTDdispSurf_vectorized(X, Y, X0, Y0, depth, strike2, dip2, DVy, nu)
    ue3, un3, uv3 = _PTDdispSurf_vectorized(X, Y, X0, Y0, depth, strike3, dip3, DVz, nu)
    
    # Sum contributions
    ue = ue1 + ue2 + ue3
    un = un1 + un2 + un3
    uv = uv1 + uv2 + uv3
    
    return ue, un, uv




def test_pCDM():
    """
    Test function that reproduces the examples from the original MATLAB code
    and plots the results.
    """
    # Example 1: Calculate and plot the vertical displacements on a regular grid
    print("Running Example 1: Vertical displacements on regular grid")
    
    x_range = np.arange(-7, 7.02, 0.01)  # Using coarser grid for faster computation
    y_range = np.arange(-5, 5.02, 0.01)
    X, Y = np.meshgrid(x_range, y_range)
    
    X0 = 0
    Y0 = 0
    depth = 1
    omegaX = 0
    omegaY = 0
    omegaZ = 0
    # DVx = 0.00144
    # DVy = 0.00128
    # DVz = 0.00072
    nu = 0.25

    # For a closing crack, all potencies should be negative
    DVx = -0.00144
    DVy = -0.00128
    DVz = -0.00072

    
    # This represents uniform volume change (inflation/deflation)
    DVx = DVy = DVz = -0.00144  # All equal for isotropy

    
    ue, un, uv = pCDM(X, Y, X0, Y0, depth, omegaX, omegaY, omegaZ, DVx, DVy, DVz, nu)
    # Convert to line-of-sight displacement
    # Parameters for satellite geometry
    ########## Descending orbit ##########
    incidence_angle = 39.4016  # degrees from vertical
    heading = -169.92834  # degrees from north (satellite heading) 
    ######################################

    ########## Ascending orbit ##########
    incidence_angle=39.2545  # degrees from vertical
    heading=-10.072947  # degrees from north (satellite heading)

    # Convert angles to radians
    inc_rad = np.radians(incidence_angle)
    head_rad = np.radians(heading)

    # Line-of-sight unit vector components
    # East component
    los_e = np.sin(inc_rad) * np.cos(head_rad)
    # North component  
    los_n = -np.sin(inc_rad) * np.sin(head_rad)
    # Up component (negative because LOS points from ground to satellite)
    los_u = -np.cos(inc_rad)

    # Calculate line-of-sight displacement
    u_los = ue * los_e + un * los_n + uv * los_u # postive is towards satellite
    u_los = -u_los


    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Vertical displacement
    im1 = axes[0].contourf(X, Y, u_los.reshape(X.shape), levels=20, cmap='RdBu_r')
    axes[0].set_title('Line-of-Sight Displacement (u_los)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])
    
    # East displacement
    im2 = axes[1].contourf(X, Y, ue.reshape(X.shape), levels=20, cmap='RdBu_r')
    axes[1].set_title('East Displacement (ue)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])
    
    # North displacement
    im3 = axes[2].contourf(X, Y, uv.reshape(X.shape), levels=20, cmap='RdBu_r')
    axes[2].set_title('Verticle Displacement (un)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Example 2: Profile comparison
    print("Running Example 2: Profile comparison")
    
    X_profile = np.arange(-10, 10.01, 0.1)
    Y_profile = np.zeros_like(X_profile)
    
    ue_prof, un_prof, uv_prof = pCDM(X_profile, Y_profile, X0, Y0, depth, 
                                     omegaX, omegaY, omegaZ, DVx, DVy, DVz, nu)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_profile, ue_prof / np.max(np.abs(uv_prof)), 'b-', linewidth=2, label='East (normalized)')
    plt.plot(X_profile, uv_prof / np.max(np.abs(uv_prof)), 'r-', linewidth=2, label='Vertical (normalized)')
    plt.xlabel('X')
    plt.ylabel('Normalized displacements')
    plt.legend()
    plt.title('Displacement Profile')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    start_time = time.time()
    test_pCDM()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
