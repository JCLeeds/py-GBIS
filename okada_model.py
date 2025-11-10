import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
from cmcrameri import cm
import numba
from numba import jit, njit, prange
from numba.core import config
import os
import shutil
import time
eps = 1e-14
# Numba-optimized Okada dislocation model
'''July 2024: Highly optimized Numba version of Okada dislocation model using Chinnery notation.
   Significant speed improvements achieved through:
   - Parallel processing with prange
   - Pre-calculation of trigonometric functions
   - Inlining small functions
   - Reducing redundant calculations
   Written by J. Condon based on Original version at "https://github.com/scottyhq/roipy and Andrew Watson's code
   first run will be slow then results are cached for future runs


    %--------------------------------------------------------------
    OKADA85 Surface deformation due to a finite rectangular source.
    [uE,uN,uZ,uZE,uZN,uNN,uNE,uEN,uEE] = OKADA85(...
       E,N,DEPTH,STRIKE,DIP,LENGTH,WIDTH,RAKE,SLIP,OPEN)
    computes displacements, tilts and strains at the surface of an elastic
    half-space, due to a dislocation defined by RAKE, SLIP, and OPEN on a
    rectangular fault defined by orientation STRIKE and DIP, and size LENGTH and
    WIDTH. The fault centroid is located (0,0,-DEPTH).

       E,N    : coordinates of observation points in a geographic referential
                (East,North,Up) relative to fault centroid (units are described below)
       DEPTH  : depth of the fault centroid (DEPTH > 0)
       STRIKE : fault trace direction (0 to 360 relative to North), defined so
                that the fault dips to the right side of the trace
       DIP    : angle between the fault and a horizontal plane (0 to 90)
       LENGTH : fault length in the STRIKE direction (LENGTH > 0)
       WIDTH  : fault width in the DIP direction (WIDTH > 0)
       RAKE   : direction the hanging wall moves during rupture, measured relative
                to the fault STRIKE (-180 to 180).
       SLIP   : dislocation in RAKE direction (length unit)
       OPEN   : dislocation in tensile component (same unit as SLIP)

    returns the following variables (same matrix size as E and N):
       uN,uE,uZ        : displacements (unit of SLIP and OPEN)
    Orginal matlab function from:
    http://www.mathworks.com/matlabcentral/fileexchange/25982-okada--surface-deformation-due-to-a-finite-rectangular-source/content/okada85.m
    
    


   "
'''
@njit(parallel=True, fastmath=True, cache=True)
def disloc3d3_numba(x, y, xoff, yoff, depth, length, width, slip, opening, 
                   strike, dip, rake, nu):
    """
    Highly optimized Numba-optimized Okada dislocation model with parallel processing.
    """
    n_points = len(x)
    
    # Pre-convert angles to radians
    strike_rad = strike * 0.017453292519943295  # np.pi/180
    dip_rad = dip * 0.017453292519943295
    rake_rad = rake * 0.017453292519943295
    
    # Pre-calculate trigonometric values
    sin_strike = np.sin(strike_rad)
    cos_strike = np.cos(strike_rad)
    sin_dip = np.sin(dip_rad)
    cos_dip = np.cos(dip_rad)
    sin_rake = np.sin(rake_rad)
    cos_rake = np.cos(rake_rad)
    
    # Check fault depth constraint
    top_depth = depth - (width * 0.5) * sin_dip
    if top_depth <= 0.0:
        depth = (width * 0.5) * sin_dip + 10.0
    
    # Slip components
    U1 = cos_rake * slip
    U2 = sin_rake * slip
    U3 = opening
    
    # Pre-calculate constants
    d = depth + sin_dip * width * 0.5
    W_cos_dip_half = cos_dip * width * 0.5
    L_half = length * 0.5
    W_cos_dip = width * cos_dip
    L_full = length
    
    # Pre-calculate transformation constants
    U1_factor = -U1 * 0.15915494309189535
    U2_factor = -U2 * 0.15915494309189535
    U3_factor = U3 * 0.15915494309189535
    
    # Output arrays
    ue = np.zeros(n_points)
    un = np.zeros(n_points)
    uz = np.zeros(n_points)
    
    # Process each observation point in parallel
    for i in prange(n_points):
        # Transform coordinates
        e = x[i] - xoff
        n = y[i] - yoff
        
        ec = e + cos_strike * W_cos_dip_half
        nc = n - sin_strike * W_cos_dip_half
        
        x_local = cos_strike * nc + sin_strike * ec + L_half
        y_local = sin_strike * nc - cos_strike * ec + W_cos_dip
        p = y_local * cos_dip + d * sin_dip
        q = y_local * sin_dip - d * cos_dip
        
        # Calculate displacements using optimized Chinnery notation
        ux = (U1_factor * chinnery_ux_ss_opt(x_local, p, L_full, width, q, dip_rad, nu) +
              U2_factor * chinnery_ux_ds_opt(x_local, p, L_full, width, q, dip_rad, nu) +
              U3_factor * chinnery_ux_tf_opt(x_local, p, L_full, width, q, dip_rad, nu))
        
        uy = (U1_factor * chinnery_uy_ss_opt(x_local, p, L_full, width, q, dip_rad, nu) +
              U2_factor * chinnery_uy_ds_opt(x_local, p, L_full, width, q, dip_rad, nu) +
              U3_factor * chinnery_uy_tf_opt(x_local, p, L_full, width, q, dip_rad, nu))
        
        uz_val = (U1_factor * chinnery_uz_ss_opt(x_local, p, L_full, width, q, dip_rad, nu) +
                  U2_factor * chinnery_uz_ds_opt(x_local, p, L_full, width, q, dip_rad, nu) +
                  U3_factor * chinnery_uz_tf_opt(x_local, p, L_full, width, q, dip_rad, nu))
        
        # Transform back to geographic coordinates
        ue[i] = sin_strike * ux - cos_strike * uy
        un[i] = cos_strike * ux + sin_strike * uy
        uz[i] = uz_val
    
    return ue, un, uz

@njit(fastmath=True, inline='always')
def chinnery_ux_ss_opt(x, p, L, W, q, dip, nu):
    return (ux_ss_opt(x, p, q, dip, nu) - ux_ss_opt(x, p - W, q, dip, nu) -
            ux_ss_opt(x - L, p, q, dip, nu) + ux_ss_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_uy_ss_opt(x, p, L, W, q, dip, nu):
    return (uy_ss_opt(x, p, q, dip, nu) - uy_ss_opt(x, p - W, q, dip, nu) -
            uy_ss_opt(x - L, p, q, dip, nu) + uy_ss_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_uz_ss_opt(x, p, L, W, q, dip, nu):
    return (uz_ss_opt(x, p, q, dip, nu) - uz_ss_opt(x, p - W, q, dip, nu) -
            uz_ss_opt(x - L, p, q, dip, nu) + uz_ss_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_ux_ds_opt(x, p, L, W, q, dip, nu):
    return (ux_ds_opt(x, p, q, dip, nu) - ux_ds_opt(x, p - W, q, dip, nu) -
            ux_ds_opt(x - L, p, q, dip, nu) + ux_ds_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_uy_ds_opt(x, p, L, W, q, dip, nu):
    return (uy_ds_opt(x, p, q, dip, nu) - uy_ds_opt(x, p - W, q, dip, nu) -
            uy_ds_opt(x - L, p, q, dip, nu) + uy_ds_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_uz_ds_opt(x, p, L, W, q, dip, nu):
    return (uz_ds_opt(x, p, q, dip, nu) - uz_ds_opt(x, p - W, q, dip, nu) -
            uz_ds_opt(x - L, p, q, dip, nu) + uz_ds_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_ux_tf_opt(x, p, L, W, q, dip, nu):
    return (ux_tf_opt(x, p, q, dip, nu) - ux_tf_opt(x, p - W, q, dip, nu) -
            ux_tf_opt(x - L, p, q, dip, nu) + ux_tf_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_uy_tf_opt(x, p, L, W, q, dip, nu):
    return (uy_tf_opt(x, p, q, dip, nu) - uy_tf_opt(x, p - W, q, dip, nu) -
            uy_tf_opt(x - L, p, q, dip, nu) + uy_tf_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def chinnery_uz_tf_opt(x, p, L, W, q, dip, nu):
    return (uz_tf_opt(x, p, q, dip, nu) - uz_tf_opt(x, p - W, q, dip, nu) -
            uz_tf_opt(x - L, p, q, dip, nu) + uz_tf_opt(x - L, p - W, q, dip, nu))

@njit(fastmath=True, inline='always')
def fast_sqrt(x):
    return np.sqrt(x)

@njit(fastmath=True, inline='always')
def ux_ss_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_eta = R + eta
    
    u = xi * q / (R * R_eta) + I1_opt(xi, eta, q, dip, nu, R) * np.sin(dip)
    if abs(q) > 1e-14:
        u += np.arctan((xi * eta) / (q * R))
    return u

@njit(fastmath=True, inline='always')
def uy_ss_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_eta = R + eta
    cos_dip = np.cos(dip)
    sin_dip = np.sin(dip)
    
    u = ((eta * cos_dip + q * sin_dip) * q / (R * R_eta) +
         q * cos_dip / R_eta + I2_opt(eta, q, dip, nu, R) * sin_dip)
    return u

@njit(fastmath=True, inline='always')
def uz_ss_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_eta = R + eta
    sin_dip = np.sin(dip)
    cos_dip = np.cos(dip)
    db = eta * sin_dip - q * cos_dip
    
    u = (db * q / (R * R_eta) + q * sin_dip / R_eta +
         I4_opt(db, eta, q, dip, nu, R) * sin_dip)
    return u

@njit(fastmath=True, inline='always')
def ux_ds_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    return q / R - I3_opt(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)

@njit(fastmath=True, inline='always')
def uy_ds_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_xi = R + xi
    cos_dip = np.cos(dip)
    sin_dip = np.sin(dip)
    
    u = ((eta * cos_dip + q * sin_dip) * q / (R * R_xi) -
         I1_opt(xi, eta, q, dip, nu, R) * sin_dip * cos_dip)
    if abs(q) > 1e-14:
        u += cos_dip * np.arctan((xi * eta) / (q * R))
    return u

@njit(fastmath=True, inline='always')
def uz_ds_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_xi = R + xi
    sin_dip = np.sin(dip)
    cos_dip = np.cos(dip)
    db = eta * sin_dip - q * cos_dip
    
    u = (db * q / (R * R_xi) - I5_opt(xi, eta, q, dip, nu, R, db) * sin_dip * cos_dip)
    if abs(q) > 1e-14:
        u += sin_dip * np.arctan((xi * eta) / (q * R))
    return u

@njit(fastmath=True, inline='always')
def ux_tf_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_eta = R + eta
    sin_dip_sq = np.sin(dip) * np.sin(dip)
    return q_sq / (R * R_eta) - I3_opt(eta, q, dip, nu, R) * sin_dip_sq

@njit(fastmath=True, inline='always')
def uy_tf_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_xi = R + xi
    R_eta = R + eta
    sin_dip = np.sin(dip)
    cos_dip = np.cos(dip)
    sin_dip_sq = sin_dip * sin_dip
    
    u = (-(eta * sin_dip - q * cos_dip) * q / (R * R_xi) -
         sin_dip * xi * q / (R * R_eta) - I1_opt(xi, eta, q, dip, nu, R) * sin_dip_sq)
    if abs(q) > 1e-14:
        u += sin_dip * np.arctan((xi * eta) / (q * R))
    return u

@njit(fastmath=True, inline='always')
def uz_tf_opt(xi, eta, q, dip, nu):
    xi_sq = xi * xi
    eta_sq = eta * eta
    q_sq = q * q
    R = fast_sqrt(xi_sq + eta_sq + q_sq)
    R_xi = R + xi
    R_eta = R + eta
    sin_dip = np.sin(dip)
    cos_dip = np.cos(dip)
    sin_dip_sq = sin_dip * sin_dip
    db = eta * sin_dip - q * cos_dip
    
    u = ((eta * cos_dip + q * sin_dip) * q / (R * R_xi) +
         cos_dip * xi * q / (R * R_eta) - I5_opt(xi, eta, q, dip, nu, R, db) * sin_dip_sq)
    if abs(q) > 1e-14:
        u -= cos_dip * np.arctan((xi * eta) / (q * R))
    return u

@njit(fastmath=True, inline='always')
def I1_opt(xi, eta, q, dip, nu, R):
    sin_dip = np.sin(dip)
    cos_dip = np.cos(dip)
    db = eta * sin_dip - q * cos_dip
    if cos_dip > 1e-14:
        return ((1 - 2*nu) * (-xi / (cos_dip * (R + db))) -
                sin_dip / cos_dip * I5_opt(xi, eta, q, dip, nu, R, db))
    else:
        R_db = R + db
        return -(1 - 2*nu) * 0.5 * xi * q / (R_db * R_db)

@njit(fastmath=True, inline='always')
def I2_opt(eta, q, dip, nu, R):
    return (1 - 2*nu) * (-np.log(R + eta)) - I3_opt(eta, q, dip, nu, R)

@njit(fastmath=True, inline='always')
def I3_opt(eta, q, dip, nu, R):
    sin_dip = np.sin(dip)
    cos_dip = np.cos(dip)
    yb = eta * cos_dip + q * sin_dip
    db = eta * sin_dip - q * cos_dip
    if cos_dip > 1e-14:
        return ((1 - 2*nu) * (yb / (cos_dip * (R + db)) - np.log(R + eta)) +
                sin_dip / cos_dip * I4_opt(db, eta, q, dip, nu, R))
    else:
        R_db = R + db
        return ((1 - 2*nu) * 0.5 * (eta / R_db + yb * q / (R_db * R_db) -
                np.log(R + eta)))

@njit(fastmath=True, inline='always')
def I4_opt(db, eta, q, dip, nu, R):
    cos_dip = np.cos(dip)
    if cos_dip > 1e-14:
        return ((1 - 2*nu) / cos_dip * (np.log(R + db) - np.sin(dip) * np.log(R + eta)))
    else:
        return -(1 - 2*nu) * q / (R + db)

@njit(fastmath=True, inline='always')
def I5_opt(xi, eta, q, dip, nu, R, db):
    cos_dip = np.cos(dip)
    xi_sq = xi * xi
    q_sq = q * q
    X = fast_sqrt(xi_sq + q_sq)
    if cos_dip > 1e-14:
        if abs(xi) < 1e-14:
            return 0.0
        R_X = R + X
        numerator = eta * (X + q*cos_dip) + X * R_X * np.sin(dip)
        denominator = xi * R_X * cos_dip
        return (1 - 2*nu) * 2 / cos_dip * np.arctan(numerator / denominator)
    else:
        return -(1 - 2*nu) * xi * np.sin(dip) / (R + db)

def disloc3d3(x, y, xoff=0, yoff=0, depth=5e3, length=1e3, width=1e3, 
              slip=0.0, opening=10.0, strike=0.0, dip=0.0, rake=0.0, nu=0.25):
    """
    Fast wrapper for highly optimized Numba Okada dislocation model.
    Returns displacements as a 3xN array for compatibility.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    ue, un, uz = disloc3d3_numba(x, y, xoff, yoff, depth, length, width, 
                                slip, opening, strike, dip, rake, nu)
    
    return ue, un, uz

def plot_enu(U, model, x, y):
    '''
    Plot East, North, and Up displacements from disloc3d3.
    '''
    
    # convert to km for better plotting
    x = x / 1000
    y = y / 1000
    
    # convert to mm
    U = U * 1000
    
    # coord grids
    xx, yy = np.meshgrid(x, y)
    
    # Regrid displacements
    xgrid = np.reshape(U[0,:],xx.shape)
    ygrid = np.reshape(U[1,:],xx.shape)
    zgrid = np.reshape(U[2,:],xx.shape)
    
    # Fault outline for plotting
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = fault_for_plotting(model)
    
    # Setup plot
    fig, ax = plt.subplots(2, 2, figsize=(20, 18))
    extent = (x[0], x[-1], y[0], y[-1])
    
    # Plot East
    im_e = ax[0,0].imshow(xgrid, extent=extent, origin='lower', cmap=cm.vik)
    ax[0,0].contour(xx, yy, xgrid, linestyles='dashed', colors='white')
    ax[0,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[0,0].scatter(end1x/1000, end1y/1000, color='Black')
    ax[0,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
    fig.colorbar(im_e, ax=ax[0,0])
    ax[0,0].set_xlabel('Easting (km)')
    ax[0,0].set_ylabel('Northing (km)')
    ax[0,0].set_title('East displacement (mm)')
    
    # Plot North
    im_n = ax[0,1].imshow(ygrid, extent=extent, cmap=cm.vik)
    ax[0,1].contour(xx, yy, ygrid, linestyles='dashed', colors='white')
    ax[0,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[0,1].scatter(end1x/1000, end1y/1000, color='Black')
    ax[0,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
    fig.colorbar(im_n, ax=ax[0,1])
    ax[0,1].set_xlabel('Easting (km)')
    ax[0,1].set_ylabel('Northing (km)')
    ax[0,1].set_title('North displacement (mm)')
    
    # Plot Up
    im_u = ax[1,0].imshow(zgrid, extent=extent, cmap=cm.vik)
    ax[1,0].contour(xx, yy, zgrid, linestyles='dashed', colors='white')
    ax[1,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[1,0].scatter(end1x/1000, end1y/1000, color='Black')
    ax[1,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
    fig.colorbar(im_u, ax=ax[1,0])
    ax[1,0].set_xlabel('Easting (km)')
    ax[1,0].set_ylabel('Northing (km)')
    ax[1,0].set_title('Vertical displacement (mm)')
    
    # Plot 3D deformation
    im_3d = ax[1,1].imshow(zgrid, extent=extent, cmap=cm.vik)
    ax[1,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[1,1].scatter(end1x/1000, end1y/1000, color='Black')
    ax[1,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
    fig.colorbar(im_3d, ax=ax[1,1], label='Vertical displacement (mm)')
    ax[1,1].quiver(xx[24::25, 24::25], yy[24::25, 24::25], xgrid[24::25, 24::25]/1000, ygrid[24::25, 24::25]/1000, scale=1, color='White')
    ax[1,1].set_xlabel('Easting (km)')
    ax[1,1].set_ylabel('Northing (km)')
    ax[1,1].set_title('3D displacement (mm)')


#-------------------------------------------------------------------------------

def fault_for_plotting(model):
    '''
    Get trace and projected corners of fault for plotting.
    '''
    
    cen_offset = model[7]/np.tan(np.deg2rad(model[3]))
    
    trace_cen_x = model[0] - (cen_offset * np.cos(np.deg2rad(model[2])))
    trace_cen_y = model[1] + (cen_offset * np.sin(np.deg2rad(model[2])))
    
    top_depth = model[7] - (model[8]/2)*np.sin(np.deg2rad(model[3]))
    bottom_depth = model[7] + (model[8]/2)*np.sin(np.deg2rad(model[3]))
    
    end1x = trace_cen_x + np.sin(np.deg2rad(model[2])) * model[6]/2
    end2x = trace_cen_x - np.sin(np.deg2rad(model[2])) * model[6]/2
    end1y = trace_cen_y + np.cos(np.deg2rad(model[2])) * model[6]/2
    end2y = trace_cen_y - np.cos(np.deg2rad(model[2])) * model[6]/2
    
    c1x = end1x + np.sin(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    c2x = end1x + np.sin(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c3x = end2x + np.sin(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c4x = end2x + np.sin(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    c1y = end1y + np.cos(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    c2y = end1y + np.cos(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c3y = end2y + np.cos(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c4y = end2y + np.cos(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    
    return end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y

#-------------------------------------------------------------------------------
    

#-------------------------------------------------------------------------------

def plot_los(U, model, x, y, e2los, n2los, u2los):
    '''
    Plot line-of-sight displacements from East, North, and Up displacements.
    '''
    
    # convert to km for better plotting
    x = x / 1000
    y = y / 1000
    
    # coord grids
    xx, yy = np.meshgrid(x, y)
    
    # Regrid displacements
    xgrid = np.reshape(U[0,:],xx.shape)
    ygrid = np.reshape(U[1,:],xx.shape)
    zgrid = np.reshape(U[2,:],xx.shape)
    
    # Convert to LOS
    los_grid = (xgrid * e2los) + (ygrid * n2los) + (zgrid * u2los)
    centroid_depth_km = model[7]/1000
    
    # Fault outline for plotting
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = fault_for_plotting(model)
    
    # Setup plot
    fig = plt.figure()
    extent = (x[0], x[-1], y[0], y[-1])
    
    # Plot Unwrapped
    im_u = plt.imshow(los_grid*1000, extent=extent, origin='lower', cmap=cm.vik,figure=fig)
    plt.plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White',figure=fig)
    plt.scatter(end1x/1000, end1y/1000, color='white',figure=fig)
    plt.plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White',figure=fig)
    cbar = fig.colorbar(im_u)
    cbar.set_label('LOS Displacement (mm)', rotation=270)
    # ax.set_clim(-100,100)
    plt.xlabel('Easting (km)')
    plt.ylabel('Northing (km)')
    plt.title('Unwrapped LOS displacement (mm): Depth %dkm ' %centroid_depth_km)


def clear_numba_cache():
    """Clear all Numba compiled function caches"""
    try:
        # Get the cache directory
        cache_dir = config.CACHE_DIR
        
        if os.path.exists(cache_dir):
            # Remove the entire cache directory
            shutil.rmtree(cache_dir)
            print(f"Numba cache cleared from: {cache_dir}")
            
            # Recreate the cache directory
            os.makedirs(cache_dir, exist_ok=True)
        else:
            print("No Numba cache directory found")
            
    except Exception as e:
        print(f"Error clearing cache: {e}")
    
#-------------------------------------------------------------------------------
def test_okada():
    '''Test the Okada dislocation model.'''

    # Observation points
    xvec = np.linspace(-20000, 20000, 100)  # Easting (m)
    yvec = np.linspace(-20000, 20000, 100)  #
    xx_vec, yy_vec = np.meshgrid(xvec, yvec)
    xx_vec = xx_vec.flatten()
    yy_vec = yy_vec.flatten()
    
    # Run 1000 iterations with varying parameters
    print("Running 1000 iterations of disloc3d3...")
    iteration_times = []
    
    for i in range(1000):
        if i == 0:
            print("Timing first iteration...")
            iter_start = time.time()

        # Vary parameters for each iteration
        strike = np.random.uniform(0, 360)  # Random strike
        dip = np.random.uniform(30, 90)     # Random dip
        rake = np.random.uniform(-180, 180) # Random rake
        slip = np.random.uniform(0.1, 2.0)  # Random slip
        depth = np.random.uniform(5e3, 15e3) # Random depth
        length = np.random.uniform(1e3, 5e3) # Random length
        width = np.random.uniform(1e3, 5e3)  # Random width
        
        # Time each iteration
        iter_start = time.time()
        ue, un, uz = disloc3d3(xx_vec, yy_vec, xoff=0, yoff=0, depth=depth,
                        length=length, width=width, slip=slip, opening=0, 
                        strike=strike, dip=dip, rake=rake, nu=0.25)
        disp = np.vstack((ue, un, uz))
        iter_end = time.time()
        iteration_times.append(iter_end - iter_start)

        if i == 0:
            print("Timing first iteration...")
            print(f"First iteration time: {iteration_times[-1]:.4f} seconds")
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/1000 iterations")
    
    print(f"Average time per iteration: {np.mean(iteration_times):.4f} seconds")
    print(f"Total time for 1000 iterations: {np.sum(iteration_times):.4f} seconds")
    
    # Use last iteration parameters for plotting
    xcent = 0.0
    ycent = 0.0
    centroid_depth = depth
    model = [xcent, ycent, strike, dip, rake, slip, length, centroid_depth, width]
    
    # Plot ENU displacements
    plot_enu(disp, model, xvec, yvec)

    # Define LOS geometry (example values for satellite line-of-sight)
    # These values represent the unit vector components for line-of-sight
    e2los = 0.1  # East component to LOS
    n2los = 0.0  # North component to LOS  
    u2los = -0.9  # Up component to LOS (negative for satellite looking down)

    # Plot LOS displacements
    plot_los(disp, model, xvec, yvec, e2los, n2los, u2los)

    plt.show()




if __name__ == "__main__":
    # Time the test function
    start_time = time.time()
    test_okada()
    end_time = time.time()
    print(f"Okada model test completed in {end_time - start_time:.4f} seconds.")