"""
Adaptive TIF Processing Tool for InSAR Data
This script processes GeoTIFF files containing InSAR displacement data with adaptive
spatial sampling. It allows users to interactively select a region of interest and
applies different downsampling rates to high-interest areas versus background regions.
Key Features:
- Interactive GUI for selecting center point and radius of high-resolution area
- Adaptive downsampling with different factors for foreground/background regions
- Processing of multi-component displacement data (East, North, Up, Phase)
- Conversion from Cartesian (E,N,U) to radar geometry (incidence, heading angles)
- Export to compressed NumPy format for further analysis
The tool is particularly useful for InSAR time series analysis where computational
efficiency requires reduced data volume while maintaining high resolution in areas
of interest (e.g., around active deformation sources).
Usage:
    python prep_data.py input.tif input_e.tif input_n.tif input_u.tif [options]
Example:
    python prep_data.py disp.tif east.tif north.tif up.tif --high_res_factor 1 --low_res_factor 4
Input Files:
    - Phase/displacement TIF file (main input)
    - East component TIF file  
    - North component TIF file
    - Up component TIF file
Output:
    - NPZ file containing processed arrays: Lat, Lons, Phase, Inc, Heading, mask
Classes:
    InteractiveSelector: GUI widget for interactive region selection
Functions:
    interactive_selection(): Launch GUI for center point and radius selection
    read_tif_file(): Load GeoTIFF data with georeferencing information
    pixel_to_coords(): Convert pixel indices to geographic coordinates
    coords_to_pixel(): Convert geographic coordinates to pixel indices  
    create_distance_mask(): Generate circular mask for high-resolution region
    downsample_data(): Reduce spatial resolution by block averaging
    process_tif_with_adaptive_sampling(): Main processing pipeline
    convert_e_n_u_to_inc_heading(): Transform Cartesian to radar geometry
Dependencies:
    - numpy: Numerical operations and array handling
    - rasterio: GeoTIFF I/O and coordinate transformations
    - matplotlib: Interactive plotting and visualization
    - scipy: Image processing (uniform filtering)
    - argparse: Command-line interface
Author: John Condon 
Date: 10/11/2025
Version: 1.0
"""
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import uniform_filter
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import matplotlib.patches as patches


class InteractiveSelector:
    def __init__(self, data, transform, extent):
        self.data = data
        self.transform = transform
        self.extent = extent
        self.center_lat = None
        self.center_lon = None
        self.radius_km = 10.0
        self.selected = False
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Display the data
        im = self.ax.imshow(data, extent=extent, cmap='viridis', aspect='auto')
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Click to select center point, then adjust radius with mouse wheel')
        plt.colorbar(im, ax=self.ax, label='Displacement')
        
        # Add buttons
        ax_button = plt.axes([0.81, 0.01, 0.08, 0.04])
        self.button = Button(ax_button, 'Confirm')
        self.button.on_clicked(self.confirm_selection)
        
        # Initialize circle patch (invisible at first)
        self.circle = patches.Circle((0, 0), 0, fill=False, color='red', linewidth=2, visible=False)
        self.ax.add_patch(self.circle)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Add text box for current radius
        self.text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        self.center_lon = event.xdata
        self.center_lat = event.ydata
        
        # Update circle
        self.circle.set_center((self.center_lon, self.center_lat))
        self.update_circle()
        self.circle.set_visible(True)
        
        self.update_text()
        self.fig.canvas.draw()
        
    def on_scroll(self, event):
        if self.center_lat is None or event.inaxes != self.ax:
            return
            
        # Adjust radius with scroll wheel
        if event.button == 'up':
            self.radius_km = min(100.0, self.radius_km + 1.0)
        elif event.button == 'down':
            self.radius_km = max(1.0, self.radius_km - 1.0)
            
        self.update_circle()
        self.update_text()
        self.fig.canvas.draw()
        
    def update_circle(self):
        # Convert radius from km to degrees (approximate)
        radius_degrees = self.radius_km / 111.0
        self.circle.set_radius(radius_degrees)
        
    def update_text(self):
        if self.center_lat is not None:
            text = f'Center: ({self.center_lat:.4f}, {self.center_lon:.4f})\nRadius: {self.radius_km:.1f} km'
            self.text.set_text(text)
        
    def confirm_selection(self, event):
        if self.center_lat is not None:
            self.selected = True
            plt.close(self.fig)
            
    def get_selection(self):
        plt.show()
        return self.center_lat, self.center_lon, self.radius_km if self.selected else (None, None, None)

def interactive_selection(filepath):
    """Allow user to interactively select center point and radius"""
    # Read the TIF file for preview
    with rasterio.open(filepath) as src:
        data = src.read(1)
        transform = src.transform
        
        # Calculate extent for imshow
        height, width = data.shape
        left = transform[2]
        top = transform[5]
        right = left + width * transform[0]
        bottom = top + height * transform[4]
        extent = [left, right, bottom, top]
    
    # Create interactive selector
    selector = InteractiveSelector(data, transform, extent)
    center_lat, center_lon, radius_km = selector.get_selection()
    
    return center_lat, center_lon, radius_km

def read_tif_file(filepath):
    """Read a .tif file and return data, transform, and CRS"""
    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return data, transform, crs, nodata

def pixel_to_coords(transform, row, col):
    """Convert pixel coordinates to lat/lon"""
    lon, lat = xy(transform, row, col)
    return lat, lon

def coords_to_pixel(transform, lat, lon):
    """Convert lat/lon to pixel coordinates"""
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    return row, col

def create_distance_mask(shape, center_row, center_col, radius_pixels):
    """Create a circular mask for the high-resolution area"""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    mask = (rows - center_row)**2 + (cols - center_col)**2 <= radius_pixels**2
    return mask

def downsample_data(data, factor, nodata_value=None):
    """Downsample data by averaging blocks of size factor x factor"""
    if factor == 1:
        return data
    
    # Pad data to make it divisible by factor
    pad_rows = (factor - data.shape[0] % factor) % factor
    pad_cols = (factor - data.shape[1] % factor) % factor
    
    if nodata_value is not None:
        padded = np.pad(data, ((0, pad_rows), (0, pad_cols)), 
                       mode='constant', constant_values=nodata_value)
    else:
        padded = np.pad(data, ((0, pad_rows), (0, pad_cols)), mode='edge')
    
    # Reshape and average
    new_shape = (padded.shape[0] // factor, factor,
                 padded.shape[1] // factor, factor)
    reshaped = padded.reshape(new_shape)
    
    if nodata_value is not None:
        # Handle nodata values properly
        mask = reshaped != nodata_value
        with np.errstate(invalid='ignore'):
            downsampled = np.where(
                mask.any(axis=(1, 3)),
                np.mean(np.where(mask, reshaped, 0), axis=(1, 3)) / 
                np.mean(mask.astype(float), axis=(1, 3)),
                nodata_value
            )
    else:
        downsampled = np.mean(reshaped, axis=(1, 3))
    
    return downsampled


def process_tif_with_adaptive_sampling(filepath,filepath_e,filepath_n,filepath_u, center_lat, center_lon, 
                                     radius_km, high_res_factor, low_res_factor):
    """
    Process TIF file with adaptive sampling rates
    
    Parameters:
    filepath: path to .tif file
    center_lat, center_lon: coordinates of center point
    radius_km: radius in kilometers for high-resolution area
    high_res_factor: downsampling factor for high-res area (1 = no downsampling)
    low_res_factor: downsampling factor for low-res area
    """
    
    # Read the TIF file
    data, transform, crs, nodata = read_tif_file(filepath)
    data_e, _, _, _ = read_tif_file(filepath_e)
    data_n, _, _, _ = read_tif_file(filepath_n)
    data_u, _, _, _ = read_tif_file(filepath_u)
    
    # Convert center coordinates to pixel coordinates
    center_row, center_col = coords_to_pixel(transform, center_lat, center_lon)
    
    # Calculate radius in pixels (approximate)
    pixel_size_x = abs(transform[0])  # degrees per pixel
    pixel_size_y = abs(transform[4])  # degrees per pixel
    avg_pixel_size = (pixel_size_x + pixel_size_y) / 2
    
    # Convert km to degrees (rough approximation: 1 degree ≈ 111 km)
    radius_degrees = radius_km / 111.0
    radius_pixels = int(radius_degrees / avg_pixel_size)
    
    # Clip data to 3 times the radius around the center point
    clip_radius_pixels = int(3 * radius_pixels)
    
    # Calculate clipping bounds
    clip_row_min = max(0, center_row - clip_radius_pixels)
    clip_row_max = min(data.shape[0], center_row + clip_radius_pixels)
    clip_col_min = max(0, center_col - clip_radius_pixels)
    clip_col_max = min(data.shape[1], center_col + clip_radius_pixels)
    
    # Clip all data arrays
    data = data[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
    data_e = data_e[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
    data_n = data_n[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
    data_u = data_u[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
    
    # Adjust center coordinates for clipped data
    center_row_clipped = center_row - clip_row_min
    center_col_clipped = center_col - clip_col_min
    
    # Adjust transform for clipped data
    clipped_transform = rasterio.Affine(transform[0], transform[1], 
                                       transform[2] + clip_col_min * transform[0],
                                       transform[3], transform[4], 
                                       transform[5] + clip_row_min * transform[4])
    
    # Create circular mask for high-resolution area on clipped data
    high_res_mask = create_distance_mask(data.shape, center_row_clipped, center_col_clipped, radius_pixels)
    
    # Process high-resolution area
    high_res_data = downsample_data(data, high_res_factor, nodata)
    high_res_data_e = downsample_data(data_e, high_res_factor, nodata)
    high_res_data_n = downsample_data(data_n, high_res_factor, nodata)
    high_res_data_u = downsample_data(data_u, high_res_factor, nodata)
    
    # Process low-resolution area  
    low_res_data = downsample_data(data, low_res_factor, nodata)
    low_res_data_e = downsample_data(data_e, low_res_factor, nodata)
    low_res_data_n = downsample_data(data_n, low_res_factor, nodata)
    low_res_data_u = downsample_data(data_u, low_res_factor, nodata)
    
    # Downsample the mask to match low-res dimensions
    low_res_mask_full = downsample_data(high_res_mask.astype(float), low_res_factor) > 0.5
    
    # Create coordinate arrays for both resolutions
    # High-res coordinates
    high_res_rows, high_res_cols = np.mgrid[0:high_res_data.shape[0], 0:high_res_data.shape[1]]
    high_res_lats = np.zeros_like(high_res_rows, dtype=float)
    high_res_lons = np.zeros_like(high_res_cols, dtype=float)
    
    # Adjust transform for high-res downsampling
    high_res_transform = rasterio.Affine(clipped_transform[0] * high_res_factor, clipped_transform[1], clipped_transform[2],
                                       clipped_transform[3], clipped_transform[4] * high_res_factor, clipped_transform[5])
    
    for i in range(high_res_data.shape[0]):
        for j in range(high_res_data.shape[1]):
            lat, lon = pixel_to_coords(high_res_transform, i, j)
            high_res_lats[i, j] = lat
            high_res_lons[i, j] = lon
    
    # Low-res coordinates
    low_res_rows, low_res_cols = np.mgrid[0:low_res_data.shape[0], 0:low_res_data.shape[1]]
    low_res_lats = np.zeros_like(low_res_rows, dtype=float)
    low_res_lons = np.zeros_like(low_res_cols, dtype=float)
    
    # Adjust transform for low-res downsampling
    low_res_transform = rasterio.Affine(clipped_transform[0] * low_res_factor, clipped_transform[1], clipped_transform[2],
                                      clipped_transform[3], clipped_transform[4] * low_res_factor, clipped_transform[5])
    
    for i in range(low_res_data.shape[0]):
        for j in range(low_res_data.shape[1]):
            lat, lon = pixel_to_coords(low_res_transform, i, j)
            low_res_lats[i, j] = lat
            low_res_lons[i, j] = lon
    
    # Combine the data: use high-res where mask is True, low-res elsewhere
    all_lats = []
    all_lons = []
    all_displacement = []
    all_e = []
    all_n = []
    all_u = []
    all_mask = []
    
    # Add high-res points where mask is True
    high_res_mask_downsampled = downsample_data(high_res_mask.astype(float), high_res_factor) > 0.5
    high_res_indices = np.where(high_res_mask_downsampled)
    
    all_lats.extend(high_res_lats[high_res_indices].flatten())
    all_lons.extend(high_res_lons[high_res_indices].flatten())
    all_displacement.extend(high_res_data[high_res_indices].flatten())
    all_mask.extend([True] * len(high_res_indices[0]))
    all_e.extend(high_res_data_e[high_res_indices].flatten())
    all_n.extend(high_res_data_n[high_res_indices].flatten())
    all_u.extend(high_res_data_u[high_res_indices].flatten())

    # Add low-res points where mask is False
    low_res_indices = np.where(~low_res_mask_full)
    
    all_lats.extend(low_res_lats[low_res_indices].flatten())
    all_lons.extend(low_res_lons[low_res_indices].flatten())
    all_displacement.extend(low_res_data[low_res_indices].flatten())
    all_e.extend(low_res_data_e[low_res_indices].flatten())
    all_n.extend(low_res_data_n[low_res_indices].flatten())
    all_u.extend(low_res_data_u[low_res_indices].flatten())
    all_mask.extend([False] * len(low_res_indices[0]))

    return (np.array(all_lats), np.array(all_lons), 
            np.array(all_displacement), np.array(all_mask),
            np.array(all_e), np.array(all_n), np.array(all_u))



def convert_e_n_u_to_inc_heading(e,n,u):
    
    # Normalize the vector to ensure it's a unit vector
    magnitude = np.sqrt(e**2 + n**2 + u**2)
    e_norm = e / magnitude
    n_norm = n / magnitude  
    u_norm = u / magnitude
    
    # Clamp u values to valid range for arccos [-1, 1]
    u_clamped = np.clip(u_norm, -1, 1)
    
    # Calculate incidence angle (angle from vertical/up direction)
    inc = np.arccos(np.abs(u_clamped))
    inc = np.degrees(inc)

    # Calculate heading (azimuth angle from east direction)
    heading = np.arctan2(n_norm, e_norm)
    heading = np.degrees(heading)
    
    # Convert to 0-360 range
    heading = (-heading) - 180
    
    print(f"Mean heading (non-zero): {np.nanmean(heading[heading != 0])}")
    print(f"Mean incidence (non-zero): {np.nanmean(inc[inc != 0])}")

    return inc, heading



def main():
    parser = argparse.ArgumentParser(description='Process TIF file with adaptive sampling')
    parser.add_argument('input_file', help='Input .tif file path')
    parser.add_argument('input_file_e', help='Input .tif file path for East component')
    parser.add_argument('input_file_n', help='Input .tif file path for North component')
    parser.add_argument('input_file_u', help='Input .tif file path for Up component')
    # parser.add_argument('--center_lat', type=float, required=True, help='Center latitude')
    # parser.add_argument('--center_lon', type=float, required=True, help='Center longitude')
    # parser.add_argument('--radius_km', type=float, default=10.0, help='Radius in km for high-res area')
    parser.add_argument('--high_res_factor', type=int, default=1, help='High-res downsampling factor')
    parser.add_argument('--low_res_factor', type=int, default=4, help='Low-res downsampling factor')
    parser.add_argument('--output', help='Output file prefix')
    args = parser.parse_args()

    # Set default output if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = f"{base_name}_processed"

    # args.input_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/Test_sight_mod/20230108_20230201.geo.unw.tif' # Temporary hardcoded for testing
    # Process the data
    # Get center coordinates interactively
    center_lat, center_lon, radius_km = interactive_selection(args.input_file)
    if center_lat is None:
        print("No selection made. Exiting.")
        return

    # Update args with interactive selection
    args.center_lat = center_lat
    args.center_lon = center_lon
    if radius_km is not None:
        args.radius_km = radius_km


    lats, lons, displacement, mask, e, n, u = process_tif_with_adaptive_sampling(
        args.input_file, args.input_file_e, args.input_file_n, args.input_file_u, args.center_lat, args.center_lon, 
        args.radius_km, args.high_res_factor, args.low_res_factor
    )
    inc, heading = convert_e_n_u_to_inc_heading(e,n,u)
    # Plot the downsampled results
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(lons, lats, c=displacement, cmap='viridis', s=1, alpha=0.7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Downsampled Results')
    plt.colorbar(scatter, ax=ax, label='Displacement')
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    if args.output:
        # Remove NaN values before saving
        valid_mask = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(displacement) | 
                       np.isnan(inc) | np.isnan(heading))

        lats = lats[valid_mask]
        lons = lons[valid_mask]
        displacement = displacement[valid_mask]
        inc = inc[valid_mask]
        heading = heading[valid_mask]
        mask = mask[valid_mask]
        np.save(f'{args.output}.npy', {
            'Lat': lats,
            'Lon': lons, 
            'Phase': displacement,
            'mask': mask,
            'Inc': inc,
            'Heading': heading,
            'center_lat': args.center_lat,
            'center_lon': args.center_lon
        })
        print(f"Results saved to: {args.output}.npy")

        # Save clipped full-resolution data as well
        # Read the TIF files again to get original data
        data, transform, crs, nodata = read_tif_file(args.input_file)
        data_e, _, _, _ = read_tif_file(args.input_file_e)
        data_n, _, _, _ = read_tif_file(args.input_file_n)
        data_u, _, _, _ = read_tif_file(args.input_file_u)
        
        # Apply the same clipping as used in processing
        center_row, center_col = coords_to_pixel(transform, args.center_lat, args.center_lon)
        pixel_size_x = abs(transform[0])
        pixel_size_y = abs(transform[4])
        avg_pixel_size = (pixel_size_x + pixel_size_y) / 2
        radius_degrees = args.radius_km / 111.0
        radius_pixels = int(radius_degrees / avg_pixel_size)
        clip_radius_pixels = int(3 * radius_pixels)
        
        clip_row_min = max(0, center_row - clip_radius_pixels)
        clip_row_max = min(data.shape[0], center_row + clip_radius_pixels)
        clip_col_min = max(0, center_col - clip_radius_pixels)
        clip_col_max = min(data.shape[1], center_col + clip_radius_pixels)
        
        # Clip the data
        clipped_data = data[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
        clipped_data_e = data_e[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
        clipped_data_n = data_n[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
        clipped_data_u = data_u[clip_row_min:clip_row_max, clip_col_min:clip_col_max]
        
        # Create coordinate arrays for clipped full resolution
        clipped_rows, clipped_cols = np.mgrid[0:clipped_data.shape[0], 0:clipped_data.shape[1]]
        clipped_lats = np.zeros_like(clipped_rows, dtype=float)
        clipped_lons = np.zeros_like(clipped_cols, dtype=float)
        
        # Adjust transform for clipped data
        clipped_transform = rasterio.Affine(transform[0], transform[1], 
                           transform[2] + clip_col_min * transform[0],
                           transform[3], transform[4], 
                           transform[5] + clip_row_min * transform[4])

        for i in range(clipped_data.shape[0]):
            for j in range(clipped_data.shape[1]):
                lat, lon = pixel_to_coords(clipped_transform, i, j)
                clipped_lats[i, j] = lat
                clipped_lons[i, j] = lon

        # Convert to radar geometry
        clipped_inc, clipped_heading = convert_e_n_u_to_inc_heading(clipped_data_e.flatten(), clipped_data_n.flatten(), clipped_data_u.flatten())

        np.save(f'{args.output}_clipped_full_resolution.npy', {
             'Lat': clipped_lats.flatten(),
             'Lon': clipped_lons.flatten(), 
             'Phase': clipped_data.flatten(),
             'Inc': clipped_inc,
             'Heading': clipped_heading,
             'center_lat': args.center_lat,
             'center_lon': args.center_lon
        })
        print(f"Clipped full resolution data saved to: {args.output}_clipped_full_resolution.npy")
    
    print(f"Processing complete. Data shape: {displacement.shape}")
    print(f"High-resolution pixels: {np.sum(mask)}")
    print(f"Low-resolution pixels: {np.sum(~mask)}")

if __name__ == "__main__":
    main()