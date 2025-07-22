# Add "import cartopy" to the top of your Jupyter notebook,
# before using these functions, or visualizations will fail.

import cartopy.crs as ccrs  # For cartographic projections in visualizations
import cartopy.feature as cfeature  # For adding geographic features (land, oceans, etc.)
import numpy as np  # For numerical computations
from matplotlib import pyplot as plt  # For plotting
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

import datetime
from datetime import date
from typing import Dict, Optional
import calendar
import warnings


def fix_quiver_bug(field, lat):
    """
    Fixes a bug in quiver vector plots where the u-component (eastward) vector is distorted
    due to the curvature of latitude in polar projections.

    Args:
        field (tuple of np.array): Tuple containing the u and v components of the vector field.
        lat (np.array): Array of latitudes corresponding to the field.

    Returns:
        np.array: Fixed vector field with adjusted u-component.
    """
    ufield, vfield = field
    # Compute the original magnitude of the vector field
    old_magnitude = np.sqrt(ufield ** 2 + vfield ** 2)
    # Adjust the u-component by accounting for latitude distortion
    ufield_fixed = ufield / np.cos(np.radians(lat))
    # Compute the new magnitude after fixing the u-component
    new_magnitude = np.sqrt(ufield_fixed ** 2 + vfield ** 2)
    # Rescale the vector field to maintain original magnitudes
    field_fixed = np.stack([ufield_fixed, vfield]) * old_magnitude / new_magnitude.clip(min=1e-6)
    return field_fixed


def create_cartopy(coastline_resolution='110m', figsize=(12, 12)):
    """
    Creates a Cartopy map using a North Polar Stereographic projection with adjustable coastline resolution.

    Parameters:
        coastline_resolution (str): Resolution of coastline data. Options are:
            - '110m' (1:110 million, lowest resolution)
            - '50m' (1:50 million, medium resolution)
            - '10m' (1:10 million, highest resolution)
            Default is '110m'.

    Returns:
        tuple: A tuple containing the figure and axis with the configured map.
    """
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=45.0)}  # North Polar projection
    )

    ax.set_facecolor(cfeature.COLORS['water'])
    
    # Add land features with adjustable resolution
    land = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale=coastline_resolution,
        edgecolor=cfeature.COLORS['land'],
        facecolor=cfeature.COLORS['land']
    )
    ax.add_feature(land, zorder=0)
    
    # Add coastlines with the same resolution (optional, for more prominent coastlines)
    # ax.coastlines(resolution=coastline_resolution, linewidth=1, color='black', zorder=1)
    
    # Add gridlines to the map
    ax.gridlines(draw_labels=True, color='gray', zorder=9)
    
    return fig, ax

def create_cartopy_grid(nrows=1, ncols=1, coastline_resolution='110m', figsize=None, ax_size=6, central_longitude=45.0):
    """
    Creates a grid of Cartopy maps using North Polar Stereographic projection.

    Parameters:
        nrows (int): Number of rows in the grid. Default is 1.
        ncols (int): Number of columns in the grid. Default is 1.
        coastline_resolution (str): Resolution of coastline data ('110m', '50m', '10m'). Default '110m'.
        figsize (tuple): Figure size (width, height) in inches. If None, scales with grid size.
        central_longitude (float): Central longitude for projection. Default 45.0.

    Returns:
        tuple: (figure, axes) where axes is a numpy array of Axes objects with the configured maps.
    """
    # Set default figure size based on grid dimensions if not specified
    if figsize is None:
        figsize = (ax_size * ncols, ax_size * nrows)
    
    # Create figure with grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=figsize,
                            subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=central_longitude)})
    
    # Ensure axes is always a 2D array for consistent handling
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Configure each axis
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor(cfeature.COLORS['water'])
            
            # Add land features
            land = cfeature.NaturalEarthFeature(
                category='physical',
                name='land',
                scale=coastline_resolution,
                edgecolor=cfeature.COLORS['land'],
                facecolor=cfeature.COLORS['land']
            )
            ax.add_feature(land, zorder=0)
            
            # Add gridlines
            ax.gridlines(draw_labels=True, color='gray', zorder=9)
    
    plt.tight_layout()
    return fig, axes

def visualize_scalar_field(ax, grid, field, if_colorbar=False, **kwargs):
    """
    Visualizes a scalar field on the map using a color mesh.

    Args:
        ax (matplotlib.axes._axes.Axes): Axis to plot on.
        grid (Grid): Grid object containing lat/lon information.
        field (np.array): Scalar field to visualize.
        vmin (float, optional): Minimum value for color scale. Defaults to None.
        vmax (float, optional): Maximum value for color scale. Defaults to None.
    """
    # Create a colored mesh plot of the scalar field, projected using Plate Carree
    layer = ax.pcolormesh(
        grid.lon,
        grid.lat,
        field,
        transform=ccrs.PlateCarree(),
        alpha=None,
        **kwargs\
    )
    if if_colorbar:
        # Add a color bar for reference
        plt.colorbar(layer)
    return layer


def block_average(arr, step, min_valid=None):
    """
    Downsample a 2D array by taking the mean over non-overlapping blocks of size step x step.
    Crops the array to make its dimensions divisible by step. Only includes averaged values where
    the number of valid (non-NaN) pixels is at least `min_valid`, otherwise sets result to NaN.
    """
    h, w = arr.shape
    h2 = (h // step) * step
    w2 = (w // step) * step
    cropped = arr[:h2, :w2]
    reshaped = cropped.reshape(h2 // step, step, w2 // step, step)
    valid_counts = np.sum(~np.isnan(reshaped), axis=(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(reshaped, axis=(1, 3))
    min_valid = min_valid if min_valid is not None else step * step / 2 
    means[valid_counts < min_valid] = np.nan
    return means


def visualize_vector_field(ax, grid, field, key_length=50, key_units='cm/s', key_color='black', 
                           from_polar=False, from_direction=True, step=64, use_pooling=True, min_valid=5,
                           scale=None, width=0.002, headwidth=3, headlength=5):
    """
    Visualizes a vector field on the map using quiver arrows, with optional block average pooling.
    Uses block-averaging for vector components but selects the geographic center of each block for
    lon/lat, ensuring no seam artifacts at the 180° meridian.
    """
    if from_polar:
        norm, angle = field
        u, v = polar_to_cartesian(norm, angle, from_direction=from_direction)
    else:
        u, v = field

    u_fixed, v_fixed = fix_quiver_bug((u, v), grid.lat)
    h, w = grid.lon.shape

    if use_pooling and step > 1:
        # Average vector components
        u_p = block_average(u_fixed, step, min_valid)
        v_p = block_average(v_fixed, step, min_valid)

        # Determine block centers for coordinates (no averaging lon/lat values)
        h2 = (h // step) * step
        w2 = (w // step) * step
        # indices at center of each block
        i_centers = (np.arange(step//2, h2, step)).astype(int)
        j_centers = (np.arange(step//2, w2, step)).astype(int)
        lon_p = grid.lon[i_centers[:, None], j_centers[None, :]]
        lat_p = grid.lat[i_centers[:, None], j_centers[None, :]]

        mask = (~np.isnan(u_p) & ~np.isnan(v_p))
        layer = ax.quiver(
            lon_p[mask], lat_p[mask], u_p[mask], v_p[mask],
            transform=ccrs.PlateCarree(), color=key_color,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength
        )
    else:
        layer = ax.quiver(
            grid.lon[::step, ::step], grid.lat[::step, ::step],
            u_fixed[::step, ::step], v_fixed[::step, ::step],
            transform=ccrs.PlateCarree(), color=key_color,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength
        )

    ax.quiverkey(layer, X=0.69, Y=0.2, U=key_length, label=f'{key_length} {key_units}',
                 labelpos='E', coordinates='axes')
    return layer


# def visualize_vector_field(ax, grid, field, key_length=50, key_units='cm/s', key_color='black', 
#                            from_polar=False, from_direction=True, step=16):
#     """
#     Visualizes a vector field on the map using quiver arrows.
    
#     Args:
#         ax (matplotlib.axes._axes.Axes): Axis to plot on.
#         grid (Grid): Grid object containing lat/lon information.
#         field (np.array): Either (u, v) or (norm, angle) depending on from_polar.
#         key_length (int, optional): Length of the quiver key. Defaults to 50.
#         key_units (str, optional): Units of the quiver key. Defaults to 'cm/s'.
#         key_color (str, optional): Color of the quiver arrows. Defaults to 'black'.
#         from_polar (bool): If True, field is treated as (norm, angle). Defaults to False.
#         from_direction (bool): If using from_polar, whether angle is a FROM direction. Defaults to True.
#     """
#     if from_polar:
#         norm, angle = field
#         u, v = polar_to_cartesian(norm, angle, from_direction=from_direction)
#     else:
#         u, v = field
    
#     field_fixed = fix_quiver_bug((u, v), grid.lat)

#     layer = ax.quiver(
#         grid.lon[::step, ::step],
#         grid.lat[::step, ::step],
#         field_fixed[0][::step, ::step],
#         field_fixed[1][::step, ::step],
#         transform=ccrs.PlateCarree(),
#         color=key_color,
#     )

#     ax.quiverkey(layer, X=0.69, Y=0.2, U=key_length, label=f'{key_length} {key_units}',
#                  labelpos='E', coordinates='axes')
#     return layer


def show_validation_table(rows, columns, data, title):
    """
    Displays a validation table with mean values and a heatmap visualization.

    Args:
        rows (list): Row labels.
        columns (list): Column labels.
        data (np.array): Data array containing the validation results.
        title (str): Title for the plot.
    """
    # Compute the mean values for each row, excluding the diagonal
    row_means = np.mean(np.array([
        [data[i, j] for j in range(data.shape[1]) if i != j]
        for i in range(data.shape[0])
    ]), axis=1).reshape(-1, 1)

    # Create a new data array with an extra column for the row means
    separator = np.full((data.shape[0], 1), -np.inf)  # Add separator (negative infinity for spacing)
    extended_data = np.hstack((data, separator, row_means))  # Add row means as a new column

    # Create the plot with adjusted figure size
    fig, ax = plt.subplots(figsize=(8, 7))

    # Determine the minimum and maximum values for color scaling
    vmin = max(0, min(data[i, j] for i in range(len(rows)) for j in range(len(columns)) if i != j))
    vmax = max(data[i, j] for i in range(len(rows)) for j in range(len(columns)) if i != j)

    # Display the extended data as an image (heatmap)
    ax.imshow(extended_data, vmin=vmin, vmax=vmax, cmap='viridis')

    # Adjust the tick labels to include the mean column
    ax.set_xticks(np.arange(len(columns) + 2))  # +2 for the separator and mean column
    ax.set_xticklabels(columns + [''] + ['Mean'])  # Add an empty label for the separator
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)

    # Rotate the x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add annotations to display the data values on the heatmap
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', color='r')

    # Add annotations for the mean values in the last column
    mean_column_index = len(columns) + 1
    for i, mean_value in enumerate(row_means):
        ax.text(mean_column_index, i, f'{mean_value[0]:.1f}', ha='center', va='center', color='r')

    # Set the title and adjust the layout
    ax.set_title(title)
    fig.tight_layout()


def get_color_params(metric_name, vmin, vmax):
    params = {
        'norm': None,
        'cmap': None,
        'vmin': None,
        'vmax': None,
    }
    if metric_name in plt.colormaps:
        params['cmap'] = plt.colormaps[metric_name]
        params['vmin'] = vmin
        params['vmax'] = vmax
    if 'mse' in metric_name or 'mae' in metric_name:
        params['cmap'] = plt.colormaps['magma_r']
        params['vmin'] = vmin
        params['vmax'] = vmax
    elif 'diff' in metric_name:
        params['cmap'] = plt.colormaps['RdBu_r']
        abs_max = max(abs(vmin), abs(vmax))
        params['norm'] = colors.TwoSlopeNorm(vmin=-abs_max,
                                             vcenter=0,
                                             vmax=abs_max)
    elif any(m in metric_name for m in ['ice', 'identity']):
        import cmocean
        params['cmap'] = cmocean.cm.ice
        params['vmin'] = vmin
        params['vmax'] = vmax
    else:
        params['vmin'] = vmin
        params['vmax'] = vmax
    return params


def plot_error_evolution(
    error_data: Dict[date, float],
    title: str = "Error Evolution Over Time",
    xlabel: str = "Date",
    ylabel: str = "Error",
    color: str = "tab:blue",
    figsize: tuple = (12, 6),
    grid: bool = True,
    marker: str = None,
    date_format: str = "%Y-%m-%d",
    label_rotation: int = 45,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one or multiple error evolution charts on the same figure.
    
    Args:
        error_data: Dictionary mapping dates to error values
        ax: Existing axes to plot on (for multiple datasets)
        label: Legend label for this dataset
        ... (other parameters remain the same)
    """
    # Extract and sort dates and errors
    start_date = min(error_data)
    end_date = max(error_data)
    dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    errors=[]
    for d in dates:
        value = error_data.get(d, np.array([np.nan]))
        try:
            value = value.item() # Convert to scalar if possible
        except AttributeError:
            pass
        errors.append(value)

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        new_plot = True
    else:
        fig = ax.figure
        new_plot = False

    # Plot the data
    ax.plot(dates, errors, marker=marker, linestyle="-", 
            color=color, label=label)
    if label:
        ax.legend()
    # Only configure axis properties for new plots
    if new_plot:
        # Configure date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        
        # Rotate and align labels
        plt.setp(ax.get_xticklabels(), rotation=label_rotation, ha="right")

        # Set labels and title
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        
        # Add grid
        if grid:
            ax.grid(True, alpha=0.3)

        # Adjust layout
        fig.tight_layout()
    return fig, ax


def plot_error_cycle(
    error_data: Dict[date, float],
    cycle: str = "month",  # "month" or "daily"
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Error",
    color: str = "tab:blue",
    figsize: tuple = (12, 6),
    grid: bool = True,
    marker: Optional[str] = None,
    linestyle: str = "-",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    interquantile_range: bool = False,
    interdecile_range: bool = False,
    std_range: bool = False,
    aggregation_func: str = 'nanmean',
    label_rotation: int = 45,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot aggregated error over a year cycle, either by calendar month or by day-of-year,
    and always align month-cycle points to the correct day-of-year positions so that
    monthly and daily curves overlay properly.

    Args:
        error_data: Dict mapping datetime.date -> float error.
        cycle: 'month' for monthly means (plotted at each month's start doy),
               'daily' for day-of-year means (1..365).
        interquantile_range: show 25–75 percentile band
        interdecile_range: show 10–90 percentile band
        std_range: show ±2 standard deviation band
        aggregation_func: NumPy function name like 'nanmean', 'nanmedian', etc.
    Returns:
        (fig, ax): the matplotlib figure and axes.
    """
    # Validate cycle
    if cycle not in {"month", "daily"}:
        raise ValueError("cycle must be either 'month' or 'daily'")

    # Prepare binning and x positions
    if cycle == "month":
        bins = list(range(1, 13))  # months
        xs = [date(2001, m, 1).timetuple().tm_yday for m in bins]
        default_title = "Monthly Error Cycle"
        default_xlabel = "Month"
        xticks = xs
        xtick_labels = [calendar.month_abbr[m] for m in bins]
    else:
        bins = list(range(1, 366))  # day-of-year
        xs = bins
        default_title = "Daily Error Cycle"
        default_xlabel = "Day of Year"
        xticks = [date(2001, m, 1).timetuple().tm_yday for m in range(1, 13)]
        xtick_labels = [calendar.month_abbr[m] for m in range(1, 13)]

    # Group errors into bins
    grouped: Dict[int, list] = {b: [] for b in bins}
    for d, err in error_data.items():
        b = d.month if cycle == "month" else d.timetuple().tm_yday
        if b in grouped:
            grouped[b].append(err)

    # Get aggregation function
    try:
        agg_func = getattr(np, aggregation_func)
    except AttributeError:
        raise ValueError(f"aggregation_func '{aggregation_func}' not found in numpy")

    # Compute aggregated values per bin
    agg_vals = [agg_func(grouped[b]) if grouped[b] else np.nan for b in bins]
    # Compute optional bands
    if interquantile_range:
        p25 = [np.nanpercentile(grouped[b], 25) if grouped[b] else np.nan for b in bins]
        p75 = [np.nanpercentile(grouped[b], 75) if grouped[b] else np.nan for b in bins]
    if interdecile_range:
        p10 = [np.nanpercentile(grouped[b], 10) if grouped[b] else np.nan for b in bins]
        p90 = [np.nanpercentile(grouped[b], 90) if grouped[b] else np.nan for b in bins]
    if std_range:
        std = [np.nanstd(grouped[b]) if grouped[b] else np.nan for b in bins]
        lower = [m - 2*s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(agg_vals, std)]
        upper = [m + 2*s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(agg_vals, std)]

    # Prepare figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot main curve at correct x positions
    ax.plot(xs, agg_vals, marker=marker, linestyle=linestyle,
            color=color, label=label or aggregation_func)

    # Shade percentile and std bands
    base_rgba = colors.to_rgba(color)
    if interdecile_range:
        ax.fill_between(xs, p10, p90, color=base_rgba, alpha=0.15, label="10–90 %")
    if interquantile_range:
        ax.fill_between(xs, p25, p75, color=base_rgba, alpha=0.25, label="25–75 %")
    if std_range:
        ax.fill_between(xs, lower, upper, color=base_rgba, alpha=0.2, label="±2 σ")

    # Formatting
    ax.set(
        title=title or default_title,
        xlabel=xlabel or default_xlabel,
        ylabel=ylabel
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=label_rotation, ha="right")
    if grid:
        ax.grid(alpha=0.3)
    if label or interquantile_range or interdecile_range or std_range:
        ax.legend()

    fig.tight_layout()
    return fig, ax

def polar_to_cartesian(norm: np.ndarray, angle_deg: np.ndarray, from_direction=True) -> tuple:
    """
    Converts polar coordinates (norm, angle) to Cartesian (u, v).
    
    Parameters:
        norm (np.ndarray): Magnitude of the vector (same shape as angle).
        angle_deg (np.ndarray): Angle in degrees.
        from_direction (bool): If True, assumes meteorological convention (angle is the direction FROM which the vector comes).
        
    Returns:
        tuple: u and v components of the vector.
    """
    angle_rad = np.radians(angle_deg)
    
    if from_direction:
        # Meteorological: wind FROM direction → invert angle
        angle_rad = np.radians(270 - angle_deg)
    
    u = norm * np.cos(angle_rad)
    v = norm * np.sin(angle_rad)
    return u, v

def cartesian_to_polar(u: np.ndarray, v: np.ndarray, to_direction=True) -> tuple:
    """
    Converts Cartesian vector components (u, v) to polar coordinates (norm, angle).
    
    Parameters:
        u (np.ndarray): Zonal (eastward) component.
        v (np.ndarray): Meridional (northward) component.
        to_direction (bool): If True, converts to meteorological direction (angle the vector comes FROM).
        
    Returns:
        tuple: 
            - norm (np.ndarray): Magnitude of the vector.
            - angle_deg (np.ndarray): Angle in degrees. 
                If to_direction=True, angle is meteorological "FROM" direction (0° = from north).
                If to_direction=False, angle is standard math angle (0° = east, counter-clockwise).
    """
    norm = np.sqrt(u**2 + v**2)
    
    # Get standard angle in radians: 0 = east, pi/2 = north, etc.
    angle_rad = np.arctan2(v, u)  # Range: [-π, π]
    angle_deg = np.degrees(angle_rad)  # Convert to degrees

    # Convert to [0, 360)
    angle_deg = (angle_deg + 360) % 360

    if to_direction:
        # Convert to meteorological FROM direction (0° = from north, clockwise)
        # u = norm * cos(theta), v = norm * sin(theta)
        # So the direction the vector comes FROM is 270 - angle
        angle_deg = (270 - angle_deg) % 360

    return norm, angle_deg

def visualize_full_vector_field(ax, grid, field, key_length=50, key_units='cm/s', key_color='black', 
                                from_polar=False, from_direction=True, step=32, use_pooling=True,
                                min_valid=5, scale=None, width=0.002, headwidth=3, headlength=5,
                                **kwargs):


    if from_polar:
        norm, angle = field
        u, v = polar_to_cartesian(norm, angle, from_direction=from_direction)
    else:
        u, v = field
        norm, angle = cartesian_to_polar(u, v, to_direction=from_direction)

    layer = visualize_scalar_field(ax, grid, norm, if_colorbar=False, **kwargs)
    visualize_vector_field(ax, grid, (u, v), key_length, key_units, key_color, False, from_direction, step,
                           use_pooling=use_pooling, min_valid=min_valid, scale=scale, width=width,
                           headwidth=headwidth, headlength=headlength)
    return layer

def plot_vector_field_scatter(errors, units='cm/s'):
    """
    Plots vector field errors with aligned marginal distributions and standard deviation circle.
    
    Parameters:
    errors (numpy.ndarray): Array of shape (N, 2) where each row is (u_error, v_error)
    """
    # Validate input shape
    if errors.shape[1] != 2:
        raise ValueError("Input array must have shape (N, 2)")
    
    # Calculate statistics
    mean_u, mean_v = np.mean(errors, axis=0)
    std_u, std_v = np.std(errors, axis=0)
    radius = np.sqrt(std_u**2 + std_v**2)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4)
    
    # Main scatter plot
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    
    # Plot error samples
    ax_scatter.scatter(errors[:, 0], errors[:, 1], s=10, alpha=0.5, color='blue')
    
    # Draw standard deviation circle
    circle = plt.Circle((0, 0), radius=radius, fill=False, color='red', 
                        linewidth=2, linestyle='-', label=f'Std Dev Circle (r={radius:.1f} {units})')
    ax_scatter.add_patch(circle)
    
    # Mark origin and mean
    ax_scatter.plot(0, 0, 'k+', markersize=12, label='No Bias (0,0)')
    ax_scatter.plot(mean_u, mean_v, 'ro', markersize=8, label=f'Mean Error ({mean_u:.1f}, {mean_v:.1f})')
    
    # Set axis limits (symmetric)
    max_val = max(15, np.max(np.abs(errors)) * 1.1, radius * 1.1)
    ax_scatter.set_xlim(-max_val, max_val)
    ax_scatter.set_ylim(-max_val, max_val)
    
    # Labels and grid
    ax_scatter.set_xlabel(f'Bias along U direction {units}', fontsize=12)
    ax_scatter.set_ylabel(f'Bias along V direction {units}', fontsize=12)
    ax_scatter.grid(True, linestyle='--', alpha=0.7)
    ax_scatter.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_scatter.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax_scatter.legend(loc='best')
    
    # Marginal histograms
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
    
    # U-error histogram (top)
    ax_histx.hist(errors[:, 0], bins=50, color='blue', alpha=0.7, density=True,
                  range=(-max_val, max_val))
    ax_histx.axvline(mean_u, color='red', linestyle='-', label=f'Mean = {mean_u:.1f}')
    ax_histx.axvline(mean_u - std_u, color='green', linestyle='--', label=f'±1σ = {std_u:.1f}')
    ax_histx.axvline(mean_u + std_u, color='green', linestyle='--')
    ax_histx.set_title('U-error Distribution', fontsize=10)
    ax_histx.legend(fontsize=8)
    ax_histx.set_yticks([])
    
    # V-error histogram (right)
    ax_histy.hist(errors[:, 1], bins=50, color='blue', alpha=0.7, 
                 orientation='horizontal', density=True,
                 range=(-max_val, max_val))
    ax_histy.axhline(mean_v, color='red', linestyle='-', label=f'Mean = {mean_v:.1f}')
    ax_histy.axhline(mean_v - std_v, color='green', linestyle='--', label=f'±1σ = {std_v:.1f}')
    ax_histy.axhline(mean_v + std_v, color='green', linestyle='--')
    ax_histy.set_title('V-error Distribution', fontsize=10)
    ax_histy.legend(fontsize=8)
    ax_histy.set_xticks([])
    
    # Remove tick labels from histograms to avoid duplication
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig, (ax_scatter, ax_histx, ax_histy)