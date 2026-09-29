# Add "import cartopy" to the top of your Jupyter notebook,
# before using these functions, or visualizations will fail.

import cartopy.crs as ccrs  # For cartographic projections in visualizations
import cartopy.feature as cfeature  # For adding geographic features (land, oceans, etc.)
import numpy as np  # For numerical computations
from matplotlib import pyplot as plt  # For plotting
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from collections.abc import Mapping

import datetime
from datetime import date
from typing import Dict, Optional
import calendar
import warnings


try:
    from cartopy.mpl.geoaxes import GeoAxes
except Exception:
    GeoAxes = None


def get_domain_projection(domain_name):
    name = domain_name.lower()
    if 'borey' in name:
        return ccrs.LambertAzimuthalEqualArea(
            central_longitude=80.0,
            central_latitude=71.0
        )
    elif 'pan' in name or 'arctic' in name:
        return ccrs.NorthPolarStereo(central_longitude=120.0)
    elif 'smp' in name or 'nestp' in name:
        return ccrs.NorthPolarStereo(central_longitude=120.0)
    elif any(k in name for k in ['glorys', 'global', 'world']):
        return ccrs.Robinson(central_longitude=120.0)
    else:
        return None

def get_domain_extent(domain_name):
    name = domain_name.lower()

    if 'borey' in name:
        return [-1850799.028266253, -169147.24810465102,
                 -390064.97663009784, 881589.4853213852]
    elif 'pan' in name or 'arctic' in name:
        return None   # todo fill when ready
    elif 'smp' in name or 'nestp' in name:
        return [-3178711.951944511, 3178711.952368256,
                -2252806.5461317636, 239082.4972454039]
    elif any(k in name for k in ['glorys', 'global', 'world']):
        return [-180, 180, -90, 90]
    else:
        return None

def set_domain_extent(ax, domain, grid=None):
    src = ccrs.PlateCarree()

    if isinstance(domain, str):
        domain_name = domain
        domain_proj = get_domain_projection(domain_name)
        if domain_proj is None:
            raise ValueError(f'Unknown domain name: {domain_name}')
    else:
        domain_name = None
        domain_proj = domain

    if grid is None:
        if domain_name is None:
            raise ValueError(
                "When grid is None, `domain` must be a string domain name "
                "so that extent can be looked up."
            )
        extent_proj = get_domain_extent(domain_name)
        if extent_proj is None:
            raise ValueError(f'No predefined extent for domain: {domain_name}')
    else:
        lat2d, lon2d = lat_lon_from_grid(grid)

        xy = domain_proj.transform_points(src, lon2d, lat2d)
        x = xy[..., 0]
        y = xy[..., 1]

        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            raise ValueError("No finite projected coordinates found for grid")

        extent_proj = [
            np.nanmin(x[mask]),
            np.nanmax(x[mask]),
            np.nanmin(y[mask]),
            np.nanmax(y[mask]),
        ]

    ax.set_extent(extent_proj, crs=domain_proj)

def lat_lon_from_grid(grid):
    lat_names = ['lat', 'latitude', 'XLAT', 'Latitude']
    lon_names = ['lon', 'long', 'longitude', 'XLON', 'XLONG', 'Longitude']

    def _looks_like_array(x):
        # Reject addict auto-created empties / dicts
        if isinstance(x, Mapping):
            return False
        # Accept numpy arrays and array-like objects
        return isinstance(x, np.ndarray) or hasattr(x, "shape")

    def _get(obj, names):
        # 1) Mapping path (safe for addict.Dict)
        if isinstance(obj, Mapping):
            for name in names:
                if name in obj:              # does NOT create in addict
                    val = obj[name]
                    if _looks_like_array(val):
                        return val

        # 2) Attribute path (for non-mapping grid objects)
        for name in names:
            try:
                val = getattr(obj, name)
            except AttributeError:
                continue
            if _looks_like_array(val):
                return val

        return None

    lat = _get(grid, lat_names)
    lon = _get(grid, lon_names)
    return lat, lon


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


def _outer_ring(lon, lat):
    """Counter-clockwise outer ring of a 2D lon/lat grid."""
    top    = np.c_[lon[0, :],          lat[0, :]]
    right  = np.c_[lon[1:, -1],        lat[1:, -1]]
    bottom = np.c_[lon[-1, -2::-1],    lat[-1, -2::-1]]
    left   = np.c_[lon[-2:0:-1, 0],    lat[-2:0:-1, 0]]
    return np.vstack([top, right, bottom, left])


def _tight_projected_limits(lon, lat, target_crs, src_crs=ccrs.PlateCarree()):
    """Compute tight (xmin,xmax,ymin,ymax) in target_crs for a lon/lat grid."""
    ring = _outer_ring(lon, lat)
    xy   = target_crs.transform_points(src_crs, ring[:, 0], ring[:, 1])
    x, y = xy[:, 0], xy[:, 1]
    return np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)


def iter_axes(x):
    if x is None:
        return
    # leaf: Axes-like
    if hasattr(x, "figure"):
        yield x
        return

    if isinstance(x, np.ndarray):
        for v in x.flat:
            yield from iter_axes(v)
        return

    if isinstance(x, (list, tuple)):
        for v in x:
            yield from iter_axes(v)
        return

    raise TypeError(f"Unsupported ax container type: {type(x)!r}")


def map_axes(x, fn):
    """Recursively map axes container -> same structure with transformed axes."""
    if x is None:
        return None
    if hasattr(x, "figure"):  # scalar axis
        return fn(x)
    if isinstance(x, np.ndarray):
        out = np.empty_like(x, dtype=object)
        for idx, v in np.ndenumerate(x):
            out[idx] = map_axes(v, fn)
        return out
    if isinstance(x, list):
        return [map_axes(v, fn) for v in x]
    if isinstance(x, tuple):
        return tuple(map_axes(v, fn) for v in x)
    raise TypeError(f"Unsupported ax container type: {type(x)!r}")


def create_cartopy_axes(
    nrows: int = 1,
    ncols: int = 1,
    *,
    coastline_resolution: str = '110m',
    central_longitude: float = 120.0,
    figsize=None,
    ax_size: float = 6.0,
    grid=None,
    add_land: bool = True,
    face_ocean: bool = True,
    add_gridlines: bool = True,
    add_coastlines: bool = True,
    proj=None,
    ax=None,  # NEW
):
    """
    If ax is None: creates new (fig, axes).
    If ax is provided: converts/replaces provided axes into Cartopy GeoAxes (if needed) and styles them.
    Works with ax being a scalar, list/tuple (nested), or np.ndarray of any ndim.
    """

    if proj is None:
        proj = ccrs.NorthPolarStereo(central_longitude=central_longitude)
    if isinstance(proj, str):
        proj = get_domain_projection(proj)
    def _is_geoaxes(a):
        if a is None:
            return False
        if GeoAxes is not None and isinstance(a, GeoAxes):
            return True
        # fallback: Cartopy GeoAxes has `.projection` (CRS), plain mpl Axes обычно нет
        return hasattr(a, "projection")

    def _ensure_geoaxes(fig, old_ax):
        """Return GeoAxes. If old_ax is not GeoAxes -> replace it in-place (remove + add) preserving location."""
        if old_ax is None:
            return None
        if _is_geoaxes(old_ax):
            return old_ax

        # same figure check
        if old_ax.figure is not fig:
            raise ValueError("All provided axes must belong to the same figure.")

        # prefer SubplotSpec (keeps gridspec layout perfectly)
        try:
            ss = old_ax.get_subplotspec()
        except Exception:
            ss = None

        if ss is not None:
            old_ax.remove()
            return fig.add_subplot(ss, projection=proj)

        # fallback: absolute position
        pos = old_ax.get_position()
        old_ax.remove()
        return fig.add_axes(pos, projection=proj)

    def _style_geoax(a):
        if a is None:
            return a

        if face_ocean:
            a.set_facecolor(cfeature.COLORS["water"])

        if add_land:
            land = cfeature.NaturalEarthFeature(
                "physical", "land", coastline_resolution,
                edgecolor=None,
                facecolor=cfeature.COLORS["land"],
            )
            a.add_feature(land, zorder=0)

        if add_gridlines:
            gl = a.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
            # gl.ylocator = mticker.FixedLocator(np.arange(-85, 90, 10))
            # позиционирование labels пока не трогаем
        
        if add_coastlines:
            a.coastlines(resolution=coastline_resolution, linewidth=0.5, color='black', zorder=8)

        return a

    def _first_axis(x):
        return next(iter_axes(x), None)

    # 1) Create axes if not provided
    if ax is None:
        if figsize is None:
            aspect = 1.0
            if grid is not None and (nrows == 1 and ncols == 1):
                # uses YOUR implementation; must exist in scope
                lat, lon = lat_lon_from_grid(grid)
                if lat is not None and lon is not None and "_tight_projected_limits" in globals():
                    xmin, xmax, ymin, ymax = _tight_projected_limits(np.asarray(lon), np.asarray(lat), proj)
                    if ymax > ymin:
                        aspect = (xmax - xmin) / (ymax - ymin)

            figsize = (ax_size * ncols * aspect, ax_size * nrows)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize,
            subplot_kw={"projection": proj},
            squeeze=False,
        )
        ax_in = axes[0, 0] if (nrows == 1 and ncols == 1) else axes

    else:
        # 2) Use provided axes
        first = _first_axis(ax)
        if first is None:
            raise ValueError("Provided ax contains no valid Axes objects.")
        fig = first.figure
        ax_in = ax

    # 3) Convert (if needed) + style recursively for all axes
    def _convert_and_style(a):
        a2 = _ensure_geoaxes(fig, a)
        return _style_geoax(a2)

    ax_out = map_axes(ax_in, _convert_and_style)

    return fig, ax_out


def create_cartopy(coastline_resolution='110m', figsize=(12, 12), fig=None, ax=None, central_longitude=45.0, proj=None, **kwargs):
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
    if proj is None:
        proj =  ccrs.NorthPolarStereo(central_longitude=central_longitude)
    if fig is None:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={'projection': proj},  # North Polar projection
            **kwargs
        )
    else:
        ax = fig.add_subplot(ax, projection=proj)

    ax.set_facecolor(cfeature.COLORS['water'])
    
    # Add land features with adjustable resolution
    land = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale=coastline_resolution,
        edgecolor='none',#cfeature.COLORS['land'],
        facecolor=cfeature.COLORS['land']
    )
    ax.add_feature(land, zorder=1)  
    ax.coastlines()
    
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
                edgecolor='none',
                facecolor=cfeature.COLORS['land']
            )
            ax.add_feature(land, zorder=1)
            
            # Add gridlines
            ax.gridlines(draw_labels=True, color='gray', zorder=9)
    
    plt.tight_layout()
    return fig, axes


def _maybe_datetime64_1d(x):
    x = np.asarray(x)
    if x.dtype == object and x.size:
        a0 = x.flat[0]
        if isinstance(a0, (datetime.date, datetime.datetime, np.datetime64)):
            return np.array([np.datetime64(t) for t in x.ravel()]).reshape(x.shape)
    return x

def _mapping_to_traj_times(traj_map, *, times):
    # sort keys for deterministic time axis
    keys = sorted(traj_map.keys())
    if times is None:
        times = np.array([np.datetime64(k) for k in keys])

    vals = [traj_map[k] for k in keys]

    # infer N from first non-None entry
    first = next((v for v in vals if v is not None), None)
    if first is None:
        raise ValueError("All mapping values are None; nothing to plot.")

    a0 = np.asarray(first)
    if a0.shape[-1] != 2:
        raise ValueError(f"Mapping values must have last dim 2, got {a0.shape}")

    N = int(np.prod(a0.shape[:-1]))
    T = len(keys)

    out = np.full((N, T, 2), np.nan, dtype=float)

    for t, v in enumerate(vals):
        if v is None:
            continue
        a = np.asarray(v, dtype=float)
        if a.shape[-1] != 2:
            raise ValueError(f"Mapping value at {keys[t]!r} must have last dim 2, got {a.shape}")
        a2 = a.reshape(-1, 2)
        if a2.shape[0] != N:
            raise ValueError(
                f"Inconsistent N at {keys[t]!r}: expected {N} points, got {a2.shape[0]}"
            )
        out[:, t, :] = a2

    return out, times


def visualize_trajectory(
    ax,
    traj,
    *,
    time_first=False,
    coord_order="latlon",          # "latlon" or "lonlat"
    times=None,                    # None or array-like length T (numeric or datetime64)
    cmap="viridis",
    linewidth=1.5,
    alpha=1,
    zorder=10,
    transform=None,                # defaults to ccrs.Geodetic()
    unwrap_longitudes=True,        # helps dateline crossings
    add_colorbar=False,
    cbar_label=None,
    cbar_kwargs=None,
    start_end_markers=False,
    marker_kwargs=None,
    fast=False,                    # if True: uses LineCollection in PlateCarree (faster, not geodesic)
):
    """
    Plot N trajectories (lat/lon) on a Cartopy axis, coloring by time along the path.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axis created with a Cartopy projection (e.g., NorthPolarStereo).
    traj : array-like, shape (N, T, 2)
        Trajectories. By default expects (..., [lat, lon]).
    coord_order : {"latlon", "lonlat"}
        Order of coordinates in traj[..., 0:2].
    times : None or array-like length T
        If provided, colors by times (numeric or numpy datetime64). Otherwise uses time index.
    transform : cartopy.crs.CRS
        How to interpret the input lon/lat. For geodesic curves, use ccrs.Geodetic().
    fast : bool
        If True, uses a LineCollection in PlateCarree (fast but not true geodesic segments).
    """

    if isinstance(traj, Mapping):
        traj, times = _mapping_to_traj_times(traj, times=times)
    # Make your existing times handling also accept python date/datetime
    if times is not None:
        times = _maybe_datetime64_1d(times)

    traj = np.asarray(traj)
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"traj must have shape (N, T, 2), got {traj.shape}")
    if time_first:
        traj = np.swapaxes(traj, 0, 1)
    N, T, _ = traj.shape
    if T < 2:
        raise ValueError("Trajectories must have T >= 2 to draw segments.")

    if coord_order not in ("latlon", "lonlat"):
        raise ValueError("coord_order must be 'latlon' or 'lonlat'")

    if transform is None:
        transform = ccrs.Geodetic()

    if times is None:
        # color per-segment by index 0..T-2
        seg_values = np.arange(T - 1, dtype=float)
        is_datetime = False
    else:
        times = np.asarray(times)
        if times.shape[0] != T:
            raise ValueError(f"times must have length T={T}, got {times.shape[0]}")
        is_datetime = np.issubdtype(times.dtype, np.datetime64)

        if is_datetime:
            # seconds since epoch as int64 for normalization
            # tnum = times.astype("datetime64[s]").astype("int64")
            tnum = mdates.date2num(times.astype("datetime64[ms]").astype(object))
            # per-segment midpoint time for nicer gradients
            seg_values = 0.5 * (tnum[:-1] + tnum[1:])
            vmin, vmax = tnum[0], tnum[-1]
        else:
            tnum = times.astype(float)
            seg_values = 0.5 * (tnum[:-1] + tnum[1:])
            vmin, vmax = np.nanmin(tnum), np.nanmax(tnum)

    vmin = np.nanmin(seg_values)
    vmax = np.nanmax(seg_values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # fallback if something degenerate happens
        vmin, vmax = 0.0, 1.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    # Accurate geodesic mode: draw each segment with ax.plot(..., transform=ccrs.Geodetic())
    def _unwrap_lon_1d(lon_1d):
        lon_rad = np.deg2rad(lon_1d)
        return np.rad2deg(np.unwrap(lon_rad, discont=np.deg2rad(180)))

    lines = []
    for i in range(N):
        if coord_order == "latlon":
            lat = traj[i, :, 0].astype(float)
            lon = traj[i, :, 1].astype(float)
        else:
            lon = traj[i, :, 0].astype(float)
            lat = traj[i, :, 1].astype(float)

        if unwrap_longitudes:
            valid = np.isfinite(lon) & np.isfinite(lat)
            lon2 = lon.copy()
            if valid.any():
                idx = np.where(valid)[0]
                lon2[idx] = _unwrap_lon_1d(lon[idx])
            lon = lon2

        # Plot each valid segment with its own color
        for t in range(T - 1):
            if not (np.isfinite(lon[t]) and np.isfinite(lat[t]) and np.isfinite(lon[t + 1]) and np.isfinite(lat[t + 1])):
                continue
            color = cmap_obj(norm(seg_values[t]))
            ln = ax.plot(
                [lon[t], lon[t + 1]],
                [lat[t], lat[t + 1]],
                transform=transform,     # Geodetic by default
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
            lines.extend(ln)

        if start_end_markers:
            _mk = {} if marker_kwargs is None else dict(marker_kwargs)
            if np.isfinite(lon[0]) and np.isfinite(lat[0]):
                ax.plot([lon[0]], [lat[0]], marker="o", transform=transform, zorder=zorder + 1, **_mk)
            if np.isfinite(lon[-1]) and np.isfinite(lat[-1]):
                ax.plot([lon[-1]], [lat[-1]], marker="s", transform=transform, zorder=zorder + 1, **_mk)

    if add_colorbar:
        _cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)
        label = cbar_label if cbar_label is not None else ("time" if times is not None else "t index")

        cbar = ax.figure.colorbar(sm, ax=ax, **_cbar_kwargs)
        cbar.set_label(label)

        if is_datetime:
            loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
            cbar.locator = loc
            cbar.formatter = mdates.DateFormatter("%Y-%m-%d")
            cbar.update_ticks()

            # optional: make labels readable
            ax_ = cbar.ax.xaxis if cbar.orientation == "horizontal" else cbar.ax.yaxis
            for lab in ax_.get_ticklabels():
                lab.set_rotation(30)
                lab.set_ha("right")
        # If times are datetime64, you can optionally pass your own ticks/formatter via cbar_kwargs.
        # Keeping it simple here because we normalized in epoch-seconds.

    return {"mappable": sm, "lines": lines}

def true_ndim(a) -> int:
    a = np.asarray(a)
    shp = a.shape
    return int(np.count_nonzero(np.array(shp) > 1))

def visualize_scalar_field(ax, grid, field, if_colorbar=False, lat=None, lon=None, **kwargs):
    """
    Visualizes a scalar field on the map using a color mesh.

    Args:
        ax (matplotlib.axes._axes.Axes): Axis to plot on.
        grid (Grid): Grid object containing lat/lon information.
        field (np.array): Scalar field to visualize.
        vmin (float, optional): Minimum value for color scale. Defaults to None.
        vmax (float, optional): Maximum value for color scale. Defaults to None.
    """
    if lat is None and lon is None:
        lat, lon = lat_lon_from_grid(grid)
    # lat = lat if lat is not None else grid.latitude
    # lon = lon if lon is not None else grid.longitude
    
    if field.ndim >= 3 and true_ndim(field) == 2:
        field = np.squeeze(field)
    assert field.ndim == 2, "Field must be 2D after squeezing"

    # Create a colored mesh plot of the scalar field, projected using Plate Carree
    layer = ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        alpha=None,
        **kwargs\
    )
    if if_colorbar:
        plt.colorbar(layer)
    return layer

def plot_countour(ax, grid, field, levels=None, lat=None, lon=None, if_label=False, colors=None, **kwargs):
    """
    Visualizes a scalar field on the map using a color mesh.

    Args:
        ax (matplotlib.axes._axes.Axes): Axis to plot on.
        grid (Grid): Grid object containing lat/lon information.
        field (np.array): Scalar field to visualize.
        vmin (float, optional): Minimum value for color scale. Defaults to None.
        vmax (float, optional): Maximum value for color scale. Defaults to None.
    """
    lat = lat if lat is not None else grid.latitude
    lon = lon if lon is not None else grid.longitude
    
    if field.ndim >= 3 and true_ndim(field) == 2:
        field = np.squeeze(field)
    assert field.ndim == 2, "Field must be 2D after squeezing"

    # Create a colored mesh plot of the scalar field, projected using Plate Carree
    fieldm = np.ma.masked_invalid(field).copy()
    jump = np.abs(np.diff(lon, axis=1)) > 180  # dateline crossings between columns

    mask = np.zeros(fieldm.shape, dtype=bool)
    mask[:, 1:] |= jump
    mask[:, :-1] |= jump
    fieldm = np.ma.masked_where(mask, fieldm)

    line_c = ax.contour(
        lon,
        lat,
        fieldm,
        transform=ccrs.PlateCarree(),
        alpha=None,
        levels=levels,
        colors=colors,
        transform_first=False,
        **kwargs
    )
    if if_label: # works bad for my case, want to move it to legend
        ax.clabel(
            line_c,  # Typically best results when labelling line contours.
            colors=colors,
            manual=False,  # Automatic placement vs manual placement.
            inline=True,  # Cut the line where the label will be placed.
            fmt='{:.0f}'.format,
            fontsize=8.0,
            )
    return line_c

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


def visualize_vector_field(ax, grid, field, key_length=50, draw_quiverkey=True, key_units='cm/s', key_color='black', 
                           from_polar=False, from_direction=True, step=64, use_pooling=True, min_valid=5,
                           scale=None, width=0.002, headwidth=3, headlength=5, **quiver_kwargs):
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
            headlength=headlength,
            **quiver_kwargs
        )
    else:
        layer = ax.quiver(
            grid.lon[::step, ::step], grid.lat[::step, ::step],
            u_fixed[::step, ::step], v_fixed[::step, ::step],
            transform=ccrs.PlateCarree(), color=key_color,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
            **quiver_kwargs
        )
    if draw_quiverkey:
        ax.quiverkey(layer, X=0.69, Y=0.2, U=key_length, label=f'{key_length} {key_units}',
                    labelpos='E', coordinates='axes')
    return layer

def plot_barbs(
    ax, grid, field,
    draw_barbkey=True,
    key_units="cm/s",
    key_color="black",
    key_X=0.69, key_Y=0.2,
    key_w=0.28, key_h=0.08,
    from_polar=False,
    from_direction=True,
    step=64,
    use_pooling=True,
    min_valid=5,
    length=6,
    **barb_kwargs
):
    if from_polar:
        norm, angle = field
        u, v = polar_to_cartesian(norm, angle, from_direction=from_direction)
    else:
        u, v = field

    u_fixed, v_fixed = fix_quiver_bug((u, v), grid.lat)
    h, w = grid.lon.shape

    if use_pooling and step > 1:
        u_p = block_average(u_fixed, step, min_valid)
        v_p = block_average(v_fixed, step, min_valid)

        h2 = (h // step) * step
        w2 = (w // step) * step
        i_centers = (np.arange(step // 2, h2, step)).astype(int)
        j_centers = (np.arange(step // 2, w2, step)).astype(int)

        lon_p = grid.lon[i_centers[:, None], j_centers[None, :]]
        lat_p = grid.lat[i_centers[:, None], j_centers[None, :]]

        mask = (~np.isnan(u_p) & ~np.isnan(v_p))
        x = lon_p[mask]
        y = lat_p[mask]
        uu = u_p[mask]
        vv = v_p[mask]
    else:
        x = grid.lon[::step, ::step].ravel()
        y = grid.lat[::step, ::step].ravel()
        uu = u_fixed[::step, ::step].ravel()
        vv = v_fixed[::step, ::step].ravel()

        mask = (~np.isnan(uu) & ~np.isnan(vv))
        x, y, uu, vv = x[mask], y[mask], uu[mask], vv[mask]

    barb_kwargs = dict(barb_kwargs)
    barb_kwargs.setdefault("barbcolor", key_color)
    barb_kwargs.setdefault("flagcolor", key_color)

    layer = ax.barbs(
        x, y, uu, vv,
        transform=ccrs.PlateCarree(),
        length=length,
        **barb_kwargs
    )

    if draw_barbkey:
        key_ax = ax.inset_axes([key_X, key_Y, key_w, key_h], transform=ax.transAxes)
        key_ax.set_axis_off()
        key_ax.set_xlim(0, 1)
        key_ax.set_ylim(0, 1)

        # Use a clean kwargs dict for the key (avoid duplicate barbcolor/flagcolor)
        key_kwargs = dict(barb_kwargs)
        key_kwargs.pop("transform", None)  # not meaningful for inset axis

        key_ax.barbs(
            [0.15]*3, [0.7, 0.5, 0.3],
            [50, 10, 5], [0.0]*3,
            length=length,
            **key_kwargs
        )
        key_ax.text(
            0.28, 0.5, f"50 {key_units}\n10 {key_units}\n5 {key_units}",
            va="center", ha="left",
            color=key_color,
            transform=key_ax.transAxes,
        )

        layer._barbkey_ax = key_ax

    return layer


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
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    color: str = "tab:blue",
    figsize: tuple = (12, 6),
    grid: bool = True,
    marker: str = None,
    date_format: str = "%Y-%m-%dT%H",
    date_step: str = "1D",
    label_rotation: int = 45,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    **plot_kwargs,
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
    
    if isinstance(start_date, (datetime.datetime, datetime.date, np.datetime64)):
        dates = np.arange(start_date, end_date + np.timedelta64(1, date_step), np.timedelta64(1, date_step)).astype(type(start_date))
    elif isinstance(start_date, (int, float)):
        dates = np.arange(start_date, end_date + 1, 1)
    else:
        raise ValueError("error_data keys must be datetime-like for date handling")

    # dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    # dates = list(error_data.keys())
    errors=[]
    for d in dates:
        value = error_data.get(d, np.array([np.nan]))
        if isinstance(value, dict) and 'sum' in value and 'count' in value:
            value = value['sum'] / value['count']
        try:
            value = value.squeeze().item() # Convert to scalar if possible
        except (AttributeError, ValueError):
            value = None
        errors.append(value)

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        new_plot = True
    else:
        fig = ax.figure
        new_plot = False

    # Plot the data
    ax.plot(dates, errors, marker=marker, 
            color=color, label=label, **plot_kwargs)
    if label:
        ax.legend()
    # Only configure axis properties for new plots
    # if new_plot:
    # Configure date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    
    # Rotate and align labels
    plt.setp(ax.get_xticklabels(), rotation=label_rotation, ha="right")

    # Set labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    # ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    
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

def visualize_full_vector_field(ax, grid, field, key_length=50, key_units='cm/s', key_color='black', draw_quiverkey=True,
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
    visualize_vector_field(ax, grid, (u, v), key_length, draw_quiverkey, key_units, key_color, False, from_direction, step,
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


def pad_extent_km(extent_proj, pad_km):
    xmin, xmax, ymin, ymax = map(float, extent_proj)

    pad_km = np.asarray(pad_km, dtype=float)
    if pad_km.size == 1:
        pad_x_km = pad_y_km = float(pad_km)
    elif pad_km.size == 2:
        pad_x_km, pad_y_km = map(float, pad_km)
    else:
        raise ValueError("pad_km must be scalar or (pad_x_km, pad_y_km).")

    pad_x_m = pad_x_km * 1000.0
    pad_y_m = pad_y_km * 1000.0
    return [xmin - pad_x_m, xmax + pad_x_m, ymin - pad_y_m, ymax + pad_y_m]


def broaden_extent_from_ax_km(ax, pad_km, *, target_crs=None):
    """
    Broaden the *current* GeoAxes extent by pad_km in target_crs coordinates.

    - If target_crs is None, uses ax.projection.
    - target_crs should be a projected CRS with meter units for "km" to be literal.
    """
    if target_crs is None:
        target_crs = ax.projection

    # extent in target_crs coordinates (meters for most projected CRSs)
    extent = ax.get_extent(crs=target_crs)   # (xmin, xmax, ymin, ymax)
    extent2 = pad_extent_km(extent, pad_km)
    ax.set_extent(extent2, crs=target_crs)
    return extent2


def _agg_per_valid_time(valid_dt, err, *, agg="nanmedian"):
    """
    Aggregate err at identical valid_dt timestamps.
    Returns sorted unique times + aggregated values.
    """
    if valid_dt.size == 0:
        return valid_dt, err

    # group by exact timestamp (ns)
    order = np.argsort(valid_dt)
    t = valid_dt[order]
    e = err[order]

    # unique groups
    uniq, idx_start, counts = np.unique(t, return_index=True, return_counts=True)

    # aggregate
    agg_fn = getattr(np, agg)
    out = np.full(uniq.shape, np.nan, dtype=float)

    for i, (s, c) in enumerate(zip(idx_start, counts)):
        out[i] = agg_fn(e[s:s+c])

    return uniq, out

def _as_date(d):
    """Coerce python date/datetime or numpy datetime64 -> python date."""
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, np.datetime64):
        return d.astype("datetime64[D]").astype(object)  # -> datetime.date
    raise TypeError(f"Unsupported key type: {type(d)!r}")

def _as_datetime(d, *, init_hour=0):
    """Key is usually a date; interpret forecast init at init_hour:00."""
    dd = _as_date(d)
    return datetime.datetime.combine(dd, datetime.time(hour=int(init_hour)))

def flatten_forecast_error_dict(
    error_by_init,
    *,
    init_hour=0,
    dt_hours=1,
    lead_max=None,
    drop_first_lead0=False,
):
    """
    Parameters
    ----------
    error_by_init : dict[date -> array(T,1) or (T,)]
        Forecast error vs lead time for each initialization date.
    init_hour : int
        Hour of initialization for each date key (commonly 0, 6, 12, 18).
    dt_hours : int/float
        Lead-time step in hours between consecutive elements in the array.
    lead_max : int/None
        If set, keep only leads <= lead_max (hours).
    drop_first_lead0 : bool
        If True, drop lead=0 point (often trivially 0).

    Returns
    -------
    valid_dt : np.ndarray[datetime64[ns]] shape (N,)
    lead_h   : np.ndarray[float] shape (N,)
    err      : np.ndarray[float] shape (N,)
    init_dt  : np.ndarray[datetime64[ns]] shape (N,)
    """
    valid_list, lead_list, err_list, init_list = [], [], [], []

    for k, v in error_by_init.items():
        if v is None:
            continue
        init_py = _as_datetime(k, init_hour=init_hour)
        init64 = np.datetime64(init_py, "ns")

        a = np.asarray(v, dtype=float).reshape(-1)  # (T,)
        T = a.size
        if T == 0:
            continue

        lead = np.arange(T, dtype=float) * float(dt_hours)

        if drop_first_lead0 and T > 0:
            a = a[1:]
            lead = lead[1:]

        if lead_max is not None:
            mask = lead <= float(lead_max)
            a = a[mask]
            lead = lead[mask]

        # valid time = init + lead
        # (use seconds to avoid fractional-hour datetime issues)
        valid64 = init64 + (lead * 3600.0).astype("timedelta64[s]").astype("timedelta64[ns]")

        valid_list.append(valid64)
        lead_list.append(lead)
        err_list.append(a)
        init_list.append(np.full(a.shape, init64, dtype="datetime64[ns]"))

    if not valid_list:
        return (
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype="datetime64[ns]"),
        )

    valid_dt = np.concatenate(valid_list)
    lead_h   = np.concatenate(lead_list)
    err      = np.concatenate(err_list)
    init_dt  = np.concatenate(init_list)

    return valid_dt, lead_h, err, init_dt

def plot_overlapping_forecast_errors(
    error_by_init,
    *,
    init_hour=0,
    dt_hours=1,
    lead_max=None,
    mode="spaghetti",          # "spaghetti" or "scatter_by_lead"
    overlay_agg=False,
    agg="nanmedian",           # "nanmedian" or "nanmean"
    overlay_kwargs=None,
    alpha=0.20,
    linewidth=1.0,
    linestyle='-',
    s=10,
    cmap="viridis",
    show_colorbar=True,
    colorbar_label="Lead time (hours)",
    ax=None,
    title=None,
    ylabel="Error",
):
    valid_dt, lead_h, err, init_dt = flatten_forecast_error_dict(
        error_by_init,
        init_hour=init_hour,
        dt_hours=dt_hours,
        lead_max=lead_max,
    )

    m = np.isfinite(err) & np.isfinite(lead_h)
    valid_dt, lead_h, err, init_dt = valid_dt[m], lead_h[m], err[m], init_dt[m]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    if valid_dt.size == 0:
        ax.set_ylabel("Error (m)")
        ax.set_title(title or "Forecast error vs valid time (overlapping horizons)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig, ax

    # one shared normalization for ALL lines / points
    norm = Normalize(vmin=np.nanmin(lead_h), vmax=np.nanmax(lead_h))

    if mode == "spaghetti":
        inits = np.unique(init_dt)

        for it in inits:
            mm = init_dt == it
            tt = valid_dt[mm]
            ee = err[mm]
            ll = lead_h[mm]

            if tt.size < 2:
                continue

            o = np.argsort(tt)
            tt = tt[o]
            ee = ee[o]
            ll = ll[o]

            x = mdates.date2num(tt.astype("datetime64[ms]").astype(object))

            points = np.column_stack([x, ee]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # color each segment by midpoint lead time
            seg_lead = 0.5 * (ll[:-1] + ll[1:])

            lc = LineCollection(
                segments,
                cmap=cmap,
                norm=norm,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )
            lc.set_array(seg_lead)
            ax.add_collection(lc)

        ax.autoscale_view()

        if show_colorbar:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.01)
            cbar.set_label(colorbar_label)

    elif mode == "scatter_by_lead":
        sc = ax.scatter(
            valid_dt.astype("datetime64[ms]").astype(object),
            err,
            c=lead_h,
            s=s,
            alpha=alpha,
            cmap=cmap,
            norm=norm,
            linewidths=0,
        )
        if show_colorbar:
            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label(colorbar_label)

    else:
        raise ValueError("mode must be 'spaghetti' or 'scatter_by_lead'")

    if overlay_agg and valid_dt.size:
        t_uniq, e_agg = _agg_per_valid_time(valid_dt, err, agg=agg)
        ok = np.isfinite(e_agg)
        _okw = dict(color="black", linewidth=2.5, alpha=0.9)
        if overlay_kwargs:
            _okw.update(overlay_kwargs)
        ax.plot(
            t_uniq[ok].astype("datetime64[ms]").astype(object),
            e_agg[ok],
            **_okw,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title or "Forecast error vs valid time (overlapping horizons)")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_ha("center")

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


def visualize_scatter(
    ax,
    coords,
    values=None,
    *,
    order='latlon',
    if_colorbar=False,
    colorbar_kwargs=None,
    **kwargs,
):
    """
    Plot scattered geographic points on a Cartopy axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Target Cartopy axis.
    coords : np.ndarray
        Array of shape (N, 2), where columns are either [lat, lon] or [lon, lat].
    values : np.ndarray or None
        Optional array of shape (N,) used for point coloring via `c=...`.
    order : str
        'latlon' or 'lonlat'.
    if_colorbar : bool
        Whether to add a colorbar.
    colorbar_kwargs : dict or None
        Extra kwargs passed to plt.colorbar.
    **kwargs
        Extra kwargs passed to ax.scatter (e.g. s, cmap, vmin, vmax, alpha, marker).

    Returns
    -------
    layer : matplotlib.collections.PathCollection
        Scatter plot handle.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"`coords` must have shape (N, 2), got {coords.shape}")

    if order == 'latlon':
        lat = coords[:, 0]
        lon = coords[:, 1]
    elif order == 'lonlat':
        lon = coords[:, 0]
        lat = coords[:, 1]
    else:
        raise ValueError("`order` must be either 'latlon' or 'lonlat'")

    mask = np.isfinite(lat) & np.isfinite(lon)

    c = values
    if values is not None:
        values = np.asarray(values)
        if values.shape[0] != coords.shape[0]:
            raise ValueError(
                f"`values` must have length {coords.shape[0]}, got {values.shape[0]}"
            )
        if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
            mask &= np.isfinite(values)
        c = values[mask]

    layer = ax.scatter(
        lon[mask],
        lat[mask],
        c=c,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )

    if if_colorbar and values is not None:
        cb_kwargs = {} if colorbar_kwargs is None else dict(colorbar_kwargs)
        plt.colorbar(layer, ax=ax, **cb_kwargs)

    return layer