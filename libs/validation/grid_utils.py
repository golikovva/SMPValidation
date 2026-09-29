import numpy as np
from collections.abc import Mapping


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


def lat_lon_to_2d(lat, lon):
    if lat is None or lon is None:
        raise ValueError("Latitude or longitude is None.")

    lat = np.asarray(lat).squeeze()
    lon = np.asarray(lon).squeeze()

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise ValueError(
                f"2D lat/lon shapes do not match: lat={lat.shape}, lon={lon.shape}"
            )
        lat2d, lon2d = lat, lon
    else:
        raise ValueError(
            "lat/lon must be either both 1D or both 2D after squeeze(). "
            f"Got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )

    return lat2d, lon2d


def grid_lat_lon_2d(grid):
    """
    Uses lat_lon_from_grid(grid) helper and returns 2D lat/lon arrays.
    """
    lat, lon = lat_lon_from_grid(grid)
    lat2d, lon2d = lat_lon_to_2d(lat, lon)
    return lat2d, lon2d
