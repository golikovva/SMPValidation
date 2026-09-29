from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

from pyproj import Geod, CRS, Transformer

from .iabp_dataset import BuoyLocationsDataset
try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except Exception:
    _HAS_KDTREE = False


# ------------------------------- helpers -------------------------------- #

import numpy as np

def _get_attr_or_key(obj, name: str):
    """Try obj[name] then getattr(obj, name). Works for dict, addict, xarray, custom classes."""
    # dict / addict / xarray Dataset-like
    try:
        return obj[name]
    except Exception:
        pass
    # attribute
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _as_numpy(a):
    if a is None:
        return None
    if hasattr(a, "values"):
        return np.asarray(a.values)
    return np.asarray(a)


def _extract_latlon_2d(dst_grid):
    """
    Robustly extract 2D lat/lon from:
      - dict/addict: grid.lat / grid.lon or grid['lat']/grid['lon'], also latitude/longitude, nav_lat/nav_lon
      - your Grid class: grid.lat / grid.lon properties
      - xarray Dataset/DataArray style: ds['lat'], ds['lon'] etc.

    Returns (lat2d, lon2d) as numpy arrays (float), both shape (H,W).
    """
    # Common name pairs to try in order
    candidates = [
        ("lat", "lon"),
        ("latitude", "longitude"),
        ("nav_lat", "nav_lon"),
    ]

    lat = lon = None
    for la, lo in candidates:
        lat = _get_attr_or_key(dst_grid, la)
        lon = _get_attr_or_key(dst_grid, lo)
        if lat is not None and lon is not None:
            break

    lat2d = _as_numpy(lat)
    lon2d = _as_numpy(lon)

    if lat2d is None or lon2d is None:
        raise ValueError(
            "Could not extract grid lat/lon. Expected one of:\n"
            "  - grid.lat & grid.lon (attributes) or grid['lat'], grid['lon']\n"
            "  - grid.latitude & grid.longitude\n"
            "  - grid.nav_lat & grid.nav_lon\n"
            "Got: "
            f"{type(dst_grid)=}"
        )

    if lat2d.ndim != 2 or lon2d.ndim != 2:
        raise ValueError(f"Grid lat/lon must be 2D arrays. Got {lat2d.ndim}D and {lon2d.ndim}D.")
    if lat2d.shape != lon2d.shape:
        raise ValueError(f"Grid lat/lon shapes differ: {lat2d.shape} vs {lon2d.shape}")

    return lat2d.astype(float), lon2d.astype(float)


from dataclasses import dataclass
from typing import Any, Tuple

try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except Exception:
    _HAS_KDTREE = False


@dataclass
class _GridIndexerLL:
    H: int
    W: int
    lat_flat: np.ndarray
    lon_flat: np.ndarray
    tree: Any  # cKDTree or None

    def ll_to_ij(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nearest grid cell in (lat, lon) space.

        Note: for Arctic / dateline issues, we use a wrapped lon distance metric by mapping
        lon to sin/cos embedding (below) rather than raw degrees.
        """
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)

        # Query points embedding: [lat, cos(lon), sin(lon)]
        lonr = np.deg2rad(lon)
        q = np.column_stack([lat, np.cos(lonr), np.sin(lonr)])

        if self.tree is not None:
            _, idx = self.tree.query(q, k=1)
        else:
            # brute force fallback
            d = self._embed(self.lat_flat, self.lon_flat)  # (HW,3)
            diff = d[None, :, :] - q[:, None, :]
            idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)

        i = (idx // self.W).astype(np.int64)
        j = (idx % self.W).astype(np.int64)
        return i, j

    @staticmethod
    def _embed(lat_flat: np.ndarray, lon_flat: np.ndarray) -> np.ndarray:
        lonr = np.deg2rad(lon_flat)
        return np.column_stack([lat_flat, np.cos(lonr), np.sin(lonr)])


def _build_grid_indexer_latlon(dst_grid: Any) -> _GridIndexerLL:
    """
    Build a nearest-neighbour indexer for a curvilinear lat/lon grid.
    Uses a longitude-wrapped embedding so buoys near +/-180 don't mismatch badly.
    """
    lat2d, lon2d = _extract_latlon_2d(dst_grid)

    H, W = lat2d.shape
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)

    tree = None
    if _HAS_KDTREE:
        emb = _GridIndexerLL._embed(lat_flat, lon_flat)  # (HW,3)
        tree = cKDTree(emb)

    return _GridIndexerLL(H=H, W=W, lat_flat=lat_flat, lon_flat=lon_flat, tree=tree)


class BuoyDriftSpeedDataset(BuoyLocationsDataset):
    """
    Returns a (T, H, W) grid of drift speed (or u/v) with NaNs everywhere except
    grid cells where buoys are located.

    Speed is computed from consecutive positions using geodesic distance on WGS84.
    Only computed when dt <= 1 hour (otherwise NaN for that step).
    """

    def __init__(
        self,
        *args,
        dst_grid: Any,
        component: str = "speed",   # "speed" / "u" / "v"
        units: str = "m/s",         # "m/s" / "cm/s"
        collision: str = "mean",    # "mean" / "last"
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._component = component
        self._units = units
        self._collision = collision

        self._geod = Geod(ellps="WGS84")
        self._grid_indexer = _build_grid_indexer_latlon(dst_grid)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self._grid_indexer.H, self._grid_indexer.W)

    def __getitem__(self, idx: Union[int, np.datetime64, str, pd.Timestamp]) -> np.ndarray:
        """
        Returns:
            grid: np.ndarray of shape (T, H, W) dtype float32, NaN except buoy cells.
        """
        # Get buoy locations from parent: {buoy_id: (T,2) [Lat,Lon]}
        coords_dict: Dict[str, np.ndarray] = super().__getitem__(idx)

        H, W = self._grid_indexer.H, self._grid_indexer.W
        T = self.T

        grid = np.full((T, H, W), np.nan, dtype=np.float32)
        if not coords_dict:
            return grid

        # Query start time (to allow dt check)
        if isinstance(idx, (int, np.integer)):
            t0 = self.times[int(idx)]
        else:
            t0 = pd.Timestamp(idx)
        time_axis = pd.date_range(t0, periods=T, freq="1h")  # integer-hour axis
        dt_seconds = np.diff(time_axis.view("i8")) / 1e9     # seconds between steps

        # Precompute per-time accumulators if collision='mean'
        if self._collision == "mean":
            acc = np.zeros((T, H * W), dtype=np.float64)
            cnt = np.zeros((T, H * W), dtype=np.int32)

        # For each buoy, compute drift from consecutive points
        for bid, ll in coords_dict.items():
            ll = np.asarray(ll)
            if ll.shape != (T, 2):
                continue

            lat = ll[:, 0].astype(float)
            lon = ll[:, 1].astype(float)

            # Geodesic inverse between consecutive points
            az, _, dist = self._geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])  # az in degrees, dist in meters

            # dt check (only if <= 1 hour)
            dt = dt_seconds.copy()
            ok = dt <= 3600.0 + 1e-6

            u = np.full(T, np.nan, dtype=np.float64)
            v = np.full(T, np.nan, dtype=np.float64)
            s = np.full(T, np.nan, dtype=np.float64)

            # u/v are placed at time k using displacement k->k+1 (so last timestep stays NaN)
            u[:-1][ok] = (dist[ok] * np.sin(np.deg2rad(az[ok]))) / dt[ok]
            v[:-1][ok] = (dist[ok] * np.cos(np.deg2rad(az[ok]))) / dt[ok]
            s[:-1][ok] = dist[ok] / dt[ok]

            if self._units == "cm/s":
                u *= 100.0
                v *= 100.0
                s *= 100.0

            if self._component == "u":
                val = u
            elif self._component == "v":
                val = v
            else:
                val = s

            # Map buoy positions (each time) onto grid indices
            # We place val[k] at the buoy cell of (lat[k], lon[k]).
            ii, jj = self._grid_indexer.ll_to_ij(lat, lon)
            flat = ii * W + jj

            # Write to grid with collision handling
            for k in range(T):
                vk = val[k]
                if np.isnan(vk):
                    continue
                fk = int(flat[k])
                if self._collision == "last":
                    grid[k].reshape(-1)[fk] = np.float32(vk)
                else:
                    acc[k, fk] += float(vk)
                    cnt[k, fk] += 1

        if self._collision == "mean":
            flat_grid = grid.reshape(T, -1).astype(np.float64)
            mask = cnt > 0
            flat_grid[mask] = acc[mask] / cnt[mask]
            grid = flat_grid.reshape(T, H, W).astype(np.float32)

        return grid
