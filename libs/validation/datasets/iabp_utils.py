from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Geod

from typing import Optional, Literal

_WGS84_GEOD = Geod(ellps="WGS84")


def interpolate_buoy_to_integer_hours_nearest(
    df: pd.DataFrame,
    *,
    time_source: Literal["POS_DOY", "DOY"] = "POS_DOY",
    id_col: str = "BuoyID",
    year_col: str = "Year",
    freq: str = "1h",
    tolerance: Optional[str] = None,
    add_datetime_col: bool = True,
    overwrite_doy_cols: bool = True,
) -> pd.DataFrame:
    required = {id_col, year_col, time_source}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    # --- Robust timestamp build (pandas Series, not NumPy) ---
    years = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    doy = pd.to_numeric(out[time_source], errors="coerce")

    year_start = pd.to_datetime(years.astype(str) + "-01-01", format="%Y-%m-%d", errors="coerce")
    out["_time"] = year_start + pd.to_timedelta(doy - 1.0, unit="D")
    # --------------------------------------------------------

    out = out.dropna(subset=["_time"]).sort_values([id_col, "_time"])

    tol = pd.Timedelta(tolerance) if tolerance is not None else None
    pieces = []

    for buoy_id, g in out.groupby(id_col, sort=False):
        g = g.copy().set_index("_time").sort_index()
        if not g.index.is_unique:
            g = g[~g.index.duplicated(keep="last")]
        if len(g) == 0:
            continue

        t0 = g.index.min().floor("h")
        t1 = g.index.max().ceil("h")
        target_index = pd.date_range(t0, t1, freq=freq)

        if tol is None:
            gi = g.reindex(target_index, method="nearest")
        else:
            gi = g.reindex(target_index, method="nearest", tolerance=tol)

        gi[id_col] = buoy_id
        gi["_time"] = gi.index
        pieces.append(gi.reset_index(drop=True))

    if not pieces:
        return out.drop(columns=["_time"])

    res = pd.concat(pieces, ignore_index=True)

    if overwrite_doy_cols:
        dt = pd.to_datetime(res["_time"])
        res[year_col] = dt.dt.year.astype(int)
        res["Hour"] = dt.dt.hour.astype(int)
        res["Min"] = dt.dt.minute.astype(int)

        frac = (dt.dt.hour * 60 + dt.dt.minute) / 1440.0
        doy_float = dt.dt.dayofyear.astype(float) + frac

        if "DOY" in res.columns:
            res["DOY"] = doy_float
        if "POS_DOY" in res.columns:
            res["POS_DOY"] = doy_float

    if add_datetime_col:
        res["datetime"] = pd.to_datetime(res["_time"])

    return res.drop(columns=["_time"])


def great_circle_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    *,
    geod: Geod = _WGS84_GEOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized wrapper around pyproj.Geod.inv.

    Returns:
        az12_deg: forward azimuth (deg) from point1 -> point2
        az21_deg: back azimuth (deg) from point2 -> point1
        dist_m:   geodesic distance (meters)
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    az12_deg, az21_deg, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return np.asarray(az12_deg), np.asarray(az21_deg), np.asarray(dist_m)


def drift_uv_cm_s_from_latlon(
    lat: np.ndarray,
    lon: np.ndarray,
    dt_s: float | np.ndarray,
    *,
    time_axis: int = 0,
    geod: Geod = _WGS84_GEOD,
    return_speed: bool = True,
    return_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Compute east/north drift velocity (cm/s) between consecutive fixes along `time_axis`.

    For each segment i -> i+1:
      az12_deg, _, dist_m = geod.inv(lon_i, lat_i, lon_{i+1}, lat_{i+1})
      dx_east_m  = dist_m * sin(az)
      dy_north_m = dist_m * cos(az)
      u = dx_east_m / dt_s * 100   (cm/s)
      v = dy_north_m / dt_s * 100

    Output arrays have the same shape as `lat`/`lon`, with NaNs on the last element
    along `time_axis` (no next fix). Invalid pairs (missing coords, dt<=0) -> NaNs.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon must have the same shape, got {lat.shape} vs {lon.shape}")

    # Move time axis to the last axis for easy slicing
    lat_t = np.moveaxis(lat, time_axis, -1)
    lon_t = np.moveaxis(lon, time_axis, -1)
    T = lat_t.shape[-1]

    u_t = np.full(lat_t.shape, np.nan, dtype=float)
    v_t = np.full(lat_t.shape, np.nan, dtype=float)
    speed_t = np.full(lat_t.shape, np.nan, dtype=float) if return_speed else None
    dist_t = np.full(lat_t.shape, np.nan, dtype=float) if return_distance else None

    if T < 2:
        # Nothing to compute
        u = np.moveaxis(u_t, -1, time_axis)
        v = np.moveaxis(v_t, -1, time_axis)
        speed = np.moveaxis(speed_t, -1, time_axis) if return_speed else None
        dist = np.moveaxis(dist_t, -1, time_axis) if return_distance else None
        return u, v, speed, dist

    lat1 = lat_t[..., :-1]
    lon1 = lon_t[..., :-1]
    lat2 = lat_t[..., 1:]
    lon2 = lon_t[..., 1:]

    # Normalize dt to per-segment array with shape broadcastable to lat1
    if np.isscalar(dt_s):
        dt_seg = float(dt_s)
        dt_seg = np.broadcast_to(dt_seg, lat1.shape).astype(float)
    else:
        dt_arr = np.asarray(dt_s, dtype=float)

        # If dt provided with same time-length T (per-fix), ignore last
        if dt_arr.ndim == 1 and dt_arr.shape[0] == T:
            dt_arr = dt_arr[:-1]
        # If dt is 1D per-segment
        if dt_arr.ndim == 1 and dt_arr.shape[0] == (T - 1):
            reshape = (1,) * (lat1.ndim - 1) + (T - 1,)
            dt_arr = dt_arr.reshape(reshape)

        # If dt matches full lat shape, align and drop last
        if dt_arr.shape == lat.shape:
            dt_arr_t = np.moveaxis(dt_arr, time_axis, -1)
            dt_arr = dt_arr_t[..., :-1]

        try:
            dt_seg = np.broadcast_to(dt_arr, lat1.shape).astype(float)
        except ValueError as e:
            raise ValueError(
                f"dt_s is not broadcastable to per-segment shape {lat1.shape}. "
                f"Got dt_s shape {np.shape(dt_s)}"
            ) from e

    valid = (
        np.isfinite(lat1) & np.isfinite(lon1) &
        np.isfinite(lat2) & np.isfinite(lon2) &
        np.isfinite(dt_seg) & (dt_seg > 0)
    )

    if np.any(valid):
        az12_deg, _, dist_m = great_circle_distance(lat1[valid], lon1[valid], lat2[valid], lon2[valid], geod=geod)
        az = np.deg2rad(az12_deg)

        dx_e = dist_m * np.sin(az)  # meters east
        dy_n = dist_m * np.cos(az)  # meters north

        u_seg = (dx_e / dt_seg[valid]) * 100.0
        v_seg = (dy_n / dt_seg[valid]) * 100.0

        u_t[..., :-1][valid] = u_seg
        v_t[..., :-1][valid] = v_seg

        if return_distance:
            dist_t[..., :-1][valid] = dist_m

        if return_speed:
            speed_t[..., :-1][valid] = np.hypot(u_seg, v_seg)

    # Move time axis back to original
    u = np.moveaxis(u_t, -1, time_axis)
    v = np.moveaxis(v_t, -1, time_axis)
    speed = np.moveaxis(speed_t, -1, time_axis) if return_speed else None
    dist = np.moveaxis(dist_t, -1, time_axis) if return_distance else None
    return u, v, speed, dist


def add_drift_uv_cm_s(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    time_col: str,
    u_col: str = "u_cm_s",
    v_col: str = "v_cm_s",
    speed_col: str | None = "speed_cm_s",      # set to None to skip
    distance_col: str | None = "distance_m",   # set to None to skip
    sort_by_time: bool = True,
    geod: Geod = _WGS84_GEOD,
) -> pd.DataFrame:
    """
    Add buoy drift velocity components (cm/s) between consecutive fixes, plus optional speed and distance.

    Behavior:
    - Converts time_col to UTC datetimes (errors -> NaT)
    - Optionally sorts by time_col
    - Computes dt per row as (t[i+1]-t[i]) in seconds
    - Fills last row with NaNs (no next fix); invalid dt/coords -> NaNs
    """
    out = df.copy()

    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    if sort_by_time:
        out = out.sort_values(time_col)

    t = out[time_col]
    dt_s = (t.shift(-1) - t).dt.total_seconds().to_numpy(dtype=float)  # length N, last is NaN

    lat = out[lat_col].to_numpy(dtype=float)
    lon = out[lon_col].to_numpy(dtype=float)

    u, v, speed, dist = drift_uv_cm_s_from_latlon(
        lat, lon, dt_s,
        time_axis=0,
        geod=geod,
        return_speed=(speed_col is not None),
        return_distance=(distance_col is not None),
    )

    out[u_col] = u
    out[v_col] = v

    if speed_col is not None and speed is not None:
        out[speed_col] = speed

    if distance_col is not None and dist is not None:
        out[distance_col] = dist

    return out


# def add_drift_uv_cm_s(
#     df: pd.DataFrame,
#     lat_col: str,
#     lon_col: str,
#     time_col: str,
#     u_col: str = "u_cm_s",
#     v_col: str = "v_cm_s",
#     speed_col: str | None = "speed_cm_s",  # set to None to skip
#     sort_by_time: bool = True,
#     geod: Geod = _WGS84_GEOD,
# ) -> pd.DataFrame:
#     """
#     Add buoy drift velocity components between consecutive fixes.

#     For each row i, compute displacement from (i) -> (i+1) using WGS84 geodesic:
#       azimuth_deg, _, dist_m = geod.inv(lon_i, lat_i, lon_{i+1}, lat_{i+1})

#     Convert to local east/north components:
#       dx_east_m  = dist_m * sin(az)
#       dy_north_m = dist_m * cos(az)

#     Then velocity (cm/s):
#       u = dx_east_m / dt_s * 100
#       v = dy_north_m / dt_s * 100
#       speed = hypot(u, v)   (optional)

#     Last row has NaNs (no next fix). Any invalid pair (missing data, dt<=0) -> NaNs.
#     """
#     out = df.copy()

#     # Ensure datetime (UTC); errors -> NaT
#     out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")

#     if sort_by_time:
#         out = out.sort_values(time_col)

#     lat1 = out[lat_col].to_numpy(dtype=float)
#     lon1 = out[lon_col].to_numpy(dtype=float)
#     lat2 = np.roll(lat1, -1)
#     lon2 = np.roll(lon1, -1)

#     t1 = out[time_col]
#     t2 = t1.shift(-1)

#     dt_s = (t2 - t1).dt.total_seconds().to_numpy(dtype=float)

#     valid = (
#         np.isfinite(lat1) & np.isfinite(lon1) &
#         np.isfinite(lat2) & np.isfinite(lon2) &
#         np.isfinite(dt_s) & (dt_s > 0)
#     )
#     valid[-1] = False  # last row has no next fix

#     u = np.full(len(out), np.nan, dtype=float)
#     v = np.full(len(out), np.nan, dtype=float)
#     distance = np.full(len(out), np.nan, dtype=float)

#     if np.any(valid):
#         az_deg, _, dist_m = geod.inv(lon1[valid], lat1[valid], lon2[valid], lat2[valid])
#         az = np.deg2rad(az_deg)

#         dx_e = dist_m * np.sin(az)  # east (m)
#         dy_n = dist_m * np.cos(az)  # north (m)
#         print(dist_m.shape, az.shape)
#         print(f"dx_e: {dx_e.shape}, dy_n: {dy_n.shape}, dt_s: {dt_s[valid].shape}")

#         u[valid] = (dx_e / dt_s[valid]) * 100.0  # cm/s
#         v[valid] = (dy_n / dt_s[valid]) * 100.0
#         distance[valid] = dist_m

#     out[u_col] = u
#     out[v_col] = v
#     out['distance_m'] = distance

#     if speed_col is not None:
#         out[speed_col] = np.hypot(out[u_col].to_numpy(), out[v_col].to_numpy())

#     return out