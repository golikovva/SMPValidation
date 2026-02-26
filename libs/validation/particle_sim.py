from __future__ import annotations

from datetime import timedelta
from typing import Optional
from pprint import pprint

import numpy as np
import xarray as xr
import pandas as pd

from opendrift.models.seaicedrift import SeaIceDrift
from opendrift.readers import reader_netCDF_CF_generic
import netCDF4  # add at top


def _circular_mean_deg(lon_deg: np.ndarray) -> float:
    lon_deg = np.asarray(lon_deg, dtype=float)
    ang = np.deg2rad(lon_deg)
    return float(np.rad2deg(np.arctan2(np.nanmean(np.sin(ang)), np.nanmean(np.cos(ang)))))


def _shift_lon_like(lon: np.ndarray, lon_center: float) -> np.ndarray:
    # Map lon into (lon_center-180, lon_center+180]
    return ((lon - lon_center + 180.0) % 360.0) - 180.0 + lon_center


class ScaledCFReader(reader_netCDF_CF_generic.Reader):
    def __init__(self, *args, scale_uv: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_uv = float(scale_uv)

    def get_variables(self, requested_variables, time=None, x=None, y=None, z=None):
        data = super().get_variables(requested_variables, time=time, x=x, y=y, z=z)
        for key in ("sea_ice_x_velocity", "sea_ice_y_velocity"):
            if key in data and data[key] is not None:
                data[key] = data[key] * self.scale_uv
        return data


def emulate_itp_track(
    model_drift_file: str,
    start_coords: np.ndarray,
    u_var: str = "u_drift",
    v_var: str = "v_drift",
    lat_var: Optional[str] = None,
    lon_var: Optional[str] = None,
    time_var: Optional[str] = None,
    assume_velocity_units: str = "cm/s",
    loglevel: int = 40,
    disable_auto_landmask: bool = False,
) -> np.ndarray:
    # IMPORTANT: keep decode_times=False so we fully control time decoding
    ds = xr.open_dataset(model_drift_file, decode_times=False)

    # --- infer coord names ---
    if time_var is None:
        if "time" in ds.coords:
            time_var = "time"
        else:
            cands = [c for c in ds.coords if "time" in c.lower()]
            if not cands:
                raise ValueError("Cannot infer time coordinate. Please pass time_var=...")
            time_var = cands[0]

    if lat_var is None:
        for cand in ["latitude", "lat", "nav_lat", "XLAT", "xlat"]:
            if cand in ds.variables or cand in ds.coords:
                lat_var = cand
                break
    if lon_var is None:
        for cand in ["longitude", "lon", "nav_lon", "XLONG", "xlong"]:
            if cand in ds.variables or cand in ds.coords:
                lon_var = cand
                break

    if lat_var is None or lon_var is None:
        raise ValueError("Cannot infer lat/lon variable names. Please pass lat_var/lon_var explicitly.")
    if u_var not in ds or v_var not in ds:
        raise KeyError(f"u_var/v_var not found in dataset: u_var={u_var}, v_var={v_var}")

    # --- time axis (numbers in seconds since 1900-01-01 for your case) ---
    tvals = ds[time_var].values
    if tvals.size < 1:
        raise ValueError("Empty time dimension.")

    # Make a *naive* python datetime (avoid tz-aware vs naive mismatches later)
    t0 = pd.to_datetime(tvals[0], unit="s", origin="1900-01-01").to_pydatetime()

    if tvals.size >= 2:
        # safest: compute dt from numeric axis directly
        dt_seconds = int(np.round(float(tvals[1] - tvals[0])))
        if dt_seconds <= 0:
            raise ValueError("Non-positive time step inferred from time axis.")
    else:
        dt_seconds = 3600

    T = int(tvals.size)
    duration_seconds = max(0, (T - 1) * dt_seconds)

    # --- normalize start_coords & detect order ---
    start_coords = np.asarray(start_coords, dtype=float)
    if start_coords.ndim != 2 or start_coords.shape[1] != 2:
        raise ValueError("start_coords must have shape (n, 2).")

    c0 = start_coords[:, 0]
    c1 = start_coords[:, 1]
    if np.nanmax(np.abs(c0)) <= 90 and np.nanmax(np.abs(c1)) > 90:
        lats, lons = c0, c1
    elif np.nanmax(np.abs(c1)) <= 90 and np.nanmax(np.abs(c0)) > 90:
        lons, lats = c0, c1
    else:
        lats, lons = c0, c1

    lons = ((lons + 180.0) % 360.0) - 180.0
    n = int(start_coords.shape[0])

    # --- convert velocities to m/s if needed ---
    scale = 1.0
    if assume_velocity_units.strip().lower() in ["cm/s", "cm s-1", "cm/s."]:
        scale = 0.01

    # --- minimal dataset for the reader (NO copying of other vars) ---
    # Key fix: rename time_counter dim/coord -> "time" and set CF time units.
    ds_min = ds[[u_var, v_var, lat_var, lon_var, time_var]].rename(
        {lat_var: "lat", lon_var: "lon", time_var: "time"}
    )

    # Make sure u/v actually use the renamed time dim (xarray usually does, but enforce)
    if "time" not in ds_min[u_var].dims:
        # if rename didn't propagate, do it explicitly
        ds_min[u_var] = ds_min[u_var].rename({ds_min[u_var].dims[0]: "time"})
    if "time" not in ds_min[v_var].dims:
        ds_min[v_var] = ds_min[v_var].rename({ds_min[v_var].dims[0]: "time"})

    # CF metadata that reader_netCDF_CF_generic relies on to detect time
    ds_min["time"].attrs.update(
        {
            "standard_name": "time",
            "units": "seconds since 1900-01-01 00:00:00",
            "calendar": "gregorian",
        }
    )

    # Lon/lat metadata (good practice; not the cause of your crash)
    ds_min["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds_min["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})

    standard_name_mapping = {
        u_var: "sea_ice_x_velocity",
        v_var: "sea_ice_y_velocity",
        "lat": "latitude",
        "lon": "longitude",
    }

    reader = ScaledCFReader(
        ds_min,
        standard_name_mapping=standard_name_mapping,
        scale_uv=scale,
    )
    reader.checked_for_overlap = True

    o = SeaIceDrift(loglevel=loglevel)
    o.add_reader(reader)

    if disable_auto_landmask:
        o.set_config("general:use_auto_landmask", False)

    for i, (lon, lat) in enumerate(zip(lons, lats)):
        o.seed_elements(lon=float(lon), lat=float(lat), time=t0, number=1, origin_marker=i)

    # Optional but useful: don’t hard-crash if some particles get missing_data
    o.run(
        duration=timedelta(seconds=duration_seconds),
        time_step=dt_seconds,
        time_step_output=dt_seconds,
        outfile=None,
        stop_on_error=False,
    )

    res = o.result
    lon_hist = res["lon"].values
    lat_hist = res["lat"].values

    track = np.stack([lat_hist, lon_hist], axis=-1).transpose(1, 0, 2)

    # pad/trim to exactly T
    T_out = track.shape[0]
    if T_out > T:
        track = track[:T, :, :]
    elif T_out < T:
        pad = np.full((T - T_out, track.shape[1], 2), np.nan, dtype=track.dtype)
        track = np.concatenate([track, pad], axis=0)

    # enforce exactly n trajectories (in case some were not produced)
    if track.shape[1] < n:
        padn = np.full((track.shape[0], n - track.shape[1], 2), np.nan, dtype=track.dtype)
        track = np.concatenate([track, padn], axis=1)
    elif track.shape[1] > n:
        track = track[:, :n, :]

    return track
