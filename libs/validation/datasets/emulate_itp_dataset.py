from __future__ import annotations

from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from torch.utils.data import Dataset

# import your function (adjust import path to your project)
# from particle_sim import emulate_itp_track
from libs.validation.particle_sim import emulate_itp_track


DateLike = Union[int, np.integer, np.datetime64, pd.Timestamp, "datetime.date", str]


def _to_np_datetime64_h(t: Any) -> np.datetime64:
    """Convert many date-like objects to np.datetime64[h]."""
    if isinstance(t, (int, np.integer)):
        raise TypeError("Integer indices are handled separately.")
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    # enforce integer hour
    if ts.minute != 0 or ts.second != 0 or ts.microsecond != 0 or ts.nanosecond != 0:
        raise ValueError(f"Time must be an integer hour, got {ts!s}")
    return np.datetime64(ts.to_datetime64(), "h")


def _default_parse_date_from_name(p: Path) -> np.datetime64:
    """
    Default filename parser:
      something_like_*_YYYY-MM-DD.nc  -> np.datetime64('YYYY-MM-DD')
    """
    token = p.stem.split("_")[1]
    date = datetime.datetime.strptime(token, 'y%Ym%md%d').date()
    return np.datetime64(date).astype("datetime64[D]")


def _infer_time_lat_lon_vars(ds: xr.Dataset) -> tuple[str, str, str]:
    # time
    if "time" in ds.coords:
        time_var = "time"
    else:
        cands = [c for c in ds.coords if "time" in c.lower()]
        if not cands:
            # also allow dims
            cands = [d for d in ds.dims if "time" in d.lower()]
        if not cands:
            raise ValueError("Cannot infer time coordinate/dimension; pass time_var=...")
        time_var = cands[0]

    # lat
    lat_var = None
    for cand in ["latitude", "lat", "nav_lat", "XLAT", "xlat"]:
        if cand in ds.variables or cand in ds.coords:
            lat_var = cand
            break

    # lon
    lon_var = None
    for cand in ["longitude", "lon", "nav_lon", "XLONG", "xlong"]:
        if cand in ds.variables or cand in ds.coords:
            lon_var = cand
            break

    if lat_var is None or lon_var is None:
        raise ValueError("Cannot infer lat/lon variable names; pass lat_var/lon_var explicitly.")
    return time_var, lat_var, lon_var


def _decode_time_axis_to_datetime64(
    ds: xr.Dataset,
    time_var: str,
) -> np.ndarray:
    """
    Robust decode for decode_times=False datasets using netCDF4.num2date when possible,
    with a fallback to your common case (seconds since 1900-01-01).
    """
    tvals = ds[time_var].values
    if tvals.size == 0:
        return np.empty((0,), dtype="datetime64[ns]")

    units = ds[time_var].attrs.get("units", None)
    cal = ds[time_var].attrs.get("calendar", "gregorian")

    # If units exist, prefer netCDF4 decode
    if isinstance(units, str) and ("since" in units):
        try:
            dts = netCDF4.num2date(
                tvals,
                units=units,
                calendar=cal,
                only_use_cftime_datetimes=False,
            )
            # convert python datetimes -> datetime64[ns]
            out = np.asarray([np.datetime64(pd.Timestamp(d).to_datetime64(), "ns") for d in dts])
            return out
        except Exception:
            pass

    # Fallback: your common case
    t0 = pd.to_datetime(tvals[0], unit="s", origin="1900-01-01")
    if tvals.size >= 2:
        dt_seconds = int(np.round(float(tvals[1] - tvals[0])))
        if dt_seconds <= 0:
            dt_seconds = 3600
    else:
        dt_seconds = 3600

    T = int(tvals.size)
    return (t0 + pd.to_timedelta(np.arange(T) * dt_seconds, unit="s")).to_numpy(dtype="datetime64[ns]")


def _extract_start_coords_from_buoy_batch(batch: Any) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract (N,2) start coords [lat, lon] and optional bids from a BuoyBatch-like object.

    Handles both conventions:
      - coords shape (N, T, 2)
      - coords shape (T, N, 2)   (common when stacking axis got swapped)
    """
    if batch is None:
        return None, None

    coords = np.asarray(batch.coords)
    bids = getattr(batch, "bids", None)
    N = int(len(bids)) if bids is not None else None

    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords with shape (*,*,2), got {coords.shape}")

    if N is not None and coords.shape[0] == N:
        # (N,T,2)
        start = coords[:, 0, :]
    elif N is not None and coords.shape[1] == N:
        # (T,N,2)
        start = coords[0, :, :]
    else:
        # best-effort: assume first axis is N
        start = coords[:, 0, :]

    start = np.asarray(start, dtype=float)
    bids_arr = np.asarray(bids, dtype=object) if bids is not None else None
    return start, bids_arr


@dataclass(frozen=True)
class EmulatedTrackSample:
    track: np.ndarray                 # (N,T,2) or (T,N,2) depending on track_order
    datetimes: Optional[np.ndarray]   # (T,) datetime64[ns] if requested
    start_coords: np.ndarray          # (N,2) [lat, lon] used for seeding
    bids: Optional[np.ndarray]        # (N,) buoy ids if from buoy dataset
    file: str                         # path to model drift file
    date_key: np.datetime64           # file key date (e.g., datetime64[D])


class EmulatedITPTrackDataset(Dataset):
    """
    Dataset that emulates Ice-Tethered Buoy tracks by integrating sea-ice drift
    from model NetCDF files using OpenDrift (via emulate_itp_track).

    Indexing:
      - int -> i-th available date (sorted)
      - np.datetime64 / datetime.date / pd.Timestamp / str -> resolves to date_key (default: datetime64[D])

    Start coords sources (priority):
      1) start_coords passed to __getitem__
      2) self.start_coords from init
      3) buoy_locations[time] first position per buoy
    """

    def __init__(
        self,
        folder: Union[str, Path],
        *,
        pattern: str = "*.nc",
        parse_date: Optional[Callable[[Path], np.datetime64]] = None,
        file_len: str = "D",  # key granularity: "D" matches your TopazOpDataset style
        # drift vars in the model files
        u_var: str = "u_drift",
        v_var: str = "v_drift",
        lat_var: Optional[str] = None,
        lon_var: Optional[str] = None,
        time_var: Optional[str] = None,
        assume_velocity_units: str = "cm/s",
        # start coords options
        start_coords: Optional[np.ndarray] = None,      # (N,2) [lat,lon] or [lon,lat] (emulate_itp_track auto-detects)
        buoy_locations: Optional[Any] = None,           # BuoyLocationsDataset-like
        # outputs
        track_order: str = "NT2",                       # "NT2" -> (N,T,2), "TN2" -> (T,N,2)
        return_datetimes: bool = True,
        fast_mode: bool = False,
        # OpenDrift configs
        loglevel: int = 40,
        disable_auto_landmask: bool = False,
        name: str = "emulated_itp_tracks",
    ) -> None:
        super().__init__()
        self.name = name
        self.folder = Path(folder)
        self.pattern = pattern
        self.file_len = str(file_len)

        self.u_var = str(u_var)
        self.v_var = str(v_var)
        self.lat_var = lat_var
        self.lon_var = lon_var
        self.time_var = time_var
        self.assume_velocity_units = str(assume_velocity_units)

        self.start_coords = None if start_coords is None else np.asarray(start_coords, dtype=float)
        self.buoy_locations = buoy_locations

        self.track_order = track_order.upper()
        if self.track_order not in {"NT2", "TN2"}:
            raise ValueError("track_order must be 'NT2' or 'TN2'.")

        self.return_datetimes = bool(return_datetimes)
        self.fast_mode = bool(fast_mode)

        self.loglevel = int(loglevel)
        self.disable_auto_landmask = bool(disable_auto_landmask)

        parser = parse_date or _default_parse_date_from_name

        files = sorted(self.folder.glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {self.pattern!r} in {self.folder.resolve()}")

        # map date_key -> list[Path] (like your NCs2sDataset)
        self.dates_dict: Dict[np.datetime64, list[Path]] = {}
        for fp in files:
            try:
                d = parser(fp).astype(f"datetime64[{self.file_len}]")
            except Exception:
                # print(f"Warning: Failed to parse date from filename {fp.name!r} using provided parser; trying fallback methods.")
                # fallback: attempt reading first time from file
                try:
                    with xr.open_dataset(fp, decode_times=False) as ds:
                        tv, _, _ = _infer_time_lat_lon_vars(ds)
                        dt = _decode_time_axis_to_datetime64(ds, tv)
                        d = dt[0].astype(f"datetime64[{self.file_len}]")
                except Exception:
                    # skip silently (or raise if you prefer)
                    continue

            self.dates_dict.setdefault(d, []).append(fp)

        self._dates = sorted(self.dates_dict.keys())
        if not self._dates:
            raise RuntimeError("No parsable dates found in files. Provide parse_date=...")

    @property
    def dates(self) -> list[np.datetime64]:
        return list(self._dates)

    def __len__(self) -> int:
        return len(self._dates)

    def _resolve_date_key(self, idx: DateLike) -> np.datetime64:
        if isinstance(idx, (int, np.integer)):
            return self._dates[int(idx)]
        dt_h = _to_np_datetime64_h(idx)
        return dt_h.astype(f"datetime64[{self.file_len}]")

    def _resolve_file(self, date_key: np.datetime64) -> Path:
        fps = self.dates_dict.get(date_key, None)
        if not fps:
            raise KeyError(f"No file for date_key={date_key} in {self.folder}")
        return fps[0]

    def _resolve_start_coords_and_bids(
        self,
        time_for_buoys: Any,
        start_coords: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if start_coords is not None:
            sc = np.asarray(start_coords, dtype=float)
            if sc.ndim != 2 or sc.shape[1] != 2:
                raise ValueError("start_coords must have shape (N,2).")
            return sc, None

        if self.start_coords is not None:
            sc = np.asarray(self.start_coords, dtype=float)
            if sc.ndim != 2 or sc.shape[1] != 2:
                raise ValueError("init start_coords must have shape (N,2).")
            return sc, None

        if self.buoy_locations is None:
            raise ValueError(
                "No start coords provided. Pass start_coords=... in init or __getitem__, "
                "or provide buoy_locations=BuoyLocationsDataset(...)."
            )

        batch = self.buoy_locations[time_for_buoys]
        if batch is None:
            return None, None

        sc, bids = _extract_start_coords_from_buoy_batch(batch)
        if sc is None or sc.size == 0:
            return None, bids
        return sc, bids

    def __getitem__(
        self,
        idx: DateLike,
        *,
        start_coords: Optional[np.ndarray] = None,
        buoy_time: Optional[Any] = None,
        # allow per-call overrides
        loglevel: Optional[int] = None,
        disable_auto_landmask: Optional[bool] = None,
    ) -> Union[EmulatedTrackSample, Dict[str, Any], np.ndarray, None]:
        """
        idx: int or datetime-like (must be integer-hour if using buoy_locations)
        start_coords: optional per-call override (N,2)
        buoy_time: if provided, used to query buoy_locations; else uses idx itself
        """
        date_key = self._resolve_date_key(idx)
        fp = self._resolve_file(date_key)

        time_for_buoys = buoy_time if buoy_time is not None else idx
        sc, bids = self._resolve_start_coords_and_bids(time_for_buoys, start_coords)
        if sc is None:
            return None

        # Run emulation (returns track as (T,N,2) [lat, lon])
        track_tn2 = emulate_itp_track(
            model_drift_file=str(fp),
            start_coords=sc,
            u_var=self.u_var,
            v_var=self.v_var,
            lat_var=self.lat_var,
            lon_var=self.lon_var,
            time_var=self.time_var,
            assume_velocity_units=self.assume_velocity_units,
            loglevel=int(self.loglevel if loglevel is None else loglevel),
            disable_auto_landmask=bool(self.disable_auto_landmask if disable_auto_landmask is None else disable_auto_landmask),
        )

        if track_tn2 is None:
            return None

        # reorder if needed
        if self.track_order == "NT2":
            track = np.asarray(track_tn2).transpose(1, 0, 2)  # (N,T,2)
        else:
            track = np.asarray(track_tn2)  # (T,N,2)

        if self.fast_mode:
            return track

        dts = None
        if self.return_datetimes:
            with xr.open_dataset(fp, decode_times=False) as ds:
                tv, _, _ = _infer_time_lat_lon_vars(ds) if self.time_var is None else (self.time_var, None, None)
                dts = _decode_time_axis_to_datetime64(ds, tv)

        sample = EmulatedTrackSample(
            track=track,
            datetimes=dts,
            start_coords=np.asarray(sc, dtype=float),
            bids=bids,
            file=str(fp),
            date_key=date_key,
        )

        # return as dict (often nicer in training code), but you can keep dataclass if you prefer
        return {
            "track": sample.track,
            "datetimes": sample.datetimes,
            "start_coords": sample.start_coords,
            "bids": sample.bids,
            "file": sample.file,
            "date_key": sample.date_key,
        }