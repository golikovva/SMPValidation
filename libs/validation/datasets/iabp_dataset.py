from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from pyproj import Geod


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

    years = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    doy = pd.to_numeric(out[time_source], errors="coerce")

    year_start = pd.to_datetime(years.astype(str) + "-01-01", format="%Y-%m-%d", errors="coerce")
    out["_time"] = year_start + pd.to_timedelta(doy - 1.0, unit="D")

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


def add_drift_uv_cm_s(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    time_col: str,
    u_col: str = "u_cm_s",
    v_col: str = "v_cm_s",
    speed_col: str | None = "speed_cm_s",
    sort_by_time: bool = True,
    geod: Geod = _WGS84_GEOD,
) -> pd.DataFrame:
    out = df.copy()

    # Parse datetime -> UTC -> make naive (so indexing is simple and consistent)
    t = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out[time_col] = t.dt.tz_convert(None)

    if sort_by_time:
        out = out.sort_values(time_col)

    lat1 = out[lat_col].to_numpy(dtype=float)
    lon1 = out[lon_col].to_numpy(dtype=float)
    lat2 = np.roll(lat1, -1)
    lon2 = np.roll(lon1, -1)

    t1 = out[time_col]
    t2 = t1.shift(-1)

    dt_s = (t2 - t1).dt.total_seconds().to_numpy(dtype=float)

    valid = (
        np.isfinite(lat1) & np.isfinite(lon1) &
        np.isfinite(lat2) & np.isfinite(lon2) &
        np.isfinite(dt_s) & (dt_s > 0)
    )
    valid[-1] = False

    u = np.full(len(out), np.nan, dtype=float)
    v = np.full(len(out), np.nan, dtype=float)

    if np.any(valid):
        az_deg, _, dist_m = geod.inv(lon1[valid], lat1[valid], lon2[valid], lat2[valid])
        az = np.deg2rad(az_deg)

        dx_e = dist_m * np.sin(az)
        dy_n = dist_m * np.cos(az)

        u[valid] = (dx_e / dt_s[valid]) * 100.0
        v[valid] = (dy_n / dt_s[valid]) * 100.0

    out[u_col] = u
    out[v_col] = v

    if speed_col is not None:
        out[speed_col] = np.hypot(out[u_col].to_numpy(), out[v_col].to_numpy())

    return out


def _to_timestamp(t: Union[np.datetime64, str, pd.Timestamp, "datetime.datetime"]) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _compute_valid_start_times(
    dt_index: pd.DatetimeIndex,
    ok_mask: np.ndarray,
    T: int,
) -> List[pd.Timestamp]:
    """
    Valid start times where:
      - all required fields are valid for T samples
      - timestamps are strictly hourly consecutive across the window
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    if len(ok_mask) != len(dt_index):
        raise ValueError("ok_mask length must match index length.")
    if len(ok_mask) < T:
        return []

    x = ok_mask.astype(np.int32)
    good_vals = np.convolve(x, np.ones(T, dtype=np.int32), mode="valid") == T

    if T == 1:
        good = good_vals
    else:
        dt64 = dt_index.to_numpy(dtype="datetime64[ns]")
        step_ok = (dt64[1:] - dt64[:-1]) == np.timedelta64(1, "h")
        step_sum = np.convolve(step_ok.astype(np.int32), np.ones(T - 1, dtype=np.int32), mode="valid")
        good_steps = step_sum == (T - 1)
        good = good_vals & good_steps  # same length (n-T+1)

    return [dt_index[i] for i, is_good in enumerate(good) if is_good]


@dataclass(frozen=True)
class BuoyTrack:
    buoy_id: str
    df: pd.DataFrame          # indexed by datetime, contains required columns
    valid_starts: frozenset   # set[pd.Timestamp]



@dataclass(frozen=True)
class BuoyWindow:
    bid: str
    coords: np.ndarray            # (T, 2) float32 [lat, lon]
    datetimes: np.ndarray         # (T,) datetime64[ns]
    variables: np.ndarray         # (T, V) float32
    var_names: Tuple[str, ...]    # (V,)

    def var(self, name: str) -> np.ndarray:
        """Return (T,) array for a variable by name."""
        try:
            j = self.var_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown variable {name!r}. Available: {list(self.var_names)}")
        return self.variables[:, j]


@dataclass(frozen=True)
class BuoyBatch:
    bids: np.ndarray              # (N,) dtype='<U' or object
    coords: np.ndarray            # (N, T, 2) float32
    datetimes: np.ndarray         # (T,) datetime64[ns] (same for all buoys in the batch)
    variables: np.ndarray         # (N, T, V) float32
    var_names: Tuple[str, ...]    # (V,)

    _bid_to_i: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        mapping = {str(b): i for i, b in enumerate(self.bids.tolist())}
        object.__setattr__(self, "_bid_to_i", mapping)

    def __len__(self) -> int:
        return int(self.bids.shape[0])

    def __getitem__(self, key: Union[int, str]) -> BuoyWindow:
        """batch[i] or batch['902002'] -> BuoyWindow"""
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0 or i >= len(self):
                raise IndexError(i)
            bid = str(self.bids[i])
        else:
            bid = str(key)
            if bid not in self._bid_to_i:
                raise KeyError(f"Buoy {bid!r} not in this batch. Available: {self.bids.tolist()[:10]} ...")
            i = self._bid_to_i[bid]

        return BuoyWindow(
            bid=bid,
            coords=self.coords[i],
            datetimes=self.datetimes,
            variables=self.variables[i],
            var_names=self.var_names,
        )

    def var(self, name: str) -> np.ndarray:
        """Return (N, T) array for a variable by name."""
        try:
            j = self.var_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown variable {name!r}. Available: {list(self.var_names)}")
        return self.variables[:, :, j]

    @classmethod
    def empty(cls, T: int, var_names: Tuple[str, ...]) -> "BuoyBatch":
        V = len(var_names)
        return cls(
            bids=np.asarray([], dtype=object),
            coords=np.empty((0, T, 2), dtype=np.float32),
            datetimes=np.empty((T,), dtype="datetime64[ns]"),
            variables=np.empty((0, T, V), dtype=np.float32),
            var_names=var_names,
        )

class BuoyLocationsDataset:
    """
    Dataset returning per-buoy windows of length T.

    __getitem__ returns:
        { buoy_id: BuoyWindow(coords=(T,2), datetimes=(T,), variables={...}) }

    If prepare=False (default), files are assumed to ALREADY contain:
        - datetime column
        - requested variables columns (if any)

    If prepare=True, raw IABP columns are accepted and we will:
        - interpolate/reindex to integer hours (nearest)
        - optionally compute drift u/v (+ optional speed) IF requested
    """

    def __init__(
        self,
        folder: Union[str, Path],
        T: int,
        *,
        pattern: str = "*.csv",
        variables: Optional[Sequence[str]] = ("u_cm_s", "v_cm_s"),
        prepare: bool = False,                    # default: assume already prepared
        compute_drift_if_needed: bool = True,     # only used when prepare=True
        datetime_col: str = "datetime",
        time_source: Literal["POS_DOY", "DOY"] = "POS_DOY",
        tolerance: Optional[str] = "45min",
        freq: str = "1h",
        id_col: str = "BuoyID",
        year_col: str = "Year",
        lat_col: str = "Lat",
        lon_col: str = "Lon",
        reader_kwargs: Optional[dict] = None,
        require_integer_hour_index: bool = True,
    ) -> None:
        self.folder = Path(folder)
        self.T = int(T)
        if self.T <= 0:
            raise ValueError("T must be a positive integer.")

        self.variables = tuple(variables) if variables is not None else tuple()

        self._tracks: Dict[str, BuoyTrack] = {}
        self._times: List[pd.Timestamp] = []

        files = sorted(self.folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {pattern!r} in {self.folder.resolve()}")

        reader_kwargs = dict(reader_kwargs or {})

        def _read_file(path: Path) -> pd.DataFrame:
            suf = path.suffix.lower()
            if suf == ".parquet":
                return pd.read_parquet(path)
            # csv vs whitespace
            if suf == ".csv":
                kw = dict(reader_kwargs)
                # allow user's sep override, otherwise comma
                kw.setdefault("sep", ",")
                return pd.read_csv(path, **kw)
            else:
                kw = dict(reader_kwargs)
                kw.setdefault("sep", r"\s+")
                kw.setdefault("comment", "#")
                kw.setdefault("engine", "python")
                return pd.read_csv(path, **kw)

        all_valid_times: set[pd.Timestamp] = set()

        for fname in files:
            df_raw = _read_file(fname)

            # Some files can contain multiple buoy IDs; support grouping anyway
            if prepare:
                # Need raw time columns
                for c in (id_col, year_col, time_source, lat_col, lon_col):
                    if c not in df_raw.columns:
                        raise ValueError(f"{fname.name}: missing column {c!r} required for prepare=True")

                df_work = interpolate_buoy_to_integer_hours_nearest(
                    df_raw,
                    time_source=time_source,
                    tolerance=tolerance,
                    freq=freq,
                    id_col=id_col,
                    year_col=year_col,
                    add_datetime_col=True,
                    overwrite_doy_cols=True,
                )

                if datetime_col not in df_work.columns:
                    raise RuntimeError(
                        f"{fname.name}: expected {datetime_col!r} after interpolation (add_datetime_col=True)."
                    )

                # Compute drift only if requested and enabled
                if compute_drift_if_needed and self.variables:
                    drift_names = {"u_cm_s", "v_cm_s", "speed_cm_s"}
                    drift_needed = any(v in drift_names for v in self.variables)
                    if drift_needed:
                        speed_col = "speed_cm_s" if ("speed_cm_s" in self.variables) else None
                        df_work = add_drift_uv_cm_s(
                            df_work,
                            lat_col=lat_col,
                            lon_col=lon_col,
                            time_col=datetime_col,
                            u_col="u_cm_s",
                            v_col="v_cm_s",
                            speed_col=speed_col,
                            sort_by_time=True,
                        )
            else:
                df_work = df_raw
                if datetime_col not in df_work.columns:
                    raise ValueError(
                        f"{fname.name}: missing {datetime_col!r}. "
                        f"Either run preparation first or set prepare=True."
                    )

            if id_col not in df_work.columns:
                raise ValueError(f"{fname.name}: missing id column {id_col!r}")

            # Parse datetime -> UTC -> naive (consistent indexing)
            dt_series = pd.to_datetime(df_work[datetime_col], utc=True, errors="coerce")
            df_work = df_work.copy()
            df_work[datetime_col] = dt_series.dt.tz_convert(None)

            for buoy_id, g in df_work.groupby(id_col, sort=False):
                g = g.copy()
                g = g.dropna(subset=[datetime_col]).sort_values(datetime_col).set_index(datetime_col)

                # Ensure unique timestamps
                if not g.index.is_unique:
                    g = g[~g.index.duplicated(keep="last")]

                if require_integer_hour_index and (g.index.minute != 0).any():
                    raise ValueError(
                        f"{fname.name} buoy {buoy_id}: non-integer-hour timestamps present."
                    )

                required_cols = [lat_col, lon_col] + list(self.variables)
                missing = [c for c in required_cols if c not in g.columns]
                if missing:
                    raise ValueError(f"{fname.name} buoy {buoy_id}: missing columns {missing}")

                g2 = g[required_cols].copy()

                ok = np.ones(len(g2), dtype=bool)
                for c in required_cols:
                    ok &= g2[c].notna().to_numpy()

                valid_starts = _compute_valid_start_times(g2.index, ok, self.T)
                if not valid_starts:
                    continue

                bid = str(buoy_id)
                self._tracks[bid] = BuoyTrack(buoy_id=bid, df=g2, valid_starts=frozenset(valid_starts))
                all_valid_times.update(valid_starts)

        self._times = sorted(all_valid_times)
        if not self._times:
            raise RuntimeError(
                "No valid (start_time, buoy) windows found. "
                "Try increasing tolerance, reducing T, or check files coverage."
            )

        self.lat_col = lat_col
        self.lon_col = lon_col

    @property
    def buoy_ids(self) -> List[str]:
        return sorted(self._tracks.keys())

    @property
    def times(self) -> List[pd.Timestamp]:
        return list(self._times)

    def __len__(self) -> int:
        return len(self._times)

    def __getitem__(self, idx: Union[int, np.integer, np.datetime64, str, pd.Timestamp]) -> BuoyBatch:
        if isinstance(idx, (int, np.integer)):
            t0 = self._times[int(idx)]
        else:
            t0 = _to_timestamp(idx)

        if t0.minute != 0 or t0.second != 0 or t0.microsecond != 0 or t0.nanosecond != 0:
            raise ValueError(f"Index time must be an integer hour, got {t0!s}")

        t1 = t0 + pd.Timedelta(hours=self.T - 1)

        bids: List[str] = []
        coords_list: List[np.ndarray] = []
        vars_list: List[np.ndarray] = []
        dt_arr: Optional[np.ndarray] = None

        var_names = tuple(self.variables)
        V = len(var_names)

        for bid, tr in self._tracks.items():
            if t0 not in tr.valid_starts:
                continue

            block = tr.df.loc[t0:t1]
            if len(block) != self.T:
                continue

            coords = block[[self.lat_col, self.lon_col]].to_numpy(dtype=np.float32)
            if np.isnan(coords).any():
                continue

            if V > 0:
                vv = block[list(var_names)].to_numpy(dtype=np.float32)  # (T, V)
                if np.isnan(vv).any():
                    continue
            else:
                vv = np.empty((self.T, 0), dtype=np.float32)

            if dt_arr is None:
                dt_arr = block.index.to_numpy(dtype="datetime64[ns]")

            bids.append(str(bid))
            coords_list.append(coords)
            vars_list.append(vv)

        if not bids:
            # Choose behavior: return empty batch (often easier for training loops)
            return BuoyBatch.empty(self.T, var_names)

        bids_arr = np.asarray(bids, dtype=object)
        coords_arr = np.stack(coords_list, axis=1)          # (T, N, 2)
        vars_arr = np.stack(vars_list, axis=-1)              # (T, V, N)

        return BuoyBatch(
            bids=bids_arr,
            coords=coords_arr,
            datetimes=dt_arr if dt_arr is not None else np.empty((self.T,), dtype="datetime64[ns]"),
            variables=vars_arr,
            var_names=var_names,
        )