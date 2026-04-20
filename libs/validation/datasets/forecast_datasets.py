from __future__ import annotations

from typing import NamedTuple, Optional, Dict, List, Tuple, Any
from datetime import datetime
import warnings

from abc import abstractmethod
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import netCDF4
import wrf
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset


def atleast_nd(arr, n):
    """
    inplace operator to expand array dims up to n dimensional
    """
    arr.shape = (1,) * (n - arr.ndim) + arr.shape
    return arr


def lon_to_180_range(lon):
    return (lon + 180) % 360 - 180


@dataclass
class ForecastRunRecord:
    init_time: np.datetime64
    files: list[Path]
    lead_times_h: np.ndarray
    valid_times: np.ndarray

@dataclass(frozen=True)
class ForecastRef:
    init_time: np.datetime64
    lead_h: int


class ForecastDatasetBase(Dataset):
    def __init__(
        self,
        data_folder,
        data_variables=None,
        expected_init_step_h=None,
        expected_lead_step_h=None,
        expected_max_lead_h=None,
        add_coords=False,
        add_time_encoding=False,
        strict=False,
    ):
        super().__init__()
        self.path = Path(data_folder)
        self.data_variables = data_variables
        self.expected_init_step_h = expected_init_step_h
        self.expected_lead_step_h = expected_lead_step_h
        self.expected_max_lead_h = expected_max_lead_h
        self.add_coords = add_coords
        self.add_time_encoding = add_time_encoding
        self.strict = strict

        self.constant_vars = {}
        self.runs_dict = self._create_runs_dict()
        self.init_times = np.array(sorted(self.runs_dict.keys()))
        self.valid_index = self._create_valid_index()

        self.src_grid = self._create_grid()
        self.src_grid["longitude"] = lon_to_180_range(self.src_grid["longitude"])

    @abstractmethod
    def _create_grid(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _parse_init_time(file):
        raise NotImplementedError

    @abstractmethod
    def _read_lead_axis_h(self, files) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _load_run_slice(self, files, lead_ids):
        raise NotImplementedError

    @property
    @abstractmethod
    def _files_template(self):
        raise NotImplementedError

    def _create_runs_dict(self):
        grouped = {}
        for file in sorted(self.path.glob(self._files_template)):
            init_time = self._parse_init_time(file)
            grouped.setdefault(init_time, []).append(file)

        runs = {}
        for init_time, files in grouped.items():
            lead_times_h = self._read_lead_axis_h(files)
            if lead_times_h is None:
                continue
            valid_times = init_time + lead_times_h.astype("timedelta64[h]")
            runs[init_time] = ForecastRunRecord(
                init_time=init_time,
                files=files,
                lead_times_h=lead_times_h,
                valid_times=valid_times,
            )
        return runs

    def _create_valid_index(self):
        index = {}
        for init_time, run in self.runs_dict.items():
            for lead_h in run.lead_times_h:
                valid_time = init_time + np.timedelta64(int(lead_h), "h")
                index.setdefault(valid_time, []).append(
                    ForecastRef(init_time=init_time, lead_h=int(lead_h))
                )
        return index

    def get_by_init_and_lead(self, init_time, lead_h):
        run = self.runs_dict.get(init_time)
        if run is None:
            return None
        ids = np.where(run.lead_times_h == lead_h)[0]
        if len(ids) == 0:
            return None
        return self._load_run_slice(run.files, ids)[0]

    def get_run(self, init_time, leads_h=None):
        run = self.runs_dict.get(init_time)
        if run is None:
            return None
        if leads_h is None:
            lead_ids = np.arange(len(run.lead_times_h))
        else:
            lead_ids = [np.where(run.lead_times_h == h)[0][0] for h in leads_h]
        return self._load_run_slice(run.files, lead_ids)

    def get_all_for_valid(self, valid_time):
        refs = self.valid_index.get(valid_time, [])
        out = []
        for ref in refs:
            data = self.get_by_init_and_lead(ref.init_time, ref.lead_h)
            if data is not None:
                out.append({
                    "data": data,
                    "init_time": ref.init_time,
                    "lead_h": ref.lead_h,
                })
        return out

    def __len__(self):
        return len(self.init_times)

    def __getitem__(self, init_time):
        return self.get_run(init_time)


class GFSGluedForecastDataset(ForecastDatasetBase):
    """
    Forecast-mode analogue of WRFs2sDataset.

    Assumptions:
    ------------
    1. One file corresponds to one forecast run.
    2. File name contains init_time, e.g. wrfout_d01_2023-01-01_00:00:00
    3. The file contains all forecast steps along Time.
    """

    @property
    def _files_template(self):
        return "**/*glued*"

    @staticmethod
    def _parse_init_time(file: Path) -> np.datetime64:
        # same parsing idea as current WRFs2sDataset
        # e.g. gfs_glued_2026-04-10_00:00:00
        parts = file.stem.split("_")
        date_part = parts[-2]
        time_part = parts[-1]
        dt = np.datetime64(f"{date_part}T{time_part}").astype("datetime64[h]")
        return dt

    def _create_grid(self):
        grid_path = sorted(self.path.glob(self._files_template))[0]
        print(f"loading WRF forecast grid from {grid_path}")
        with xr.open_dataset(grid_path, cache=False) as ds:
            lon = ds["XLONG"].values[0]
            lat = ds["XLAT"].values[0]
            grid = {"longitude": lon, "latitude": lat}
        return grid

    def _read_lead_axis_h(self, files: List[Path]) -> np.ndarray:
        """
        Read actual lead axis from the file Time/Times information.
        Returns integer lead hours for this run.
        """
        if len(files) == 0:
            raise ValueError("Empty files list passed to _read_lead_axis_h")

        file = files[0]
        init_time = self._parse_init_time(file)

        valid_times = None

        # First try xarray decoded time coordinates
        try:
            with xr.open_dataset(file, cache=False, decode_times=True) as ds:
                lead_h = ds.coords["time_counter"].values.astype(np.int32)
        except:
            return None
        # print(f"Parsed valid times for {file}: {valid_times}")
        # print(f"Parsed lead hours for {file}: {lead_h}")

        # Basic sanity checks
        if np.any(lead_h < 0):
            raise ValueError(
                f"Negative lead times found in {file}. "
                f"Parsed init_time={init_time}, valid_times[0]={valid_times[0]}"
            )

        # Optional consistency checks against expected config
        if self.expected_max_lead_h is not None and lead_h.max() > self.expected_max_lead_h:
            msg = (
                f"Run {file.name} has max lead {lead_h.max()}h, "
                f"but expected_max_lead_h={self.expected_max_lead_h}"
            )
            if self.strict:
                raise ValueError(msg)
            warnings.warn(msg)

        if self.expected_lead_step_h is not None and len(lead_h) > 1:
            diffs = np.diff(lead_h)
            bad = diffs != self.expected_lead_step_h
            if np.any(bad):
                msg = (
                    f"Run {file.name} has irregular lead spacing {np.unique(diffs)}, "
                    f"expected {self.expected_lead_step_h}h"
                )
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg)

        return lead_h

    def _load_run_slice(self, files: List[Path], lead_ids) -> np.ndarray:
        """
        Load selected forecast steps from one WRF forecast run.

        Returns
        -------
        np.ndarray of shape (T, C, H, W)
        """
        if len(files) == 0:
            raise ValueError("Empty files list passed to _load_run_slice")

        file = str(files[0])
        lead_ids = np.asarray(list(lead_ids), dtype=int)

        npy = []
        with netCDF4.Dataset(file, "r") as ncf:
            for variable in self.data_variables:
                var = wrf.getvar(
                    ncf,
                    variable,
                    timeidx=lead_ids,
                    meta=False,
                    squeeze=False,
                )
                # Bring to shape (T, 1, H, W)
                var = atleast_nd(var, 5)[:, 0]
                npy.append(var)

        # concat over channel axis -> (C, T, H, W) or similar intermediate
        npy = np.concatenate(npy, axis=0)
        # to (T, C, H, W)
        return np.transpose(npy, (1, 0, 2, 3))
    

class ForecastWindowSample(NamedTuple):
    """
    Fixed-shape sample for forecast-window tasks.

    forecast:
        shape (T, R, C, H, W)
        T = seq_len along valid_time
        R = run slots (newest -> oldest)
    lead_h:
        shape (T, R), int32
        lead time in hours, -1 where unavailable
    avail_mask:
        shape (T, R), bool
        True where forecast[t, r] is valid
    valid_time_unix_s:
        shape (T,), int64
        valid times encoded as unix seconds
    init_time_unix_s:
        shape (T, R), int64
        init times encoded as unix seconds, -1 where unavailable
    """
    forecast: np.ndarray
    lead_h: np.ndarray
    avail_mask: np.ndarray
    valid_time_unix_s: np.ndarray
    init_time_unix_s: np.ndarray


def _to_datetime64_h(x) -> np.datetime64:
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[h]")
    if isinstance(x, pd.Timestamp):
        return x.to_datetime64().astype("datetime64[h]")
    if isinstance(x, datetime):
        return np.datetime64(x).astype("datetime64[h]")
    return np.datetime64(x, "h")


def _dt64_to_unix_s(x: np.datetime64) -> np.int64:
    return x.astype("datetime64[s]").astype(np.int64)


class ForecastWindowDataset(Dataset):
    """
    Window view over a run-centric forecast dataset.

    This dataset is indexed by the window start valid_time ("anchor").
    It returns a dense tensorized representation over:
        valid_time x run_slot

    Slot semantics (current implementation):
        slot 0 = newest available forecast for this valid_time
        slot 1 = next newest
        ...
        i.e. sorted by lead_h ascending.

    Required base_ds interface:
        - base_ds.valid_index: dict[valid_time] -> list[ForecastRef]
              where ForecastRef has .init_time and .lead_h
        - base_ds.get_by_init_and_lead(init_time, lead_h) -> field (C,H,W) or None

    Optional fast path:
        - base_ds.get_run(init_time, leads_h=[...]) -> (K,C,H,W)
          preserving the order of requested leads_h
    """

    def __init__(
        self,
        base_ds,
        seq_len: int,
        valid_step_h: int = 1,
        *,
        max_run_slots: Optional[int] = None,
        include_future_inits_in_window: bool = False,
        drop_incomplete_windows: bool = True,
        fill_value: float = np.nan,
        name: Optional[str] = None,
    ):
        super().__init__()

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if valid_step_h <= 0:
            raise ValueError("valid_step_h must be positive")

        self.base_ds = base_ds
        self.seq_len = int(seq_len)
        self.valid_step_h = int(valid_step_h)
        self.include_future_inits_in_window = include_future_inits_in_window
        self.drop_incomplete_windows = drop_incomplete_windows
        self.fill_value = fill_value
        self.name = name or f"{getattr(base_ds, 'name', 'forecast')}_window"
        self.src_grid = getattr(base_ds, "src_grid", None)

        self.field_shape, self.field_dtype = self._infer_field_meta()
        self.max_run_slots = (
            int(max_run_slots)
            if max_run_slots is not None
            else self._infer_max_run_slots()
        )

        self.anchor_times = self._build_anchor_times()

    # ---------- public API ----------

    def __len__(self) -> int:
        return len(self.anchor_times)

    def __getitem__(self, index) -> ForecastWindowSample:
        anchor = self._normalize_index(index)
        valid_times = self._make_window_valid_times(anchor)

        T = self.seq_len
        R = self.max_run_slots
        C, H, W = self.field_shape

        forecast = np.full(
            (T, R, C, H, W),
            self.fill_value,
            dtype=self.field_dtype,
        )
        lead_h = np.full((T, R), -1, dtype=np.int32)
        avail_mask = np.zeros((T, R), dtype=bool)
        valid_time_unix_s = np.array(
            [_dt64_to_unix_s(vt) for vt in valid_times], dtype=np.int64
        )
        init_time_unix_s = np.full((T, R), -1, dtype=np.int64)

        # Group requested fields by init_time to allow bulk loading if supported.
        requests_by_init: Dict[np.datetime64, List[Tuple[int, int, int]]] = {}

        for t, vt in enumerate(valid_times):
            refs = self._get_refs_for_valid(vt, anchor)
            for r, ref in enumerate(refs):
                requests_by_init.setdefault(ref.init_time, []).append(
                    (t, r, int(ref.lead_h))
                )

        for init_time, items in requests_by_init.items():
            # Keep deterministic order
            items = sorted(items, key=lambda x: x[2])  # sort by lead_h
            requested_leads = [lead for _, _, lead in items]

            batch = self._fetch_many_from_init(init_time, requested_leads)
            if batch is None:
                continue

            if len(batch) != len(items):
                raise RuntimeError(
                    f"Bulk fetch returned {len(batch)} fields, "
                    f"expected {len(items)} for init={init_time}"
                )

            init_s = _dt64_to_unix_s(init_time)
            for field, (t, r, lead) in zip(batch, items):
                forecast[t, r] = field
                lead_h[t, r] = lead
                avail_mask[t, r] = True
                init_time_unix_s[t, r] = init_s
        return MetricField(
            np.swapaxes(forecast, 0, 1),
            lead_h=np.swapaxes(lead_h, 0, 1),
            avail_mask=np.swapaxes(avail_mask, 0, 1)    ,
            # valid_time_unix_s=valid_time_unix_s,
            # init_time_unix_s=init_time_unix_s,
        )
        # return ForecastWindowSample(
        #     forecast=forecast,
        #     lead_h=lead_h,
        #     avail_mask=avail_mask,
        #     valid_time_unix_s=valid_time_unix_s,
        #     init_time_unix_s=init_time_unix_s,
        # )

    # ---------- helpers ----------

    def _normalize_index(self, index) -> np.datetime64:
        if isinstance(index, (int, np.integer)):
            return self.anchor_times[int(index)]
        return _to_datetime64_h(index)

    def _make_window_valid_times(self, anchor: np.datetime64) -> np.ndarray:
        return anchor + np.arange(self.seq_len) * np.timedelta64(self.valid_step_h, "h")

    def _infer_field_meta(self) -> Tuple[Tuple[int, int, int], np.dtype]:
        # Find first actually loadable field
        for valid_time in sorted(self.base_ds.valid_index.keys()):
            refs = self.base_ds.valid_index[valid_time]
            for ref in refs:
                field = self.base_ds.get_by_init_and_lead(ref.init_time, int(ref.lead_h))
                if field is None:
                    continue
                if field.ndim != 3:
                    raise ValueError(
                        f"Forecast field must have shape (C,H,W), got {field.shape}"
                    )
                dtype = field.dtype
                if not np.issubdtype(dtype, np.floating):
                    dtype = np.float32
                return field.shape, dtype
        raise RuntimeError("Could not infer forecast field shape from base_ds")

    def _infer_max_run_slots(self) -> int:
        if not getattr(self.base_ds, "valid_index", None):
            raise RuntimeError("base_ds.valid_index is empty or missing")
        return max(len(refs) for refs in self.base_ds.valid_index.values())

    def _get_refs_for_valid(self, valid_time: np.datetime64, anchor: np.datetime64):
        valid_time = _to_datetime64_h(valid_time)
        refs = list(self.base_ds.valid_index.get(valid_time, []))

        if not self.include_future_inits_in_window:
            refs = [ref for ref in refs if ref.init_time <= anchor]

        return refs[: self.max_run_slots]

    def _fetch_many_from_init(
        self,
        init_time: np.datetime64,
        leads_h: List[int],
    ) -> Optional[np.ndarray]:
        """
        Try fast bulk path first, fall back to per-lead loading.
        Returns array of shape (K,C,H,W) or None.
        """
        if hasattr(self.base_ds, "get_run"):
            try:
                batch = self.base_ds.get_run(init_time, leads_h=leads_h)
                if batch is not None:
                    batch = np.asarray(batch)
                    if batch.ndim != 4:
                        raise ValueError(
                            f"base_ds.get_run must return (K,C,H,W), got {batch.shape}"
                        )
                    return batch
            except TypeError:
                # base_ds.get_run exists but has another signature
                pass

        fields = []
        for lead in leads_h:
            field = self.base_ds.get_by_init_and_lead(init_time, int(lead))
            if field is None:
                return None
            field = np.asarray(field)
            if field.ndim != 3:
                raise ValueError(
                    f"base_ds.get_by_init_and_lead must return (C,H,W), got {field.shape}"
                )
            if field.dtype != self.field_dtype:
                field = field.astype(self.field_dtype, copy=False)
            fields.append(field)

        return np.stack(fields, axis=0)

    def _build_anchor_times(self) -> np.ndarray:
        raw_valid_times = sorted(
            {_to_datetime64_h(vt) for vt in self.base_ds.valid_index.keys()}
        )

        anchor_times = []
        for anchor in raw_valid_times:
            if not self.drop_incomplete_windows:
                anchor_times.append(anchor)
                continue

            ok = True
            for vt in self._make_window_valid_times(anchor):
                refs = self._get_refs_for_valid(vt, anchor)
                if len(refs) == 0:
                    ok = False
                    break

            if ok:
                anchor_times.append(anchor)

        if len(anchor_times) == 0:
            raise RuntimeError(
                "No valid anchors were found. "
                "Try drop_incomplete_windows=False or inspect forecast coverage."
            )

        return np.array(anchor_times, dtype="datetime64[h]")
    
class MetricField(np.ndarray):
    """
    ndarray with attached metadata.
    Ordinary aggregators can use it as a normal array.
    Special aggregators can read .meta.
    """

    def __new__(cls, input_array, **meta):
        obj = np.asarray(input_array).view(cls)
        obj.meta = dict(meta)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.meta = getattr(obj, "meta", {})

    def get_meta(self, key=None, default=None):
        if key is None:
            return self.meta
        return self.meta.get(key, default)
