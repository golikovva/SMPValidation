from __future__ import annotations

from typing import NamedTuple, Optional, Dict, List, Tuple, Any
from datetime import datetime
import warnings
import re

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
        transform=None,
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
        self.transform = transform

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

    @property
    def grid(self):
        # if self.dst_grid is not None:  # todo
        #     return self.dst_grid
        return self.src_grid

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
        npy = np.transpose(npy, (1, 0, 2, 3))
        # to (T, C, H, W)
        if self.transform:
            npy = self.transform(npy)
        return npy
    

class GFSGribForecastDataset(ForecastDatasetBase):
    """
    Forecast dataset for per-lead GFS GRIB files.

    Supported layouts include Herbie subset files:
        20260301/subset_80ef7196__gfs.t00z.pgrb2.0p25.f000

    and NOMADS-style subset files:
        date=20260425/cycle=00/gfs.t00z.pgrb2.0p25.f000.subset.grib2

    The matching .idx inventory files are ignored.
    """

    _gfs_file_re = re.compile(
        r"(?:^|__)gfs\.t(?P<cycle>\d{2})z\.(?P<product>.+?)\.f(?P<lead>\d{3})(?:$|[._])"
    )

    def __init__(self, *args, product: str = "pgrb2.0p25", **kwargs):
        self.product = product
        super().__init__(*args, **kwargs)

    @property
    def _files_template(self):
        return "**/*gfs*.f*"

    @staticmethod
    def _import_pygrib():
        try:
            import pygrib
        except ImportError as exc:
            raise ImportError(
                "GFSGribForecastDataset requires pygrib to read GRIB files."
            ) from exc
        return pygrib

    @classmethod
    def _match_gfs_file(cls, file: Path):
        return cls._gfs_file_re.search(file.name)

    def _is_data_file(self, file: Path) -> bool:
        if not file.is_file() or file.name.endswith(".idx"):
            return False

        match = self._match_gfs_file(file)
        if match is None:
            return False

        return self.product is None or match.group("product") == self.product

    def _iter_grib_files(self):
        for file in sorted(self.path.glob(self._files_template)):
            if self._is_data_file(file):
                yield file

    @staticmethod
    def _date_from_path(file: Path) -> str:
        for part in reversed(file.parts[:-1]):
            candidate = part.split("=", 1)[-1]
            if re.fullmatch(r"\d{8}", candidate):
                return candidate

        match = re.search(r"(?:^|[._-])(?P<date>\d{8})(?:[._-]|$)", file.name)
        if match is not None:
            return match.group("date")

        raise ValueError(
            f"Could not parse GFS init date from {file}. "
            "Expected an ancestor folder like 20260301 or date=20260301."
        )

    @classmethod
    def _parse_lead_h(cls, file: Path) -> int:
        match = cls._match_gfs_file(file)
        if match is None:
            raise ValueError(f"Could not parse GFS lead time from {file.name}")
        return int(match.group("lead"))

    @classmethod
    def _parse_cycle(cls, file: Path) -> str:
        match = cls._match_gfs_file(file)
        if match is None:
            raise ValueError(f"Could not parse GFS cycle from {file.name}")
        return match.group("cycle")

    @classmethod
    def _parse_init_time(cls, file: Path) -> np.datetime64:
        date_part = cls._date_from_path(file)
        cycle = cls._parse_cycle(file)
        dt = datetime.strptime(f"{date_part}{cycle}", "%Y%m%d%H")
        return np.datetime64(dt).astype("datetime64[h]")

    def _create_runs_dict(self):
        grouped: Dict[np.datetime64, Dict[int, Path]] = {}

        for file in self._iter_grib_files():
            init_time = self._parse_init_time(file)
            lead_h = self._parse_lead_h(file)
            run_files = grouped.setdefault(init_time, {})

            if lead_h in run_files:
                msg = (
                    f"Duplicate GFS file for init={init_time}, lead={lead_h}h: "
                    f"{run_files[lead_h]} and {file}. Keeping the first one."
                )
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg)
                continue

            run_files[lead_h] = file

        runs = {}
        for init_time, lead_to_file in sorted(grouped.items()):
            files = [file for _, file in sorted(lead_to_file.items())]
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

    def _create_grid(self):
        grid_path = next(self._iter_grib_files(), None)
        if grid_path is None:
            raise RuntimeError(
                f"No GFS GRIB files found under {self.path} with product={self.product!r}"
            )

        print(f"loading GFS GRIB forecast grid from {grid_path}")
        pygrib = self._import_pygrib()
        with pygrib.open(str(grid_path)) as ds:
            messages = ds.read(1)
            if len(messages) == 0:
                raise ValueError(f"No GRIB messages found in {grid_path}")
            lat, lon = messages[0].latlons()

        return {"longitude": lon, "latitude": lat}

    @staticmethod
    def _normalize_date_query(date):
        if isinstance(date, str) and re.fullmatch(r"\d{8}", date):
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

        dt = np.datetime64(date)
        unit = np.datetime_data(dt.dtype)[0]
        day = dt.astype("datetime64[D]")

        if unit in {"Y", "M", "W", "D"}:
            return day, None

        return day, dt.astype("datetime64[h]")

    @staticmethod
    def _file_for_lead(run: ForecastRunRecord, lead_h: Optional[int]) -> Path:
        if lead_h is None:
            return run.files[0]

        ids = np.where(run.lead_times_h == int(lead_h))[0]
        if len(ids) == 0:
            raise KeyError(
                f"Run {run.init_time} has no lead_h={lead_h}. "
                f"Available leads: {run.lead_times_h.tolist()}"
            )

        return run.files[int(ids[0])]

    def _resolve_inventory_file(self, date=None, lead_h: Optional[int] = None) -> Path:
        if not self.runs_dict:
            raise RuntimeError("No GFS forecast runs were parsed.")

        if date is None:
            init_time = sorted(self.runs_dict.keys())[0]
            return self._file_for_lead(self.runs_dict[init_time], lead_h)

        query_day, query_time = self._normalize_date_query(date)

        if query_time is not None and query_time in self.runs_dict:
            return self._file_for_lead(self.runs_dict[query_time], lead_h)

        matches = [
            run
            for init_time, run in sorted(self.runs_dict.items())
            if init_time.astype("datetime64[D]") == query_day
        ]
        if not matches:
            raise KeyError(f"No GFS forecast runs found for date={date!r}")

        return self._file_for_lead(matches[0], lead_h)

    @staticmethod
    def _safe_grib_attr(message, attr: str, default=None):
        try:
            return getattr(message, attr)
        except (AttributeError, RuntimeError, ValueError):
            return default

    @staticmethod
    def _selector_for_inventory_row(row: Dict[str, Any]) -> Dict[str, Any]:
        selector = {}
        for key in ("shortName", "typeOfLevel", "level"):
            value = row.get(key)
            if value is not None:
                selector[key] = value
        return selector

    def list_data_variables(self, date=None, lead_h: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List GRIB variables available in the first matching GFS file.

        Parameters
        ----------
        date:
            Optional init date or init time. Examples: "20260301",
            "2026-03-01", "2026-03-01T00". If omitted, the first parsed run
            is inspected.
        lead_h:
            Optional forecast lead hour to inspect. If omitted, the first
            available lead in the selected run is inspected.

        Returns
        -------
        list[dict]
            Each row includes GRIB metadata and a "data_variable" entry that
            can be passed to data_variables. Unique shortNames are returned as
            strings; duplicate shortNames are returned as selector dicts.
        """
        file = self._resolve_inventory_file(date=date, lead_h=lead_h)
        pygrib = self._import_pygrib()

        rows = []
        with pygrib.open(str(file)) as ds:
            ds.seek(0)
            for i, message in enumerate(ds, start=1):
                row = {
                    "message": i,
                    "shortName": self._safe_grib_attr(message, "shortName"),
                    "name": self._safe_grib_attr(message, "name"),
                    "parameterName": self._safe_grib_attr(message, "parameterName"),
                    "typeOfLevel": self._safe_grib_attr(message, "typeOfLevel"),
                    "level": self._safe_grib_attr(message, "level"),
                    "units": self._safe_grib_attr(message, "units"),
                    "forecastTime": self._safe_grib_attr(message, "forecastTime"),
                    "stepRange": self._safe_grib_attr(message, "stepRange"),
                    "file": file,
                }
                rows.append(row)

        short_name_counts = {}
        for row in rows:
            short_name = row["shortName"]
            if short_name is not None:
                short_name_counts[short_name] = short_name_counts.get(short_name, 0) + 1

        for row in rows:
            short_name = row["shortName"]
            if short_name is not None and short_name_counts[short_name] == 1:
                row["data_variable"] = short_name
            else:
                row["data_variable"] = self._selector_for_inventory_row(row)

        return rows

    def _read_lead_axis_h(self, files: List[Path]) -> np.ndarray:
        if len(files) == 0:
            raise ValueError("Empty files list passed to _read_lead_axis_h")

        lead_h = np.array([self._parse_lead_h(file) for file in files], dtype=np.int32)

        if np.any(lead_h < 0):
            raise ValueError(f"Negative GFS lead times found in {files[0].parent}")

        if self.expected_max_lead_h is not None and lead_h.max() > self.expected_max_lead_h:
            msg = (
                f"Run {self._parse_init_time(files[0])} has max lead {lead_h.max()}h, "
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
                    f"Run {self._parse_init_time(files[0])} has irregular lead spacing "
                    f"{np.unique(diffs)}, expected {self.expected_lead_step_h}h"
                )
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg)

        return lead_h

    @staticmethod
    def _variable_selectors(variable: Any) -> List[Dict[str, Any]]:
        if isinstance(variable, dict):
            return [variable]

        if isinstance(variable, str):
            return [
                {"shortName": variable},
                {"name": variable},
                {"parameterName": variable},
            ]

        raise TypeError(
            "GFS GRIB data_variables must contain strings or pygrib selector dicts"
        )

    @classmethod
    def _select_grib_message(cls, ds, variable: Any, file: Path):
        for selector in cls._variable_selectors(variable):
            try:
                messages = ds.select(**selector)
            except (RuntimeError, ValueError):
                continue

            if len(messages) == 1:
                return messages[0]

            if len(messages) > 1:
                raise ValueError(
                    f"{file.name}: variable {variable!r} matched {len(messages)} "
                    f"GRIB messages with selector {selector}. Use a dict selector "
                    "with enough GRIB keys to make it unique."
                )

        raise KeyError(f"{file.name}: no GRIB message matched variable {variable!r}")

    @staticmethod
    def _grib_values(message) -> np.ndarray:
        values = message.values
        if np.ma.isMaskedArray(values):
            values = values.filled(np.nan)
        return np.asarray(values)

    def _load_file_vars(self, file: Path, pygrib) -> np.ndarray:
        with pygrib.open(str(file)) as ds:
            if self.data_variables is None:
                ds.seek(0)
                messages = list(ds)
            else:
                messages = [
                    self._select_grib_message(ds, variable, file)
                    for variable in self.data_variables
                ]

            if len(messages) == 0:
                raise ValueError(f"No GRIB messages found in {file}")

            return np.stack([self._grib_values(message) for message in messages], axis=0)

    def _load_run_slice(self, files: List[Path], lead_ids) -> np.ndarray:
        if len(files) == 0:
            raise ValueError("Empty files list passed to _load_run_slice")

        lead_ids = np.asarray(list(lead_ids), dtype=int)
        pygrib = self._import_pygrib()

        npy = []
        for lead_id in lead_ids:
            npy.append(self._load_file_vars(files[int(lead_id)], pygrib))

        npy = np.stack(npy, axis=0)
        if self.transform:
            npy = self.transform(npy)
        return npy


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

    @property
    def grid(self):
        # if self.dst_grid is not None:  # todo
        #     return self.dst_grid
        return self.src_grid


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
