"""Readers for native buoy products and individual oceanographic profiles.

Readers retain native timestamps.  Alignment to a requested time grid belongs
to :class:`BuoyObservationDataset`, not to the file format adapters.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
from hashlib import sha256
from io import StringIO
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .buoy_types import (
    BuoyDescriptor,
    NativeBuoyData,
    PositionSeries,
    ScalarSeries,
    VariableSpec,
)
from .buoy_utils import merge_native_data, normalize_coordinates, normalize_datetimes, slice_native_data


_VARIABLES = ("ice_thickness", "snow_thickness")
_SIMBA_COLUMNS = {
    "ice_thickness": "EsEs [m]",
    "snow_thickness": "Snow thick [m]",
}
_SIMBA_UNCERTAINTY = {
    "ice_thickness": "EsEs unc [m]",
    "snow_thickness": "Snow thick unc [m]",
}
_CRREL_COLUMNS = {"ice_thickness": "hi", "snow_thickness": "hs"}


def _checked_times(values: object) -> np.ndarray:
    times = np.asarray(normalize_datetimes(values))
    if times.ndim != 1 or np.isnat(times).any():
        raise ValueError("timestamps must be a one-dimensional array without NaT")
    return times


def _simba_header(path: Path) -> tuple[list[str], list[str]]:
    """Find the table by its metadata terminator, never by a fixed row count."""
    metadata: list[str] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            metadata.append(line.rstrip("\r\n"))
            if line.strip() == "*/":
                break
        else:
            raise ValueError("SIMBA metadata terminator '*/' is missing")
        for line in stream:
            if line.strip():
                columns = line.rstrip("\r\n").split("\t")
                break
        else:
            raise ValueError("SIMBA table header is missing")
    if len(columns) != len(set(columns)):
        raise ValueError("SIMBA table contains duplicate column names")
    missing = {"Date/Time", "Latitude", "Longitude"} - set(columns)
    if missing:
        raise ValueError(f"SIMBA table is missing required columns: {sorted(missing)}")
    return metadata, columns


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)


def _crrel_values(dataset: xr.Dataset, name: str) -> np.ndarray:
    variable = dataset[name]
    if variable.dims != ("time",):
        raise ValueError(f"{name!r} must have dimensions ('time',), got {variable.dims}")
    values = np.asarray(variable.values, dtype=np.float64).copy()
    # Some CRREL files do not declare these known missing-value sentinels.
    values[values == -999.0] = np.nan
    for fill in (9.969209968386869e36, float(np.float32(9.969209968386869e36))):
        values[values == fill] = np.nan
    for key in ("_FillValue", "missing_value"):
        if key in variable.attrs:
            for fill in np.asarray(variable.attrs[key]).reshape(-1):
                values[values == float(fill)] = np.nan
    return values


def _metre_factor(variable: xr.DataArray) -> float:
    raw = variable.attrs.get("units")
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", errors="strict")
    units = str(raw).strip().lower()
    if units in {"m", "meter", "meters", "metre", "metres"}:
        return 1.0
    if units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return 0.01
    raise ValueError(f"{variable.name!r} has unsupported thickness units {raw!r}")


def _validate_crrel_schema(dataset: xr.Dataset) -> None:
    missing = {"time", "lat", "lon"} - set(dataset.variables)
    if missing:
        raise ValueError(f"CRREL dataset is missing required variables: {sorted(missing)}")
    for name in ("time", "lat", "lon"):
        if dataset[name].dims != ("time",):
            raise ValueError(f"{name!r} must have dimensions ('time',)")


class _ScalarFileSource:
    """Shared file discovery; subclasses own only their storage conventions."""

    source: str
    suffix: str
    default_pattern: str
    variable_specs = {
        name: VariableSpec(name=name, units="m", geometry="scalar")
        for name in _VARIABLES
    }

    def __init__(self, folder: str | Path, pattern: str | None = None) -> None:
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Buoy source folder does not exist: {self.folder}")
        self.pattern = pattern or self.default_pattern
        self._catalog: tuple[BuoyDescriptor, ...] | None = None

    def discover(self) -> tuple[BuoyDescriptor, ...]:
        """Return stable, namespaced IDs, grouping matching IDs across folders."""
        if self._catalog is not None:
            return self._catalog
        groups: dict[str, list[Path]] = {}
        for path in sorted(self.folder.rglob(self.pattern)):
            if path.is_file():
                buoy_id = path.stem.removesuffix(self.suffix)
                if not buoy_id:
                    raise ValueError(f"Cannot derive a buoy identifier from {path}")
                groups.setdefault(buoy_id, []).append(path)
        if not groups:
            raise FileNotFoundError(
                f"No {self.source} files matching {self.pattern!r} in {self.folder}"
            )
        descriptors: list[BuoyDescriptor] = []
        for buoy_id, files in sorted(groups.items()):
            file_metadata: dict[str, dict] = {}
            available: set[str] = set()
            for path in files:
                try:
                    variables, metadata = self._inspect_file(path)
                except Exception as exc:
                    raise ValueError(f"Cannot inspect {self.source} file {path}: {exc}") from exc
                available.update(variables)
                file_metadata[str(path)] = {**metadata, "variables": tuple(variables)}
            descriptors.append(
                BuoyDescriptor(
                    key=f"{self.source}:{buoy_id}",
                    source=self.source,
                    source_buoy_id=buoy_id,
                    files=tuple(files),
                    variables=tuple(name for name in _VARIABLES if name in available),
                    metadata={
                        "source": self.source,
                        "files": tuple(str(path) for path in files),
                        "file_metadata": file_metadata,
                    },
                )
            )
        self._catalog = tuple(descriptors)
        return self._catalog

    def read(
        self,
        key: str,
        *,
        variables: Sequence[str] = _VARIABLES,
        start: object = None,
        stop: object = None,
    ) -> NativeBuoyData:
        """Read native observations in the half-open interval [start, stop)."""
        requested = tuple(dict.fromkeys(variables))
        unknown = set(requested) - set(_VARIABLES)
        if unknown:
            raise ValueError(f"Unsupported {self.source} scalar variables: {sorted(unknown)}")
        descriptor = next((item for item in self.discover() if item.key == key), None)
        if descriptor is None:
            raise KeyError(f"Unknown {self.source} buoy key: {key!r}")
        parts: list[NativeBuoyData] = []
        for path in descriptor.files:
            try:
                parts.append(self._read_file(path, descriptor, requested))
            except Exception as exc:
                raise ValueError(f"Cannot read {self.source} file {path}: {exc}") from exc
        return slice_native_data(merge_native_data(parts), start=start, stop=stop)

    def _inspect_file(self, path: Path) -> tuple[tuple[str, ...], dict]:
        raise NotImplementedError

    def _read_file(
        self, path: Path, descriptor: BuoyDescriptor, requested: tuple[str, ...]
    ) -> NativeBuoyData:
        raise NotImplementedError


class SimbaTabSource(_ScalarFileSource):
    """Derived SIMBA ice/snow thickness with reported uncertainty in metres."""

    source = "simba"
    suffix = "_icethick"
    default_pattern = "*.tab"

    def _inspect_file(self, path: Path) -> tuple[tuple[str, ...], dict]:
        metadata_lines, columns = _simba_header(path)
        variables = tuple(name for name, column in _SIMBA_COLUMNS.items() if column in columns)
        metadata = {
            "format": "pangaea_tab",
            "product": "derived_ice_and_snow_thickness",
            "source_metadata_lines": tuple(metadata_lines),
            "preprocessing": {
                "ice_thickness": "source product: 3-day running mean",
                "snow_thickness": "source product: 3-day running mean",
            },
            "uncertainty_columns": {
                name: column for name, column in _SIMBA_UNCERTAINTY.items() if column in columns
            },
            "negative_snow_meaning": "may indicate ice surface melt after snow disappearance",
        }
        return variables, metadata

    def _read_file(
        self, path: Path, descriptor: BuoyDescriptor, requested: tuple[str, ...]
    ) -> NativeBuoyData:
        _, columns = _simba_header(path)
        text = path.read_text(encoding="utf-8-sig")
        frame = pd.read_csv(StringIO(text.split("*/", 1)[1].lstrip()), sep="\t")
        if list(frame.columns) != columns:
            raise ValueError("SIMBA table header does not match its parsed columns")
        # Parse explicitly as ISO timestamps; blank/malformed times are errors.
        parsed = pd.to_datetime(frame["Date/Time"], format="ISO8601", utc=True, errors="raise")
        times = _checked_times(parsed.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"))
        coords = np.column_stack(
            (_numeric_column(frame, "Latitude"), _numeric_column(frame, "Longitude"))
        )
        file_metadata = descriptor.metadata["file_metadata"][str(path)]
        series: dict[str, ScalarSeries] = {}
        for name in requested:
            column = _SIMBA_COLUMNS[name]
            if name not in file_metadata["variables"]:
                continue
            if column not in frame:
                raise ValueError(f"Declared scalar column {column!r} is missing")
            uncertainty_column = _SIMBA_UNCERTAINTY[name]
            if name in file_metadata["uncertainty_columns"] and uncertainty_column not in frame:
                raise ValueError(f"Declared uncertainty column {uncertainty_column!r} is missing")
            uncertainty = (
                _numeric_column(frame, uncertainty_column) if uncertainty_column in frame else None
            )
            series[name] = ScalarSeries(
                datetimes=times,
                values=_numeric_column(frame, column),
                uncertainty=uncertainty,
            )
        return NativeBuoyData(
            descriptor=descriptor,
            positions=PositionSeries(datetimes=times, coords=coords),
            series=series,
            metadata={
                "source": self.source,
                "files": (str(path),),
                "file_metadata": {str(path): file_metadata},
                "units": {name: "m" for name in series},
            },
        )


class CrrelNetCDFSource(_ScalarFileSource):
    """Processed CRREL scalar thickness; thermistor arrays are never loaded."""

    source = "crrel"
    suffix = "_updated"
    default_pattern = "*.nc"

    def _inspect_file(self, path: Path) -> tuple[tuple[str, ...], dict]:
        with xr.open_dataset(path, decode_times=False, engine="h5netcdf") as dataset:
            _validate_crrel_schema(dataset)
            variables = tuple(name for name, column in _CRREL_COLUMNS.items() if column in dataset)
            metadata = {
                "format": "netcdf",
                "product": "processed_ice_and_snow_thickness",
                "time_units": dataset["time"].attrs.get("units"),
                "source_attributes": dict(dataset.attrs),
                "source_variable_attributes": {
                    name: dict(dataset[_CRREL_COLUMNS[name]].attrs) for name in variables
                },
                "alternative_west_product_available": any(
                    name in dataset for name in ("hi_west", "hs_west")
                ),
                "selected_product": "hi/hs (processed)",
                "uncertainty": "not provided by this scalar product",
            }
        return variables, metadata

    def _read_file(
        self, path: Path, descriptor: BuoyDescriptor, requested: tuple[str, ...]
    ) -> NativeBuoyData:
        file_metadata = descriptor.metadata["file_metadata"][str(path)]
        with xr.open_dataset(path, decode_times=False, engine="h5netcdf") as dataset:
            _validate_crrel_schema(dataset)
            if not dataset["time"].attrs.get("units"):
                raise ValueError("CRREL time variable must declare CF time units")
            decoded_time = xr.decode_cf(dataset[["time"]], decode_times=True)["time"].values
            if not np.issubdtype(np.asarray(decoded_time).dtype, np.datetime64):
                raise ValueError("CRREL time cannot be decoded to Gregorian datetime64 timestamps")
            times = _checked_times(decoded_time)
            coords = np.column_stack((_crrel_values(dataset, "lat"), _crrel_values(dataset, "lon")))
            # This CRREL product uses the pair (0, 0) for missing positions.
            # Mask before duplicate resolution, keeping the timestamp as a gap.
            sentinel_coords = np.all(coords == (0.0, 0.0), axis=1)
            diagnostics = [{
                "kind": "invalid_coordinates", "source": self.source,
                "file": str(path), "index": int(index), "time": str(times[index]),
                "reason": "crrel_missing_position_sentinel", "coords": [0.0, 0.0],
            } for index in np.flatnonzero(sentinel_coords)]
            coords[sentinel_coords] = np.nan
            series: dict[str, ScalarSeries] = {}
            for name in requested:
                if name not in file_metadata["variables"]:
                    continue
                column = _CRREL_COLUMNS[name]
                if column not in dataset:
                    raise ValueError(f"Declared scalar variable {column!r} is missing")
                factor = _metre_factor(dataset[column])
                series[name] = ScalarSeries(
                    datetimes=times,
                    values=_crrel_values(dataset, column) * factor,
                    uncertainty=None,
                )
        return NativeBuoyData(
            descriptor=descriptor,
            positions=PositionSeries(datetimes=times, coords=coords),
            series=series,
            metadata={
                "source": self.source,
                "files": (str(path),),
                "file_metadata": {str(path): file_metadata},
                "units": {name: "m" for name in series},
                "diagnostics": diagnostics,
                "diagnostic_counts": {"invalid_coordinates": len(diagnostics)},
            },
        )


def _iabp_table(path: Path, *, ids_only: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Read selected columns and retain physical line numbers for diagnostics."""
    lines = [
        (number, line)
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("IABP table is empty")
    columns = lines[0][1].split()
    if len(columns) != len(set(columns)):
        raise ValueError("IABP table contains duplicate column names")
    position_columns = [name for name in ("POS_DOY", "PosDOY") if name in columns]
    if len(position_columns) != 1:
        raise ValueError("IABP table must contain exactly one of POS_DOY or PosDOY")
    required = {"BuoyID", "Year", "DOY", "Lat", "Lon"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"IABP table is missing required columns: {sorted(missing)}")
    selected = ["BuoyID"] if ids_only else [
        "BuoyID", "Year", "DOY", position_columns[0], "Lat", "Lon"
    ]
    frame = pd.read_csv(
        StringIO("\n".join(line for _, line in lines)),
        sep=r"\s+", usecols=selected,
        dtype={name: "string" for name in ("BuoyID", "DOY", position_columns[0]) if name in selected},
        keep_default_na=False, index_col=False,
    ).rename(columns={"PosDOY": "POS_DOY"})
    line_numbers = np.asarray([number for number, _ in lines[1:]], dtype=np.int64)
    if frame.empty:
        raise ValueError("IABP table has a header but no data rows")
    if len(frame) != len(line_numbers):
        raise ValueError("IABP data rows could not be matched to physical source lines")
    valid_ids = frame["BuoyID"].str.fullmatch(r"[0-9]+").fillna(False).to_numpy(dtype=bool)
    if not valid_ids.all():
        index = int(np.flatnonzero(~valid_ids)[0])
        raise ValueError(f"invalid BuoyID at line {line_numbers[index]}: {frame['BuoyID'].iloc[index]!r}")
    metadata = {
        "format": "iabp_whitespace_table",
        "product": "native_buoy_positions",
        "columns": tuple(columns),
        "position_time_column": position_columns[0],
        "position_year_rule": "closest valid Year-1/Year/Year+1 candidate to Year+DOY",
        "coordinate_order": ("latitude", "longitude"),
    }
    return frame, line_numbers, metadata


def _iabp_position_times(
    frame: pd.DataFrame, line_numbers: np.ndarray, path: Path
) -> tuple[np.ndarray, list[dict], int]:
    """Resolve each position clock independently of its report's year boundary."""
    size = len(frame)
    times = np.full(size, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if not size:
        return times, [], 0
    year = pd.to_numeric(frame["Year"], errors="coerce").to_numpy(dtype=np.float64)
    report_doy = pd.to_numeric(frame["DOY"], errors="coerce").to_numpy(dtype=np.float64)
    position_doy = pd.to_numeric(frame["POS_DOY"], errors="coerce").to_numpy(dtype=np.float64)
    # Keeping all three candidate years within complete datetime64[ns] years
    # prevents silent NumPy integer overflow for malformed archive years.
    valid_year = np.isfinite(year) & (year == np.floor(year)) & (year >= 1679) & (year <= 2260)
    safe_year = np.where(valid_year, year, 2000).astype(np.int64)

    def days_in_year(years: np.ndarray) -> np.ndarray:
        leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
        return 365 + leap.astype(np.int64)

    def starts(years: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(years, return_inverse=True)
        epochs = np.asarray([np.datetime64(f"{value:04d}-01-01", "ns") for value in unique])
        return epochs.astype(np.int64)[inverse]

    valid_report = valid_year & np.isfinite(report_doy) & (report_doy >= 1)
    valid_report &= report_doy < days_in_year(safe_year) + 1
    valid_position = np.isfinite(position_doy) & (position_doy >= 1) & (position_doy < 367)
    def day_offsets(values: pd.Series, valid: np.ndarray) -> np.ndarray:
        # DOY is a decimal in the source.  Multiplying its binary float by a
        # year's worth of nanoseconds invents several ns of timestamp noise.
        tokens = values.astype(str).to_numpy()
        offsets = np.zeros(size, dtype=np.int64)
        offsets[valid] = np.fromiter((
            int(((Decimal(token) - 1) * 86_400_000_000_000).to_integral_value(rounding=ROUND_HALF_EVEN))
            for token in tokens[valid]
        ), dtype=np.int64, count=int(valid.sum()))
        return offsets

    report_offset = day_offsets(frame["DOY"], valid_report)
    position_offset = day_offsets(frame["POS_DOY"], valid_position)
    report_ns = starts(safe_year) + report_offset
    candidate_years = safe_year[:, None] + np.asarray([-1, 0, 1])
    candidate_ns = starts(candidate_years.reshape(-1)).reshape(size, 3)
    candidate_ns += position_offset[:, None]
    valid_candidates = (
        valid_report[:, None] & valid_position[:, None]
        & (position_doy[:, None] < days_in_year(candidate_years) + 1)
    )
    distances = np.abs(candidate_ns - report_ns[:, None])
    distances[~valid_candidates] = np.iinfo(np.int64).max
    closest = distances.min(axis=1)
    choice = distances.argmin(axis=1)
    has_candidate = valid_candidates.any(axis=1)
    ambiguous = has_candidate & ((distances == closest[:, None]).sum(axis=1) != 1)
    usable = has_candidate & ~ambiguous
    times[usable] = candidate_ns[np.flatnonzero(usable), choice[usable]].astype("datetime64[ns]")
    chosen_years = candidate_years[np.arange(size), choice]
    adjusted = usable & (chosen_years != safe_year)
    diagnostics: list[dict] = []
    for index in np.flatnonzero(~usable):
        if not valid_year[index]:
            reason = "invalid_year"
        elif not valid_report[index]:
            reason = "invalid_report_doy"
        elif ambiguous[index]:
            reason = "ambiguous_position_year"
        else:
            reason = "invalid_position_doy"
        diagnostics.append({
            "kind": "invalid_position_time", "reason": reason,
            "file": str(path), "line": int(line_numbers[index]),
            "year": str(frame["Year"].iloc[index]),
            "doy": str(frame["DOY"].iloc[index]),
            "pos_doy": str(frame["POS_DOY"].iloc[index]),
        })
    for index in np.flatnonzero(adjusted):
        diagnostics.append({
            "kind": "position_year_adjusted", "file": str(path),
            "line": int(line_numbers[index]), "time": str(times[index]),
            "report_year": int(safe_year[index]), "position_year": int(chosen_years[index]),
        })
    return times, diagnostics, int(adjusted.sum())


class IabpTabSource:
    """Native IABP trajectories; meteorology and drift derivation stay outside.

    IDs come from rows, not filenames.  Discovery scans only the ID column;
    reading loads only the six columns needed to interpret position observations.
    Invalid times are reported and skipped.  Invalid coordinates remain at their
    known timestamp so that later drift calculations can break the track there.
    """

    source = "iabp"
    variable_specs: dict[str, VariableSpec] = {}

    def __init__(self, folder: str | Path, pattern: str = "*.dat") -> None:
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Buoy source folder does not exist: {self.folder}")
        self.pattern = pattern
        self._catalog: tuple[BuoyDescriptor, ...] | None = None
        self._descriptors: dict[str, BuoyDescriptor] = {}

    def discover(self) -> tuple[BuoyDescriptor, ...]:
        if self._catalog is not None:
            return self._catalog
        groups: dict[str, list[Path]] = {}
        metadata_by_file: dict[str, dict] = {}
        for path in sorted(self.folder.rglob(self.pattern)):
            if not path.is_file():
                continue
            try:
                ids, _, metadata = _iabp_table(path, ids_only=True)
            except Exception as exc:
                raise ValueError(f"Cannot inspect IABP file {path}: {exc}") from exc
            metadata_by_file[str(path)] = metadata
            for buoy_id in ids["BuoyID"].unique():
                groups.setdefault(str(buoy_id), []).append(path)
        if not groups:
            raise FileNotFoundError(
                f"No IABP files matching {self.pattern!r} in {self.folder}"
            )
        descriptors = []
        for buoy_id, files in sorted(groups.items()):
            descriptors.append(BuoyDescriptor(
                key=f"{self.source}:{buoy_id}", source=self.source,
                source_buoy_id=buoy_id, files=tuple(files), variables=(),
                metadata={
                    "source": self.source, "files": tuple(str(path) for path in files),
                    "position_time_source": "POS_DOY", "native_clock": True,
                    "file_metadata": {str(path): metadata_by_file[str(path)] for path in files},
                },
            ))
        self._catalog = tuple(descriptors)
        self._descriptors = {descriptor.key: descriptor for descriptor in descriptors}
        return self._catalog

    def read(
        self, key: str, *, variables: Sequence[str] = (), start: object = None, stop: object = None
    ) -> NativeBuoyData:
        if variables:
            raise ValueError("IABP provides native positions only; no scalar variables are available")
        self.discover()
        try:
            descriptor = self._descriptors[key]
        except KeyError as exc:
            raise KeyError(f"Unknown IABP buoy key: {key!r}") from exc
        parts = []
        for path in descriptor.files:
            try:
                parts.append(self._read_file(path, descriptor))
            except Exception as exc:
                raise ValueError(f"Cannot read IABP file {path}: {exc}") from exc
        return slice_native_data(merge_native_data(parts), start=start, stop=stop)

    def _read_file(self, path: Path, descriptor: BuoyDescriptor) -> NativeBuoyData:
        frame, line_numbers, file_metadata = _iabp_table(path)
        belongs = (frame["BuoyID"] == descriptor.source_buoy_id).to_numpy(dtype=bool)
        if not belongs.any():
            raise ValueError(f"Declared buoy {descriptor.source_buoy_id!r} no longer occurs in file")
        frame = frame.loc[belongs].reset_index(drop=True)
        line_numbers = line_numbers[belongs]
        times, diagnostics, adjusted_count = _iabp_position_times(frame, line_numbers, path)
        valid_time = ~np.isnat(times)
        coords = np.column_stack([
            pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
            for name in ("Lat", "Lon")
        ])
        invalid_coords = ~np.isfinite(coords).all(axis=1)
        invalid_coords |= (np.abs(coords[:, 0]) > 90) | (np.abs(coords[:, 1]) > 360)
        # The IABP missing-position pair lies inside geographical bounds, so
        # reject it explicitly before merging duplicates or deriving drift.
        sentinel_coords = np.all(coords == (-90.0, -180.0), axis=1)
        invalid_coords |= sentinel_coords
        for index in np.flatnonzero(invalid_coords & valid_time):
            diagnostics.append({
                "kind": "invalid_coordinates", "source": self.source, "file": str(path),
                "line": int(line_numbers[index]), "time": str(times[index]),
                "reason": ("iabp_missing_position_sentinel" if sentinel_coords[index]
                           else "missing_or_out_of_range_position"),
                "coords": coords[index].tolist(),
            })
        coords[invalid_coords] = np.nan
        row_counts = {
            "rows_read": len(frame), "rows_kept": int(valid_time.sum()),
            "rows_skipped_invalid_time": int((~valid_time).sum()),
            "invalid_coordinate_rows": int((invalid_coords & valid_time).sum()),
            "position_year_adjustments": adjusted_count,
        }
        file_metadata = {**file_metadata, "row_counts": row_counts}
        return NativeBuoyData(
            descriptor=descriptor,
            positions=PositionSeries(times[valid_time], coords[valid_time]),
            series={},
            metadata={
                "source": self.source, "files": (str(path),),
                "file_metadata": {str(path): file_metadata},
                "row_counts": row_counts, "diagnostics": diagnostics,
            },
        )


_SURFACE_VARIABLES = ("sst", "sss")
_SURFACE_SPECS = {
    "sst": VariableSpec("sst", units="degC"),
    "sss": VariableSpec("sss", units="psu"),
}
_UPTEMP_PROVENANCE = {
    "depth_interpretation": "published source estimate; may be calculated or nominal",
    "processing": "published Level 2 QC and surface-sensor selection are preserved",
    "source_doc": "https://psc.apl.washington.edu/UpTempO/Level2_QC_doc_2025.php",
}
_AOTD_PROVENANCE = {
    "depth_interpretation": "standard depth levels of the processed source product",
    "surface_estimate": "upper available standard level; no vertical interpolation or extrapolation",
    "source_doc": "https://www.nature.com/articles/s41597-025-05855-3",
}


def _surface_depth_limit(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_surface_depth must be a finite number >= 0 metres") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(result) or result < 0:
        raise ValueError("max_surface_depth must be a finite number >= 0 metres")
    return result


def _surface_variables(variables: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(variables))
    unknown = set(requested) - set(_SURFACE_VARIABLES)
    if unknown:
        raise ValueError(f"Unsupported surface variables: {sorted(unknown)}")
    return requested


def _aotd_surface_units(dataset):
    """Validate numerical meanings before labelling values with canonical units."""
    allowed = {
        "temp": {"degc", "celsius", "degreecelsius", "degreescelsius", "°c"},
        "salt": {"psu", "1"},
        "depth": {"m", "meter", "meters", "metre", "metres"},
    }
    for name, accepted in allowed.items():
        units = dataset[name].attrs.get("units")
        normalized = re.sub(r"[\s_]", "", str(units).lower())
        if normalized not in accepted:
            raise ValueError(f"{name}.units={units!r} is missing or incompatible with "
                             f"{'degC' if name == 'temp' else 'psu' if name == 'salt' else 'm'}")
    positive = dataset["depth"].attrs.get("positive", "down")
    if str(positive).lower() != "down":
        raise ValueError("depth.positive must be 'down'")


_UPTEMP_COLUMN_NAMES = {
    "year": "year", "month": "month", "day": "day", "hour (gmt)": "hour",
    "latitude (n)": "lat", "longitude (e)": "lon",
    "sea surface temperature": "sst", "sea surface temperature depth": "sst_depth",
    "sea surface salinity": "sss", "sea surface salinity depth": "sss_depth",
}


def _uptempo_header(path: Path) -> dict:
    columns, numbered, comments, ids = {}, {}, [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if line == "END":
                if not any(item.strip() for item in stream):
                    raise ValueError("table has no observation rows after END")
                break
            comments.append(line)
            identity = re.match(r"%\s*Iridium\s+ID\s*:\s*(\d+)\s*$", line, re.I)
            if identity:
                ids.add(identity.group(1))
            column = re.match(r"%\s*(\d+)\s*=\s*(.*?)\s*$", line)
            if column:
                index, description = int(column.group(1)), column.group(2)
                if index in numbered:
                    raise ValueError(f"duplicate numbered column {index}")
                numbered[index] = description
                name = _UPTEMP_COLUMN_NAMES.get(" ".join(description.lower().split()))
                if name:
                    if name in columns:
                        raise ValueError(f"ambiguous column for {name}")
                    columns[name] = index
        else:
            raise ValueError("missing END header terminator")
    if len(ids) != 1:
        raise ValueError("header requires one unambiguous Iridium ID")
    missing = set(("year", "month", "day", "hour", "lat", "lon")) - set(columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    for name in _SURFACE_VARIABLES:
        if (name in columns) != (f"{name}_depth" in columns):
            raise ValueError(f"published {name} requires both value and depth columns")
    return {
        **_UPTEMP_PROVENANCE,
        "iridium_id": next(iter(ids)), "columns": columns,
        "numbered_columns": numbered, "header_lines": tuple(comments),
        "data_start_line": line_number + 1, "product": "published_surface",
        "variables": tuple(name for name in _SURFACE_VARIABLES
                           if name in columns and f"{name}_depth" in columns),
    }


def _uptempo_times(rows: list[list[str]], indexes: dict[str, int], line_numbers, path):
    times = np.full(len(rows), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    diagnostics = []
    for index, row in enumerate(rows):
        try:
            date = [Decimal(row[indexes[name]]) for name in ("year", "month", "day")]
            if any(not value.is_finite() or value != value.to_integral_value() for value in date):
                raise ValueError("year, month and day must be finite integers")
            hour = Decimal(row[indexes["hour"]])
            if not hour.is_finite() or not 0 <= hour < 24:
                raise ValueError("GMT hour must be in [0, 24)")
            base = pd.Timestamp(year=int(date[0]), month=int(date[1]), day=int(date[2])).value
            offset = int((hour * Decimal(3_600_000_000_000)).to_integral_value(rounding=ROUND_HALF_EVEN))
            total = base + offset
            if not np.iinfo(np.int64).min < total <= np.iinfo(np.int64).max:
                raise ValueError("timestamp is outside datetime64[ns] range")
            times[index] = np.datetime64(total, "ns")
        except Exception as exc:
            diagnostics.append({"kind": "invalid_time", "file": str(path),
                                "line": int(line_numbers[index]), "reason": str(exc)})
    return times, diagnostics


class UpTempOTabSource:
    """Published UpTempO surface products, bounded by their measured depths.

    Sensor-chain channels are never used to replace missing published SST/SSS.
    Known observation times and raw positions survive rejected scalar values.
    """

    source = "uptempo"
    variable_specs = _SURFACE_SPECS

    def __init__(self, folder: str | Path, *, max_surface_depth, pattern: str = "*.dat"):
        self.folder = Path(folder).expanduser().resolve()
        self.max_surface_depth = _surface_depth_limit(max_surface_depth)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"UpTempO source folder does not exist: {self.folder}")
        self.pattern = pattern
        self._catalog = None
        self._descriptors = {}
        self._headers = {}

    def discover(self) -> tuple[BuoyDescriptor, ...]:
        if self._catalog is not None:
            return self._catalog
        files = sorted(path for path in self.folder.rglob(self.pattern) if path.is_file())
        if not files:
            raise FileNotFoundError(f"No UpTempO files matching {self.pattern!r} in {self.folder}")
        groups = {}
        for path in files:
            try:
                header = _uptempo_header(path)
            except Exception as exc:
                raise ValueError(f"Cannot inspect UpTempO file {path}: {exc}") from exc
            self._headers[path] = header
            groups.setdefault(header["iridium_id"], []).append(path)
        descriptors = []
        for buoy_id, paths in sorted(groups.items()):
            variables = tuple(name for name in _SURFACE_VARIABLES
                              if any(name in self._headers[path]["variables"] for path in paths))
            descriptors.append(BuoyDescriptor(
                f"{self.source}:{buoy_id}", self.source, buoy_id, tuple(paths), variables,
                metadata={**_UPTEMP_PROVENANCE, "source": self.source, "entity_kind": "buoy",
                          "files": tuple(str(path) for path in paths),
                          "max_surface_depth": self.max_surface_depth,
                          "surface_selection": "published_value_and_depth",
                          "file_metadata": {str(path): self._headers[path] for path in paths}},
            ))
        self._catalog = tuple(descriptors)
        self._descriptors = {item.key: item for item in descriptors}
        return self._catalog

    def read(self, key: str, *, variables: Sequence[str] = _SURFACE_VARIABLES,
             start=None, stop=None) -> NativeBuoyData:
        requested = _surface_variables(variables)
        self.discover()
        if key not in self._descriptors:
            raise KeyError(f"Unknown UpTempO buoy key: {key!r}")
        descriptor = self._descriptors[key]
        parts = []
        for path in descriptor.files:
            try:
                parts.append(self._read_file(path, descriptor, requested))
            except Exception as exc:
                raise ValueError(f"Cannot read UpTempO file {path}: {exc}") from exc
        return slice_native_data(merge_native_data(parts), start=start, stop=stop)

    def _read_file(self, path, descriptor, requested):
        header = self._headers[path]
        available = tuple(name for name in requested if name in header["variables"])
        names = ["year", "month", "day", "hour", "lat", "lon"]
        names += [column for name in available for column in (name, f"{name}_depth")]
        selected = [header["columns"][name] for name in names]
        indexes = {name: index for index, name in enumerate(names)}
        rows, line_numbers = [], []
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, raw in enumerate(stream, 1):
                if line_number < header["data_start_line"] or not raw.strip():
                    continue
                fields = raw.split()
                if len(fields) <= max(selected):
                    raise ValueError(f"line {line_number} lacks a required numbered column")
                rows.append([fields[index] for index in selected])
                line_numbers.append(line_number)
        times, diagnostics = _uptempo_times(rows, indexes, line_numbers, path)
        good_time = ~np.isnat(times)

        def numbers(name):
            return pd.to_numeric(pd.Series([row[indexes[name]] for row in rows], dtype=str),
                                 errors="coerce").to_numpy(dtype=np.float64)

        coords = np.column_stack([numbers("lat"), numbers("lon")])
        bad_coords = (~np.isfinite(coords).all(axis=1) | (np.abs(coords[:, 0]) > 90)
                      | (np.abs(coords[:, 1]) > 360))
        coords[bad_coords] = np.nan
        coords = normalize_coordinates(coords)
        for index in np.flatnonzero(bad_coords & good_time):
            diagnostics.append({"kind": "invalid_coordinates", "file": str(path),
                                "line": line_numbers[index], "time": str(times[index])})
        series = {}
        for name in available:
            values, depths = numbers(name), numbers(f"{name}_depth")
            bad_value = ~np.isfinite(values) | np.isin(values, (-999, -9999))
            bad_depth = ~np.isfinite(depths) | (depths < 0) | (depths > self.max_surface_depth)
            for reason, mask in (("missing_value", bad_value), ("surface_depth_rejected", bad_depth)):
                if np.any(mask & good_time):
                    lines = np.asarray(line_numbers, dtype=np.int64)[mask & good_time]
                    breaks = np.flatnonzero(np.diff(lines) > 1) + 1
                    line_ranges = tuple((int(group[0]), int(group[-1]))
                                        for group in np.split(lines, breaks))
                    diagnostics.append({"kind": reason, "file": str(path), "variable": name,
                                        "count": int(np.count_nonzero(mask & good_time)),
                                        "line_ranges": line_ranges})
            values[bad_value | bad_depth] = np.nan
            depths[bad_value | bad_depth] = np.nan
            series[name] = ScalarSeries(times[good_time], values[good_time], depths=depths[good_time])
        counts = {"rows_read": len(rows), "rows_kept": int(good_time.sum()),
                  "rows_skipped_invalid_time": int((~good_time).sum()),
                  "invalid_coordinate_rows": int(np.count_nonzero(bad_coords & good_time))}
        return NativeBuoyData(
            descriptor, PositionSeries(times[good_time], coords[good_time]), series,
            metadata={**_UPTEMP_PROVENANCE, "source": self.source, "files": (str(path),),
                      "max_surface_depth": self.max_surface_depth,
                      "surface_selection": "published_value_and_depth",
                      "file_metadata": {str(path): {**header, "row_counts": counts}},
                      "row_counts": counts, "diagnostics": diagnostics},
        )


class AotdNetCDFSource:
    """One entity per AOTD profile, with lazily cached surface extraction.

    Coordinates and profile fields are joined by their original row positions,
    regardless of the dimension names used in the NetCDF file.  No platform
    identity or trajectory is inferred from proximity or repeated timestamps.
    """

    source = "aotd"
    variable_specs = _SURFACE_SPECS

    def __init__(self, path: str | Path, *, max_surface_depth, time_units_override=None):
        self.path = Path(path).expanduser().resolve()
        self.max_surface_depth = _surface_depth_limit(max_surface_depth)
        if not self.path.is_file():
            raise FileNotFoundError(f"AOTD source file does not exist: {self.path}")
        if time_units_override is not None and (
            not isinstance(time_units_override, str) or not time_units_override.strip()
        ):
            raise ValueError("time_units_override must be a nonempty CF time-units string")
        self.time_units_override = time_units_override
        self._catalog = None
        self._descriptors = {}
        self._rows = {}
        self._surface_cache = {}
        self.metadata = {}

    def _decode_times(self, data, units, calendar):
        if not isinstance(units, str) or not re.match(r"\s*\w+\s+since\s+\S", units):
            raise ValueError("time.units is not valid CF time units; supply time_units_override explicitly")
        try:
            # Check units independently, so invalid metadata cannot silently omit every profile.
            xr.coding.times.decode_cf_datetime(np.array([0.0]), units, calendar=calendar, use_cftime=False)
        except Exception as exc:
            raise ValueError(f"invalid CF time units/calendar {units!r}/{calendar!r}; "
                             "supply a valid time_units_override explicitly") from exc
        times = np.full(data.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        finite = np.isfinite(data)
        try:
            times[finite] = xr.coding.times.decode_cf_datetime(
                data[finite], units, calendar=calendar, use_cftime=False
            ).astype("datetime64[ns]")
        except Exception:
            for row in np.flatnonzero(finite):
                try:
                    times[row] = xr.coding.times.decode_cf_datetime(
                        data[row:row + 1], units, calendar=calendar, use_cftime=False
                    ).astype("datetime64[ns]")[0]
                except Exception:
                    pass
        return times

    def discover(self) -> tuple[BuoyDescriptor, ...]:
        if self._catalog is not None:
            return self._catalog
        try:
            self._load_index()
        except Exception as exc:
            raise ValueError(f"Cannot inspect AOTD file {self.path}: {exc}") from exc
        return self._catalog

    def _load_index(self):
        with xr.open_dataset(self.path, engine="h5netcdf", decode_times=False) as dataset:
            required = {"lat", "lon", "time", "depth", "temp", "salt", "pres"}
            missing = required - set(dataset.variables)
            if missing:
                raise ValueError(f"missing required variables: {sorted(missing)}")
            _aotd_surface_units(dataset)
            if any(dataset[name].ndim != 1 for name in ("lat", "lon", "time", "depth")):
                raise ValueError("lat, lon, time and depth must each be one-dimensional")
            n = dataset["time"].size
            if dataset["lat"].size != n or dataset["lon"].size != n:
                raise ValueError("lat, lon and time must have equal row counts")
            depth = np.asarray(dataset["depth"].values, dtype=np.float64)
            if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(np.diff(depth) <= 0):
                raise ValueError("depth must be finite, nonnegative and strictly increasing")
            for name in ("temp", "salt", "pres"):
                if dataset[name].ndim != 2 or dataset[name].shape != (n, len(depth)):
                    raise ValueError(f"{name} must have positional shape (profiles, depth)")
            self._depth_dim = dataset["temp"].dims[1]
            if any(dataset[name].dims[1] != self._depth_dim for name in ("salt", "pres")):
                raise ValueError("temp, salt and pres must use the same depth dimension")
            if dataset["depth"].dims != (self._depth_dim,):
                raise ValueError("depth must describe the shared profile depth dimension")
            self._shallow_indices = np.flatnonzero(depth <= self.max_surface_depth)
            self._depth = depth
            original_units = dataset["time"].attrs.get("units")
            units = self.time_units_override if self.time_units_override is not None else original_units
            calendar = dataset["time"].attrs.get("calendar", "standard")
            self._times = self._decode_times(np.asarray(dataset["time"].values, dtype=np.float64), units, calendar)
            coords = np.column_stack([np.asarray(dataset[name].values, dtype=np.float64)
                                      for name in ("lat", "lon")])
            invalid_coords = (~np.isfinite(coords).all(axis=1) | (np.abs(coords[:, 0]) > 90)
                              | (np.abs(coords[:, 1]) > 360))
            coords[invalid_coords] = np.nan
            self._coords = normalize_coordinates(coords)
            file_metadata = {
                **_AOTD_PROVENANCE,
                "format": "AOTD NetCDF", "entity_kind": "profile",
                "time_units_original": original_units, "time_units_effective": units,
                "time_units_override": self.time_units_override,
                "calendar": calendar, "profile_count": n,
                "source_variable_attrs": {name: dict(dataset[name].attrs)
                                          for name in ("temp", "salt", "pres", "depth")},
                "surface_selection": "shallowest_valid_level_independently",
                "source_fill_rule": "temp missing and salt == 0 and pres == 0",
                "max_surface_depth": self.max_surface_depth,
            }
        digest = sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self._file_hash = digest.hexdigest()
        self._file_stat = (self.path.stat().st_size, self.path.stat().st_mtime_ns)
        file_metadata["file_sha256"] = self._file_hash
        self._file_metadata = file_metadata
        diagnostics, profile_diagnostics, descriptors = [], {}, []
        for row in range(len(self._times)):
            if np.isnat(self._times[row]):
                diagnostics.append({"kind": "invalid_time", "file": str(self.path), "profile_row": row})
                continue
            current = []
            if invalid_coords[row]:
                current.append({"kind": "invalid_coordinates", "file": str(self.path),
                                "profile_row": row, "time": str(self._times[row])})
                diagnostics.extend(current)
            profile_diagnostics[row] = current
            identity = f"{self.path.stem}:{self._file_hash[:12]}:{row}"
            descriptor = BuoyDescriptor(
                f"{self.source}:{identity}", self.source, identity, (self.path,), _SURFACE_VARIABLES,
                metadata={**_AOTD_PROVENANCE, "source": self.source, "entity_kind": "profile", "profile_row": row,
                          "files": (str(self.path),), "file_sha256": self._file_hash,
                          "max_surface_depth": self.max_surface_depth,
                          "file_metadata": {str(self.path): file_metadata}},
                start_time=self._times[row], end_time=self._times[row],
            )
            descriptors.append(descriptor)
            self._rows[descriptor.key] = row
        self._profile_diagnostics = profile_diagnostics
        self._catalog = tuple(descriptors)
        self._descriptors = {item.key: item for item in descriptors}
        self.metadata = {
            **_AOTD_PROVENANCE, "source": self.source, "file_metadata": {str(self.path): file_metadata},
            "row_counts": {"rows_read": len(self._times), "rows_kept": len(descriptors),
                           "rows_skipped_invalid_time": len(self._times) - len(descriptors)},
            "diagnostics": diagnostics, "diagnostic_counts": dict(Counter(d["kind"] for d in diagnostics)),
        }

    def _ensure_surface(self, requested):
        missing = tuple(name for name in requested if name not in self._surface_cache)
        if not missing:
            return
        n = len(self._times)
        if not len(self._shallow_indices):
            for name in missing:
                self._surface_cache[name] = (np.full(n, np.nan), np.full(n, np.nan))
            return
        current_stat = self.path.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != self._file_stat:
            raise ValueError("file changed after discovery; construct a new AotdNetCDFSource")
        with xr.open_dataset(self.path, engine="h5netcdf", decode_times=False) as dataset:
            fields = {"temp"}
            if "sss" in missing:
                fields.update(("salt", "pres"))
            arrays = {name: np.asarray(dataset[name].isel(
                {self._depth_dim: self._shallow_indices}).values, dtype=np.float64)
                for name in fields}
        if "sss" in missing:
            fill = np.isnan(arrays["temp"]) & (arrays["salt"] == 0) & (arrays["pres"] == 0)
            arrays["salt"][fill] = np.nan
            self._source_fill_counts = fill.sum(axis=1)
            self.metadata["source_fill_cells_masked"] = int(fill.sum())
            self._file_metadata["source_fill_cells_masked"] = int(fill.sum())
        for name in missing:
            values = arrays["temp" if name == "sst" else "salt"]
            finite = np.isfinite(values)
            has_value = finite.any(axis=1)
            index = finite.argmax(axis=1)
            result = np.full(n, np.nan)
            depths = np.full(n, np.nan)
            rows = np.flatnonzero(has_value)
            result[rows] = values[rows, index[rows]]
            depths[rows] = self._depth[self._shallow_indices[index[rows]]]
            self._surface_cache[name] = (result, depths)

    def read(self, key: str, *, variables: Sequence[str] = _SURFACE_VARIABLES,
             start=None, stop=None) -> NativeBuoyData:
        requested = _surface_variables(variables)
        self.discover()
        if key not in self._descriptors:
            raise KeyError(f"Unknown AOTD profile key: {key!r}")
        try:
            self._ensure_surface(requested)
        except Exception as exc:
            raise ValueError(f"Cannot read AOTD file {self.path}: {exc}") from exc
        row = self._rows[key]
        times = self._times[row:row + 1]
        series = {name: ScalarSeries(times, self._surface_cache[name][0][row:row + 1],
                                     depths=self._surface_cache[name][1][row:row + 1])
                  for name in requested}
        diagnostics = list(self._profile_diagnostics[row])
        if "sss" in requested and getattr(self, "_source_fill_counts", None) is not None:
            if self._source_fill_counts[row]:
                diagnostics.append({"kind": "source_fill_masked", "file": str(self.path),
                                    "profile_row": row, "time": str(times[0]), "variable": "sss",
                                    "count": int(self._source_fill_counts[row])})
        for name in requested:
            if not np.isfinite(self._surface_cache[name][0][row]):
                diagnostics.append({"kind": "no_surface_value", "file": str(self.path),
                                    "profile_row": row, "time": str(times[0]), "variable": name})
        native = NativeBuoyData(
            self._descriptors[key], PositionSeries(times, self._coords[row:row + 1]), series,
            metadata={**_AOTD_PROVENANCE, "source": self.source, "entity_kind": "profile", "profile_row": row,
                      "files": (str(self.path),), "file_metadata": {str(self.path): self._file_metadata},
                      "row_counts": {"rows_read": 1, "rows_kept": 1}, "diagnostics": diagnostics},
        )
        return slice_native_data(native, start=start, stop=stop)
