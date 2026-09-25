"""Readers for native SIMBA and CRREL scalar buoy products.

Readers retain native timestamps.  Alignment to a requested time grid belongs
to :class:`BuoyDataset`, not to the file format adapters.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
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
from .buoy_utils import merge_native_data, normalize_datetimes, slice_native_data


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
            },
        )
