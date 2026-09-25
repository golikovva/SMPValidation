"""Source-independent contracts for irregular buoy observations and aligned batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class VariableSpec:
    name: str
    units: str = "m"
    geometry: str = "scalar"


SCALAR_VARIABLES = {
    name: VariableSpec(name)
    for name in ("ice_thickness", "snow_thickness")
}


@dataclass(frozen=True)
class BuoyDescriptor:
    key: str
    source: str
    source_buoy_id: str
    files: tuple[Path, ...]
    variables: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    start_time: np.datetime64 | None = None
    end_time: np.datetime64 | None = None


@dataclass(frozen=True)
class PositionSeries:
    """Native position clock and [latitude, longitude] pairs, without resampling."""

    datetimes: np.ndarray
    coords: np.ndarray

    def __post_init__(self):
        times = np.asarray(self.datetimes, dtype="datetime64[ns]")
        coords = np.asarray(self.coords, dtype=np.float64)
        if times.ndim != 1 or coords.shape != (len(times), 2):
            raise ValueError("PositionSeries requires datetimes (J,) and coords (J, 2).")
        object.__setattr__(self, "datetimes", times)
        object.__setattr__(self, "coords", coords)


@dataclass(frozen=True)
class ScalarSeries:
    """One physical variable on its own native clock; unknown uncertainty is NaN."""

    datetimes: np.ndarray
    values: np.ndarray
    uncertainty: np.ndarray | None = None

    def __post_init__(self):
        times = np.asarray(self.datetimes, dtype="datetime64[ns]")
        values = np.asarray(self.values, dtype=np.float64)
        uncertainty = (
            np.full(values.shape, np.nan, dtype=np.float64)
            if self.uncertainty is None
            else np.asarray(self.uncertainty, dtype=np.float64)
        )
        if times.ndim != 1 or values.shape != times.shape or uncertainty.shape != times.shape:
            raise ValueError("ScalarSeries times, values and uncertainty must all have shape (J,).")
        object.__setattr__(self, "datetimes", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True)
class NativeBuoyData:
    """Irregular streams; positions and each measurement retain independent clocks."""

    descriptor: BuoyDescriptor
    positions: PositionSeries
    series: Mapping[str, ScalarSeries]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BuoySource(Protocol):
    def discover(self) -> Iterable[BuoyDescriptor]: ...

    def read(
        self,
        key: str,
        *,
        variables: Sequence[str],
        start=None,
        stop=None,
    ) -> NativeBuoyData: ...


@dataclass(frozen=True)
class BuoyWindow:
    bid: str
    coords: np.ndarray
    datetimes: np.ndarray
    variables: np.ndarray
    var_names: tuple[str, ...]
    units: tuple[str, ...]
    coord_valid: np.ndarray
    valid: np.ndarray
    coord_times: np.ndarray
    value_times: np.ndarray
    uncertainty: np.ndarray
    metadata: Mapping[str, Any]

    def var(self, name: str) -> np.ndarray:
        try:
            return self.variables[:, self.var_names.index(name)]
        except ValueError:
            raise KeyError(f"Unknown variable {name!r}; available: {self.var_names}") from None


@dataclass(frozen=True)
class BuoyBatch:
    """Aligned point observations. All times are timezone-naive UTC nanoseconds.

    Dimensions are always buoy, query time, variable. Source clocks remain in
    coord_times/value_times. Arrays are NumPy arrays, not deeply immutable.
    """

    bids: np.ndarray
    coords: np.ndarray
    datetimes: np.ndarray
    variables: np.ndarray
    var_names: tuple[str, ...]
    units: tuple[str, ...]
    coord_valid: np.ndarray
    valid: np.ndarray
    coord_times: np.ndarray
    value_times: np.ndarray
    uncertainty: np.ndarray
    metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    _bid_to_i: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        dtypes = {
            "bids": str, "coords": np.float32, "datetimes": "datetime64[ns]",
            "variables": np.float32, "coord_valid": bool, "valid": bool,
            "coord_times": "datetime64[ns]", "value_times": "datetime64[ns]",
            "uncertainty": np.float32,
        }
        for name, dtype in dtypes.items():
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        object.__setattr__(self, "var_names", tuple(self.var_names))
        object.__setattr__(self, "units", tuple(self.units))
        if self.bids.ndim != 1 or self.datetimes.ndim != 1:
            raise ValueError("bids and datetimes must be one-dimensional.")
        n, t, v = len(self.bids), len(self.datetimes), len(self.var_names)
        if len(set(self.bids)) != n or len(set(self.var_names)) != v:
            raise ValueError("Buoy IDs and variable names must be unique.")
        if len(self.units) != v:
            raise ValueError("units must have one entry per variable.")
        shapes = {
            "coords": (n, t, 2), "variables": (n, t, v),
            "coord_valid": (n, t), "valid": (n, t, v),
            "coord_times": (n, t), "value_times": (n, t, v),
            "uncertainty": (n, t, v),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}.")
        if np.isnat(self.datetimes).any() or (
            t > 1 and np.any(self.datetimes[1:] <= self.datetimes[:-1])
        ):
            raise ValueError("Query datetimes must be valid, increasing and unique.")
        if not np.array_equal(self.valid, np.isfinite(self.variables)):
            raise ValueError("valid must match finite measurements.")
        if not np.array_equal(self.coord_valid, np.isfinite(self.coords).all(axis=-1)):
            raise ValueError("coord_valid must match finite position pairs.")
        if not np.array_equal(self.valid, ~np.isnat(self.value_times)):
            raise ValueError("value_times must be NaT exactly where measurements are missing.")
        if not np.array_equal(self.coord_valid, ~np.isnat(self.coord_times)):
            raise ValueError("coord_times must be NaT exactly where positions are missing.")
        object.__setattr__(self, "_bid_to_i", {bid: i for i, bid in enumerate(self.bids)})

    def __len__(self) -> int:
        return len(self.bids)

    def __getitem__(self, key: int | str) -> BuoyWindow:
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            i = int(key)
            if i < 0:
                i += len(self)
            if i < 0 or i >= len(self):
                raise IndexError(key)
        else:
            try:
                i = self._bid_to_i[str(key)]
            except KeyError:
                raise KeyError(f"Buoy {key!r} is not in this batch.") from None
        bid = str(self.bids[i])
        return BuoyWindow(
            bid=bid, coords=self.coords[i], datetimes=self.datetimes,
            variables=self.variables[i], var_names=self.var_names, units=self.units,
            coord_valid=self.coord_valid[i], valid=self.valid[i],
            coord_times=self.coord_times[i], value_times=self.value_times[i],
            uncertainty=self.uncertainty[i], metadata=self.metadata.get(bid, {}),
        )

    def var(self, name: str) -> np.ndarray:
        try:
            return self.variables[:, :, self.var_names.index(name)]
        except ValueError:
            raise KeyError(f"Unknown variable {name!r}; available: {self.var_names}") from None

    @classmethod
    def empty(cls, datetimes, var_names, units) -> BuoyBatch:
        times = np.asarray(datetimes, dtype="datetime64[ns]")
        t, v = len(times), len(var_names)
        return cls(
            bids=np.empty(0, dtype=str), coords=np.empty((0, t, 2), np.float32),
            datetimes=times, variables=np.empty((0, t, v), np.float32),
            var_names=tuple(var_names), units=tuple(units),
            coord_valid=np.empty((0, t), bool), valid=np.empty((0, t, v), bool),
            coord_times=np.empty((0, t), dtype="datetime64[ns]"),
            value_times=np.empty((0, t, v), dtype="datetime64[ns]"),
            uncertainty=np.empty((0, t, v), np.float32),
        )
