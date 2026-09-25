"""Time-window selection of scalar buoy observations, independent of model grids."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .buoy_types import BuoyBatch, BuoySource, NativeBuoyData, SCALAR_VARIABLES
from .buoy_utils import (
    fixed_timedelta,
    merge_native_data,
    normalize_datetimes,
    select_time_indices,
    slice_native_data,
    utc_time,
)


class BuoyObservationDataset:
    """Combine source-native observations into arrays ordered (buoy, time, variable).

    Native scalars are loaded once. No spatial gridding, temporal interpolation,
    aggregation or drift calculation is performed. Bare timestamps denote UTC.
    ``nearest`` requires an explicit tolerance and selects earlier ties. It
    searches finite observations separately for each variable and for positions.

    ``times`` indexes candidate window starts, not necessarily complete windows.
    Partial coverage requires a position and at least one required measurement
    at the same query time. ``valid`` itself describes measurements independently
    of the position mask. All query methods return a BuoyBatch, including misses.
    """

    def __init__(
        self,
        sources: Iterable[BuoySource],
        variables: Sequence[str] = ("ice_thickness", "snow_thickness"),
        *,
        required_variables: Sequence[str] | None = None,
        T: int = 1,
        step="1h",
        time_method="exact",
        tolerance=None,
        coord_tolerance=None,
        coverage="partial",
        start_times=None,
        name="buoy_observations",
    ):
        self.name = name
        self.variables = self._variable_names(variables, "variables")
        unknown = set(self.variables) - SCALAR_VARIABLES.keys()
        if unknown:
            raise ValueError(f"Unsupported scalar variables: {sorted(unknown)}.")
        self.required_variables = (
            self.variables if required_variables is None
            else self._variable_names(required_variables, "required_variables")
        )
        if not set(self.required_variables).issubset(self.variables):
            raise ValueError("required_variables must be a nonempty subset of variables.")
        if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)) or T <= 0:
            raise ValueError("T must be a positive integer sample count.")
        if time_method not in ("exact", "nearest"):
            raise ValueError("time_method must be 'exact' or 'nearest'.")
        if coverage not in ("partial", "complete"):
            raise ValueError("coverage must be 'partial' or 'complete'.")
        self.T = int(T)
        self.step = fixed_timedelta(step)
        self.time_method = time_method
        self.coverage = coverage
        self.tolerance = (
            fixed_timedelta(tolerance, name="tolerance", allow_zero=True)
            if tolerance is not None else None
        )
        if time_method == "nearest" and self.tolerance is None:
            raise ValueError("nearest requires an explicit finite tolerance.")
        self.coord_tolerance = (
            fixed_timedelta(coord_tolerance, name="coord_tolerance", allow_zero=True)
            if coord_tolerance is not None else self.tolerance
        )
        self.units = tuple(SCALAR_VARIABLES[name].units for name in self.variables)
        self._required_indices = [self.variables.index(name) for name in self.required_variables]
        supplied_times = (
            normalize_datetimes(start_times, strict=True) if start_times is not None else None
        )
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("At least one BuoySource is required.")

        parts = defaultdict(list)
        for source in self.sources:
            descriptors = tuple(source.discover())
            for descriptor in descriptors:
                selected = tuple(name for name in self.variables if name in descriptor.variables)
                native = source.read(descriptor.key, variables=selected)
                if native.descriptor.key != descriptor.key:
                    raise ValueError(f"Source returned the wrong buoy for {descriptor.key!r}.")
                missing = set(selected) - native.series.keys()
                if missing:
                    raise ValueError(f"{descriptor.key}: source omitted declared variables {sorted(missing)}.")
                parts[descriptor.key].append(NativeBuoyData(
                    native.descriptor, native.positions,
                    {name: native.series[name] for name in selected}, native.metadata,
                ))
        self._data = {key: merge_native_data(parts[key]) for key in sorted(parts)}
        self.catalog = {key: data.descriptor for key, data in self._data.items()}
        if supplied_times is not None:
            self._times = supplied_times
        else:
            available = [
                scalar.datetimes[np.isfinite(scalar.values)]
                for data in self._data.values() for scalar in data.series.values()
            ]
            self._times = (
                np.unique(np.concatenate(available)) if available
                else np.empty(0, dtype="datetime64[ns]")
            )

    @staticmethod
    def _variable_names(names, parameter):
        if isinstance(names, str):
            raise ValueError(f"{parameter} must be a sequence of variable names, not a string.")
        result = tuple(names)
        if not result or any(not isinstance(name, str) for name in result) or len(set(result)) != len(result):
            raise ValueError(f"{parameter} must contain unique variable names and be nonempty.")
        return result

    @property
    def times(self) -> np.ndarray:
        """Sorted candidate starts. A selected start may yield an empty batch."""
        return self._times.copy()

    @property
    def buoy_ids(self) -> tuple[str, ...]:
        return tuple(self._data)

    def __len__(self) -> int:
        return len(self._times)

    def __getitem__(self, key) -> BuoyBatch:
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            start = self._times[int(key)]
        else:
            start = utc_time(key)
        try:
            times = pd.date_range(pd.Timestamp(start), periods=self.T, freq=self.step)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Requested window is outside supported datetime bounds: {exc}") from exc
        return self.at(times)

    def at(self, times) -> BuoyBatch:
        """Select observations on a nonempty, strictly increasing query clock."""
        return self._batch(normalize_datetimes(times, allow_empty=False, strict=True))

    def between(self, start, stop, step=None) -> BuoyBatch:
        """Select [start, stop) at a fixed cadence (the configured step by default)."""
        left, right = utc_time(start), utc_time(stop)
        cadence = self.step if step is None else fixed_timedelta(step)
        if right < left:
            raise ValueError("stop must not precede start.")
        if right == left:
            return BuoyBatch.empty(np.empty(0, dtype="datetime64[ns]"), self.variables, self.units)
        times = pd.date_range(pd.Timestamp(left), pd.Timestamp(right), freq=cadence, inclusive="left")
        return self._batch(normalize_datetimes(times, strict=True))

    def read_native(self, bid: str, start=None, stop=None) -> NativeBuoyData:
        """Return copied native streams in [start, stop); clocks remain independent."""
        try:
            data = self._data[bid]
        except KeyError:
            raise KeyError(f"Unknown buoy {bid!r}; available: {self.buoy_ids}") from None
        return slice_native_data(data, start=start, stop=stop)

    def _select(self, native_times, values, times, tolerance):
        eligible = np.isfinite(values)
        if values.ndim == 2:
            eligible = eligible.all(axis=1)
        candidates = np.flatnonzero(eligible)
        selected = select_time_indices(
            native_times[candidates], times, method=self.time_method, tolerance=tolerance,
        )
        found = selected >= 0
        rows = candidates[selected[found]]
        return found, rows

    def _batch(self, times):
        t, v = len(times), len(self.variables)
        if not t:
            return BuoyBatch.empty(times, self.variables, self.units)
        output = []
        for bid, data in self._data.items():
            coords = np.full((t, 2), np.nan, dtype=np.float32)
            values = np.full((t, v), np.nan, dtype=np.float32)
            uncertainty = np.full((t, v), np.nan, dtype=np.float32)
            coord_times = np.full(t, np.datetime64("NaT", "ns"))
            value_times = np.full((t, v), np.datetime64("NaT", "ns"))
            position = data.positions
            found, rows = self._select(position.datetimes, position.coords, times, self.coord_tolerance)
            coords[found] = position.coords[rows]
            coord_times[found] = position.datetimes[rows]
            for j, name in enumerate(self.variables):
                scalar = data.series.get(name)
                if scalar is None:
                    continue
                found, rows = self._select(scalar.datetimes, scalar.values, times, self.tolerance)
                values[found, j] = scalar.values[rows]
                uncertainty[found, j] = scalar.uncertainty[rows]
                value_times[found, j] = scalar.datetimes[rows]
            valid = np.isfinite(values)
            coord_valid = np.isfinite(coords).all(axis=1)
            required = valid[:, self._required_indices]
            if self.coverage == "partial":
                include = np.any(coord_valid & required.any(axis=1))
            else:
                include = np.all(coord_valid & required.all(axis=1))
            if include:
                output.append((bid, coords, values, uncertainty, coord_valid, valid, coord_times, value_times))
        if not output:
            return BuoyBatch.empty(times, self.variables, self.units)
        bids = [row[0] for row in output]
        return BuoyBatch(
            bids=np.asarray(bids), coords=np.stack([r[1] for r in output]), datetimes=times.copy(),
            variables=np.stack([r[2] for r in output]), var_names=self.variables, units=self.units,
            uncertainty=np.stack([r[3] for r in output]), coord_valid=np.stack([r[4] for r in output]),
            valid=np.stack([r[5] for r in output]), coord_times=np.stack([r[6] for r in output]),
            value_times=np.stack([r[7] for r in output]),
            metadata={bid: deepcopy(dict(self._data[bid].metadata)) for bid in bids},
        )
