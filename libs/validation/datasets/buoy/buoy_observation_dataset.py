"""Time-window selection of scalar buoy observations, independent of model grids."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .buoy_types import (
    BuoyBatch, BuoySource, DriftSupport, NativeBuoyData, SCALAR_VARIABLES, DRIFT_VARIABLES,
)
from .buoy_utils import (
    derive_drift,
    fixed_timedelta,
    merge_native_data,
    normalize_datetimes,
    select_time_indices,
    slice_native_data,
    utc_time,
    validate_drift_max_speed,
)


class BuoyObservationDataset:
    """Combine source-native observations into arrays ordered (buoy, time, variable).

    Native scalars are loaded once. No spatial gridding, temporal interpolation,
    or aggregation is performed. Drift is derived from consecutive native fixes
    before temporal matching. Bare timestamps denote UTC.
    ``nearest`` requires an explicit tolerance and selects earlier ties. It
    searches finite scalars and positions separately. For drift, it selects the
    closest native segment start first, then checks validity; rejected segments
    remain gaps. Drift magnitudes above drift_max_speed (100 cm/s by default)
    are rejected jointly with both components. None disables the speed limit.

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
        drift_max_gap=None,
        drift_max_speed=100.0,
        coverage="partial",
        start_times=None,
        name="buoy_observations",
    ):
        self.name = name
        self.variables = self._variable_names(variables, "variables")
        specs = {**SCALAR_VARIABLES, **DRIFT_VARIABLES}
        unknown = set(self.variables) - specs.keys()
        if unknown:
            raise ValueError(f"Unsupported observation variables: {sorted(unknown)}.")
        self._drift_indices = [j for j, name in enumerate(self.variables) if name in DRIFT_VARIABLES]
        self.drift_max_speed = validate_drift_max_speed(drift_max_speed)
        self.drift_max_gap = (
            fixed_timedelta(drift_max_gap, name="drift_max_gap")
            if drift_max_gap is not None else None
        )
        if self._drift_indices and self.drift_max_gap is None:
            raise ValueError("Drift variables require an explicit positive drift_max_gap.")
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
        self.units = tuple(specs[name].units for name in self.variables)
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
                selected = tuple(name for name in self.variables
                                 if name in descriptor.variables and name not in DRIFT_VARIABLES)
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
        self._drift = ({key: derive_drift(data.positions, self.drift_max_gap,
                                        max_speed=self.drift_max_speed)
                        for key, data in self._data.items()} if self._drift_indices else {})
        self.catalog = {key: data.descriptor for key, data in self._data.items()}
        # Conservative bounds on usable requested measurements. In particular,
        # do not allocate a query-sized row for every profile in a large archive.
        self._observation_bounds = {}
        for key, data in self._data.items():
            clocks = [scalar.datetimes[np.isfinite(scalar.values)]
                      for scalar in data.series.values()]
            if key in self._drift:
                drift = self._drift[key]
                clocks.append(drift.datetimes[drift.support.valid])
            clocks = [clock for clock in clocks if len(clock)]
            self._observation_bounds[key] = (
                (min(int(clock.min().astype(np.int64)) for clock in clocks),
                 max(int(clock.max().astype(np.int64)) for clock in clocks))
                if clocks else None
            )
        if supplied_times is not None:
            self._times = supplied_times
        else:
            available = [
                scalar.datetimes[np.isfinite(scalar.values)]
                for data in self._data.values() for scalar in data.series.values()
            ]
            available.extend(drift.datetimes[drift.support.valid] for drift in self._drift.values())
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

    def read_drift_diagnostics(self, bid: str | None = None) -> list[dict]:
        """Return copied speed-rejection records for one buoy, or all buoys.

        Endpoints are (latitude, longitude); speed and limit are in cm/s,
        distance in metres and duration in seconds. Records cover the complete
        native history. Batch metadata contains counters only to avoid copying
        this history into every validation window and metric.
        """
        if bid is not None and bid not in self._data:
            raise KeyError(f"Unknown buoy {bid!r}; available: {self.buoy_ids}")
        keys = self.buoy_ids if bid is None else (bid,)
        return [
            {"buoy_id": key, **deepcopy(entry)}
            for key in keys if key in self._drift
            for entry in self._drift[key].rejected_speed_segments
        ]

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

    def _select_drift(self, drift, times):
        # Keep invalid anchors in the search to prevent nearest from filling
        # rejected speed, coordinate or long-gap segments with other values.
        selected = select_time_indices(
            drift.datetimes, times, method=self.time_method, tolerance=self.tolerance,
        )
        found = selected >= 0
        found[found] &= drift.support.valid[selected[found]]
        return found, selected[found]

    def _batch(self, times):
        t, v = len(times), len(self.variables)
        if not t:
            return BuoyBatch.empty(times, self.variables, self.units)
        output = []
        margin = int(self.tolerance.value) if self.time_method == "nearest" else 0
        left = int(times[0].astype(np.int64)) - margin
        right = int(times[-1].astype(np.int64)) + margin
        for bid, data in self._data.items():
            bounds = self._observation_bounds[bid]
            if bounds is None or bounds[1] < left or bounds[0] > right:
                continue
            coords = np.full((t, 2), np.nan, dtype=np.float32)
            values = np.full((t, v), np.nan, dtype=np.float32)
            uncertainty = np.full((t, v), np.nan, dtype=np.float32)
            depths = np.full((t, v), np.nan, dtype=np.float32)
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
                depths[found, j] = scalar.depths[rows]
                value_times[found, j] = scalar.datetimes[rows]
            support = None
            if self._drift_indices:
                drift = self._drift[bid]
                support = DriftSupport.empty((t,))
                found, rows = self._select_drift(drift, times)
                for j in self._drift_indices:
                    column = tuple(DRIFT_VARIABLES).index(self.variables[j])
                    values[found, j] = drift.values[rows, column]
                    value_times[found, j] = drift.datetimes[rows]
                support.coords[found] = drift.support.coords[rows]
                support.datetimes[found] = drift.support.datetimes[rows]
                support.valid[found] = True
            valid = np.isfinite(values)
            coord_valid = np.isfinite(coords).all(axis=1)
            required = valid[:, self._required_indices]
            if self.coverage == "partial":
                include = np.any(coord_valid & required.any(axis=1))
            else:
                include = np.all(coord_valid & required.all(axis=1))
            if include:
                output.append((bid, coords, values, uncertainty, coord_valid, valid,
                               coord_times, value_times, support, depths))
        if not output:
            return BuoyBatch.empty(times, self.variables, self.units)
        bids = [row[0] for row in output]
        metadata = {bid: deepcopy(dict(self._data[bid].metadata)) for bid in bids}
        support = None
        if self._drift_indices:
            support = DriftSupport(
                coords=np.stack([r[8].coords for r in output]),
                datetimes=np.stack([r[8].datetimes for r in output]),
                valid=np.stack([r[8].valid for r in output]),
            )
            for bid in bids:
                metadata[bid]["drift"] = deepcopy(self._drift[bid].metadata)
        return BuoyBatch(
            bids=np.asarray(bids), coords=np.stack([r[1] for r in output]), datetimes=times.copy(),
            variables=np.stack([r[2] for r in output]), var_names=self.variables, units=self.units,
            uncertainty=np.stack([r[3] for r in output]), coord_valid=np.stack([r[4] for r in output]),
            valid=np.stack([r[5] for r in output]), coord_times=np.stack([r[6] for r in output]),
            value_times=np.stack([r[7] for r in output]), metadata=metadata, drift_support=support,
            depths=np.stack([r[9] for r in output]),
        )
