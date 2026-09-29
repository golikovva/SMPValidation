"""Shared normalization, duplicate reconciliation and bounded time selection."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Number, Real
from typing import Sequence

import numpy as np
import pandas as pd

from .buoy_types import DriftSupport, NativeBuoyData, PositionSeries, ScalarSeries


def normalize_datetimes(values, *, allow_empty=True, strict=False) -> np.ndarray:
    """Convert a sequence to UTC ns; never interpret bare numbers as epoch times."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("Times must be a one-dimensional sequence.")
    if not len(raw):
        if not allow_empty:
            raise ValueError("At least one query time is required.")
        return np.empty(0, dtype="datetime64[ns]")
    if raw.dtype.kind in "biuf" or any(isinstance(v, Number) for v in raw):
        raise ValueError("Times must be datetimes, not bare numbers.")
    try:
        times = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="mixed"))
        result = times.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid datetime sequence: {exc}") from exc
    if np.isnat(result).any():
        raise ValueError("NaT is not an observation or query time.")
    if strict and len(result) > 1 and np.any(result[1:] <= result[:-1]):
        raise ValueError("Times must be strictly increasing and unique.")
    return result


def utc_time(value) -> np.datetime64:
    return normalize_datetimes([value], allow_empty=False)[0]


def fixed_timedelta(value, *, name="step", allow_zero=False) -> pd.Timedelta:
    if value is None or isinstance(value, (Number, bool)):
        raise ValueError(f"{name} must be an explicit fixed time duration.")
    try:
        duration = pd.Timedelta(value)
        ns = duration.value
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite fixed time duration.") from exc
    if pd.isna(duration) or ns < 0 or (ns == 0 and not allow_zero):
        adjective = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {adjective} duration.")
    return duration


def normalize_coordinates(coords) -> np.ndarray:
    """Normalize longitude to [-180, 180); an invalid position invalidates the pair."""
    result = np.asarray(coords, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("Coordinates must have shape (J, 2) in [latitude, longitude] order.")
    good = np.isfinite(result).all(axis=1) & (np.abs(result[:, 0]) <= 90)
    result[~good] = np.nan
    result[good, 1] = (result[good, 1] + 180.0) % 360.0 - 180.0
    return result


def _duplicate_groups(times):
    unique, first, counts = np.unique(times, return_index=True, return_counts=True)
    return unique, first, counts


def _merge_positions(parts, diagnostics):
    times = normalize_datetimes(np.concatenate([p.datetimes for p in parts]))
    coords = normalize_coordinates(np.concatenate([p.coords for p in parts]))
    order = np.argsort(times, kind="stable")
    times, coords = times[order], coords[order]
    unique, first, counts = _duplicate_groups(times)
    out = coords[first].copy()
    for i in np.flatnonzero(counts > 1):
        rows = coords[first[i]:first[i] + counts[i]]
        rows = rows[np.isfinite(rows).all(axis=1)]
        distinct = np.unique(rows, axis=0)
        if len(distinct) == 1:
            out[i] = distinct[0]
        else:
            out[i] = np.nan
            if len(distinct) > 1:
                diagnostics.append({"kind": "conflicting_coordinates", "time": str(unique[i])})
    return PositionSeries(unique, out)


def _merge_scalar(parts, name, diagnostics):
    times = normalize_datetimes(np.concatenate([p.datetimes for p in parts]))
    values = np.concatenate([p.values for p in parts]).astype(np.float64)
    uncertainty = np.concatenate([p.uncertainty for p in parts]).astype(np.float64)
    depths = np.concatenate([p.depths for p in parts]).astype(np.float64)
    values[~np.isfinite(values)] = np.nan
    uncertainty[~np.isfinite(uncertainty) | ~np.isfinite(values)] = np.nan
    depths[~np.isfinite(depths) | ~np.isfinite(values)] = np.nan
    order = np.argsort(times, kind="stable")
    times, values, uncertainty = times[order], values[order], uncertainty[order]
    depths = depths[order]
    unique, first, counts = _duplicate_groups(times)
    out, error = values[first].copy(), uncertainty[first].copy()
    out_depth = depths[first].copy()
    for i in np.flatnonzero(counts > 1):
        block = slice(first[i], first[i] + counts[i])
        vals = values[block]
        distinct = np.unique(vals[np.isfinite(vals)])
        if len(distinct) == 1:
            out[i] = distinct[0]
            known_depths = np.unique(depths[block][np.isfinite(depths[block])])
            out_depth[i] = known_depths[0] if len(known_depths) == 1 else np.nan
            if len(known_depths) > 1:
                out[i], error[i] = np.nan, np.nan
                diagnostics.append({"kind": "conflicting_depths", "variable": name, "time": str(unique[i])})
                continue
            errors = uncertainty[block][np.isfinite(vals)]
            errors = np.unique(errors[np.isfinite(errors)])
            error[i] = errors[0] if len(errors) == 1 else np.nan
            if len(errors) > 1:
                diagnostics.append({
                    "kind": "conflicting_uncertainty", "variable": name, "time": str(unique[i]),
                })
        else:
            out[i], error[i], out_depth[i] = np.nan, np.nan, np.nan
            if len(distinct) > 1:
                diagnostics.append({"kind": "conflicting_values", "variable": name, "time": str(unique[i])})
    return ScalarSeries(unique, out, error, depths=out_depth)


def merge_native_data(parts: Sequence[NativeBuoyData]) -> NativeBuoyData:
    """Combine files of one source key, preserving conflicts and native stream clocks.

    Conflicting finite duplicates are invalidated, not arbitrarily overwritten.
    Existing conflict diagnostics act as tombstones when already merged data is
    combined again, so a later file cannot silently resurrect a rejected value.
    """
    if not parts:
        raise ValueError("At least one native record is required.")
    descriptor = parts[0].descriptor
    if any(p.descriptor.key != descriptor.key for p in parts):
        raise ValueError("Only records of the same source-qualified buoy can be merged.")
    if any((p.descriptor.source, p.descriptor.source_buoy_id) !=
           (descriptor.source, descriptor.source_buoy_id) for p in parts):
        raise ValueError(f"Inconsistent identity metadata for {descriptor.key}.")
    files = tuple(sorted({f for p in parts for f in p.descriptor.files}, key=str))
    names = sorted({name for p in parts for name in p.series})
    diagnostics = [dict(d) for p in parts for d in p.metadata.get("diagnostics", ())]
    positions = _merge_positions([p.positions for p in parts], diagnostics)
    series = {
        name: _merge_scalar([p.series[name] for p in parts if name in p.series], name, diagnostics)
        for name in names
    }
    # Preserve invalidation when merging a previously normalized result. Apply
    # groups in bulk: IABP can contain thousands of conflicts in a single track.
    tombstones = {}
    for diagnostic in diagnostics:
        when = diagnostic.get("time")
        kind = diagnostic.get("kind")
        if when is not None and kind in (
            "conflicting_coordinates", "conflicting_values", "conflicting_uncertainty", "conflicting_depths",
        ):
            tombstones.setdefault((kind, diagnostic.get("variable")), []).append(when)
    for (kind, name), rejected in tombstones.items():
        rejected_times = np.asarray(rejected, dtype="datetime64[ns]")
        if kind == "conflicting_coordinates":
            positions.coords[np.isin(positions.datetimes, rejected_times)] = np.nan
        else:
            scalar = series.get(name)
            if scalar is not None:
                selected = np.isin(scalar.datetimes, rejected_times)
                scalar.uncertainty[selected] = np.nan
                if kind in ("conflicting_values", "conflicting_depths"):
                    scalar.values[selected] = np.nan
                    scalar.depths[selected] = np.nan
    # Deduplicate diagnostics produced by repeated normalization.
    unique_diagnostics = []
    seen = set()
    for diagnostic in diagnostics:
        identity = repr(sorted(diagnostic.items()))
        if identity not in seen:
            unique_diagnostics.append(diagnostic)
            seen.add(identity)
    metadata = dict(descriptor.metadata)
    metadata.update(parts[0].metadata)
    metadata.update({
        "source": descriptor.source, "source_buoy_id": descriptor.source_buoy_id,
        "files": tuple(str(f) for f in files), "diagnostics": unique_diagnostics,
    })
    if any("row_counts" in part.metadata for part in parts):
        row_counts = Counter()
        for part in parts:
            row_counts.update(part.metadata.get("row_counts", {}))
        metadata["row_counts"] = dict(row_counts)
    if any("diagnostic_counts" in part.metadata or "row_counts" in part.metadata for part in parts):
        diagnostic_counts = Counter()
        for diagnostic in unique_diagnostics:
            diagnostic_counts[diagnostic["kind"]] += diagnostic.get("count", 1)
        metadata["diagnostic_counts"] = dict(diagnostic_counts)
    file_metadata = {}
    for part in parts:
        file_metadata.update(part.descriptor.metadata.get("file_metadata", {}))
        file_metadata.update(part.metadata.get("file_metadata", {}))
    if file_metadata:
        metadata["file_metadata"] = file_metadata
    # Keep per-file processing information, including differing headers/versions.
    records = []
    for part in parts:
        if "source_records" in part.metadata:
            records.extend(part.metadata["source_records"])
        else:
            records.append({
                "files": tuple(str(f) for f in part.metadata.get("files", part.descriptor.files)),
                "metadata": {**part.descriptor.metadata,
                             **{k: v for k, v in part.metadata.items() if k != "diagnostics"}},
            })
    metadata["source_records"] = records
    axes = [positions.datetimes] + [s.datetimes for s in series.values()]
    all_times = np.concatenate(axes)
    descriptor = replace(
        descriptor, files=files,
        variables=tuple(sorted({name for p in parts for name in p.descriptor.variables})),
        start_time=all_times.min() if len(all_times) else None,
        end_time=all_times.max() if len(all_times) else None,
    )
    return NativeBuoyData(descriptor, positions, series, metadata)


def slice_native_data(data: NativeBuoyData, start=None, stop=None) -> NativeBuoyData:
    """Copy each native stream within [start, stop), without aligning its clock."""
    left = utc_time(start) if start is not None else None
    right = utc_time(stop) if stop is not None else None
    if left is not None and right is not None and right < left:
        raise ValueError("stop must not precede start.")

    def mask(times):
        selected = np.ones(len(times), dtype=bool)
        if left is not None:
            selected &= times >= left
        if right is not None:
            selected &= times < right
        return selected

    take = mask(data.positions.datetimes)
    positions = PositionSeries(data.positions.datetimes[take], data.positions.coords[take])
    series = {}
    for name, scalar in data.series.items():
        take = mask(scalar.datetimes)
        series[name] = ScalarSeries(
            scalar.datetimes[take], scalar.values[take], scalar.uncertainty[take],
            depths=scalar.depths[take],
        )
    return NativeBuoyData(deepcopy(data.descriptor), positions, series, deepcopy(dict(data.metadata)))


def select_time_indices(source_times, query_times, *, method="exact", tolerance=None):
    """Indices into a sorted, valid source clock; -1 means no eligible observation.

    Nearest ties select the earlier record. Unsigned differences avoid nanosecond
    overflow for distant dates at opposite ends of the datetime64[ns] range.
    """
    result = np.full(len(query_times), -1, dtype=np.int64)
    if not len(source_times) or not len(query_times):
        return result
    at = np.searchsorted(source_times, query_times)
    if method == "exact":
        bounded = np.minimum(at, len(source_times) - 1)
        good = (at < len(source_times)) & (source_times[bounded] == query_times)
        result[good] = bounded[good]
        return result
    if method != "nearest" or tolerance is None:
        raise ValueError("Time selection requires exact, or nearest with a tolerance.")
    before = np.maximum(at - 1, 0)
    after = np.minimum(at, len(source_times) - 1)
    offset = np.uint64(1 << 63)
    source = source_times.view(np.int64).astype(np.uint64) ^ offset
    query = query_times.view(np.int64).astype(np.uint64) ^ offset
    maximum = np.iinfo(np.uint64).max
    distance_before = np.where(at > 0, query - source[before], maximum)
    distance_after = np.where(at < len(source), source[after] - query, maximum)
    earlier = distance_before <= distance_after
    chosen = np.where(earlier, before, after)
    distance = np.minimum(distance_before, distance_after)
    good = distance <= np.uint64(tolerance.value)
    result[good] = chosen[good]
    return result


@dataclass(frozen=True)
class _NativeDrift:
    """Native segment starts, including invalid anchors that preserve gaps."""

    datetimes: np.ndarray
    values: np.ndarray  # speed, eastward, northward, all in cm/s
    support: DriftSupport
    metadata: dict
    rejected_speed_segments: tuple[dict, ...] = ()


def validate_drift_max_speed(value) -> float | None:
    """Validate an optional positive drift magnitude limit in cm/s."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("drift_max_speed must be a positive finite number in cm/s or None.")
    try:
        limit = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("drift_max_speed must be a positive finite number in cm/s or None.") from exc
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("drift_max_speed must be a positive finite number in cm/s or None.")
    return limit


def derive_drift(positions: PositionSeries, max_gap, *, max_speed=100.0) -> _NativeDrift:
    """Compute WGS84 drift in cm/s before resampling, preserving position breaks.

    Positions must already be merged and sorted, including invalid records.
    Magnitudes strictly above max_speed are rejected (None disables this check).
    Rejected segment starts remain in the cache so nearest selection cannot
    silently substitute a neighbouring usable segment.
    """
    # Keep the existing numerical kernel and its cm/s API unchanged. Lazy import
    # also avoids adding pyproj to scalar-only source reading.
    from ..iabp_utils import drift_uv_cm_s_from_latlon

    gap = fixed_timedelta(max_gap, name="drift_max_gap")
    limit = validate_drift_max_speed(max_speed)
    times = positions.datetimes
    coords = positions.coords
    if np.isnat(times).any() or np.any(times[1:] <= times[:-1]):
        raise ValueError("Drift requires sorted, unique, valid position times.")
    # Unsigned subtraction is exact even across the full datetime64[ns] range.
    ticks = times.view(np.int64).astype(np.uint64)
    durations = ticks[1:] - ticks[:-1]
    dt_seconds = durations.astype(np.float64) / 1e9
    gap_ns = gap.value
    good_coords = np.isfinite(coords).all(axis=1)
    good_pairs = good_coords[:-1] & good_coords[1:]
    within_gap = durations <= gap_ns
    east, north, speed, distance = drift_uv_cm_s_from_latlon(
        coords[:, 0], coords[:, 1], dt_seconds,
        return_distance=True,
    )
    values = np.column_stack((speed[:-1], east[:-1], north[:-1]))
    finite = np.isfinite(values).all(axis=1)
    eligible = good_pairs & within_gap & finite
    speed_exceeded = (eligible & (speed[:-1] > limit) if limit is not None
                      else np.zeros(len(durations), dtype=bool))
    usable = eligible & ~speed_exceeded
    rows = np.flatnonzero(usable)
    support = DriftSupport.empty((len(durations),))
    support.coords[rows] = np.stack((coords[rows], coords[rows + 1]), axis=1)
    support.datetimes[rows] = np.stack((times[rows], times[rows + 1]), axis=1)
    support.valid[rows] = True
    rejected = tuple({
        "kind": "drift_speed_exceeded",
        "t0": times[row], "t1": times[row + 1],
        "coords0": tuple(coords[row]), "coords1": tuple(coords[row + 1]),
        "dt_seconds": float(dt_seconds[row]),
        "distance_m": float(distance[row]),
        "speed_cm_s": float(speed[row]), "limit_cm_s": limit,
    } for row in np.flatnonzero(speed_exceeded))
    metadata = {
        "method": "consecutive_native_positions",
        "ellipsoid": "WGS84", "anchor": "start", "units": "cm/s",
        "components": "local_east_north_at_start",
        "drift_max_gap": str(gap), "drift_max_gap_ns": gap_ns,
        "drift_max_speed_cm_s": limit,
        "candidate_segments": len(durations), "valid_segments": len(rows),
        "rejected_segments": int((~usable).sum()),
        "invalid_position_segments": int((~good_pairs).sum()),
        "gap_exceeded_segments": int((good_pairs & ~within_gap).sum()),
        "nonfinite_velocity_segments": int((good_pairs & within_gap & ~finite).sum()),
        "speed_exceeded_segments": int(speed_exceeded.sum()),
    }
    values[~usable] = np.nan
    return _NativeDrift(times[:-1].copy(), values, support, metadata, rejected)
