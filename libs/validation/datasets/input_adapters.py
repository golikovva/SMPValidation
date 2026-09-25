"""Prepare metric arguments without coupling numerical metrics to data readers.

ArrayInputAdapter is deliberately transparent for existing grid validation.
BuoyInputAdapter aligns already sampled point fields to one reference batch; it
does not interpolate space/time, read datasets, or infer axes from array sizes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


POINT_DIMS = ("buoy", "time", "variable")


@dataclass(frozen=True)
class ComparisonContext:
    geometry: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_dims: tuple[str, ...] | None = None
    result_units: tuple[str | None, ...] | None = None


@dataclass(frozen=True)
class PreparedInputs:
    arrays: tuple[np.ndarray, ...]
    valid: np.ndarray | None
    context: ComparisonContext
    input_dims: tuple[str, ...] | None = None

    @property
    def has_valid_samples(self) -> bool:
        if self.valid is not None:
            return bool(np.any(self.valid))
        # Legacy metrics own their missing-data semantics. Even an all-NaN grid
        # can be meaningful to a categorical/counting metric; do not mask it.
        return bool(self.arrays) and all(np.size(a) > 0 for a in self.arrays)


class MetricInputAdapter(Protocol):
    def prepare(self, inputs: Sequence[Any], *, metric) -> PreparedInputs: ...


class ArrayInputAdapter:
    """Preserve legacy arrays, NumPy broadcasting, subclasses and metadata merge.

    No joint finite mask is applied. Metrics keep their own missing-data rules;
    metadata keys from later arguments take precedence, as in the old Validator.
    """

    def prepare(self, inputs, *, metric) -> PreparedInputs:
        arrays, metadata = [], {}
        for value in inputs:
            if hasattr(value, "bids") and hasattr(value, "variables"):
                raise TypeError("Buoy batches require an explicit BuoyInputAdapter.")
            array = value if isinstance(value, np.ndarray) else np.asarray(value)
            if array.dtype.kind not in "biufc":
                raise TypeError("ArrayInputAdapter expects numeric arrays.")
            arrays.append(array)
            metadata.update(getattr(value, "meta", {}) or {})
        return PreparedInputs(tuple(arrays), None, ComparisonContext("array", metadata))


def _names(values, label):
    if isinstance(values, str):
        raise ValueError(f"{label} must be a sequence, not a string.")
    if np.asarray(values).ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional sequence.")
    result = tuple(str(v) for v in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique.")
    return result


def _times(values):
    # pandas is only needed in the opt-in buoy path.
    import pandas as pd
    raw = np.asarray(values)
    if raw.ndim != 1 or (len(raw) and raw.dtype.kind in "biuf"):
        raise ValueError("datetimes must be a one-dimensional datetime sequence.")
    times = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="mixed"))
    times = times.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    if np.isnat(times).any() or len(np.unique(times)) != len(times):
        raise ValueError("datetimes must be valid and unique.")
    return times


def _canonical(values, dims, label):
    dims = tuple(dims)
    values = np.asarray(values)
    if len(dims) != 3 or set(dims) != set(POINT_DIMS) or values.ndim != 3:
        raise ValueError(f"{label} needs three explicit axes: {POINT_DIMS}.")
    return np.transpose(values, [dims.index(dim) for dim in POINT_DIMS])


_UNIT_SCALES = {
    "m": ("length", 1.0), "cm": ("length", 0.01), "mm": ("length", 0.001),
    "m/s": ("speed", 1.0), "cm/s": ("speed", 0.01), "mm/s": ("speed", 0.001),
    "1": ("dimensionless", 1.0),
}
_UNIT_ALIASES = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "m s-1": "m/s", "m s^-1": "m/s", "cm s-1": "cm/s", "cm s^-1": "cm/s",
    "dimensionless": "1",
}


def _unit_name(unit):
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Every selected variable needs an explicit unit.")
    unit = unit.strip()
    return _UNIT_ALIASES.get(unit, unit)


def _unit_factor(source, target):
    source, target = _unit_name(source), _unit_name(target)
    if source == target:
        return 1.0
    left, right = _UNIT_SCALES.get(source), _UNIT_SCALES.get(target)
    if left is None or right is None or left[0] != right[0]:
        raise ValueError(f"Cannot convert incompatible or unsupported units {source!r} to {target!r}.")
    return left[1] / right[1]


def _squared(unit):
    if unit is None:
        return None
    return f"({unit})^2" if any(mark in unit for mark in ("/", "^", " ")) else f"{unit}^2"


def _metric_contract(metric, dims, units):
    """Known point-metric contracts; never guess reductions from output shape."""
    stages = getattr(metric, "_metrics", None)
    if stages is not None:
        joint = False
        for stage in stages:
            dims, units, stage_joint = _metric_contract(stage, dims, units)
            joint |= stage_joint
        return dims, units, joint
    classes = {cls.__name__ for cls in type(metric).__mro__}
    incompatible = classes & {"LocalStd", "DriftSuccess", "SIEvsSICConfusion", "VectorAngle", "GreatCircleDistance"}
    if incompatible:
        raise ValueError(f"{type(metric).__name__} has no scalar buoy-point contract; use a suitable point metric.")
    if classes & {"AngleError", "VectorNorm"}:
        axis = int(getattr(metric, "var_axis", -3))
        if "variable" not in dims or not -len(dims) <= axis < len(dims) or dims[axis] != "variable":
            raise ValueError("Buoy vector metrics require var_axis=-1 on the variable axis.")
        if len(set(units)) > 1:
            raise ValueError("Vector components must use the same units.")
        output_units = ("degree",) if "AngleError" in classes else units[:1]
        return tuple(dim for dim in dims if dim != "variable"), output_units, True
    if classes & {"MSE", "StatSquared", "Times"}:
        units = tuple(_squared(u) for u in units)
    elif classes & {"SkillScore", "ReversedSkillScore", "SicSuccess", "CategoricalSicSuccess", "ThickSuccess"}:
        units = tuple("1" for _ in units)
    elif not classes & {"MAE", "Difference", "IdentityStat", "IdentityOver", "CircularDifference"}:
        # Arbitrary transforms may change physical dimensions; do not mislabel.
        units = tuple(None for _ in units)
    declared = getattr(metric, "output_dims", None)
    if declared is not None:
        dims = tuple(declared)
    return dims, units, False


def _reindex(array, buoy_indices, time_indices, channels=None, *, fill=np.nan):
    shape = (len(buoy_indices), len(time_indices)) + (
        (len(channels),) if channels is not None else array.shape[2:]
    )
    dtype = "datetime64[ns]" if np.asarray(array).dtype.kind == "M" else float
    result = np.full(shape, fill, dtype=dtype)
    rows, cols = np.flatnonzero(buoy_indices >= 0), np.flatnonzero(time_indices >= 0)
    if len(rows) and len(cols):
        if channels is None:
            result[np.ix_(rows, cols)] = array[np.ix_(buoy_indices[rows], time_indices[cols])]
        else:
            result[np.ix_(rows, cols, np.arange(len(channels)))] = array[
                np.ix_(buoy_indices[rows], time_indices[cols], channels)
            ]
    return result


class BuoyInputAdapter:
    """Align BuoyBatch or annotated ndarray inputs to a reference observation batch.

    Annotated arrays need meta['dims', 'bids', 'datetimes', 'var_names', 'units'].
    Optional meta['valid'] follows value axes; coords/coord_valid always use
    (buoy,time,2)/(buoy,time). Matching is exact on query timestamps, never nearest.

    A bare array is rejected unless assume_aligned=True: the caller then asserts
    identical reference IDs, times, variable order, units and sampling positions.
    array_dims declares its axes explicitly (e.g. time,variable,buoy for the old
    interpolation wrapper). This opt-in cannot verify identities the array lost.

    variable_maps maps input index -> {canonical_name: source_name}; target_units
    maps canonical_name -> output unit, defaulting to reference units. Unknown
    reductions need explicit result_dims or metric.output_dims; custom result
    units can be provided explicitly. Supported result axes are NTV and NT.
    """

    def __init__(
        self, reference_index=1, variables=None, *, variable_maps=None,
        target_units=None, assume_aligned=False, array_dims=POINT_DIMS,
        component_policy="auto", result_dims=None, result_units=None,
    ):
        if not isinstance(reference_index, (int, np.integer)) or isinstance(reference_index, bool):
            raise ValueError("reference_index must be an integer.")
        if component_policy not in ("auto", "independent", "joint"):
            raise ValueError("component_policy must be auto, independent or joint.")
        if len(array_dims) != 3 or set(array_dims) != set(POINT_DIMS):
            raise ValueError(f"array_dims must be a permutation of {POINT_DIMS}.")
        self.reference_index = int(reference_index)
        self.variables = None if variables is None else _names(variables, "variables")
        if self.variables == ():
            raise ValueError("Select at least one variable.")
        self.variable_maps = dict(variable_maps or {})
        self.target_units = dict(target_units or {})
        self.assume_aligned = bool(assume_aligned)
        self.array_dims = tuple(array_dims)
        self.component_policy = component_policy
        self.result_dims = None if result_dims is None else tuple(result_dims)
        self.result_units = None if result_units is None else tuple(result_units)

    @staticmethod
    def _unpack(value):
        if hasattr(value, "bids") and hasattr(value, "variables"):
            required = ("bids", "datetimes", "var_names", "units", "coords")
            for name in required:
                if not hasattr(value, name):
                    raise ValueError(f"Buoy batch is missing {name!r}.")
            meta = {name: getattr(value, name) for name in required}
            for name in ("valid", "coord_valid", "value_times", "coord_times", "uncertainty"):
                if hasattr(value, name):
                    meta[name] = getattr(value, name)
            meta.update(dims=POINT_DIMS, provenance=getattr(value, "metadata", {}))
            return np.asarray(value.variables), meta
        return np.asarray(value), dict(getattr(value, "meta", {}) or {})

    @staticmethod
    def _field(values, meta):
        required = {"dims", "bids", "datetimes", "var_names", "units"}
        missing = required - meta.keys()
        if missing:
            raise ValueError(f"Point array requires alignment metadata {sorted(missing)}; a bare array needs assume_aligned=True.")
        values = _canonical(values, meta["dims"], "values")
        if values.dtype.kind not in "biuf":
            raise TypeError("Point values must be real numeric arrays.")
        values = values.astype(float, copy=False)
        bids = _names(meta["bids"], "bids")
        times = _times(meta["datetimes"])
        names = _names(meta["var_names"], "var_names")
        units = tuple(_unit_name(u) for u in meta["units"])
        shape = (len(bids), len(times), len(names))
        if values.shape != shape or len(units) != len(names):
            raise ValueError(f"Point values/units do not match declared IDs, times and variables: expected {shape}.")
        valid = np.isfinite(values)
        if "valid" in meta:
            declared = _canonical(meta["valid"], meta["dims"], "valid")
            if declared.shape != shape:
                raise ValueError("valid must have the same axes and shape as values.")
            valid &= np.asarray(declared, dtype=bool)
        coords = None
        coord_valid = np.ones(shape[:2], dtype=bool)
        if "coords" in meta:
            coords = np.asarray(meta["coords"], dtype=float)
            if coords.shape != shape[:2] + (2,):
                raise ValueError("coords must have shape (N,T,2) in [latitude, longitude] order.")
            coord_valid &= np.isfinite(coords).all(axis=-1)
        if "coord_valid" in meta:
            declared = np.asarray(meta["coord_valid"], dtype=bool)
            if declared.shape != shape[:2]:
                raise ValueError("coord_valid must have shape (N,T).")
            coord_valid &= declared
        return values, bids, times, names, units, valid, coords, coord_valid

    def prepare(self, inputs, *, metric) -> PreparedInputs:
        inputs = tuple(inputs)
        if not inputs or not -len(inputs) <= self.reference_index < len(inputs):
            raise ValueError("reference_index must identify an input observation batch.")
        reference_index = self.reference_index % len(inputs)
        raw_reference, reference_meta = self._unpack(inputs[reference_index])
        reference = self._field(raw_reference, reference_meta)
        _, ref_bids, ref_times, ref_names, ref_units, _, ref_coords, ref_coord_valid = reference
        if ref_coords is None:
            raise ValueError("The reference input must provide observation coords.")
        names = self.variables if self.variables is not None else ref_names
        if not names:
            raise ValueError("Select at least one scalar variable.")
        ref_map = self.variable_maps.get(reference_index, {})
        try:
            ref_channels = [ref_names.index(ref_map.get(name, name)) for name in names]
        except ValueError as exc:
            raise ValueError("A selected variable is absent from the reference input.") from exc
        units = tuple(_unit_name(self.target_units.get(name, ref_units[j])) for name, j in zip(names, ref_channels))
        dims, result_units, needs_joint = _metric_contract(metric, POINT_DIMS, units)
        dims = self.result_dims if self.result_dims is not None else dims
        result_units = self.result_units if self.result_units is not None else result_units
        if dims not in (POINT_DIMS, POINT_DIMS[:2]):
            raise ValueError("A point metric must return (buoy,time,variable) or (buoy,time); aggregate separately.")
        if needs_joint and self.component_policy == "independent":
            raise ValueError("A vector metric requires jointly valid components.")
        joint_components = needs_joint or self.component_policy == "joint" or dims == POINT_DIMS[:2]
        if self.result_units is None and len(result_units) != (len(names) if len(dims) == 3 else 1):
            result_units = tuple(None for _ in range(len(names) if len(dims) == 3 else 1))
        if len(result_units) != (len(names) if len(dims) == 3 else 1):
            raise ValueError("result_units must describe each output channel (one for a vector reduction).")

        arrays, masks, input_contexts = [], [], []
        for index, value in enumerate(inputs):
            raw, meta = self._unpack(value)
            identity_keys = {"dims", "bids", "datetimes", "var_names", "units"}
            if self.assume_aligned and not identity_keys.intersection(meta):
                meta = {
                    "dims": self.array_dims, "bids": ref_bids, "datetimes": ref_times,
                    "var_names": ref_names, "units": ref_units, "coords": ref_coords,
                    "coord_valid": ref_coord_valid,
                    **meta,
                }
            values, bids, times, source_names, source_units, valid, coords, coord_valid = self._field(raw, meta)
            bid_lookup = {bid: i for i, bid in enumerate(bids)}
            time_lookup = {time: i for i, time in enumerate(times)}
            bi = np.asarray([bid_lookup.get(bid, -1) for bid in ref_bids], dtype=int)
            ti = np.asarray([time_lookup.get(time, -1) for time in ref_times], dtype=int)
            variable_map = self.variable_maps.get(index, {})
            try:
                vi = [source_names.index(variable_map.get(name, name)) for name in names]
            except ValueError as exc:
                raise ValueError(f"Input {index} is missing a selected variable; declare variable_maps or select shared variables.") from exc
            aligned = _reindex(values, bi, ti, vi)
            aligned_valid = _reindex(valid, bi, ti, vi, fill=0).astype(bool)
            aligned_coord_valid = _reindex(coord_valid, bi, ti, fill=0).astype(bool)
            if coords is not None:
                aligned_coords = _reindex(coords, bi, ti)
                same_position = np.isclose(aligned_coords[..., 0], ref_coords[..., 0], rtol=0, atol=1e-5)
                lon_delta = (aligned_coords[..., 1] - ref_coords[..., 1] + 180) % 360 - 180
                same_position &= np.isclose(lon_delta, 0, rtol=0, atol=1e-5)
                if np.any(aligned_coord_valid & ref_coord_valid & ~same_position):
                    raise ValueError(f"Input {index} was sampled at different coordinates; interpolate on the reference observations first.")
            factors = np.asarray([_unit_factor(source_units[j], unit) for j, unit in zip(vi, units)])
            aligned = aligned * factors
            mask = aligned_valid & np.isfinite(aligned) & aligned_coord_valid[..., None] & ref_coord_valid[..., None]
            arrays.append(aligned)
            masks.append(mask)
            info = {
                "metadata": deepcopy(meta), "buoy_indices": bi, "time_indices": ti,
                "variable_indices": tuple(vi), "valid": mask.copy(),
            }
            for key in ("value_times", "uncertainty"):
                if key in meta:
                    original = _canonical(meta[key], meta["dims"], key)
                    if original.shape != values.shape:
                        raise ValueError(f"{key} must have the same axes and shape as values.")
                    if key == "value_times":
                        aligned_extra = _reindex(original.astype("datetime64[ns]"), bi, ti, vi, fill=np.datetime64("NaT", "ns"))
                    else:
                        aligned_extra = _reindex(original, bi, ti, vi) * factors
                    info[key] = aligned_extra
            if "coord_times" in meta:
                original = np.asarray(meta["coord_times"], dtype="datetime64[ns]")
                if original.shape != values.shape[:2]:
                    raise ValueError("coord_times must have shape (N,T).")
                info["coord_times"] = _reindex(original, bi, ti, fill=np.datetime64("NaT", "ns"))
            input_contexts.append(info)
        joint_valid = np.logical_and.reduce(masks)
        if joint_components:
            joint_valid = np.broadcast_to(joint_valid.all(axis=-1, keepdims=True), joint_valid.shape).copy()
        arrays = tuple(np.where(joint_valid, array, np.nan) for array in arrays)
        metadata = {
            "bids": np.asarray(ref_bids, dtype=str), "coords": ref_coords.copy(),
            "datetimes": ref_times.copy(), "reference_index": reference_index,
            "input_var_names": tuple(names), "input_units": units,
            "joint_valid": joint_valid.copy(), "inputs": tuple(input_contexts),
        }
        return PreparedInputs(
            arrays, joint_valid,
            ComparisonContext("points", metadata, tuple(dims), tuple(result_units)), POINT_DIMS,
        )


def make_metric_field(errors, *, metric, context, input_dims=None):
    """Attach result-aware context using the existing, public MetricField class."""
    from ..validator import MetricField

    if context.geometry == "array":
        return MetricField(errors, **context.metadata)
    if context.geometry != "points" or tuple(input_dims or ()) != POINT_DIMS:
        raise ValueError("Unsupported comparison geometry or input dimension contract.")
    meta = deepcopy(dict(context.metadata))
    dims = context.result_dims
    sizes = {
        "buoy": len(meta["bids"]), "time": len(meta["datetimes"]),
        "variable": len(meta["input_var_names"]),
    }
    expected = tuple(sizes[dim] for dim in dims)
    values = np.asarray(errors)
    if values.shape != expected or values.dtype.kind not in "biuf":
        raise ValueError(
            f"Metric {getattr(metric, 'name', type(metric).__name__)!r} returned {values.shape}; "
            f"declared point result {dims} requires {expected}. Declare result_dims for custom reductions."
        )
    mask = meta["joint_valid"] if dims == POINT_DIMS else meta["joint_valid"].all(axis=-1)
    values = np.where(mask, values, np.nan)
    meta.update(geometry="points", dims=dims, input_dims=POINT_DIMS,
                valid=mask & np.isfinite(values), units=context.result_units)
    if "variable" in dims:
        meta["var_names"] = meta["input_var_names"]
    else:
        meta["result_name"] = getattr(metric, "name", type(metric).__name__)
    return MetricField(values, **meta)


__all__ = [
    "ArrayInputAdapter", "BuoyInputAdapter", "MetricInputAdapter",
    "PreparedInputs", "ComparisonContext", "make_metric_field",
]
