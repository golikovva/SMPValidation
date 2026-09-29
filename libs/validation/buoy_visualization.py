"""Plots for point metrics collected by ``RawFieldAggregator``.

Pass ``validator.results`` (or the mapping loaded from ``Validator.save``).
The fields must retain their ``.meta``: IDs, query times, coordinates, axes,
validity and units. This module never loads datasets or reconstructs missing
metadata. In particular, re-saving an old pickle without metadata cannot repair
it; save the original in-memory results again after updating MetricField.

Examples (names of datasets/metrics must match the result dictionary)::

    plot_buoy_scatter(results, "model", "buoys", variable="sst")
    plot_buoy_metric_map(results, "difference", ("model", "buoys"),
                         variable="sst", start="2023-01-01", end="2023-12-31")
    plot_buoy_track(results, "buoy-id", "buoys", sources=["model"], panels=[
        BuoyPanel("identity", variable="sst", label="SST"),
        BuoyPanel("mae", variable="sst", label="Absolute error"),
    ], show_summary=True)

For drift, select components explicitly in eastward, northward order::

    uv = ("drift_eastward", "drift_northward")
    plot_buoy_scatter(results, "model", "buoys", variable=uv, reduction="norm")
    plot_buoy_scatter(results, "model", "buoys", variable=uv,
                      reduction="direction", circular_view="nearest")
    plot_buoy_track(results, "buoy-id", "buoys", sources=["model"], panels=[
        BuoyPanel("identity", uv, "norm", "Drift speed"),
        BuoyPanel("difference\u2192norm", label="Vector error"),
        BuoyPanel("angle_error", label="Angle error"),
    ])

All selected stored samples contribute equally; overlapping validation windows
remain separate observations. Times are query times in UTC, not original sensor
clocks. Date-only bounds include the entire day. No implicit temporal/spatial
averaging, unit conversion, file output or ``plt.show()`` is performed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd


__all__ = ["BuoyPanel", "plot_buoy_scatter", "plot_buoy_track", "plot_buoy_metric_map"]


@dataclass(frozen=True)
class BuoyPanel:
    """A scalar time panel; ``variable`` is a name or two component names.

    ``identity`` shows each source and the reference. Other metrics use the
    exact stored key ``(source, reference)``. ``reduction`` is None, ``norm`` or
    ``direction``; directions use atan2(northward, eastward), in [0, 360).
    """

    metric: str
    variable: str | tuple[str, str] | None = None
    reduction: str | None = None
    label: str | None = None


def _time_bound(value, *, end=False):
    if value is None:
        return None
    whole_day = (isinstance(value, date) and not isinstance(value, datetime)) or (
        isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None
    ) or (isinstance(value, np.datetime64) and np.datetime_data(value.dtype)[0] == "D")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Time bounds must not be NaT.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    if end and whole_day:
        stamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return stamp


def _raw_fields(results, metric, dataset_key):
    if not isinstance(results, Mapping):
        raise TypeError("results must be validator.results or its saved mapping.")
    if "results" in results and "processed_dates" in results:
        results = results["results"]
    key = (dataset_key,) if isinstance(dataset_key, str) else tuple(dataset_key)
    if metric not in results:
        raise ValueError(f"Metric {metric!r} is absent; available: {list(results)}.")
    if key not in results[metric]:
        hint = ""
        if len(key) == 2 and key[::-1] in results[metric]:
            hint = f" Only the reverse key {key[::-1]!r} exists; values are not sign-flipped automatically."
        raise ValueError(f"Metric {metric!r} has no dataset key {key!r}.{hint}")
    raw = results[metric][key].get("RawFieldAggregator")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{metric!r}, {key!r} requires RawFieldAggregator results.")
    return raw


def _scalar_field(field, variable, reduction):
    """Validate a point field and select a scalar without changing its metadata."""
    meta = getattr(field, "meta", None)
    required = {"dims", "bids", "datetimes", "coords", "valid", "units"}
    if not isinstance(meta, Mapping) or not required.issubset(meta):
        missing = sorted(required - set(meta or {}))
        raise ValueError(
            f"Buoy field is missing metadata {missing}. Use in-memory results or "
            "save them again with the updated MetricField; an old metadata-free "
            "pickle cannot supply IDs, times and positions."
        )
    dims = tuple(meta["dims"])
    canonical = ("buoy", "time", "variable") if "variable" in dims else ("buoy", "time")
    values = np.asarray(field)
    if (len(dims) != len(canonical) or set(dims) != set(canonical)
            or values.ndim != len(dims) or values.dtype.kind not in "biuf"
            or meta.get("geometry", "points") != "points"):
        raise ValueError("Expected numeric point data with buoy/time[/variable] dimensions.")
    order = [dims.index(dim) for dim in canonical]
    valid = np.asarray(meta["valid"], dtype=bool)
    if valid.shape != values.shape:
        raise ValueError("The valid mask must have the same shape as the field.")
    values = np.transpose(values, order).astype(float, copy=False)
    valid = np.transpose(valid, order) & np.isfinite(values)
    bids = np.asarray(meta["bids"], dtype=str)
    raw_times = np.asarray(meta["datetimes"])
    if bids.ndim != 1 or raw_times.ndim != 1 or raw_times.dtype.kind in "biuf":
        raise ValueError("bids and datetimes must be one-dimensional ID/time arrays.")
    times = pd.DatetimeIndex(pd.to_datetime(raw_times, utc=True)).tz_localize(None)
    if times.hasnans or not times.is_unique or len(set(bids)) != len(bids):
        raise ValueError("IDs and query times must be valid and unique within each field.")
    coords = np.asarray(meta["coords"], dtype=float)
    if values.shape[:2] != (len(bids), len(times)) or coords.shape != (*values.shape[:2], 2):
        raise ValueError("Field dimensions, IDs, times and (buoy,time,2) coordinates disagree.")
    units = meta["units"]
    if isinstance(units, str) or units is None:
        raise ValueError("units must describe each output channel (a sequence, even for one channel).")
    units = tuple(units)
    if reduction not in (None, "norm", "direction"):
        raise ValueError("reduction must be None, 'norm' or 'direction'.")
    if values.ndim == 2:
        if variable is not None or reduction is not None or len(units) != 1:
            raise ValueError("An already reduced metric needs one unit and no variable/reduction selection.")
        selected, mask, unit = values, valid, units[0]
    else:
        names = tuple(meta.get("var_names", ()))
        if len(names) != values.shape[-1] or len(set(names)) != len(names) or len(units) != len(names):
            raise ValueError("var_names and units must uniquely describe the variable axis.")
        if reduction is None:
            if variable is None:
                if len(names) != 1:
                    raise ValueError(f"Select a variable explicitly from {names}.")
                variable = names[0]
            if not isinstance(variable, str) or variable not in names:
                raise ValueError(f"Select one variable from {names}.")
            i = names.index(variable)
            selected, mask, unit = values[..., i], valid[..., i], units[i]
        else:
            if (isinstance(variable, str) or variable is None or len(variable) != 2
                    or len(set(variable)) != 2 or any(v not in names for v in variable)):
                raise ValueError("Vector reduction requires two explicit component names, in east/north order.")
            i, j = (names.index(v) for v in variable)
            if units[i] != units[j]:
                raise ValueError("Vector components must have matching units; convert before plotting.")
            u, v = values[..., i], values[..., j]
            speed = np.hypot(u, v)
            mask = valid[..., i] & valid[..., j]
            if reduction == "norm":
                selected, unit = speed, units[i]
            else:
                selected = np.degrees(np.arctan2(v, u)) % 360
                mask = mask & (speed > 0)
                unit = "degrees"
    coord_valid = np.isfinite(coords).all(axis=-1) & (np.abs(coords[..., 0]) <= 90)
    selected = np.where(mask & coord_valid, selected, np.nan)
    coords = np.where(coord_valid[..., None], coords, np.nan)
    return selected, bids, times, coords, unit


def _collect_records(results, metric, dataset_key, *, variable=None, reduction=None,
                     start=None, end=None, buoy_ids=None):
    start, end = _time_bound(start), _time_bound(end, end=True)
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be later than end.")
    if isinstance(buoy_ids, (str, int, np.integer)):
        buoy_ids = [buoy_ids]
    keep_ids = None if buoy_ids is None else {str(bid) for bid in buoy_ids}
    frames, all_units = [], []
    for window, field in _raw_fields(results, metric, dataset_key).items():
        values, bids, times, coords, unit = _scalar_field(field, variable, reduction)
        bi = np.arange(len(bids)) if keep_ids is None else np.flatnonzero(np.isin(bids, list(keep_ids)))
        ti = np.ones(len(times), dtype=bool)
        if start is not None:
            ti &= times >= start
        if end is not None:
            ti &= times <= end
        ti = np.flatnonzero(ti)
        if not len(bi) or not len(ti):
            continue
        all_units.append(unit)
        positions = coords[np.ix_(bi, ti)].reshape(-1, 2)
        frames.append(pd.DataFrame({
            "window": _time_bound(window),
            "bid": np.repeat(bids[bi], len(ti)),
            "time": np.tile(times.to_numpy()[ti], len(bi)),
            "lat": positions[:, 0], "lon": (positions[:, 1] + 180) % 360 - 180,
            "value": values[np.ix_(bi, ti)].ravel(),
        }))
    if any(unit != all_units[0] for unit in all_units):
        raise ValueError("Units change between selected fields; convert before plotting.")
    columns = ["window", "bid", "time", "lat", "lon", "value"]
    records = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if records.duplicated(["window", "bid", "time"]).any():
        raise ValueError("Duplicate (validation window, buoy ID, time) records are ambiguous.")
    records = records.sort_values(["time", "window", "bid"], kind="stable").reset_index(drop=True)
    records.attrs["units"] = all_units[0] if all_units else None
    return records


def _paired_records(source, reference):
    if source.attrs.get("units") != reference.attrs.get("units") and len(source) and len(reference):
        raise ValueError("Source and reference units differ; convert before plotting.")
    pairs = source.merge(reference, on=["window", "bid", "time"], how="inner",
                         suffixes=("_source", "_reference"), validate="one_to_one")
    if pairs.empty:
        pairs.attrs["units"] = source.attrs.get("units")
        return pairs
    usable = np.isfinite(pairs[["value_source", "value_reference"]].to_numpy(dtype=float)).all(axis=1)
    lat_equal = np.isclose(pairs.lat_source, pairs.lat_reference, rtol=0, atol=1e-5)
    lon_delta = (pairs.lon_source - pairs.lon_reference + 180) % 360 - 180
    if np.any(usable & ~(lat_equal & np.isclose(lon_delta, 0, rtol=0, atol=1e-5))):
        raise ValueError("Source and reference samples have different coordinates; interpolate first.")
    pairs.attrs["units"] = source.attrs.get("units")
    return pairs


def _label(text, unit):
    return f"{text} ({unit})" if unit else str(text)


def plot_buoy_scatter(results, source, reference, *, variable=None, reduction=None,
                      start=None, end=None, buoy_ids=None, circular=False, period=360,
                      circular_view="wrapped", ax=None, show_ci=False,
                      show_bin_means=False, title=None, xlabel=None, ylabel=None,
                      **scatter_kwargs):
    """Compare identity values: X=source, Y=reference, Bias=source-reference.

    ``variable=(east, north)`` selects components for a vector reduction.
    Direction reduction enables circular statistics in degrees automatically;
    for stored angles use ``circular=True`` and specify their period if needed.
    All finite matched pairs contribute to statistics independently of xlim/ylim.
    Optional CI estimates assume independent samples (no track autocorrelation
    correction). ``show_bin_means`` uses circular means for angular data.
    Remaining keywords are passed to ``plot_pair_scatter``.
    """
    from .scatter_identity_plots import plot_pair_scatter

    selection = dict(variable=variable, reduction=reduction, start=start, end=end, buoy_ids=buoy_ids)
    pairs = _paired_records(
        _collect_records(results, "identity", (source,), **selection),
        _collect_records(results, "identity", (reference,), **selection),
    )
    x = pairs.value_source.to_numpy(dtype=float)
    y = pairs.value_reference.to_numpy(dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    if not len(x):
        raise ValueError("No finite matched buoy observations in the selected period/IDs.")
    circular = circular or reduction == "direction"
    if reduction == "direction" and period != 360:
        raise ValueError("Direction reduction returns degrees; period must be 360.")
    options = dict(scatter_kwargs)
    reserved = {"source_axis", "stats_scope", "show_mae", "units"} & options.keys()
    if reserved:
        raise ValueError(f"The buoy interface sets {sorted(reserved)} from data and its source/reference contract.")
    if not circular:
        lower, upper = min(x.min(), y.min()), max(x.max(), y.max())
        if lower == upper:
            pad = max(abs(lower) * 0.05, 0.5)
            lower, upper = lower - pad, upper + pad
        options.setdefault("xymin", lower)
        options.setdefault("xymax", upper)
    return plot_pair_scatter(
        x, y, ax=ax, source_axis="x", stats_scope="all", show_mae=True,
        show_ci=show_ci, show_bin_means=show_bin_means, circular=circular,
        period=period, circular_view=circular_view, units=pairs.attrs["units"] or "",
        title=title or f"{source} vs {reference}",
        xlabel=xlabel or source, ylabel=ylabel or reference, **options,
    )


def _map_axes(*, ax=None, proj=None, figsize=None, map_kwargs=None):
    # Keep geospatial imports out of scalar preparation/scatter paths.
    from . import visualization

    options = dict(map_kwargs or {})
    if proj is not None:
        options["proj"] = proj
    if figsize is not None:
        options["figsize"] = figsize

    fig, ax = visualization.create_cartopy_axes(ax=ax, **options)
    return fig, ax, visualization


def _map_extent(ax, coords, extent, visualization):
    if extent is not None:
        ax.set_extent(extent, crs=visualization.ccrs.PlateCarree())
        return
    coords = np.asarray(coords, dtype=float)
    projected = ax.projection.transform_points(visualization.ccrs.PlateCarree(), coords[:, 1], coords[:, 0])
    xy = projected[np.isfinite(projected[:, :2]).all(axis=1), :2]
    if not len(xy):
        raise ValueError("No selected buoy coordinates are visible in this projection.")
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    # Cartopy geodetic/PlateCarree axes use degrees; projected maps use metres.
    floor = 0.5 if isinstance(ax.projection, visualization.ccrs.PlateCarree) else 50_000.0
    pad = np.maximum((upper - lower) * 0.05, floor * 0.5)
    ax.set_xlim(lower[0] - pad[0], upper[0] + pad[0])
    ax.set_ylim(lower[1] - pad[1], upper[1] + pad[1])


def plot_buoy_metric_map(results, metric, dataset_key, *, variable=None, reduction=None,
                         start=None, end=None, buoy_ids=None, ax=None, proj=None,
                         extent=None, figsize=None, map_kwargs=None, cmap=None,
                         norm=None, vmin=None, vmax=None, s=12, alpha=0.8,
                         title=None, label=None, colorbar_kwargs=None, **scatter_kwargs):
    """Color each selected buoy sample by its stored metric, without aggregation.

    ``dataset_key`` is the exact tuple in results; its order/sign is preserved.
    ``map_kwargs`` goes to visualization.create_cartopy_axes (e.g. add_land,
    add_coastlines, central_longitude). ``extent`` is lon_min/lon_max/lat_min/lat_max
    in degrees; automatic bounds are computed in the selected map projection.
    """
    records = _collect_records(results, metric, dataset_key, variable=variable,
                               reduction=reduction, start=start, end=end, buoy_ids=buoy_ids)
    valid = np.isfinite(records[["value", "lat", "lon"]].to_numpy(dtype=float)).all(axis=1)
    shown = records.loc[valid]
    if shown.empty:
        raise ValueError("No finite buoy metric samples in the selected period/IDs.")
    values = shown.value.to_numpy(dtype=float)
    mixed = values.min() < 0 < values.max()
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Specify either norm or vmin/vmax, not both.")
    if norm is None and mixed:
        lo = np.percentile(values, 1) if vmin is None else vmin
        hi = np.percentile(values, 99) if vmax is None else vmax
        if lo < 0 < hi:
            # norm, vmin, vmax = colors.TwoSlopeNorm(0, lo, hi), None, None
            norm, vmin, vmax = colors.CenteredNorm(0, max(abs(lo), abs(hi))), None, None
    if cmap is None:
        cmap = "RdBu_r" if mixed or isinstance(norm, colors.TwoSlopeNorm) else "viridis"
    fig, ax, visualization = _map_axes(ax=ax, proj=proj, figsize=figsize, map_kwargs=map_kwargs)
    coords = shown[["lat", "lon"]].to_numpy()
    options = dict(scatter_kwargs)
    options.setdefault("zorder", 10)
    options.setdefault("rasterized", True)
    layer = visualization.visualize_scatter(
        ax, coords, values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        s=s, alpha=alpha, **options,
    )
    cbar = fig.colorbar(layer, ax=ax, **dict(colorbar_kwargs or {}))
    cbar.set_label(_label(label or metric, records.attrs.get("units")))
    _map_extent(ax, coords, extent, visualization)
    key = (dataset_key,) if isinstance(dataset_key, str) else tuple(dataset_key)
    ax.set_title(title or f"{metric}: {' vs '.join(key)} (N={len(shown)})")
    return fig, ax


def _gap_threshold(times, max_gap):
    if max_gap is not None:
        gap = pd.Timedelta(max_gap)
        if pd.isna(gap) or gap <= pd.Timedelta(0):
            raise ValueError("max_gap must be a positive timedelta or string such as '6h'.")
        return gap
    unique = np.unique(np.asarray(times, dtype="datetime64[ns]"))
    diffs = np.diff(unique.astype(np.int64))
    return pd.Timedelta(int(1.5 * np.median(diffs)), unit="ns") if len(diffs) else None


def _break_gaps(times, values, max_gap):
    times = np.asarray(times, dtype="datetime64[ns]")
    values = np.asarray(values, dtype=float)
    gap = _gap_threshold(times, max_gap)
    if gap is None or len(times) < 2:
        return times, values
    stops = np.flatnonzero(np.diff(times.astype(np.int64)) > gap.value) + 1
    midpoints = times[stops - 1] + (times[stops] - times[stops - 1]) // 2
    return np.insert(times, stops, midpoints), np.insert(values, stops, np.nan, axis=0)


def _track_positions(frames):
    """Collapse repeated metadata positions, not the plotted metric observations."""
    records = pd.concat(frames, ignore_index=True)
    positions = []
    for time, group in records.groupby("time", sort=True):
        coords = group[["lat", "lon"]].to_numpy(dtype=float)
        coords = coords[np.isfinite(coords).all(axis=1)]
        if len(coords):
            delta = (coords[:, 1] - coords[0, 1] + 180) % 360 - 180
            if not (np.allclose(coords[:, 0], coords[0, 0], rtol=0, atol=1e-5)
                    and np.allclose(delta, 0, rtol=0, atol=1e-5)):
                raise ValueError(f"Conflicting buoy coordinates at {time} across selected panels/windows.")
            positions.append(coords[0])
        else:
            positions.append([np.nan, np.nan])
    return np.asarray(sorted(records.time.unique()), dtype="datetime64[ns]"), np.asarray(positions)


def _mean_value(values, circular=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    if not circular:
        return float(values.mean())
    z = np.exp(1j * np.deg2rad(values)).mean()
    return float(np.rad2deg(np.angle(z)) % 360) if abs(z) > 1e-12 else np.nan


def plot_buoy_track(results, buoy_id, reference, *, sources, panels, start=None, end=None,
                    show_summary=False, max_gap=None, figsize=None, source_labels=None,
                    source_styles=None, proj=None, extent=None, map_kwargs=None, title=None):
    """Draw time panels, the buoy trajectory and an optional table of means.

    ``panels`` contains BuoyPanel objects. Gaps/invalid samples break lines;
    max_gap defaults to 1.5 times the median positive query-time step per series.
    The returned axes dictionary has ``panels`` (list), ``map`` and ``summary``
    (None unless requested). Summary cells average the shown stored values,
    using a circular mean for direction reductions. No metric is recomputed.
    """
    if isinstance(sources, str):
        sources = [sources]
    sources, panels = list(sources), list(panels)
    if not sources or len(set(sources)) != len(sources) or reference in sources:
        raise ValueError("sources must be nonempty, unique and exclude the reference.")
    if not panels or any(not isinstance(panel, BuoyPanel) for panel in panels):
        raise ValueError("panels must be a nonempty sequence of BuoyPanel objects.")
    _gap_threshold([], max_gap)  # validate even for an empty/single-point series
    data, frames = [], []
    for panel in panels:
        by_source = {}
        for source in sources + ([reference] if panel.metric == "identity" else []):
            key = (source,) if panel.metric == "identity" else (source, reference)
            records = _collect_records(results, panel.metric, key, variable=panel.variable,
                                       reduction=panel.reduction, start=start, end=end,
                                       buoy_ids=[buoy_id])
            by_source[source] = records
            if len(records):
                frames.append(records)
        units = [df.attrs.get("units") for df in by_source.values() if len(df)]
        if any(unit != units[0] for unit in units):
            raise ValueError(f"Panel {panel.metric!r} contains inconsistent units.")
        data.append(by_source)
    if not frames or not any(np.isfinite(df.value.to_numpy(dtype=float)).any() for df in frames):
        raise ValueError(f"No finite observations for buoy {buoy_id!r} in the selected panels/period.")
    track_times, positions = _track_positions(frames)
    labels, styles = dict(source_labels or {}), dict(source_styles or {})
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    styles = {source: {"color": palette[i % len(palette)], **styles.get(source, {})}
              for i, source in enumerate(sources)} | {
                  reference: {"color": "black", **styles.get(reference, {})}}
    n = len(panels)
    fig = plt.figure(figsize=figsize or (15, max(5, 3 * n)), layout="constrained")
    grid = fig.add_gridspec(n, 2, width_ratios=[1.8, 1])
    time_axes, summary_values, column_labels = [], {}, []
    for i, (panel, by_source) in enumerate(zip(panels, data)):
        ax = fig.add_subplot(grid[i, 0], sharex=time_axes[0] if time_axes else None)
        time_axes.append(ax)
        unit = next((df.attrs.get("units") for df in by_source.values() if len(df)), None)
        heading = panel.label or panel.metric
        column_labels.append(_label(heading, unit))
        panel_has_values = False
        for source, records in by_source.items():
            values = records.value.to_numpy(dtype=float)
            if not len(values):
                summary_values[source, i] = np.nan
                continue
            style = dict(styles[source])
            if np.isfinite(values).sum() == 1:
                style.setdefault("marker", "o")
            style.setdefault("label", labels.get(source, source))
            # Overlapping windows are independent stored observations. Keep
            # their curves separate instead of connecting duplicate timestamps.
            groups = (list(records.groupby("window", sort=True))
                      if records.time.duplicated().any() else [(None, records)])
            for group_index, (_, group) in enumerate(groups):
                times, line_values = _break_gaps(group.time.to_numpy(), group.value.to_numpy(), max_gap)
                group_style = dict(style)
                if group_index:
                    group_style["label"] = "_nolegend_"
                if np.isfinite(group.value.to_numpy()).sum() == 1:
                    group_style.setdefault("marker", "o")
                ax.plot(times, line_values, **group_style)
            panel_has_values |= bool(np.isfinite(values).any())
            summary_values[source, i] = _mean_value(values, panel.reduction == "direction")
        if not panel_has_values:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(heading)
        ax.set_ylabel(_label(heading, unit))
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    time_axes[-1].set_xlabel("Time (UTC)")
    if show_summary:
        right = grid[:, 1].subgridspec(2, 1, height_ratios=[3, 1])
        map_slot, table_slot = right[0], right[1]
    else:
        map_slot, table_slot = grid[:, 1], None
    _, map_ax, visualization = _map_axes(ax=fig.add_subplot(map_slot), proj=proj, map_kwargs=map_kwargs)
    finite_coords = np.isfinite(positions).all(axis=1)
    if finite_coords.sum() == 1:
        numeric_time = mdates.date2num(track_times[finite_coords])
        layer = visualization.visualize_scatter(map_ax, positions[finite_coords], numeric_time,
                                                cmap="viridis", s=35, zorder=10)
        cbar = fig.colorbar(layer, ax=map_ax, orientation="horizontal", pad=0.08)
        cbar.set_ticks(numeric_time)
        cbar.set_ticklabels([pd.Timestamp(track_times[finite_coords][0]).strftime("%Y-%m-%d %H:%M")])
        cbar.set_label("Time (UTC)")
    else:
        times, trajectory = _break_gaps(track_times, positions, max_gap)
        trajectory_artists = visualization.visualize_trajectory(
            map_ax, trajectory[None, ...], times=times, add_colorbar=True,
            cbar_label="Time (UTC)", cbar_kwargs={"orientation": "horizontal", "pad": 0.08},
            start_end_markers=True,
        )
        # The shared trajectory helper draws segments only. Preserve isolated
        # observations too, including an entirely disconnected track.
        usable = np.isfinite(trajectory).all(axis=1)
        connected = np.zeros(len(usable), dtype=bool)
        adjacent = usable[:-1] & usable[1:]
        connected[:-1] |= adjacent
        connected[1:] |= adjacent
        isolated = usable & ~connected
        if isolated.any():
            mappable = trajectory_artists["mappable"]
            visualization.visualize_scatter(
                map_ax, trajectory[isolated], mdates.date2num(times[isolated]),
                cmap=mappable.cmap, norm=mappable.norm, s=25, zorder=12,
            )
    _map_extent(map_ax, positions[finite_coords], extent, visualization)
    map_ax.set_title(f"Buoy trajectory: {buoy_id}")
    summary_ax = None
    if show_summary:
        summary_ax = fig.add_subplot(table_slot)
        summary_ax.axis("off")
        rows = sources + ([reference] if any(p.metric == "identity" for p in panels) else [])
        cells = [[f"{summary_values.get((source, i), np.nan):.3g}"
                  if np.isfinite(summary_values.get((source, i), np.nan)) else "—"
                  for i in range(n)] for source in rows]
        table = summary_ax.table(cellText=cells, rowLabels=[labels.get(s, s) for s in rows],
                                 colLabels=column_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        summary_ax.set_title("Mean values over selected samples")
    fig.suptitle(title or f"Buoy {buoy_id}")
    return fig, {"panels": time_axes, "map": map_ax, "summary": summary_ax}
