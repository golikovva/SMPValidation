import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from scipy.stats import circmean
from functools import partial

# ============================================================
# Basic utilities
# ============================================================
def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "values"):
        x = x.values
    return np.asarray(x)


def _as_1d_scalar_field(arr):
    """
    Convert scalar field arrays like:
      (1, 1, N), (1, N), (N, 1), (N,)
    to shape (N,)
    """
    a = _to_numpy(arr)
    a = np.squeeze(a)

    if a.ndim == 0:
        return a[None]
    if a.ndim == 1:
        return a
    if a.ndim == 2 and 1 in a.shape:
        return a.reshape(-1)

    raise ValueError(f"Cannot convert scalar field with shape {a.shape} to 1D.")


def _as_uv_2n(arr):
    """
    Convert vector field arrays to shape (2, N), axis 0 = [u, v].

    Supported common cases:
      (1, 2, N)
      (2, N)
      (N, 2)
      (1, N, 2)
    """
    a = _to_numpy(arr)

    if a.ndim == 3:
        if a.shape[0] == 1 and a.shape[1] == 2:
            return a[0]
        if a.shape[0] == 1 and a.shape[2] == 2:
            return np.moveaxis(a[0], -1, 0)

    if a.ndim == 2:
        if a.shape[0] == 2:
            return a
        if a.shape[1] == 2:
            return a.T

    raise ValueError(f"Unsupported UV array shape: {a.shape}")


def _speed_from_uv(arr, scale=1.0):
    uv = _as_uv_2n(arr)
    return np.hypot(uv[0], uv[1]) * scale


def _pretty_key(key):
    if isinstance(key, tuple) and len(key) == 1:
        return key[0]
    return str(key)


# ============================================================
# Direction-aware stats
# ============================================================
def _wrap_deg(diff_deg, period=360.0):
    """Wrap a cyclic difference to [-period / 2, period / 2)."""
    return (diff_deg + period / 2.0) % period - period / 2.0


def _circular_corr_deg(x_deg, y_deg, period=360.0):
    """
    Circular-circular correlation (Jammalamadaka-Sarma style).
    """
    x = np.asarray(x_deg) * (2.0 * np.pi / period)
    y = np.asarray(y_deg) * (2.0 * np.pi / period)

    if x.size < 2:
        return np.nan
    # A circular mean (and hence this correlation) is undefined for a
    # distribution with a vanishing mean resultant.
    if any(np.hypot(np.mean(np.sin(a)), np.mean(np.cos(a))) < 1e-12
           for a in (x, y)):
        return np.nan

    x_bar = np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x)))
    y_bar = np.arctan2(np.mean(np.sin(y)), np.mean(np.cos(y)))

    sx = np.sin(x - x_bar)
    sy = np.sin(y - y_bar)

    den = np.sqrt(np.sum(sx**2) * np.sum(sy**2))
    if den <= np.finfo(float).eps * x.size:
        return np.nan
    return np.sum(sx * sy) / den


def _summary_stats(x, y, *, circular=False, period=360.0):
    """
    x = reference
    y = model/source
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite.")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = x.size
    if n == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "bias_ci": (np.nan, np.nan),
            "rmse": np.nan,
            "mae": np.nan,
            "rmse_ci": (np.nan, np.nan),
            "cor": np.nan,
        }

    if circular:
        err = _wrap_deg(y - x, period=period)
        bias = np.mean(err)
        rmse = np.sqrt(np.mean(err**2))
        cor = _circular_corr_deg(x, y, period=period)
    else:
        err = y - x
        bias = np.mean(err)
        rmse = np.sqrt(np.mean(err**2))
        cor = (np.corrcoef(x, y)[0, 1]
               if n > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan)

    # Simple large-sample normal CI for bias
    if n > 1:
        bias_se = np.std(err, ddof=1) / np.sqrt(n)
        bias_ci = (bias - 1.96 * bias_se, bias + 1.96 * bias_se)
    else:
        bias_ci = (np.nan, np.nan)

    # Delta-method-ish approximation for RMSE CI
    e2 = err**2
    if n > 1 and rmse > 0:
        var_e2 = np.var(e2, ddof=1)
        rmse_se = np.sqrt(var_e2 / (4 * n * rmse**2))
        rmse_ci = (rmse - 1.96 * rmse_se, rmse + 1.96 * rmse_se)
    elif n > 1 and rmse == 0:
        # The empirical error distribution is a point mass at zero.
        rmse_ci = (0.0, 0.0)
    else:
        rmse_ci = (np.nan, np.nan)

    return {
        "n": n,
        "bias": bias,
        "bias_ci": bias_ci,
        "rmse": rmse,
        "mae": np.mean(np.abs(err)),
        "rmse_ci": rmse_ci,
        "cor": cor,
    }


# ============================================================
# Generic collectors
# ============================================================
def collect_pairs_from_date_dicts(
    pred_by_date,
    ref_by_date,
    *,
    pred_transform=None,
    ref_transform=None,
):
    """
    Generic collector from:
      {date -> array_like}
    and returns two concatenated 1D arrays across all common dates.

    pred_transform / ref_transform should convert a raw stored array
    to a 1D array of values.
    """
    if pred_transform is None:
        pred_transform = _as_1d_scalar_field
    if ref_transform is None:
        ref_transform = _as_1d_scalar_field

    common_dates = sorted(set(pred_by_date.keys()) & set(ref_by_date.keys()))

    xs, ys = [], []
    for d in common_dates:
        y = pred_transform(pred_by_date[d])  # model/source
        x = ref_transform(ref_by_date[d])    # reference

        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]

        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            xs.append(x[mask])
            ys.append(y[mask])

    if not xs:
        return np.array([]), np.array([]), common_dates

    return np.concatenate(xs), np.concatenate(ys), common_dates


def collect_pairs_from_time_dicts(
    pred_by_time,
    ref_by_time,
    *,
    pred_transform=None,
    ref_transform=None,
):
    """
    Generic collector from:
      {actual_time -> [array_like, ...]}
    and returns two concatenated 1D arrays across all common actual times.
    """
    if pred_transform is None:
        pred_transform = _as_1d_scalar_field
    if ref_transform is None:
        ref_transform = _as_1d_scalar_field

    common_times = sorted(set(pred_by_time.keys()) & set(ref_by_time.keys()))

    xs, ys = [], []
    for t in common_times:
        pred_values = pred_by_time[t]
        ref_values = ref_by_time[t]

        if not isinstance(pred_values, list):
            pred_values = [pred_values]
        if not isinstance(ref_values, list):
            ref_values = [ref_values]

        for pred_raw in pred_values:
            for ref_raw in ref_values:
                y = pred_transform(pred_raw)  # model/source
                x = ref_transform(ref_raw)    # reference

                n = min(x.size, y.size)
                x = x[:n]
                y = y[:n]

                mask = np.isfinite(x) & np.isfinite(y)
                if np.any(mask):
                    xs.append(x[mask])
                    ys.append(y[mask])

    if not xs:
        return np.array([]), np.array([]), common_times

    return np.concatenate(xs), np.concatenate(ys), common_times


# ============================================================
# Accessors for specific validator structures
# ============================================================
def get_flat_date_dict(
    results,
    *,
    metric,
    dataset_key,
    aggregator="RawFieldAggregator",
):
    """
    Old drift-like structure:
      results[metric][dataset_key][aggregator][date] -> array
    """
    return results[metric][dataset_key][aggregator]


def get_nested_lead_date_dict(
    results,
    *,
    metric,
    dataset_key,
    lead,
    nested_key="NestedLeadTimeAggregator",
    by_lead_key="by_lead",
    raw_key="RawFieldAggregator__1",
):
    """
    New wind-like structure:
      results[metric][dataset_key][nested_key][by_lead_key][lead][raw_key][date] -> array
    """
    return results[metric][dataset_key][nested_key][by_lead_key][lead][raw_key]


def get_nested_by_lead_dict(
    results,
    *,
    metric,
    dataset_key,
    nested_key="NestedLeadTimeAggregator",
    by_lead_key="by_lead",
):
    """
    Return:
      results[metric][dataset_key][nested_key][by_lead_key]
    """
    return results[metric][dataset_key][nested_key][by_lead_key]


def _is_all_leads(lead):
    return lead is None or (isinstance(lead, str) and lead.strip().lower() == "all")


def _lead_to_timedelta64(lead):
    lead_hours = float(lead)
    seconds = int(round(lead_hours * 3600.0))
    return np.timedelta64(seconds, "s")


def _actual_time_key(date, lead):
    try:
        return np.datetime64(date, "s") + _lead_to_timedelta64(lead)
    except (TypeError, ValueError):
        return date


def _lead_date_dict_to_actual_time_dict(by_lead, *, raw_key):
    """
    Convert nested lead buckets to actual-time buckets using:
      actual_time = date + lead hours
    """
    by_actual_time = {}

    for lead_value, lead_bucket in by_lead.items():
        if raw_key not in lead_bucket:
            continue

        for date, raw_value in lead_bucket[raw_key].items():
            actual_time = _actual_time_key(date, lead_value)
            by_actual_time.setdefault(actual_time, []).append(raw_value)

    return by_actual_time


# ============================================================
# Shared plotting core
# ============================================================
def plot_pair_scatter(
    x,
    y,
    *,
    ax=None,
    title=None,
    xlabel="Reference",
    ylabel="Model",
    units="",
    circular=False,
    xymax=None,
    hist_bins=140,
    mean_bin_width=None,
    min_points_per_bin=30,
    scatter_sample=25000,
    show_colorbar=True,
    xymin=None,
    source_axis="y",
    stats_scope="visible",
    show_ci=True,
    show_bin_means=True,
    show_mae=False,
    period=360.0,
    circular_view="wrapped",
    xlim=None,
    ylim=None,
):
    """Plot paired values with binned density and source-minus-reference errors.

    Existing defaults retain x=reference, y=source, a nonnegative plotting
    range, and statistics restricted to visible pairs. Set source_axis="x"
    for x=source/y=reference, and stats_scope="all" to retain every finite
    selected pair in the statistics when zooming the axes. Negative linear
    values require an explicit negative xymin or xlim/ylim.

    xymin/xymax set common limits; xlim and ylim override each axis separately.
    Circular values are reduced modulo period. The "nearest" view moves y by
    whole periods to its nearest representation to x; its default y limits
    extend the common range by half a period at either end. This does not
    duplicate observations or change their errors. Circular bin means use the
    same representation as the displayed data.

    Returns (figure, axes) and never calls show. Empty finite input raises
    ValueError; a viewport containing no pairs remains a valid plot.
    Confidence intervals, when requested, use a normal approximation treating
    observations as independent.
    """
    if source_axis not in {"x", "y"}:
        raise ValueError("source_axis must be 'x' or 'y'.")
    if stats_scope not in {"visible", "all"}:
        raise ValueError("stats_scope must be 'visible' or 'all'.")
    if circular_view not in {"wrapped", "nearest"}:
        raise ValueError("circular_view must be 'wrapped' or 'nearest'.")
    if not circular and circular_view != "wrapped":
        raise ValueError("circular_view='nearest' requires circular=True.")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite.")
    for name, value in (("hist_bins", hist_bins),
                        ("min_points_per_bin", min_points_per_bin),
                        ("scatter_sample", scatter_sample)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        raise ValueError("No valid points to plot.")

    if circular:
        x, y = x % period, y % period
        if xymax is None:
            xymax = period
        if mean_bin_width is None:
            mean_bin_width = period / 24.0
    else:
        if xymax is None:
            xymax = max(1.0, math.ceil(np.percentile(np.r_[x, y], 99.8)))
        if mean_bin_width is None:
            mean_bin_width = 2.0
    if not np.isfinite(mean_bin_width) or mean_bin_width <= 0:
        raise ValueError("mean_bin_width must be positive and finite.")

    xymin = 0.0 if xymin is None else float(xymin)
    xymax = float(xymax)
    if not np.isfinite([xymin, xymax]).all() or xymin > xymax:
        raise ValueError("xymin and xymax must be finite and ordered.")
    if xymin == xymax:
        padding = max(1.0, abs(xymin) * 0.01)
        xymin, xymax = xymin - padding, xymax + padding

    def _limits(value, default, name):
        bounds = np.asarray(default if value is None else value, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must contain two increasing finite limits.")
        return tuple(bounds)

    xbounds = _limits(xlim, (xymin, xymax), "xlim")
    ydefault = ((xymin - period / 2, xymax + period / 2)
                if circular and circular_view == "nearest" else (xymin, xymax))
    ybounds = _limits(ylim, ydefault, "ylim")
    y_display = x + _wrap_deg(y - x, period) if circular and circular_view == "nearest" else y
    keep = ((x >= xbounds[0]) & (x <= xbounds[1]) &
            (y_display >= ybounds[0]) & (y_display <= ybounds[1]))
    x_plot, y_plot = x[keep], y_display[keep]

    stats_x, stats_y = (x, y) if stats_scope == "all" else (x[keep], y[keep])
    if source_axis == "x":
        stats_x, stats_y = stats_y, stats_x
    st = _summary_stats(stats_x, stats_y, circular=circular, period=period)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 7.0))
    else:
        fig = ax.figure
    rng = np.random.default_rng(0)
    idx = (rng.choice(x_plot.size, size=scatter_sample, replace=False)
           if x_plot.size > scatter_sample else np.arange(x_plot.size))
    ax.scatter(
        x_plot[idx], y_plot[idx],
        s=12, c="0.75", alpha=0.18, linewidths=0,
        rasterized=True, zorder=1,
    )

    xedges = np.linspace(*xbounds, hist_bins + 1)
    yedges = np.linspace(*ybounds, hist_bins + 1)
    H, xe, ye = np.histogram2d(x_plot, y_plot, bins=[xedges, yedges])
    H = H.T
    Hm = np.ma.masked_where(H == 0, H)
    cmap = LinearSegmentedColormap.from_list(
        "counts_cmap",
        ["#6d6bb8", "#9b78c5", "#ca86bc", "#e49aa8", "#e9b58f", "#e4d99c"],
        N=256,
    )
    count_levels = np.array([1, 2, 5, 10, 25, 50, 75, 100, 120], dtype=float)
    last_boundary = max(float(H.max()) + 1.0, 121.0)
    boundaries = np.r_[count_levels, last_boundary]
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=False)
    mesh = ax.pcolormesh(
        xe, ye, Hm, cmap=cmap, norm=norm, shading="auto", zorder=2,
    )
    if show_colorbar:
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        tick_pos = 0.5 * (boundaries[:-1] + boundaries[1:])
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels(["1", "2", "5", "10", "25", "50", "75", "100", "120+"])
        cbar.set_label("Counts")

    identity_limits = (max(xbounds[0], ybounds[0]), min(xbounds[1], ybounds[1]))
    if identity_limits[0] < identity_limits[1]:
        ax.plot(identity_limits, identity_limits, "--", color="k", lw=1.2, alpha=0.7, zorder=3)

    if show_bin_means:
        bins = np.arange(xbounds[0], xbounds[1] + mean_bin_width, mean_bin_width)
        digit = np.digitize(x_plot, bins) - 1
        # Match histogram2d's closed final bin, including the upper endpoint.
        digit = np.minimum(digit, len(bins) - 2)
        xm, ym, xerr = [], [], []
        for i in range(len(bins) - 1):
            m = digit == i
            if np.sum(m) < min_points_per_bin:
                continue
            if circular:
                # Opposing/uniform directions have no circular mean. SciPy
                # otherwise returns a floating-point-dependent direction.
                angles = (x_plot[m] * (2 * np.pi / period),
                          y_plot[m] * (2 * np.pi / period))
                if any(np.hypot(np.mean(np.sin(a)), np.mean(np.cos(a))) < 1e-12
                       for a in angles):
                    continue
            mean_f = partial(circmean, high=period, low=0) if circular else np.mean
            xmean, ymean = mean_f(x_plot[m]), mean_f(y_plot[m])
            if circular and circular_view == "nearest":
                # Keep means near this bin, e.g. x=359, y=361 rather than y=1.
                bin_midpoint = 0.5 * (bins[i] + bins[i + 1])
                xmean = bin_midpoint + _wrap_deg(xmean - bin_midpoint, period)
                ymean = xmean + _wrap_deg(ymean - xmean, period)
            xm.append(xmean)
            ym.append(ymean)
            xerr.append(0.5 * (bins[i + 1] - bins[i]))
        if xm:
            ax.errorbar(
                xm, ym, xerr=xerr, fmt="o", color="k", ms=8, lw=0,
                elinewidth=1.8, capsize=0, zorder=4,
            )

    unit_suffix = f" {units}" if units else ""
    lines = [f"Num={st['n']}", f"Bias={st['bias']:.2f}{unit_suffix}"]
    if show_ci:
        lines.append(f"(95% CI: {st['bias_ci'][0]:.3f}, {st['bias_ci'][1]:.3f})")
    lines.append(f"RMSE={st['rmse']:.2f}{unit_suffix}")
    if show_ci:
        lines.append(f"(95% CI: {st['rmse_ci'][0]:.3f}, {st['rmse_ci'][1]:.3f})")
    if show_mae:
        lines.append(f"MAE={st['mae']:.2f}{unit_suffix}")
    lines.append(f"Cor={st['cor']:.2f}" if np.isfinite(st["cor"]) else "Cor=N/A")
    ax.text(
        0.03, 0.96, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=4),
        zorder=5,
    )
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlabel(f"{xlabel}{f' ({units})' if units else ''}", fontsize=16)
    ax.set_ylabel(f"{ylabel}{f' ({units})' if units else ''}", fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=16)
    return fig, ax



def plot_wind_metric_vs_stations(
    results,
    *,
    metric="speed",                      # "speed" or "dir"
    lead=0,                              # int lead, None, or "all"
    reference_key=("StationsCSVDataset",),
    source_keys=None,
    nested_key="NestedLeadTimeAggregator",
    by_lead_key="by_lead",
    raw_key="RawFieldAggregator__1",
    units=None,
    save_dir=None,
    dpi=200,
):
    """
    Plot one figure per source/model against stations.

    If lead is None or "all", all common lead times are aligned by actual time
    and concatenated into one scatter plot.

    Expected structure:
      results[metric][dataset_key]['NestedLeadTimeAggregator']['by_lead'][lead]['RawFieldAggregator__1'][date]
    """
    if metric not in {"speed", "dir"}:
        raise ValueError("metric must be 'speed' or 'dir'")

    circular = (metric == "dir")

    if units is None:
        units = "deg" if circular else ""

    all_keys = list(results[metric].keys())

    if source_keys is None:
        source_keys = [k for k in all_keys if k != reference_key]

    use_all_leads = _is_all_leads(lead)
    lead_label = "all" if use_all_leads else str(lead)

    figs = {}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for src_key in source_keys:
        if use_all_leads:
            ref_by_lead = get_nested_by_lead_dict(
                results,
                metric=metric,
                dataset_key=reference_key,
                nested_key=nested_key,
                by_lead_key=by_lead_key,
            )
            pred_by_lead = get_nested_by_lead_dict(
                results,
                metric=metric,
                dataset_key=src_key,
                nested_key=nested_key,
                by_lead_key=by_lead_key,
            )

            ref_by_actual_time = _lead_date_dict_to_actual_time_dict(
                ref_by_lead,
                raw_key=raw_key,
            )
            pred_by_actual_time = _lead_date_dict_to_actual_time_dict(
                pred_by_lead,
                raw_key=raw_key,
            )

            x, y, _ = collect_pairs_from_time_dicts(
                pred_by_actual_time,
                ref_by_actual_time,
                pred_transform=_as_1d_scalar_field,
                ref_transform=_as_1d_scalar_field,
            )
        else:
            ref_by_date = get_nested_lead_date_dict(
                results,
                metric=metric,
                dataset_key=reference_key,
                lead=lead,
                nested_key=nested_key,
                by_lead_key=by_lead_key,
                raw_key=raw_key,
            )
            pred_by_date = get_nested_lead_date_dict(
                results,
                metric=metric,
                dataset_key=src_key,
                lead=lead,
                nested_key=nested_key,
                by_lead_key=by_lead_key,
                raw_key=raw_key,
            )

            x, y, _ = collect_pairs_from_date_dicts(
                pred_by_date,
                ref_by_date,
                pred_transform=_as_1d_scalar_field,
                ref_transform=_as_1d_scalar_field,
            )

        if x.size == 0:
            print(f"Skipping {_pretty_key(src_key)} at lead={lead_label}: no valid data.")
            continue

        title = f"{_pretty_key(src_key)} | {metric} | lead={lead_label}"

        fig, ax = plt.subplots(figsize=(7.2, 7.0))
        plot_pair_scatter(
            x, y,
            ax=ax,
            title=title,
            xlabel="Station",
            ylabel="Model",
            units=units,
            circular=circular,
            xymax=360.0 if circular else None,
            hist_bins=36 if circular else 56,
            mean_bin_width=15.0 if circular else 2.0,
            min_points_per_bin=20 if circular else 30,
            scatter_sample=25000,
            show_colorbar=True,
        )

        figs[src_key] = fig

        if save_dir is not None:
            safe_name = _pretty_key(src_key).replace("/", "_").replace(" ", "_")
            fig.savefig(
                save_dir / f"{metric}__lead_{lead_label}__{safe_name}.png",
                dpi=dpi,
                bbox_inches="tight",
            )

        plt.show()

    return figs
