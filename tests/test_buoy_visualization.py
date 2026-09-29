"""Buoy plots retain metadata alignment, physical values and finite sample counts."""

from copy import deepcopy
from datetime import date, datetime
import importlib
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALIAS = "_buoy_visualization_test_package"
_previous_modules = {
    key: value for key, value in sys.modules.items()
    if key == ALIAS or key.startswith(ALIAS + ".")
}
_package = ModuleType(ALIAS)
_package.__path__ = [str(ROOT / "libs" / "validation")]
sys.modules[ALIAS] = _package
visualization = importlib.import_module(ALIAS + ".buoy_visualization")
HAS_CARTOPY = importlib.util.find_spec("cartopy") is not None


def tearDownModule():
    for key in tuple(sys.modules):
        if key == ALIAS or key.startswith(ALIAS + "."):
            sys.modules.pop(key)
    sys.modules.update(_previous_modules)


class Field(np.ndarray):
    """Only the public ndarray plus metadata protocol is needed by the plotter."""

    def __new__(cls, values, **metadata):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.meta = metadata
        return obj

    def __array_finalize__(self, original):
        self.meta = getattr(original, "meta", {})


def times(*hours):
    return np.datetime64("2023-01-01", "ns") + np.asarray(hours).astype("timedelta64[h]")


def field(values, *, bids=("a",), at=None, names=("sst",), units=("degC",),
          valid=None, coords=None):
    values = np.asarray(values, dtype=float)
    if at is None:
        at = times(*range(values.shape[1]))
    if coords is None:
        # Match a location by ID and timestamp, independent of fixture ordering.
        coords = np.array([
            [[80 + (ord(bid[0]) - ord("a")) * .1,
              20 + float((t - np.datetime64("2023-01-01")) / np.timedelta64(1, "h")) * .1]
             for t in at]
            for bid in bids
        ])
    metadata = dict(
        dims=("buoy", "time", "variable") if values.ndim == 3 else ("buoy", "time"),
        bids=np.asarray(bids), datetimes=np.asarray(at), coords=np.asarray(coords),
        valid=np.ones(values.shape, dtype=bool) if valid is None else np.asarray(valid),
        units=units,
    )
    if values.ndim == 3:
        metadata["var_names"] = names
    else:
        metadata["result_name"] = names[0]
    return Field(values, **metadata)


def results_for(source, reference=None, *, window="2023-01-01"):
    result = {"identity": {("source",): {"RawFieldAggregator": {window: source}}}}
    if reference is not None:
        result["identity"][("reference",)] = {"RawFieldAggregator": {window: reference}}
    return result


def annotation(axis):
    return "\n".join(text.get_text() for text in axis.texts)


def stat(axis, name):
    text = annotation(axis)
    match = re.search(rf"\b{name}\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
    if match is None:
        raise AssertionError(f"Statistic {name!r} absent from {text!r}")
    return float(match.group(1))


class BuoyRecordTests(unittest.TestCase):
    def test_pairing_uses_ids_timestamps_and_windows(self):
        source = field([[[10], [11], [12]], [[20], [21], [22]]],
                       bids=("a", "b"), at=times(0, 1, 2))
        reference = field([[[220], [210], [230]], [[120], [110], [130]]],
                          bids=("b", "a"), at=times(2, 1, 3))
        result = results_for(source, reference)
        # A second validation window must not cross-join the same ID and time.
        result["identity"][("source",)]["RawFieldAggregator"]["2023-01-02"] = source
        left = visualization._collect_records(result, "identity", ("source",))
        right = visualization._collect_records(result, "identity", ("reference",))
        paired = visualization._paired_records(left, right)
        self.assertEqual(len(paired), 4)
        actual = set(zip(paired["value_source"], paired["value_reference"]))
        self.assertEqual(actual, {(11, 110), (12, 120), (21, 210), (22, 220)})

    def test_scalar_forms_and_variable_selection(self):
        for data in ([[-2., -1., 0.]], [[[-2.], [-1.], [0.]]]):
            with self.subTest(shape=np.shape(data)):
                frame = visualization._collect_records(results_for(field(data)), "identity", ("source",))
                np.testing.assert_allclose(frame["value"], [-2, -1, 0])
        multi = results_for(field([[[1, 11], [2, 12]]], names=("sst", "thickness"), units=("degC", "m")))
        with self.assertRaisesRegex(ValueError, "variable"):
            visualization._collect_records(multi, "identity", ("source",))
        selected = visualization._collect_records(multi, "identity", ("source",), variable="thickness")
        np.testing.assert_allclose(selected["value"], [11, 12])
        self.assertEqual(selected.attrs["units"], "m")

    def test_vector_norm_direction_masks_and_zero_vectors(self):
        data = field([[[3, 4], [0, 1], [0, 0], [-1, 0], [1, 2]]],
                     names=("eastward", "northward"), units=("cm/s", "cm/s"),
                     valid=[[[True, True], [True, True], [True, True],
                             [True, True], [True, False]]])
        result = results_for(data)
        with self.assertRaisesRegex(ValueError, "variable|component"):
            visualization._collect_records(result, "identity", ("source",), reduction="norm")
        kwargs = dict(variable=("eastward", "northward"))
        norm = visualization._collect_records(result, "identity", ("source",), reduction="norm", **kwargs)
        direction = visualization._collect_records(result, "identity", ("source",), reduction="direction", **kwargs)
        np.testing.assert_allclose(norm["value"], [5, 1, 0, 1, np.nan], equal_nan=True)
        np.testing.assert_allclose(direction["value"], [np.degrees(np.arctan2(4, 3)), 90, np.nan, 180, np.nan], equal_nan=True)

    def test_validity_masks_and_nan_keep_rows_to_break_tracks(self):
        data = field([[[1], [2], [np.nan], [4]]], valid=[[[True], [False], [True], [True]]])
        records = visualization._collect_records(results_for(data), "identity", ("source",))
        self.assertEqual(len(records), 4)
        np.testing.assert_allclose(records["value"], [1, np.nan, np.nan, 4], equal_nan=True)

    def test_date_bounds_cover_whole_day_and_datetimes_are_exact(self):
        data = field([[[0], [1], [2], [3], [4]]], at=times(0, 12, 23, 24, 25))
        result = results_for(data)
        for start, end in ((date(2023, 1, 1), date(2023, 1, 1)),
                           ("2023-01-01", "2023-01-01")):
            records = visualization._collect_records(result, "identity", ("source",), start=start, end=end)
            np.testing.assert_allclose(records["value"], [0, 1, 2])
        exact = visualization._collect_records(
            result, "identity", ("source",),
            start=datetime(2023, 1, 1, 12), end=datetime(2023, 1, 2),
        )
        np.testing.assert_allclose(exact["value"], [1, 2, 3])

    def test_metadata_errors_are_actionable(self):
        result = results_for(np.ones((1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "meta|metadata"):
            visualization._collect_records(result, "identity", ("source",))
        for change in ({"datetimes": times(0, 1)}, {"coords": np.zeros((1, 1))},
                       {"dims": ("lat", "lon", "variable")}):
            with self.subTest(change=change):
                data = field([[[1]]])
                data.meta.update(change)
                with self.assertRaises(ValueError):
                    visualization._collect_records(results_for(data), "identity", ("source",))

    def test_pairing_rejects_coordinate_and_unit_mismatch(self):
        source = field([[[1]]])
        for reference in (field([[[2]]], units=("K",)),
                          field([[[2]]], coords=[[[79., 20.]]])):
            result = results_for(source, reference)
            left = visualization._collect_records(result, "identity", ("source",))
            right = visualization._collect_records(result, "identity", ("reference",))
            with self.assertRaisesRegex(ValueError, "unit|coord|location"):
                visualization._paired_records(left, right)

    def test_wrapped_equivalent_longitudes_align(self):
        result = results_for(field([[[1]]], coords=[[[80, 181]]]),
                             field([[[2]]], coords=[[[80, -179]]]))
        left = visualization._collect_records(result, "identity", ("source",))
        right = visualization._collect_records(result, "identity", ("reference",))
        self.assertEqual(len(visualization._paired_records(left, right)), 1)

    def test_gap_policy_preserves_samples_and_breaks_long_intervals(self):
        at = times(0, 1, 2, 10, 11)
        values = np.arange(5, dtype=float)
        broken_times, broken_values = visualization._break_gaps(at, values, None)
        self.assertEqual(len(broken_times), 6)
        np.testing.assert_allclose(broken_values, [0, 1, 2, np.nan, 3, 4], equal_nan=True)
        self.assertTrue(at[2] < broken_times[3] < at[3])
        _, unbroken = visualization._break_gaps(at, values, "9h")
        np.testing.assert_array_equal(unbroken, values)
        _, positions = visualization._break_gaps(at, np.column_stack([values, values]), None)
        self.assertTrue(np.isnan(positions[3]).all())
        for invalid in ("0h", "-1h"):
            with self.assertRaisesRegex(ValueError, "max_gap"):
                visualization._break_gaps(at, values, invalid)


class BuoyScatterTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_source_on_x_negative_sst_bias_and_no_automatic_output(self):
        source = field([[[-2], [-1], [20]]])
        reference = field([[[-3], [-3], [17]]])
        source_meta = deepcopy(source.meta)
        result = results_for(source, reference)
        fig, ax = plt.subplots()
        with patch.object(plt, "show") as show, patch.object(Figure, "savefig") as save:
            returned = visualization.plot_buoy_scatter(result, "source", "reference", ax=ax)
        self.assertEqual(returned, (fig, ax))
        self.assertIn("source", ax.get_xlabel())
        self.assertIn("reference", ax.get_ylabel())
        self.assertEqual(ax.get_xlabel().count("degC"), 1)
        self.assertEqual(ax.get_ylabel().count("degC"), 1)
        self.assertEqual(stat(ax, "Num"), 3)
        self.assertEqual(stat(ax, "Bias"), 2)
        self.assertAlmostEqual(stat(ax, "RMSE"), np.sqrt(14 / 3), places=2)
        self.assertEqual(stat(ax, "MAE"), 2)
        self.assertNotIn("95%", annotation(ax))
        self.assertLess(ax.get_xlim()[0], -2)
        np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
        self.assertEqual(ax.get_aspect(), 1)
        np.testing.assert_array_equal(source, [[[-2], [-1], [20]]])
        for key in source_meta:
            np.testing.assert_equal(source.meta[key], source_meta[key])
        show.assert_not_called()
        save.assert_not_called()

    def test_statistics_do_not_change_when_outliers_are_outside_plot_bounds(self):
        result = results_for(field([[[-2], [1], [100]]]), field([[[-3], [0], [90]]]))
        _, full = visualization.plot_buoy_scatter(result, "source", "reference")
        _, limited = visualization.plot_buoy_scatter(result, "source", "reference", xymax=2)
        self.assertEqual(annotation(full), annotation(limited))
        self.assertEqual(stat(limited, "Num"), 3)
        self.assertEqual(stat(limited, "Bias"), 4)

    def test_only_finite_jointly_valid_pairs_contribute(self):
        result = results_for(
            field([[[1], [2], [3], [np.nan]]], valid=[[[True], [False], [True], [True]]]),
            field([[[0], [1], [np.nan], [2]]]),
        )
        _, axis = visualization.plot_buoy_scatter(result, "source", "reference")
        self.assertEqual(stat(axis, "Num"), 1)
        self.assertEqual(stat(axis, "Bias"), 1)
        self.assertRegex(annotation(axis), r"Cor\s*=\s*(?:N/A|n/a|unavailable)")

    def test_circular_shortest_errors_and_view_independence(self):
        result = results_for(field([[[359], [1]]], units=("degrees",)),
                             field([[[1], [359]]], units=("degrees",)))
        texts = []
        for view in ("wrapped", "nearest"):
            _, axis = visualization.plot_buoy_scatter(
                result, "source", "reference", circular=True, circular_view=view,
            )
            self.assertEqual(stat(axis, "Num"), 2)
            self.assertEqual(stat(axis, "Bias"), 0)
            self.assertEqual(stat(axis, "RMSE"), 2)
            texts.append(annotation(axis))
        self.assertEqual(texts[0], texts[1])

    def test_custom_period_and_ci_option(self):
        result = results_for(field([[[23], [1]]], units=("hours",)),
                             field([[[1], [23]]], units=("hours",)))
        _, axis = visualization.plot_buoy_scatter(
            result, "source", "reference", circular=True, period=24, show_ci=True,
        )
        self.assertEqual(stat(axis, "RMSE"), 2)
        self.assertIn("95% CI", annotation(axis))
        np.testing.assert_allclose(axis.get_xlim(), (0, 24))

    def test_constant_circular_correlation_is_unavailable(self):
        result = results_for(field([[[359], [359]]], units=("degrees",)),
                             field([[[1], [1]]], units=("degrees",)))
        _, axis = visualization.plot_buoy_scatter(result, "source", "reference", circular=True)
        self.assertEqual(stat(axis, "Bias"), -2)
        self.assertRegex(annotation(axis), r"Cor\s*=\s*(?:N/A|n/a|unavailable)")

    def test_empty_selection_fails_without_silent_truncation(self):
        result = results_for(field([[[1]]]), field([[[2]]], at=times(1)))
        with self.assertRaisesRegex(ValueError, "point|sample|match|overlap|empty"):
            visualization.plot_buoy_scatter(result, "source", "reference")
        result = results_for(field([[[1]]]), field([[[2]]]))
        for kwargs in ({"buoy_ids": ["missing"]}, {"start": "2024-01-01"}):
            with self.assertRaises(ValueError):
                visualization.plot_buoy_scatter(result, "source", "reference", **kwargs)


@unittest.skipUnless(HAS_CARTOPY, "Cartopy is optional for buoy data and scatter tests")
class BuoyMapTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        # Exercise actual projections/helpers without downloading Natural Earth data.
        self.map_kwargs = dict(add_land=False, add_coastlines=False, add_gridlines=False)

    def test_map_keeps_every_selected_valid_error_and_uses_diverging_scale(self):
        data = field([[[1], [-2], [3]], [[-4], [5], [np.nan]]],
                     bids=("a", "b"), at=times(0, 24, 48))
        result = {"diff": {("source", "reference"): {"RawFieldAggregator": {"2023-01-01": data}}}}
        fig, axis = visualization.plot_buoy_metric_map(
            result, "diff", ("source", "reference"), end="2023-01-02", map_kwargs=self.map_kwargs,
        )
        dots = [item for item in axis.collections if isinstance(item, PathCollection)]
        self.assertEqual(sum(len(item.get_offsets()) for item in dots), 4)
        np.testing.assert_allclose(np.sort(dots[0].get_array()), [-4, -2, 1, 5])
        self.assertAlmostEqual(dots[0].norm(0), .5)
        self.assertIn("degC", fig.axes[-1].get_ylabel())
        fig.canvas.draw()

    def test_track_multiple_sources_panels_summary_and_gaps(self):
        at = times(0, 1, 2, 10, 11)
        source = field([[[1], [2], [np.nan], [3], [4]]], at=at)
        ref = field([[[2], [3], [4], [4], [5]]], at=at)
        result = results_for(source, ref)
        result["identity"][("second",)] = {"RawFieldAggregator": {"2023-01-01": source + 1}}
        result["diff"] = {
            (name, "reference"): {"RawFieldAggregator": {"2023-01-01": field([[-1, -1, np.nan, -1, -1]], at=at)}}
            for name in ("source", "second")
        }
        fig, axes = visualization.plot_buoy_track(
            result, "a", "reference", sources=["source", "second"],
            panels=[visualization.BuoyPanel("identity"), visualization.BuoyPanel("diff")],
            show_summary=True, map_kwargs=self.map_kwargs,
        )
        self.assertEqual(len(axes["panels"]), 2)
        self.assertIsNotNone(axes["summary"])
        panel0 = {line.get_label(): line for line in axes["panels"][0].lines}
        panel1 = {line.get_label(): line for line in axes["panels"][1].lines}
        self.assertIn("reference", panel0)
        self.assertNotIn("reference", panel1)
        for name in ("source", "second"):
            self.assertEqual(panel0[name].get_color(), panel1[name].get_color())
            self.assertTrue(np.isnan(panel0[name].get_ydata()).any())
        fig.canvas.draw()

    def test_single_location_and_dateline_have_finite_local_extent(self):
        for lons in ([179.5], [179.5, -179.5]):
            coords = np.array([[[80, lon] for lon in lons]])
            data = field(np.ones((1, len(lons), 1)), coords=coords)
            result = results_for(data, data)
            fig, axes = visualization.plot_buoy_track(
                result, "a", "reference", sources=["source"],
                panels=[visualization.BuoyPanel("identity")], map_kwargs=self.map_kwargs,
            )
            axis = axes["map"]
            self.assertTrue(np.isfinite([*axis.get_xlim(), *axis.get_ylim()]).all())
            self.assertLess(np.diff(axis.get_xlim())[0], 1e6)
            self.assertTrue(axis.collections or axis.lines)
            fig.canvas.draw()

    def test_overlapping_windows_remain_separate_time_curves(self):
        result = results_for(field([[[1], [2], [3]]], at=times(0, 1, 2)),
                             field([[[2], [3], [4]]], at=times(0, 1, 2)))
        result["identity"][("source",)]["RawFieldAggregator"]["2023-01-02"] = field(
            [[[10], [20], [30]]], at=times(1, 2, 3),
        )
        result["identity"][("reference",)]["RawFieldAggregator"]["2023-01-02"] = field(
            [[[3], [4], [5]]], at=times(1, 2, 3),
        )
        fig, axes = visualization.plot_buoy_track(
            result, "a", "reference", sources=["source"],
            panels=[visualization.BuoyPanel("identity")], map_kwargs=self.map_kwargs,
        )
        panel = axes["panels"][0]
        source_color = next(line.get_color() for line in panel.lines if line.get_label() == "source")
        source_curves = [line for line in panel.lines if line.get_color() == source_color]
        self.assertEqual(len(source_curves), 2)
        actual_values = {tuple(line.get_ydata()) for line in source_curves}
        self.assertEqual(actual_values, {(1, 2, 3), (10, 20, 30)})
        for line in source_curves:
            self.assertEqual(len(np.unique(line.get_xdata())), 3)
        legend_labels = [text.get_text() for text in panel.get_legend().get_texts()]
        self.assertEqual(legend_labels.count("source"), 1)
        self.assertEqual(legend_labels.count("reference"), 1)
        fig.canvas.draw()

    def test_map_trajectory_never_bridges_time_or_coordinate_gaps(self):
        at = times(0, 1, 2, 3, 10, 11)
        coords = np.array([[[80., 20.], [80., 20.1], [np.nan, np.nan],
                            [80., 20.3], [80., 21.], [80., 21.1]]])
        data = field(np.ones((1, len(at), 1)), at=at, coords=coords)
        fig, axes = visualization.plot_buoy_track(
            results_for(data, data), "a", "reference", sources=["source"],
            panels=[visualization.BuoyPanel("identity")], map_kwargs=self.map_kwargs,
        )
        segments = [line for line in axes["map"].lines if len(line.get_xdata()) > 1]
        self.assertEqual(len(segments), 2)
        actual = np.asarray([line.get_xdata() for line in segments])
        np.testing.assert_allclose(actual, [[20., 20.1], [21., 21.1]])
        fig.canvas.draw()

    def test_isolated_trajectory_locations_are_visible_without_segments(self):
        cases = (
            (times(0, 1, 2, 3, 4),
             [[[80., 20.], [np.nan, np.nan], [80., 20.2], [np.nan, np.nan], [80., 20.4]]], None),
            (times(0, 3, 6), [[[80., 20.], [80., 20.2], [80., 20.4]]], "1h"),
        )
        for at, coords, max_gap in cases:
            with self.subTest(max_gap=max_gap):
                data = field(np.ones((1, len(at), 1)), at=at, coords=coords)
                fig, axes = visualization.plot_buoy_track(
                    results_for(data, data), "a", "reference", sources=["source"],
                    panels=[visualization.BuoyPanel("identity")], max_gap=max_gap,
                    map_kwargs=self.map_kwargs,
                )
                map_axis = axes["map"]
                shown = []
                for layer in map_axis.collections:
                    if isinstance(layer, PathCollection):
                        shown.extend(np.asarray(layer.get_offsets()).tolist())
                for line in map_axis.lines:
                    if line.get_marker() not in (None, "None", "", " "):
                        shown.extend(zip(line.get_xdata(), line.get_ydata()))
                    self.assertLessEqual(len(line.get_xdata()), 1)
                self.assertTrue(shown, "Isolated buoy positions must remain visible")
                np.testing.assert_allclose(np.unique(np.asarray(shown), axis=0),
                                           [[20., 80.], [20.2, 80.], [20.4, 80.]])
                time_bars = [axis for axis in fig.axes
                             if axis not in axes["panels"] and "Time" in axis.get_xlabel()]
                self.assertEqual(len(time_bars), 1)
                fig.canvas.draw()

    def test_reverse_metric_pair_is_not_used_implicitly(self):
        data = field([[[1], [2]]])
        result = results_for(data, data)
        result["diff"] = {("reference", "source"): {"RawFieldAggregator": {"2023-01-01": data}}}
        with self.assertRaisesRegex(ValueError, "reverse|orientation|order"):
            visualization.plot_buoy_track(
                result, "a", "reference", sources=["source"],
                panels=[visualization.BuoyPanel("diff")], map_kwargs=self.map_kwargs,
            )


if __name__ == "__main__":
    unittest.main()
