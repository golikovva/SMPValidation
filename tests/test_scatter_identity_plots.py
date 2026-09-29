"""Shared paired scatter behavior, independent of optional geospatial imports."""

import importlib.util
from pathlib import Path
import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_scatter_identity_plots_test", ROOT / "libs/validation/scatter_identity_plots.py",
)
scatter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scatter)


class PairScatterTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def plot(self, x, y, **kwargs):
        kwargs.setdefault("show_colorbar", False)
        kwargs.setdefault("hist_bins", 10)
        return scatter.plot_pair_scatter(x, y, **kwargs)

    def test_legacy_defaults_keep_orientation_style_and_visible_statistics(self):
        fig, ax = self.plot([1, 2, 3, 4, -1, 100], [2, 3, 4, 5, 0, 120],
                            xymax=5, min_points_per_bin=1, show_colorbar=True)
        self.assertEqual(ax.texts[0].get_text(),
                         "Num=4\nBias=1.00\n(95% CI: 1.000, 1.000)\n"
                         "RMSE=1.00\n(95% CI: 1.000, 1.000)\nCor=1.00")
        self.assertEqual(ax.get_xlim(), (0., 5.))
        self.assertEqual(ax.get_ylim(), (0., 5.))
        self.assertEqual(ax.get_xlabel(), "Reference")
        self.assertEqual(ax.get_ylabel(), "Model")
        self.assertEqual(fig.axes[-1].get_ylabel(), "Counts")
        self.assertEqual(ax.collections[1].get_array().sum(), 4)
        np.testing.assert_allclose(ax.lines[-1].get_xdata(orig=False), [1, 2.5, 4])
        np.testing.assert_allclose(ax.lines[-1].get_ydata(orig=False), [2, 3.5, 5])

    def test_source_on_x_preserves_negative_data_and_reports_mae(self):
        _, ax = self.plot([-3, -2, -1], [-2, -1, 0], xymin=-4, xymax=1,
                          source_axis="x", stats_scope="all", show_ci=False,
                          show_bin_means=False, show_mae=True)
        self.assertEqual(ax.texts[0].get_text(),
                         "Num=3\nBias=-1.00\nRMSE=1.00\nMAE=1.00\nCor=1.00")
        np.testing.assert_array_equal(ax.collections[0].get_offsets(),
                                      [[-3, -2], [-2, -1], [-1, 0]])
        self.assertEqual(ax.collections[1].get_array().sum(), 3)
        self.assertEqual(len(ax.lines), 1)  # Identity only; no mean markers.

    def test_axis_limits_do_not_change_full_statistics(self):
        x, y = [1, 2, 20, np.nan], [2, 4, 25, 0]
        options = dict(source_axis="x", stats_scope="all", show_ci=False,
                       show_bin_means=False, show_mae=True)
        _, complete = self.plot(x, y, xymax=30, **options)
        _, zoom = self.plot(x, y, xlim=(0, 3), ylim=(0, 5), **options)
        self.assertEqual(complete.texts[0].get_text(), zoom.texts[0].get_text())
        self.assertTrue(zoom.texts[0].get_text().startswith("Num=3\n"))
        self.assertEqual(zoom.collections[1].get_array().sum(), 2)
        self.assertEqual(zoom.get_xlim(), (0, 3))
        self.assertEqual(zoom.get_ylim(), (0, 5))

    def test_wrapped_and_nearest_views_have_identical_statistics_and_counts(self):
        options = dict(circular=True, source_axis="x", stats_scope="all",
                       show_ci=False, show_bin_means=False, show_mae=True)
        _, wrapped = self.plot([359, 1], [1, 359], **options)
        _, nearest = self.plot([359, 1], [1, 359], circular_view="nearest", **options)
        self.assertEqual(wrapped.texts[0].get_text(), nearest.texts[0].get_text())
        self.assertIn("RMSE=2.00\nMAE=2.00", nearest.texts[0].get_text())
        np.testing.assert_array_equal(wrapped.collections[0].get_offsets(),
                                      [[359, 1], [1, 359]])
        np.testing.assert_array_equal(nearest.collections[0].get_offsets(),
                                      [[359, 361], [1, -1]])
        self.assertEqual(nearest.get_ylim(), (-180, 540))
        for ax in (wrapped, nearest):
            self.assertEqual(ax.collections[1].get_array().sum(), 2)

    def test_circular_period_and_nearest_bin_means(self):
        _, ax = self.plot([358, 359], [0, 1], circular=True,
                          circular_view="nearest", min_points_per_bin=1)
        np.testing.assert_allclose(ax.lines[-1].get_xdata(orig=False), [358.5])
        np.testing.assert_allclose(ax.lines[-1].get_ydata(orig=False), [360.5])
        period = 2 * np.pi
        _, radians = self.plot([period - .1], [.1], circular=True, period=period,
                               circular_view="nearest", source_axis="x", show_ci=False)
        self.assertIn("Bias=-0.20", radians.texts[0].get_text())
        np.testing.assert_allclose(radians.collections[0].get_offsets(),
                                   [[period - .1, period + .1]])
        self.assertEqual(radians.get_xlim(), (0, period))

    def test_constant_singleton_and_empty_view_are_safe(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for x, y, options in (([0], [0], {}), ([2, 2], [2, 2], {}),
                                  ([359, 359], [1, 1], {"circular": True})):
                with self.subTest(x=x, options=options):
                    _, ax = self.plot(x, y, **options)
                    self.assertIn("Cor=N/A", ax.texts[0].get_text())
            _, constant = self.plot([2, 2], [2, 2], xymin=2, xymax=2)
            self.assertLess(constant.get_xlim()[0], 2)
            self.assertGreater(constant.get_xlim()[1], 2)
            _, empty = self.plot([1], [2], xlim=(4, 5), ylim=(6, 7))
            self.assertTrue(empty.texts[0].get_text().startswith("Num=0\n"))
            self.assertEqual(empty.collections[0].get_offsets().shape, (0, 2))
            _, full = self.plot([1], [2], xlim=(4, 5), ylim=(6, 7), stats_scope="all")
            self.assertTrue(full.texts[0].get_text().startswith("Num=1\n"))

    def test_invalid_inputs_are_rejected_before_creating_a_figure(self):
        options = [{"source_axis": "z"}, {"stats_scope": "some"}, {"period": 0},
                   {"period": np.inf}, {"hist_bins": 0}, {"hist_bins": 1.5},
                   {"scatter_sample": -1}, {"min_points_per_bin": 0},
                   {"mean_bin_width": np.nan}, {"mean_bin_width": 0},
                   {"xymin": 4, "xymax": 2}, {"xlim": (1, 1)},
                   {"ylim": (2, 1)}, {"xlim": (0, np.inf)},
                   {"circular_view": "other"}, {"circular_view": "nearest"}]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plot([1], [2], **kwargs)
        for x, y in (([], []), ([np.nan], [1]), ([1, 2], [1])):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                self.plot(x, y)
        self.assertFalse(plt.get_fignums())

    def test_summary_handles_empty_and_undefined_circular_correlation(self):
        self.assertEqual(scatter._summary_stats([], [])["n"], 0)
        stats = scatter._summary_stats([0, 180], [10, 190], circular=True)
        self.assertTrue(np.isnan(stats["cor"]))
        self.assertAlmostEqual(stats["bias"], 10)
        self.assertAlmostEqual(stats["mae"], 10)

    def test_bin_means_include_the_final_endpoint(self):
        _, ax = self.plot([4], [3], xymax=4, mean_bin_width=2,
                          min_points_per_bin=1)
        self.assertEqual(ax.collections[1].get_array().sum(), 1)
        np.testing.assert_allclose(ax.lines[-1].get_xdata(orig=False), [4])
        np.testing.assert_allclose(ax.lines[-1].get_ydata(orig=False), [3])

    def test_undefined_circular_bin_means_do_not_draw_arbitrary_markers(self):
        for view in ("wrapped", "nearest"):
            with self.subTest(view=view):
                _, ax = self.plot([0, 0], [90, 270], circular=True,
                                  circular_view=view, min_points_per_bin=1)
                self.assertEqual(ax.collections[1].get_array().sum(), 2)
                self.assertEqual(len(ax.lines), 1)  # Only the identity line.
                self.assertIn("Cor=N/A", ax.texts[0].get_text())

    def test_zero_error_confidence_interval_and_singleton_uncertainty(self):
        for circular, x, y in ((False, [1, 2], [1, 2]),
                               (True, [0, 90], [360, 450])):
            with self.subTest(circular=circular):
                stats = scatter._summary_stats(x, y, circular=circular)
                self.assertEqual(stats["bias_ci"], (0, 0))
                self.assertEqual(stats["rmse_ci"], (0, 0))
        singleton = scatter._summary_stats([1], [1])
        self.assertTrue(np.isnan(singleton["bias_ci"]).all())
        self.assertTrue(np.isnan(singleton["rmse_ci"]).all())


if __name__ == "__main__":
    unittest.main()
