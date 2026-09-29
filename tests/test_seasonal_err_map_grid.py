"""Seasonal table regressions using real aggregation and Matplotlib, without Cartopy."""

import ast
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Tuple
import unittest
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.figure import Figure
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative_path, names, namespace):
    """Load production definitions without unrelated optional geospatial imports."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing production definitions: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


aggregation = load_definitions(
    "libs/validation/aggregators.py",
    ["Aggregator", "SeasonalSpatialAggregator"],
    dict(np=np, ABC=ABC, abstractmethod=abstractmethod, Any=Any,
         Dict=Dict, Sequence=Sequence, Tuple=Tuple),
)
SeasonalSpatialAggregator = aggregation["SeasonalSpatialAggregator"]
get_color_params = load_definitions(
    "libs/validation/visualization.py", ["get_color_params"],
    dict(plt=plt, colors=colors),
)["get_color_params"]


def seasonal_result(**fields):
    fields = {
        name: np.full((2, 2), value, dtype=float) if np.ndim(value) == 0
        else np.asarray(value, dtype=float)
        for name, value in fields.items()
    }
    aggregator = SeasonalSpatialAggregator(
        {name: (month,) for month, name in enumerate(fields, 1)}
    )
    accumulator = aggregator.init_accumulator((2, 2))
    for month, field in enumerate(fields.values(), 1):
        # Two samples exercise averaging, instead of pre-finalized fixtures.
        for _ in range(2):
            aggregator.accumulate(accumulator, field, date(2024, month, 1))
    return {"SeasonalSpatialAggregator": accumulator}


class SeasonalErrMapGridTests(unittest.TestCase):
    def setUp(self):
        self.plots = []

        def create_axes(nrows, ncols, **kwargs):
            # Match the production helper's scalar result for the 1 x 1 case.
            return plt.subplots(nrows, ncols, figsize=kwargs.get("figsize"))

        def scalar_field(ax, grid, data, **kwargs):
            self.plots.append((ax, grid, np.asarray(data), kwargs))
            return ax.pcolormesh(
                data, **{key: kwargs[key] for key in ("norm", "cmap", "vmin", "vmax")
                         if key in kwargs}
            )

        self.visualization = SimpleNamespace(
            create_cartopy_axes=Mock(side_effect=create_axes),
            create_cartopy_grid=Mock(side_effect=create_axes),
            visualize_scalar_field=Mock(side_effect=scalar_field),
            get_color_params=get_color_params,
        )
        self.plot = load_definitions(
            "libs/validation/results_visualization.py", ["plot_seasonal_err_map_grid"],
            dict(np=np, plt=plt, colors=colors, visualization=self.visualization,
                 SeasonalSpatialAggregator=SeasonalSpatialAggregator),
        )["plot_seasonal_err_map_grid"]
        self.addCleanup(plt.close, "all")

    def test_auto_selection_season_union_and_stored_pair_orientation(self):
        results = {"diff": {
            ("target",): {},
            ("target", "a"): seasonal_result(Winter=-2, Summer=-4),
            ("b", "target"): seasonal_result(Summer=3, Autumn=5),
            ("a", "unrelated"): seasonal_result(Winter=1000),
            ("target", "target"): seasonal_result(Winter=1000),
            ("target", "a", "b"): seasonal_result(Winter=1000),
        }}
        fig, axes = self.plot(results, None, "target", "diff")

        self.assertEqual(axes.shape, (2, 3))
        self.assertEqual(len(self.plots), 4)
        for (_, _, field, _), value in zip(self.plots, [-2, -4, 3, 5]):
            np.testing.assert_array_equal(field, np.full((2, 2), value))
        self.assertIn("target vs a", axes[0, 0].get_title())
        self.assertIn("Winter", axes[0, 0].get_title())
        self.assertIn("b vs target", axes[1, 2].get_title())
        self.assertIn("Autumn", axes[1, 2].get_title())
        for axis in (axes[0, 2], axes[1, 0]):
            self.assertFalse(axis.collections)
            self.assertTrue(any("No data" in text.get_text() for text in axis.texts))
        self.assertEqual(len(fig.axes), 7)  # Six panels and one shared colorbar.

    def test_selection_order_and_shared_scale_exclude_unselected_data(self):
        results = {"mae": {
            ("a", "target"): seasonal_result(
                Winter=[[0, 10], [20, 30]], Summer=10, Autumn=1000000,
            ),
            ("b", "target"): seasonal_result(
                Winter=20, Summer=[[30, 40], [50, 60]], Autumn=1000000,
            ),
            ("unselected", "target"): seasonal_result(Winter=1000000),
        }}
        with patch.object(
            SeasonalSpatialAggregator, "finalize", wraps=SeasonalSpatialAggregator.finalize,
        ) as finalize:
            _, axes = self.plot(
                results, None, "target", "mae", ds_names=["b", "a"],
                seasons=["Summer", "Winter"],
            )

        self.assertEqual(finalize.call_count, 2)
        self.assertEqual(axes.shape, (2, 2))
        self.assertIn("b vs target", axes[0, 0].get_title())
        self.assertIn("Summer", axes[0, 0].get_title())
        self.assertIn("a vs target", axes[1, 1].get_title())
        self.assertIn("Winter", axes[1, 1].get_title())
        for axis in axes.flat:
            np.testing.assert_allclose(axis.collections[0].get_clim(), (0.3, 59.7))

    def test_single_panel_forwards_options_and_saves_once(self):
        results = {"mae": {("a", "target"): seasonal_result(Winter=2)}}
        lat, lon = np.zeros((2, 2)), np.ones((2, 2))
        projection, grid = object(), object()
        with patch.object(Figure, "savefig", autospec=True) as save, patch.object(plt, "show") as show:
            fig, axes = self.plot(
                results, grid, "target", "mae", variable="SIC error, fraction",
                figsize=(4, 3), suptitle="Seasonal errors", filename="seasonal.jpeg",
                axes_names=[["Custom title"]], proj=projection, lat=lat, lon=lon,
                cmap="viridis", vmin=0, vmax=12,
            )

        self.assertEqual(axes.shape, (1, 1))
        self.assertEqual(axes[0, 0].get_title(), "Custom title")
        self.assertEqual(fig._suptitle.get_text(), "Seasonal errors")
        self.assertEqual(fig.axes[-1].get_ylabel(), "SIC error, fraction")
        np.testing.assert_allclose(fig.get_size_inches(), (4, 3))
        _, received_grid, _, options = self.plots[0]
        self.assertIs(received_grid, grid)
        self.assertIs(options["lat"], lat)
        self.assertIs(options["lon"], lon)
        self.assertEqual(axes[0, 0].collections[0].get_cmap().name, "viridis")
        self.assertEqual(axes[0, 0].collections[0].get_clim(), (0, 12))
        self.assertIs(self.visualization.create_cartopy_axes.call_args.kwargs["proj"], projection)
        save.assert_called_once_with(fig, "seasonal.jpeg", dpi=600, bbox_inches="tight")
        show.assert_not_called()

    def test_nan_and_zero_difference_maps_have_valid_color_scale(self):
        for value in (np.nan, 0.0):
            with self.subTest(value=value):
                results = {"diff": {("a", "target"): seasonal_result(Winter=value)}}
                _, axes = self.plot(results, None, "target", "diff")
                lower, upper = axes[0, 0].collections[0].get_clim()
                self.assertTrue(np.isfinite([lower, upper]).all())
                self.assertLess(lower, upper)
                self.assertAlmostEqual(lower, -upper)

    def test_custom_normalization_uses_shared_bounds_and_preserves_explicit_bounds(self):
        results = {"diff": {
            ("a", "target"): seasonal_result(Winter=[[1, 2], [3, 4]]),
            ("b", "target"): seasonal_result(Winter=[[10, 20], [30, 40]]),
        }}
        cases = [
            (colors.Normalize(), (1.03, 39.7)),
            ("linear", (1.03, 39.7)),
            (colors.Normalize(vmin=-10, vmax=60), (-10, 60)),
        ]
        for norm, expected in cases:
            with self.subTest(norm=norm):
                _, axes = self.plot(results, None, "target", "diff", norm=norm)
                for axis in axes.flat:
                    np.testing.assert_allclose(axis.collections[0].get_clim(), expected)

    def test_invalid_selections_fail_informatively(self):
        pair = ("a", "target")
        results = {"mae": {pair: seasonal_result(Winter=2)}}
        cases = [
            (results, "target", "missing", {}, "metric"),
            (results, "absent", "mae", {}, "absent|target|dataset"),
            (results, "target", "mae", {"ds_names": []}, "dataset|ds_names"),
            (results, "target", "mae", {"ds_names": ["missing"]}, "missing"),
            (results, "target", "mae", {"seasons": []}, "season"),
            (results, "target", "mae", {"seasons": ["missing"]}, "missing|season"),
            ({"mae": {pair: {}}}, "target", "mae", {}, "SeasonalSpatialAggregator|aggregator"),
            ({"mae": {pair: seasonal_result(Winter=2),
                      ("target", "a"): seasonal_result(Winter=-2)}},
             "target", "mae", {}, "ambiguous|multiple|more than one"),
        ]
        for data, target, metric, options, message in cases:
            with self.subTest(target=target, metric=metric, options=options, message=message):
                with self.assertRaisesRegex(ValueError, "(?i)" + message):
                    self.plot(data, None, target, metric, **options)


if __name__ == "__main__":
    unittest.main()
