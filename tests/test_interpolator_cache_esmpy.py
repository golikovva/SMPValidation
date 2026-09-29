"""Disk-cache integration tests using real ESMPy and NetCDF files.

Run in the project scientific environment; missing native dependencies produce
an explicit skip. No model data or optional plotting/geospatial imports are used.
"""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_ERROR = None
try:
    import esmpy
    import netCDF4
except (ImportError, OSError) as exc:
    DEPENDENCY_ERROR = str(exc)


def load_interpolator():
    """Import production modules without running libs.validation.__init__."""
    package_name = "_interpolation_cache_integration"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "libs" / "validation")]
        sys.modules[package_name] = package
    module_name = package_name + ".interpolator"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "libs" / "validation" / "interpolator.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].Interpolator


class RectilinearGrid:
    """Minimal project Grid interface backed by a genuine spherical ESMF Grid."""

    def __init__(self, size, bounds):
        self.shape = (size, size)
        west, east, south, north = bounds
        x_edges = np.linspace(west, east, size + 1)
        y_edges = np.linspace(south, north, size + 1)
        self.lon, self.lat = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
            indexing="ij",
        )
        self.lon_corners, self.lat_corners = np.meshgrid(
            x_edges, y_edges, indexing="ij"
        )
        self.grid = esmpy.Grid(
            np.array(self.shape, dtype=np.int32),
            staggerloc=[esmpy.StaggerLoc.CENTER, esmpy.StaggerLoc.CORNER],
            coord_sys=esmpy.CoordSys.SPH_DEG,
        )
        for stagger, lon, lat in (
            (esmpy.StaggerLoc.CENTER, self.lon, self.lat),
            (esmpy.StaggerLoc.CORNER, self.lon_corners, self.lat_corners),
        ):
            self.grid.get_coords(0, staggerloc=stagger)[...] = lon
            self.grid.get_coords(1, staggerloc=stagger)[...] = lat

    def cell_areas(self):
        field = esmpy.Field(self.grid)
        try:
            field.get_area()
            return field.data.copy()
        finally:
            field.destroy()


@contextmanager
def grid_pair(method, offset=0.0):
    src_size, dst_size = (12, 6) if method == "CONSERVE" else (6, 12)
    source = RectilinearGrid(src_size, (-5, 5, 40, 50))
    try:
        destination = RectilinearGrid(dst_size, (-4 + offset, 4 + offset, 41, 49))
        try:
            yield source, destination
        finally:
            destination.grid.destroy()
    finally:
        source.grid.destroy()


def sample_results(interpolator, source):
    smooth = 1.0 + source.lon * 0.01 + source.lat * 0.001
    missing = smooth.copy()
    midpoint = source.shape[0] // 2
    missing[midpoint - 1 : midpoint + 1, midpoint - 1 : midpoint + 1] = np.nan
    return {
        "region": interpolator.dst_region.copy(),
        "constant": interpolator(np.ones(source.shape)),
        "smooth": interpolator(smooth),
        "missing": interpolator(missing),
    }


def fresh_process_result(cache_dir, output_path, method):
    """A new interpreter must load weights without calling the weight builder."""
    manager = esmpy.Manager()
    interpolator_class = load_interpolator()
    with grid_pair(method, offset=5.0) as (source, destination):
        operator = interpolator_class(source, destination)
        try:
            with patch.object(
                esmpy, "Regrid", side_effect=AssertionError("Weights were recalculated")
            ):
                operator.initialize(cache_dir=cache_dir)
                results = sample_results(operator, source)
            np.savez(output_path, **results)
        finally:
            operator.destroy()
    # Keep the Manager alive until all ESMF resources above have been released.
    assert manager is not None


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"ESMPy/netCDF4 unavailable: {DEPENDENCY_ERROR}")
class InterpolatorCacheESMPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = esmpy.Manager()
        if esmpy.pet_count() != 1:
            raise unittest.SkipTest("These integration fixtures require one ESMF process")
        cls.interpolator_class = load_interpolator()

    def evaluate(self, method, cache_dir, offset=0.0, warm=False, allow_cache_fallback=False):
        with grid_pair(method, offset) as (source, destination):
            operator = self.interpolator_class(source, destination)
            try:
                if warm:
                    with patch.object(
                        esmpy,
                        "Regrid",
                        side_effect=AssertionError("Warm cache recalculated weights"),
                    ):
                        operator.initialize(cache_dir=cache_dir)
                else:
                    with patch.object(esmpy, "Regrid", wraps=esmpy.Regrid) as build:
                        operator.initialize(cache_dir=cache_dir)
                    if not allow_cache_fallback:
                        self.assertEqual(build.call_count, 1)
                    if build.call_count:
                        self.assertEqual(
                            build.call_args.kwargs["regrid_method"],
                            getattr(esmpy.RegridMethod, method),
                        )
                return sample_results(operator, source)
            finally:
                operator.destroy()

    def assert_same_results(self, expected, actual):
        np.testing.assert_array_equal(expected["region"], actual["region"])
        for field in ("constant", "smooth", "missing"):
            np.testing.assert_array_equal(np.isnan(expected[field]), np.isnan(actual[field]))
            np.testing.assert_allclose(
                expected[field], actual[field], rtol=1e-11, atol=1e-12, equal_nan=True
            )

    def test_cold_warm_and_uncached_preserve_values_and_nan_masks(self):
        for method in ("BILINEAR", "CONSERVE"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as cache_dir:
                baseline = self.evaluate(method, None)
                cold = self.evaluate(method, cache_dir)
                warm = self.evaluate(method, cache_dir, warm=True)
                self.assert_same_results(baseline, cold)
                self.assert_same_results(baseline, warm)
                self.assertTrue(baseline["region"].all())
                np.testing.assert_allclose(baseline["constant"], 1.0, atol=1e-11)
                self.assertTrue(np.isnan(baseline["missing"]).any())
                self.assertTrue(np.isfinite(baseline["missing"]).any())
                files = list(Path(cache_dir).glob("*.nc"))
                self.assertEqual(len(files), 1)
                self.assertRegex(files[0].stem, r"^[0-9a-f]{64}$")
                with netCDF4.Dataset(files[0]) as weights:
                    self.assertTrue({"row", "col", "S"}.issubset(weights.variables))

    def test_unmapped_destinations_remain_nan_for_both_methods(self):
        for method in ("BILINEAR", "CONSERVE"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as cache_dir:
                baseline = self.evaluate(method, None, offset=5.0)
                cold = self.evaluate(method, cache_dir, offset=5.0)
                warm = self.evaluate(method, cache_dir, offset=5.0, warm=True)
                self.assert_same_results(baseline, cold)
                self.assert_same_results(baseline, warm)
                region = baseline["region"]
                self.assertTrue(region.any())
                self.assertTrue((~region).any())
                for field in ("constant", "smooth", "missing"):
                    self.assertTrue(np.isnan(warm[field][~region]).all())
                np.testing.assert_array_equal(np.isfinite(warm["constant"]), region)

    def test_weights_are_reused_by_a_fresh_python_process(self):
        for method in ("BILINEAR", "CONSERVE"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                cache_dir = Path(directory) / "weights"
                expected = self.evaluate(method, cache_dir, offset=5.0)
                output_path = Path(directory) / "fresh-process.npz"
                script = (
                    "import runpy, sys; "
                    "ns = runpy.run_path(sys.argv[1]); "
                    "ns['fresh_process_result'](*sys.argv[2:])"
                )
                process = subprocess.run(
                    [sys.executable, "-B", "-c", script, str(Path(__file__).resolve()),
                     str(cache_dir), str(output_path), method],
                    cwd=directory,
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                with np.load(output_path, allow_pickle=False) as result:
                    self.assert_same_results(expected, result)
                self.assertEqual(len(list(cache_dir.glob("*.nc"))), 1)

    def test_disjoint_grids_remain_all_nan_with_or_without_cache(self):
        for method in ("BILINEAR", "CONSERVE"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as cache_dir:
                baseline = self.evaluate(method, None, offset=30.0)
                # Some ESMF versions cannot serialize an empty weight matrix.
                # Both successful cache reuse and an in-memory fallback are valid.
                cold = self.evaluate(
                    method, cache_dir, offset=30.0, allow_cache_fallback=True
                )
                repeated = self.evaluate(
                    method, cache_dir, offset=30.0, allow_cache_fallback=True
                )
                self.assert_same_results(baseline, cold)
                self.assert_same_results(baseline, repeated)
                for result in (baseline, cold, repeated):
                    self.assertFalse(result["region"].any())
                    for field in ("constant", "smooth", "missing"):
                        self.assertTrue(np.isnan(result[field]).all())


if __name__ == "__main__":
    unittest.main()
