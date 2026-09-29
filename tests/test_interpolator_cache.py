"""Cache regressions without the optional native ESMF installation.

The fake backend models weight-file I/O, route-handle ownership and the ESMF
SELECT/TOTAL destination semantics. Numerical interpolation is covered by the
separate native integration tests; these tests exercise cache orchestration.
"""

from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class RegridMethod(IntEnum):
    BILINEAR = 0
    CONSERVE = 2


class Region(IntEnum):
    TOTAL = 0
    SELECT = 1


class StaggerLoc(IntEnum):
    CENTER = 0
    CORNER = 3


class CoordSys(IntEnum):
    SPH_DEG = 0


class FakeGrid:
    def __init__(self, shape=(2, 2), area=1.0):
        self.shape = shape
        self._cell_area = area
        self.lat = np.arange(np.prod(shape), dtype=float).reshape(shape) + 50
        self.lon = np.arange(np.prod(shape), dtype=float).reshape(shape) + 10
        corner_shape = tuple(size + 1 for size in shape)
        self.lat_corners = np.arange(np.prod(corner_shape), dtype=float).reshape(corner_shape) + 49.5
        self.lon_corners = np.arange(np.prod(corner_shape), dtype=float).reshape(corner_shape) + 9.5
        self.grid = self
        self.coord_sys = CoordSys.SPH_DEG
        self.num_peri_dims = 0
        self.periodic_dim = None
        self.pole_dim = None
        self.rank = len(shape)
        self.max_index = np.array(shape)
        self.mask = {}

    def cell_areas(self):
        return np.full(self.shape, self._cell_area)

    def get_coords(self, coord_dim, staggerloc=StaggerLoc.CENTER):
        if staggerloc == StaggerLoc.CENTER:
            return (self.lon, self.lat)[coord_dim]
        return (self.lon_corners, self.lat_corners)[coord_dim]


class FakeNetCDFDataset:
    """Minimal standard netCDF attribute API over a real on-disk file."""

    def __init__(self, filename, mode="r", **kwargs):
        object.__setattr__(self, "filename", Path(filename))
        object.__setattr__(self, "mode", mode)
        if mode.startswith("w"):
            document = {"attrs": {}}
        else:
            document = json.loads(self.filename.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or "weights" not in document:
                raise OSError("Not a weight file")
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "closed", False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self.closed and self.mode != "r":
            self.filename.write_text(json.dumps(self.document), encoding="utf-8")
        object.__setattr__(self, "closed", True)

    def ncattrs(self):
        return list(self.document.get("attrs", {}))

    def getncattr(self, name):
        try:
            return self.document.get("attrs", {})[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def setncattr(self, name, value):
        self.document.setdefault("attrs", {})[name] = value

    def setncatts(self, attrs):
        self.document.setdefault("attrs", {}).update(attrs)

    def __getattr__(self, name):
        return self.getncattr(name)

    def __setattr__(self, name, value):
        self.setncattr(name, value)


class FakeBackend:
    def __init__(self):
        self.fields = []
        self.operators = []
        self.generated = []
        self.loaded = []
        self.applied = []
        self.write_error = None
        self.load_error = None
        self.compute_error = None
        self.apply_error = None
        self.write_barrier = None
        self.constructor_fill = 42.0
        self.pet_count = 1
        self.module = ModuleType("esmpy")
        constants = SimpleNamespace(
            RegridMethod=RegridMethod,
            UnmappedAction=SimpleNamespace(IGNORE=1),
            Region=Region,
            StaggerLoc=StaggerLoc,
            CoordSys=CoordSys,
            NormType=SimpleNamespace(DSTAREA=0),
            LineType=SimpleNamespace(GREAT_CIRCLE=1),
            _ESMF_VERSION="fake-esmf-1",
        )
        self.module.api = SimpleNamespace(constants=constants)
        self.module.__dict__.update(vars(constants))
        self.module.__version__ = "fake-esmpy-1"
        self.module.interface = SimpleNamespace(cbindings=SimpleNamespace(_ESMF_VERSION_STRING="fake-esmf-1"))
        self.module.pet_count = lambda: self.pet_count
        self.module.local_pet = lambda: 0
        self.module.Field = self.field
        self.module.Regrid = self.regrid
        self.module.RegridFromFile = self.regrid_from_file

    def field(self, grid, **kwargs):
        field = SimpleNamespace(grid=grid, data=np.zeros(grid.shape), destroyed=False)

        def destroy():
            field.destroyed = True

        field.destroy = destroy
        self.fields.append(field)
        return field

    def operator(self, src_field, dst_field):
        backend = self

        class Operator:
            destroyed = False

            def __call__(self, source, destination, zero_region=Region.TOTAL, **kwargs):
                backend.applied.append(zero_region)
                if backend.apply_error is not None:
                    raise backend.apply_error
                if zero_region == Region.TOTAL:
                    destination.data[:] = 0.0
                # Leave the last destination cell unmapped. This deliberately
                # differs from the constructor's modification of dst_field.
                count = destination.data.size - 1
                destination.data.flat[:count] = np.resize(source.data.ravel(), count)
                return destination

            def destroy(self):
                self.destroyed = True

        route = Operator()
        self.operators.append(route)
        dst_field.data[:] = self.constructor_fill
        return route

    def regrid(self, src_field, dst_field, **kwargs):
        self.generated.append(kwargs.copy())
        if self.compute_error is not None:
            raise self.compute_error
        filename = kwargs.get("filename")
        if filename is not None:
            if self.write_error is not None:
                raise self.write_error
            if self.write_barrier is not None:
                self.write_barrier.wait(timeout=10)
            document = {
                "weights": [1.0] * (dst_field.data.size - 1),
                "src_shape": list(src_field.data.shape),
                "dst_shape": list(dst_field.data.shape),
                "attrs": {},
            }
            Path(filename).write_text(json.dumps(document), encoding="utf-8")
        return self.operator(src_field, dst_field)

    def regrid_from_file(self, src_field, dst_field, filename, **kwargs):
        self.loaded.append(Path(filename))
        if self.load_error is not None:
            raise self.load_error
        document = json.loads(Path(filename).read_text(encoding="utf-8"))
        if document["src_shape"] != list(src_field.data.shape):
            raise ValueError("Incompatible source shape")
        if document["dst_shape"] != list(dst_field.data.shape):
            raise ValueError("Incompatible destination shape")
        return self.operator(src_field, dst_field)


class InterpolatorCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache_dir = Path(self.temp.name) / "weights"
        self.backend = FakeBackend()
        self.package_name = "_interpolator_cache_test_" + uuid.uuid4().hex
        package = ModuleType(self.package_name)
        package.__path__ = [str(ROOT / "libs" / "validation")]
        netcdf = ModuleType("netCDF4")
        netcdf.Dataset = FakeNetCDFDataset
        self.modules_patch = patch.dict(sys.modules, {
            self.package_name: package,
            "esmpy": self.backend.module,
            "netCDF4": netcdf,
        })
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        spec = importlib.util.spec_from_file_location(
            self.package_name + ".interpolator",
            ROOT / "libs" / "validation" / "interpolator.py",
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)
        self.src = FakeGrid(area=2.0)
        self.dst = FakeGrid(area=1.0)
        self.interpolators = []
        self.addCleanup(self.destroy_interpolators)
        self.cache_logger = self.package_name + ".interpolation_cache"

    def destroy_interpolators(self):
        for interpolator in self.interpolators:
            interpolator.destroy()

    def create(self, *, src=None, dst=None, enabled=True):
        interpolator = self.module.Interpolator(src or self.src, dst or self.dst)
        self.interpolators.append(interpolator)
        interpolator.initialize(cache_dir=self.cache_dir if enabled else None)
        return interpolator

    def files(self):
        return list(self.cache_dir.glob("*.nc"))

    def test_disabled_cache_creates_no_files_and_still_interpolates(self):
        interpolator = self.create(enabled=False)
        result = interpolator(np.arange(4.0).reshape(2, 2))
        np.testing.assert_equal(result, [[0, 1], [2, np.nan]])
        self.assertFalse(self.cache_dir.exists())
        self.assertEqual(len(self.backend.generated), 1)
        self.assertNotIn("filename", self.backend.generated[0])
        self.assertFalse(self.backend.loaded)

    def test_miss_publishes_one_complete_file_and_new_instance_loads(self):
        first = self.create()
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(len(self.backend.generated), 1)
        written_to = Path(self.backend.generated[0]["filename"])
        self.assertNotEqual(written_to, self.files()[0])
        self.assertEqual(written_to.parent, self.cache_dir)
        self.assertFalse(written_to.exists())
        first.destroy()
        second = self.create(src=FakeGrid(area=2), dst=FakeGrid(area=1))
        self.assertEqual(len(self.backend.generated), 1)
        self.assertEqual(len(self.backend.loaded), 1)
        self.assertEqual(self.backend.loaded[0], self.files()[0])
        np.testing.assert_equal(second([[4, 3], [2, 1]]), [[4, 3], [2, np.nan]])

    def test_geometry_method_and_version_changes_do_not_reuse_weights(self):
        self.create()
        changes = [
            lambda: self.src.lon.__setitem__((0, 0), self.src.lon[0, 0] + 0.5),
            lambda: self.dst.lat.__setitem__((0, 0), self.dst.lat[0, 0] + 0.5),
            lambda: self.src.lon_corners.__setitem__((0, 0), self.src.lon_corners[0, 0] + 0.5),
            lambda: self.dst.lat_corners.__setitem__((0, 0), self.dst.lat_corners[0, 0] + 0.5),
            lambda: setattr(self.src, "lon", self.src.lon[::-1].copy()),
            lambda: setattr(self.src, "_cell_area", 0.1),
            lambda: setattr(self.backend.module, "__version__", "fake-esmpy-2"),
        ]
        for index, change in enumerate(changes, start=2):
            with self.subTest(change=index):
                change()
                self.create()
                self.assertEqual(len(self.backend.generated), index)
                self.assertFalse(self.backend.loaded)
        self.create(src=FakeGrid(shape=(1, 4)))
        self.assertEqual(len(self.backend.generated), len(changes) + 2)

    def test_input_field_values_and_nan_values_do_not_change_weights(self):
        first = self.create()
        np.testing.assert_equal(first([[1, np.nan], [3, 4]]), [[1, np.nan], [3, np.nan]])
        second = self.create()
        np.testing.assert_equal(second([[9, 8], [7, 6]]), [[9, 8], [7, np.nan]])
        self.assertEqual(len(self.backend.generated), 1)

    def test_memory_layout_does_not_change_grid_identity(self):
        self.create()
        for grid in (self.src, self.dst):
            for name in ("lat", "lon", "lat_corners", "lon_corners"):
                setattr(grid, name, np.asfortranarray(getattr(grid, name)))
        self.create()
        self.assertEqual(len(self.backend.generated), 1)
        self.assertEqual(len(self.backend.loaded), 1)

    def test_cache_format_and_esmf_version_invalidate_weights(self):
        self.create()
        cache_module = sys.modules[self.package_name + ".interpolation_cache"]
        with patch.object(cache_module, "_CACHE_VERSION", cache_module._CACHE_VERSION + 1):
            self.create()
        self.backend.module.api.constants._ESMF_VERSION = "fake-esmf-2"
        self.create()
        self.assertEqual(len(self.backend.generated), 3)
        self.assertFalse(self.backend.loaded)

    def test_coverage_is_explicitly_probed_after_both_constructors(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                interpolator = self.create()
                np.testing.assert_array_equal(interpolator.dst_region, [[True, True], [True, False]])
                self.assertEqual(self.backend.applied[-1], Region.SELECT)

    def test_truncated_weight_file_is_replaced_and_then_reused(self):
        self.create()
        self.files()[0].write_text("{broken", encoding="utf-8")
        with self.assertLogs(self.cache_logger, level="WARNING"):
            second = self.create()
        self.assertEqual(len(self.backend.generated), 2)
        self.assertEqual(len(self.files()), 1)
        self.assertIn("weights", json.loads(self.files()[0].read_text(encoding="utf-8")))
        self.assertTrue(np.isnan(second([[1, 2], [3, 4]])[-1, -1]))
        self.create()
        self.assertEqual(len(self.backend.generated), 2)

    def test_load_failure_recomputes_and_preserves_usable_result(self):
        self.create()
        self.backend.load_error = OSError("Cannot read weight file")
        with self.assertLogs(self.cache_logger, level="WARNING"):
            interpolator = self.create()
        self.assertEqual(len(self.backend.generated), 2)
        np.testing.assert_equal(interpolator([[1, 2], [3, 4]]), [[1, 2], [3, np.nan]])

    def test_mismatched_embedded_key_is_not_loaded(self):
        self.create()
        path = self.files()[0]
        with FakeNetCDFDataset(path, "a") as weights:
            weights.setncattr("smp_interpolation_cache_key", "different-grid")
        with self.assertLogs(self.cache_logger, level="WARNING"):
            self.create()
        self.assertFalse(self.backend.loaded)
        self.assertEqual(len(self.backend.generated), 2)

    def test_failed_publication_keeps_previous_file_and_usable_operator(self):
        self.create()
        path = self.files()[0]
        previous = "incomplete previous file"
        path.write_text(previous, encoding="utf-8")
        with patch.object(os, "replace", side_effect=PermissionError("File is read-only")):
            with self.assertLogs(self.cache_logger, level="WARNING"):
                interpolator = self.create()
        self.assertEqual(path.read_text(encoding="utf-8"), previous)
        self.assertEqual(list(self.cache_dir.iterdir()), [path])
        np.testing.assert_equal(interpolator([[1, 2], [3, 4]]), [[1, 2], [3, np.nan]])

    def test_attached_esmf_mask_bypasses_cache(self):
        self.src.mask = {StaggerLoc.CENTER: np.zeros(self.src.shape, dtype=np.int32)}
        with self.assertLogs(self.cache_logger, level="WARNING"):
            self.create()
        self.assertFalse(self.cache_dir.exists())
        self.assertNotIn("filename", self.backend.generated[-1])

    def test_weight_write_failure_falls_back_without_cache(self):
        self.backend.write_error = OSError("No space left on device")
        with self.assertLogs(self.cache_logger, level="WARNING"):
            interpolator = self.create()
        self.assertNotIn("filename", self.backend.generated[-1])
        np.testing.assert_equal(interpolator([[1, 2], [3, 4]]), [[1, 2], [3, np.nan]])
        self.assertFalse(self.files())

    def test_unwritable_cache_directory_falls_back_without_cache(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("read-only")):
            with self.assertLogs(self.cache_logger, level="WARNING"):
                interpolator = self.create()
        self.assertNotIn("filename", self.backend.generated[-1])
        self.assertFalse(self.cache_dir.exists())
        self.assertEqual(interpolator([[1, 2], [3, 4]])[0, 0], 1)

    def test_computation_errors_propagate_and_fields_are_released(self):
        error = ValueError("Invalid cell geometry")
        self.backend.compute_error = error
        with self.assertLogs(self.cache_logger, level="WARNING"):
            with self.assertRaises(ValueError) as caught:
                self.create()
        self.assertIs(caught.exception, error)
        self.assertTrue(all(field.destroyed for field in self.backend.fields))
        self.assertFalse(self.files())

    def test_failed_coverage_probe_releases_operator_and_fields(self):
        self.backend.apply_error = RuntimeError("Probe failed")
        with self.assertRaisesRegex(RuntimeError, "Probe failed"):
            self.create(enabled=False)
        self.assertTrue(all(field.destroyed for field in self.backend.fields))
        self.assertTrue(all(route.destroyed for route in self.backend.operators))

    def test_failed_application_releases_temporary_fields(self):
        interpolator = self.create()
        before = len(self.backend.fields)
        self.backend.apply_error = RuntimeError("Apply failed")
        with self.assertRaisesRegex(RuntimeError, "Apply failed"):
            interpolator([[1, 2], [3, 4]])
        self.assertTrue(all(field.destroyed for field in self.backend.fields[before:]))

    def test_destroy_releases_ownership_and_is_idempotent(self):
        interpolator = self.create()
        interpolator.destroy()
        interpolator.destroy()
        self.assertTrue(all(field.destroyed for field in self.backend.fields))
        self.assertTrue(all(route.destroyed for route in self.backend.operators))
        self.assertIsNone(interpolator.regrid)

    def test_reinitialization_releases_previous_operator_and_fields(self):
        interpolator = self.create()
        previous_operator = interpolator.regrid
        previous_fields = list(self.backend.fields)
        interpolator.initialize(cache_dir=self.cache_dir)
        self.assertTrue(previous_operator.destroyed)
        self.assertTrue(all(field.destroyed for field in previous_fields))
        self.assertIsNot(interpolator.regrid, previous_operator)
        self.assertEqual(len(self.backend.generated), 1)
        self.assertEqual(len(self.backend.loaded), 1)
        np.testing.assert_equal(interpolator([[4, 3], [2, 1]]), [[4, 3], [2, np.nan]])

    def test_parallel_first_writers_publish_only_a_complete_final_file(self):
        self.backend.write_barrier = Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.create) for _ in range(2)]
            interpolators = [future.result(timeout=15) for future in futures]
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(len(list(self.cache_dir.iterdir())), 1)
        filenames = [call["filename"] for call in self.backend.generated]
        self.assertEqual(len(set(map(str, filenames))), 2)
        self.backend.write_barrier = None
        self.create()
        self.assertEqual(len(self.backend.generated), 2)
        for interpolator in interpolators:
            self.assertEqual(interpolator([[1, 2], [3, 4]])[0, 0], 1)

    def test_mpi_bypasses_disk_cache(self):
        self.backend.pet_count = 2
        with self.assertLogs(self.cache_logger, level="WARNING"):
            self.create()
        self.assertFalse(self.cache_dir.exists())
        self.assertNotIn("filename", self.backend.generated[-1])


if __name__ == "__main__":
    unittest.main()
