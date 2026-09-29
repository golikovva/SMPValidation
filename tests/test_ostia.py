"""Exercise OSTIA readers with real NetCDF and a substituted ESMF boundary."""

import ast
import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DAY = datetime.date(2023, 1, 2)


class RecordingGrid:
    def __init__(self, lat, lon):
        self.lat = np.asarray(lat)
        self.lon = np.asarray(lon)
        self.shape = self.lat.shape


class RecordingInterpolator:
    instances = []

    def __init__(self, src_grid, dst_grid):
        self.src_grid = src_grid
        self.dst_grid = dst_grid
        self.calls = []
        self.cache_dir = None
        self.instances.append(self)

    def initialize(self, *, cache_dir=None):
        self.cache_dir = cache_dir

    def __call__(self, field):
        self.calls.append(np.array(field, copy=True))
        return np.asarray(field[:1, :2], dtype=np.float64)


def load_dataset_classes():
    """Retain production imports and methods except optional native imports."""
    namespace = {
        "__name__": "isolated_ostia_tests",
        "Grid": RecordingGrid,
        "Interpolator": RecordingInterpolator,
    }
    paths = (
        ROOT / "libs/validation/grid_utils.py",
        ROOT / "libs/validation/datasets/base.py",
        ROOT / "libs/validation/datasets/ostia.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        tree.body = [
            node for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and (
                    (node.module or "").startswith("libs.validation")
                    or (node.level and node.module in {"base", "grid", "interpolator"})
                )
            )
        ]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class OstiaDatasetTests(unittest.TestCase):
    def setUp(self):
        classes = load_dataset_classes()
        self.base = classes["Dataset"]
        self.ostia = classes["OstiaDataset"]
        self.sic = classes["OstiaSicDataset"]
        self.sst = classes["OstiaSstDataset"]
        RecordingInterpolator.instances = []
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_file(
        self, *, sic=None, sst=None, mask=None, include_mask=True,
        day=DAY, folder="2023", times=None, latitude=None, longitude=None,
        order=None, encoding=None, sst_units="K",
    ):
        fields = {name: np.asarray(value) for name, value in (
            ("sea_ice_fraction", sic), ("analysed_sst", sst),
        ) if value is not None}
        shape = next(iter(fields.values())).shape[-2:]
        height, width = shape
        if times is None:
            times = [np.datetime64(f"{day.isoformat()}T23:59:00")]
        latitude = np.arange(height, dtype=float) + 70 if latitude is None else np.asarray(latitude)
        longitude = np.arange(width, dtype=float) + 20 if longitude is None else np.asarray(longitude)
        if latitude.ndim == 2:
            spatial_dims = ("y", "x")
            coords = {
                "time": times,
                "latitude": (spatial_dims, latitude),
                "longitude": (spatial_dims, longitude),
            }
        else:
            spatial_dims = ("latitude", "longitude")
            coords = {"time": times, "latitude": latitude, "longitude": longitude}
        dims = ("time", *spatial_dims)
        if include_mask:
            fields["mask"] = np.ones(shape, dtype=np.int16) if mask is None else np.asarray(mask)
        variables = {}
        for name, values in fields.items():
            if values.ndim == 2:
                values = np.broadcast_to(values[None], (len(times), *shape)).copy()
            variables[name] = (dims, values)
        dataset = xr.Dataset(variables, coords=coords)
        if sic is not None:
            dataset["sea_ice_fraction"].attrs["units"] = "1"
        if sst is not None and sst_units is not None:
            dataset["analysed_sst"].attrs["units"] = sst_units
        if order is not None:
            for name in dataset.data_vars:
                dataset[name] = dataset[name].transpose(*order)
        path = self.root / folder / f"{day:%Y%m%d}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(path, engine="scipy", encoding=encoding or {})
        return path

    def test_common_base_is_abstract_and_daily_access_stays_inherited(self):
        with self.assertRaises(TypeError):
            self.ostia(self.root)
        self.assertIs(self.sic.__getitem__, self.base.__getitem__)
        self.assertIs(self.sst.__getitem__, self.base.__getitem__)
        self.write_file(sic=[[0, 0.5], [1, np.nan]])
        dataset = self.sic(self.root)
        self.assertIsNone(dataset.average_times)
        self.assertEqual(dataset[DAY].shape, (1, 1, 2, 2))
        self.assertIsNone(dataset[DAY + datetime.timedelta(days=1)])

    def test_sic_only_file_converts_fraction_and_checks_bounds_before_float32(self):
        self.write_file(sic=[[0, 0.125, 1], [-1e-50, 1 + 1e-8, np.nan]])
        result = self.sic(self.root)[DAY]
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(
            result[0, 0], [[0, 12.5, 100], [np.nan, np.nan, np.nan]], equal_nan=True,
        )

    def test_sst_only_file_supports_celsius_and_kelvin_without_precision_loss(self):
        values = np.array([[273.15, 273.151, 280.125], [np.nan, np.inf, -np.inf]], dtype=np.float64)
        self.write_file(sst=values, sst_units="kelvin")
        celsius = self.sst(self.root)[DAY]
        kelvin = self.sst(self.root, units="kelvin")[DAY]
        expected_celsius = (values - 273.15).astype(np.float32)
        expected_kelvin = values.astype(np.float32)
        expected_celsius[~np.isfinite(values)] = np.nan
        expected_kelvin[~np.isfinite(values)] = np.nan
        np.testing.assert_array_equal(celsius[0, 0], expected_celsius)
        np.testing.assert_array_equal(kelvin[0, 0], expected_kelvin)
        self.assertEqual(celsius.dtype, np.float32)
        self.assertEqual(kelvin.dtype, np.float32)

    def test_cf_scale_offset_and_fill_are_applied_once_for_both_fields(self):
        self.write_file(
            sic=[[0, 0.12], [1, np.nan]],
            sst=[[273.15, 273.16], [280.25, np.nan]],
            encoding={
                "sea_ice_fraction": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32767},
                "analysed_sst": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 273.15, "_FillValue": -32767},
            },
        )
        np.testing.assert_allclose(self.sic(self.root)[DAY][0, 0], [[0, 12], [100, np.nan]], equal_nan=True)
        np.testing.assert_allclose(
            self.sst(self.root)[DAY][0, 0], [[0, 0.01], [7.1, np.nan]], atol=1e-6, equal_nan=True,
        )

    def test_mask_keeps_ocean_and_ice_and_excludes_invalid_or_inland_values(self):
        mask = np.array([[1, 8, 9, 2, 4, 16, 3], [10, 0, 32, 1.5, np.nan, -1, np.inf]])
        self.write_file(sic=np.full(mask.shape, 0.5), sst=np.full(mask.shape, 274.15), mask=mask)
        valid = np.zeros(mask.shape, dtype=bool)
        valid[0, :3] = True
        for dataset_class in (self.sic, self.sst):
            with self.subTest(dataset=dataset_class.__name__):
                result = dataset_class(self.root)[DAY][0, 0]
                np.testing.assert_array_equal(np.isfinite(result), valid)

    def test_mask_is_required_and_an_unselected_variable_does_not_substitute(self):
        self.write_file(sic=[[0, 0.5], [1, 0.1]], include_mask=False, folder="no_mask")
        with self.assertRaises((KeyError, ValueError)):
            dataset = self.sic(self.root, files_template="no_mask/*.nc")
            dataset[DAY]
        self.write_file(sic=[[0, 0.5], [1, 0.1]], folder="sic_only")
        with self.assertRaises((KeyError, ValueError)):
            dataset = self.sst(self.root, files_template="sic_only/*.nc")
            dataset[DAY]

    def test_invalid_output_units_are_rejected_before_file_discovery(self):
        with self.assertRaises(ValueError):
            self.sst(self.root / "does-not-exist", units="fahrenheit")

    def test_sst_rejects_non_kelvin_source_units(self):
        self.write_file(sst=[[1, 2], [3, 4]], sst_units="degree_Celsius")
        with self.assertRaises(ValueError):
            dataset = self.sst(self.root)
            dataset[DAY]

    def test_filename_and_time_must_agree_on_date_not_hour(self):
        self.write_file(sic=[[0, 0.1], [0.5, 1]], folder="same_day")
        result = self.sic(self.root, files_template="same_day/*.nc")[DAY]
        self.assertEqual(result[0, 0, 0, 1], 10)
        self.write_file(
            sic=[[0, 0.1], [0.5, 1]], folder="wrong_day",
            times=[np.datetime64("2023-01-03T00:00:00")],
        )
        with self.assertRaises(ValueError):
            dataset = self.sic(self.root, files_template="wrong_day/*.nc")
            dataset[DAY]

    def test_duplicate_files_and_multiple_time_layers_are_rejected(self):
        for folder in ("first", "second"):
            self.write_file(sic=[[0, 0.1], [0.5, 1]], folder=folder)
        with self.assertRaises(ValueError):
            self.sic(self.root)
        self.write_file(
            sic=[[0, 0.1], [0.5, 1]], folder="multi_time",
            times=[np.datetime64("2023-01-02T00:00"), np.datetime64("2023-01-02T12:00")],
        )
        with self.assertRaises(ValueError):
            dataset = self.sic(self.root, files_template="multi_time/*.nc")
            dataset[DAY]

    def test_rectilinear_and_curvilinear_grids_preserve_field_orientation(self):
        values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        latitude = np.array([[71, 71, 71], [70, 70, 70]])
        longitude = np.array([[-20, 0, 20], [-20, 0, 20]])
        for folder, lat, lon in (
            ("rectilinear", latitude[:, 0], longitude[0]),
            ("curvilinear", latitude, longitude),
        ):
            with self.subTest(grid=folder):
                self.write_file(sic=values, folder=folder, latitude=lat, longitude=lon)
                dataset = self.sic(self.root, files_template=f"{folder}/*.nc")
                np.testing.assert_array_equal(dataset.src_grid.lat, latitude)
                np.testing.assert_array_equal(dataset.src_grid.lon, longitude)
                np.testing.assert_allclose(dataset[DAY][0, 0], np.asarray(values) * 100)

    def test_coordinate_changes_and_unexpected_field_axis_order_are_rejected(self):
        self.write_file(sic=[[0, 0.1], [0.5, 1]])
        tomorrow = DAY + datetime.timedelta(days=1)
        self.write_file(sic=[[0, 0.1], [0.5, 1]], day=tomorrow, latitude=[70, 72])
        with self.assertRaises(ValueError):
            dataset = self.sic(self.root)
            dataset[tomorrow]
        self.write_file(
            sic=[[0, 0.1], [0.5, 1]], folder="reordered",
            order=("longitude", "time", "latitude"),
        )
        with self.assertRaises(ValueError):
            dataset = self.sic(self.root, files_template="reordered/*.nc")
            dataset[DAY]

    def test_recursive_default_and_template_override_select_dates(self):
        selected = self.write_file(sic=[[0, 0.1], [0.5, 1]], folder="nested/selected")
        tomorrow = DAY + datetime.timedelta(days=1)
        self.write_file(sic=[[0, 0.1], [0.5, 1]], day=tomorrow, folder="other")
        self.assertEqual(set(self.sic(self.root).dates_dict), {DAY, tomorrow})
        dataset = self.sic(self.root, files_template="nested/selected/*.nc")
        self.assertEqual(dataset.dates_dict, {DAY: [selected]})
        self.assertIsNone(dataset[tomorrow])

    def test_inherited_interpolation_preserves_daily_axes_dtype_and_cache(self):
        self.write_file(sic=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], sst=np.full((2, 3), 274.15))
        destination = object()
        cache_dir = self.root / "weights"
        for dataset_class in (self.sic, self.sst):
            with self.subTest(dataset=dataset_class.__name__):
                dataset = dataset_class(self.root, destination, "OSTIA", interpolation_cache_dir=cache_dir)
                result = dataset[DAY]
                self.assertEqual(result.shape, (1, 1, 1, 2))
                self.assertEqual(result.dtype, np.float32)
                operator = RecordingInterpolator.instances[-1]
                self.assertIs(operator.src_grid, dataset.src_grid)
                self.assertIs(operator.dst_grid, destination)
                self.assertEqual(operator.cache_dir, cache_dir)
                self.assertEqual(len(operator.calls), 1)
                np.testing.assert_array_equal(result[0, 0], operator.calls[0][:1, :2])


if __name__ == "__main__":
    unittest.main()
