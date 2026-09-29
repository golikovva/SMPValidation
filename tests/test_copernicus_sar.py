"""Read real miniature NetCDF files; isolate only the native ESMF boundary."""

import ast
import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DAY = datetime.date(2023, 1, 2)
DAY_START = np.datetime64("2023-01-02T00:00:00", "s")
EPOCH_START = int(DAY_START.astype(np.int64))


class RecordingGrid:
    instances = []

    def __init__(self, lat, lon):
        self.lat = np.asarray(lat)
        self.lon = np.asarray(lon)
        self.shape = self.lat.shape
        self.instances.append(self)


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

    def __call__(self, values):
        self.calls.append(np.array(values, copy=True))
        # Native ESMF produces float64. A smaller destination also verifies axes.
        return np.asarray(values[:1, :2], dtype=np.float64)


def load_dataset_classes():
    """Keep production methods and imports without importing optional packages."""
    namespace = {
        "__name__": "isolated_copernicus_sar_tests",
        "Grid": RecordingGrid,
        "Interpolator": RecordingInterpolator,
    }
    paths = (
        ROOT / "libs/validation/grid_utils.py",
        ROOT / "libs/validation/datasets/base.py",
        ROOT / "libs/validation/datasets/copernicus_sar.py",
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
    return namespace["Dataset"], namespace["CopernicusSarSicDataset"]


class CopernicusSarSicDatasetTests(unittest.TestCase):
    def setUp(self):
        self.base_class, self.dataset_class = load_dataset_classes()
        RecordingGrid.instances = []
        RecordingInterpolator.instances = []
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_file(
        self,
        values,
        *,
        day=DAY,
        folder="2023",
        acquisition=None,
        acquisition_units="s",
        status=None,
        latitude=None,
        longitude=None,
        order=("time", "latitude", "longitude"),
        encoding=None,
        times=None,
    ):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 2:
            values = values[None]
        _, height, width = values.shape
        latitude = np.arange(height, dtype=float) + 70 if latitude is None else latitude
        longitude = np.arange(width, dtype=float) + 20 if longitude is None else longitude
        if times is None:
            times = [np.datetime64(f"{day.isoformat()}T19:29:04")]
        coords = {"time": times, "latitude": latitude, "longitude": longitude}
        dims = ("time", "latitude", "longitude")
        dataset = xr.Dataset({"sic": (dims, values)}, coords=coords)
        dataset["sic"].attrs["units"] = "%"
        for name, data in (("acq_time", acquisition), ("status_flag", status)):
            if data is not None:
                data = np.asarray(data)
                if data.ndim == 2:
                    data = data[None]
                dataset[name] = (dims, data)
        if acquisition is not None and acquisition_units is not None:
            dataset["acq_time"].attrs["units"] = acquisition_units
        for name in dataset.data_vars:
            dataset[name] = dataset[name].transpose(*order)
        path = self.root / folder / f"sic_{day:%Y%m%d}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(path, engine="scipy", encoding=encoding or {})
        return path

    def test_daily_preserves_percentages_zero_and_missing_values(self):
        self.write_file(
            [[0, 20, 100, -1], [101, np.nan, 30, 40]],
            status=[[0, 1, 2, 0], [0, 0, 128, 129]],
        )
        dataset = self.dataset_class(self.root, name="SAR")
        result = dataset[DAY]
        expected = np.array([[0, np.nan, 100, np.nan], [np.nan, np.nan, 30, np.nan]])
        self.assertEqual(result.shape, (1, 1, 2, 4))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[0, 0], expected, equal_nan=True)
        self.assertIsNone(dataset.average_times)
        self.assertIsNone(dataset[DAY + datetime.timedelta(days=1)])
        self.assertIs(self.dataset_class.__getitem__, self.base_class.__getitem__)

    def test_daily_works_without_acquisition_status_or_uncertainty(self):
        self.write_file([[0, 25], [50, 100]])
        result = self.dataset_class(self.root, mode="daily")[DAY]
        np.testing.assert_array_equal(result[0, 0], [[0, 25], [50, 100]])

    def test_cf_scale_and_fill_value_are_decoded_before_sic_validation(self):
        self.write_file(
            [[0, 12.5], [100, np.nan]],
            encoding={"sic": {"dtype": "int16", "scale_factor": 0.5, "_FillValue": -32767}},
        )
        result = self.dataset_class(self.root)[DAY]
        np.testing.assert_allclose(result[0, 0], [[0, 12.5], [100, np.nan]], equal_nan=True)

    def test_hourly_emits_all_twenty_four_layers_without_averaging(self):
        values = np.arange(1, 25, dtype=np.float32).reshape(2, 12)
        acquisition = EPOCH_START + np.arange(24).reshape(2, 12) * 3600 + 1800
        self.write_file(values, acquisition=acquisition.astype(np.float64))
        result = self.dataset_class(self.root, mode="hourly")[DAY]
        self.assertEqual(result.shape, (24, 1, 2, 12))
        self.assertEqual(result.dtype, np.float32)
        for hour in range(24):
            with self.subTest(hour=hour):
                self.assertEqual(np.isfinite(result[hour]).sum(), 1)
                self.assertEqual(result[hour, 0].ravel()[hour], hour + 1)

    def test_hour_boundaries_are_utc_and_exclude_other_dates_and_missing_times(self):
        acquisition = np.array([
            [EPOCH_START, EPOCH_START + 3599, EPOCH_START + 3600, EPOCH_START + 86399],
            [EPOCH_START + 86400, EPOCH_START - 1, 0, np.nan],
        ], dtype=np.float64)
        self.write_file(np.arange(1, 9).reshape(2, 4) * 10, acquisition=acquisition)
        result = self.dataset_class(self.root, mode="hourly")[DAY]
        self.assertEqual(np.isfinite(result).sum(), 4)
        np.testing.assert_array_equal(result[0, 0, 0, :2], [10, 20])
        self.assertEqual(result[1, 0, 0, 2], 30)
        self.assertEqual(result[23, 0, 0, 3], 40)
        self.assertTrue(np.isnan(result[2:23]).all())
        self.assertTrue(np.isnan(result[:, :, 1]).all())

    def test_cf_acquisition_units_allow_zero_relative_to_target_midnight(self):
        self.write_file(
            [[10, 20], [30, 40]],
            acquisition=[[0, 1], [23.5, np.nan]],
            acquisition_units="hours since 2023-01-02 00:00:00",
        )
        result = self.dataset_class(self.root, mode="hourly")[DAY]
        self.assertEqual(result[0, 0, 0, 0], 10)
        self.assertEqual(result[1, 0, 0, 1], 20)
        self.assertEqual(result[23, 0, 1, 0], 30)
        self.assertEqual(np.isfinite(result).sum(), 3)

    def test_datetime_acquisition_roundtrips_through_netcdf_and_masks_nat(self):
        acquisition = np.array([
            ["2023-01-02T00:30", "2023-01-02T02:00"],
            ["2023-01-03T00:00", "NaT"],
        ], dtype="datetime64[ns]")
        self.write_file([[10, 20], [30, 40]], acquisition=acquisition, acquisition_units=None)
        result = self.dataset_class(self.root, mode="hourly")[DAY]
        self.assertEqual(result[0, 0, 0, 0], 10)
        self.assertEqual(result[2, 0, 0, 1], 20)
        self.assertEqual(np.isfinite(result).sum(), 2)

    def test_grid_and_field_axes_agree_and_unexpected_axis_order_is_rejected(self):
        self.write_file(
            [[1, 2, 3], [4, 5, 6]],
            latitude=[71, 70], longitude=[-20, 0, 20],
        )
        dataset = self.dataset_class(self.root)
        np.testing.assert_array_equal(dataset.src_grid.lat, [[71, 71, 71], [70, 70, 70]])
        np.testing.assert_array_equal(dataset.src_grid.lon, [[-20, 0, 20], [-20, 0, 20]])
        np.testing.assert_array_equal(dataset[DAY][0, 0], [[1, 2, 3], [4, 5, 6]])
        self.write_file(
            [[1, 2, 3], [4, 5, 6]], folder="reordered",
            order=("longitude", "time", "latitude"),
        )
        with self.assertRaises(ValueError):
            reordered = self.dataset_class(self.root, files_template="reordered/*.nc")
            reordered[DAY]

    def test_rejects_multiple_files_or_multiple_time_records_for_one_date(self):
        self.write_file([[10, 20], [30, 40]], folder="first")
        self.write_file([[10, 20], [30, 40]], folder="second")
        with self.assertRaises(ValueError):
            self.dataset_class(self.root)
        # An explicit template selects one file, but its time axis must be singleton.
        self.write_file(
            np.ones((2, 2, 2)), folder="multi",
            times=[np.datetime64("2023-01-02T00:00"), np.datetime64("2023-01-02T12:00")],
        )
        with self.assertRaises(ValueError):
            dataset = self.dataset_class(self.root, files_template="multi/*.nc")
            dataset[DAY]

    def test_rejects_coordinate_changes_between_days(self):
        self.write_file([[10, 20], [30, 40]])
        following_day = DAY + datetime.timedelta(days=1)
        self.write_file([[10, 20], [30, 40]], day=following_day, latitude=[70, 72])
        with self.assertRaises(ValueError):
            dataset = self.dataset_class(self.root)
            dataset[following_day]

    def test_template_override_filters_files_and_default_search_is_recursive(self):
        first = self.write_file([[10, 20], [30, 40]], folder="nested/selected")
        other_day = DAY + datetime.timedelta(days=1)
        self.write_file([[10, 20], [30, 40]], folder="other", day=other_day)
        dataset = self.dataset_class(self.root)
        self.assertEqual(set(dataset.dates_dict), {DAY, other_day})
        selected = self.dataset_class(self.root, files_template="nested/selected/*.nc")
        self.assertEqual(selected.dates_dict, {DAY: [first]})
        self.assertIsNone(selected[other_day])

    def test_existing_interpolation_handles_every_hour_and_forwards_cache(self):
        self.write_file(
            [[10, 20, 30], [40, 50, 60]],
            acquisition=np.full((2, 3), EPOCH_START + 3600, dtype=np.float64),
        )
        destination = object()
        cache_dir = self.root / "weights"
        dataset = self.dataset_class(
            self.root, destination, "SAR", mode="hourly", interpolation_cache_dir=cache_dir,
        )
        result = dataset[DAY]
        self.assertEqual(result.shape, (24, 1, 1, 2))
        self.assertEqual(result.dtype, np.float32)
        operator, = RecordingInterpolator.instances
        self.assertIs(operator.src_grid, dataset.src_grid)
        self.assertIs(operator.dst_grid, destination)
        self.assertEqual(operator.cache_dir, cache_dir)
        self.assertEqual(len(operator.calls), 24)
        np.testing.assert_array_equal(result[1, 0], [[10, 20]])
        self.assertTrue(np.isnan(result[0]).all())

    def test_identity_coarsening_preserves_daily_and_hourly_results(self):
        self.write_file(
            [[0, 20, np.nan], [40, 100, 101]],
            acquisition=np.array([[0, 1, 2], [3, 23, 24]]) * 3600.0 + EPOCH_START,
            status=[[0, 0, 0], [1, 0, 0]],
        )
        for mode in ("daily", "hourly"):
            expected = self.dataset_class(self.root, mode=mode)[DAY]
            for coarsen in (1, (1, 1)):
                with self.subTest(mode=mode, coarsen=coarsen):
                    dataset = self.dataset_class(self.root, mode=mode, coarsen=coarsen)
                    np.testing.assert_array_equal(dataset[DAY], expected)
                    self.assertEqual(dataset[DAY].dtype, np.float32)
                    self.assertEqual(dataset.src_grid.shape, (2, 3))

    def test_area_weights_and_polar_bounds_work_in_both_axis_directions(self):
        latitudes = np.array([60, 70, 80, 90], dtype=float)
        longitudes = np.array([-20, -10, 0, 10], dtype=float)
        values = np.repeat([[0], [100], [25], [75]], 4, axis=1)
        sin55, sin65, sin75, sin85, sin90 = np.sin(np.deg2rad([55, 65, 75, 85, 90]))
        expected = np.repeat([
            [100 * (sin75 - sin65) / (sin75 - sin55)],
            [(25 * (sin85 - sin75) + 75 * (sin90 - sin85)) / (sin90 - sin75)],
        ], 2, axis=1)
        for descending in (False, True):
            with self.subTest(descending=descending):
                axis_slice = slice(None, None, -1) if descending else slice(None)
                self.write_file(
                    values[axis_slice, axis_slice],
                    latitude=latitudes[axis_slice],
                    longitude=longitudes[axis_slice],
                )
                dataset = self.dataset_class(self.root, coarsen=2)
                result = dataset[DAY]
                self.assertEqual(result.shape, (1, 1, 2, 2))
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_allclose(result[0, 0], expected[axis_slice, axis_slice], rtol=1e-6)
                np.testing.assert_allclose(dataset.src_grid.lat[:, 0], np.array([65, 82.5])[axis_slice])
                np.testing.assert_allclose(dataset.src_grid.lon[0], np.array([-15, 5])[axis_slice])

    def test_partial_blocks_keep_edges_and_exclude_land_and_invalid_sic(self):
        self.write_file(
            [
                [0, 0, 10, 100, 90],
                [0, 0, 20, 30, 90],
                [50, 50, np.nan, np.nan, 25],
                [50, 50, 101, -1, 25],
                [10, 30, 40, 60, 80],
            ],
            status=[
                [0, 0, 1, 129, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            longitude=np.arange(5),
        )
        dataset = self.dataset_class(self.root, coarsen=(2, 2))
        result = dataset[DAY]
        np.testing.assert_allclose(
            result[0, 0], [[0, 25, 90], [50, np.nan, 25], [20, 50, 80]], equal_nan=True,
        )
        self.assertEqual(result.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(dataset.src_grid.lat[:, 0], [70.5, 72.5, 74])
        np.testing.assert_allclose(dataset.src_grid.lon[0], [0.5, 2.5, 4])
        self.assertEqual(len(RecordingGrid.instances), 1)
        self.assertEqual(RecordingGrid.instances[0].shape, (3, 3))

    def test_separate_latitude_longitude_factors(self):
        self.write_file(
            np.repeat(np.arange(7)[None, :], 4, axis=0),
            longitude=np.arange(7),
        )
        dataset = self.dataset_class(self.root, coarsen=(2, 3))
        self.assertEqual(dataset[DAY].shape, (1, 1, 2, 3))
        np.testing.assert_allclose(dataset[DAY][0, 0], [[1, 4, 6], [1, 4, 6]])
        np.testing.assert_allclose(dataset.src_grid.lon[0], [1, 4, 6])

    def test_hourly_coarsening_averages_only_observations_in_each_hour(self):
        self.write_file(
            [[10, 30, 20, 40], [50, 70, 60, 80], [0, 100, 25, 75], [20, 40, 60, 80]],
            latitude=[60, 70, 80, 90],
            acquisition=EPOCH_START + np.array([
                [0, 1, 0, 0], [0, 1, 1, 1], [2, 2, 23, 23], [2, 2, 24, -1],
            ], dtype=float) * 3600,
        )
        result = self.dataset_class(self.root, mode="hourly", coarsen=2)[DAY]
        weights = np.diff(np.sin(np.deg2rad([55, 65, 75, 85, 90])))
        expected = np.full((24, 1, 2, 2), np.nan, dtype=np.float32)
        expected[0, 0, 0] = [np.average([10, 50], weights=weights[:2]), 30]
        expected[1, 0, 0] = [np.average([30, 70], weights=weights[:2]), 70]
        expected[2, 0, 1, 0] = np.average([50, 30], weights=weights[2:])
        expected[23, 0, 1, 1] = 50
        self.assertEqual(result.shape, (24, 1, 2, 2))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6, equal_nan=True)

    def test_coarse_grid_is_used_by_inherited_interpolation(self):
        self.write_file(
            np.arange(24).reshape(4, 6),
            acquisition=np.full((4, 6), EPOCH_START + 3600, dtype=float),
        )
        destination = object()
        cache_dir = self.root / "coarse_weights"
        dataset = self.dataset_class(
            self.root, destination, mode="hourly", coarsen=2,
            interpolation_cache_dir=cache_dir,
        )
        result = dataset[DAY]
        self.assertIs(self.dataset_class.__getitem__, self.base_class.__getitem__)
        self.assertEqual(result.shape, (24, 1, 1, 2))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(len(RecordingGrid.instances), 1)
        self.assertEqual(dataset.src_grid.shape, (2, 3))
        operator, = RecordingInterpolator.instances
        self.assertIs(operator.src_grid, dataset.src_grid)
        self.assertEqual(operator.cache_dir, cache_dir)
        self.assertEqual(len(operator.calls), 24)
        self.assertTrue(all(values.shape == (2, 3) for values in operator.calls))
        np.testing.assert_array_equal(result[1, 0], operator.calls[1][:1, :2])
        self.assertTrue(np.isnan(result[0]).all())

    def test_coarsening_checks_original_coordinates_for_every_file(self):
        self.write_file(np.full((4, 4), 25))
        following_day = DAY + datetime.timedelta(days=1)
        self.write_file(
            np.full((4, 4), 25), day=following_day,
            # Coarse pair means would be unchanged; source coordinates differ.
            latitude=[69.9, 71.1, 71.9, 73.1],
        )
        with self.assertRaises(ValueError):
            dataset = self.dataset_class(self.root, coarsen=2)
            dataset[following_day]

    def test_invalid_coarsening_factors_fail_before_grid_creation(self):
        self.write_file(np.ones((4, 4)))
        invalid = (None, True, False, 0, -1, 1.5, (2,), (2, 2, 2), (2, 0), (True, 2), (2.0, 2))
        for coarsen in invalid:
            with self.subTest(coarsen=coarsen):
                with self.assertRaises((TypeError, ValueError)):
                    self.dataset_class(self.root, coarsen=coarsen)
                self.assertFalse(RecordingGrid.instances)
        for coarsen in (4, (1, 4), (4, 1), 100):
            with self.subTest(too_small=coarsen):
                with self.assertRaises(ValueError):
                    self.dataset_class(self.root, coarsen=coarsen)
                self.assertFalse(RecordingGrid.instances)

    def test_coarsening_rejects_nonregular_or_nonmonotonic_axes(self):
        for latitude, longitude in (
            ([70, 71, 73, 74], [0, 1, 2, 3]),
            ([70, 71, 72, 73], [0, 1, 3, 4]),
            ([70, 71, 71, 72], [0, 1, 2, 3]),
            ([70, 72, 71, 73], [0, 1, 2, 3]),
        ):
            with self.subTest(latitude=latitude, longitude=longitude):
                self.write_file(np.ones((4, 4)), latitude=latitude, longitude=longitude)
                with self.assertRaises(ValueError):
                    self.dataset_class(self.root, coarsen=2)
                self.assertFalse(RecordingGrid.instances)

    def test_two_dimensional_coordinates_remain_supported_without_coarsening(self):
        latitude, longitude = np.meshgrid(
            np.arange(4) + 70.0, np.arange(4) + 20.0, indexing="ij",
        )
        values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        source = xr.Dataset(
            {"sic": (("time", "y", "x"), values)},
            coords={
                "time": [DAY_START],
                "latitude": (("y", "x"), latitude),
                "longitude": (("y", "x"), longitude),
            },
        )
        source.to_netcdf(self.root / f"sic_{DAY:%Y%m%d}.nc", engine="scipy")
        for coarsen in (1, (1, 1)):
            with self.subTest(coarsen=coarsen):
                dataset = self.dataset_class(self.root, coarsen=coarsen)
                np.testing.assert_array_equal(dataset[DAY], values[:, None])
                np.testing.assert_array_equal(dataset.src_grid.lat, latitude)
                np.testing.assert_array_equal(dataset.src_grid.lon, longitude)
        RecordingGrid.instances = []
        with self.assertRaises(ValueError):
            self.dataset_class(self.root, coarsen=2)
        self.assertFalse(RecordingGrid.instances)


if __name__ == "__main__":
    unittest.main()
