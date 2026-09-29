"""Surface products preserve their sampled depths and source-specific semantics."""

import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "libs" / "validation" / "datasets" / "buoy"
ALIAS = "_buoy_surface_test_parent"
_previous_alias_modules = {
    name: module for name, module in sys.modules.items()
    if name == ALIAS or name.startswith(ALIAS + ".")
}
_parent_package = ModuleType(ALIAS)
_parent_package.__path__ = [str(PACKAGE.parent)]
sys.modules[ALIAS] = _parent_package
SPEC = importlib.util.spec_from_file_location(
    ALIAS + ".buoy", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)],
)
buoy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = buoy
SPEC.loader.exec_module(buoy)
utils = importlib.import_module(ALIAS + ".buoy.buoy_utils")


def tearDownModule():
    for name in tuple(sys.modules):
        if name == ALIAS or name.startswith(ALIAS + "."):
            sys.modules.pop(name)
    sys.modules.update(_previous_alias_modules)


def times(*hours):
    return np.datetime64("2023-01-01", "ns") + np.asarray(hours).astype("timedelta64[h]")


def record(key, series, position_times=None):
    if position_times is None:
        position_times = times(0, 1, 2)
    source, identity = key.split(":", 1)
    return buoy.NativeBuoyData(
        buoy.BuoyDescriptor(key, source, identity, (), tuple(series)),
        buoy.PositionSeries(position_times, np.tile([80., 20.], (len(position_times), 1))),
        series,
    )


class MemorySource:
    def __init__(self, *records):
        self.records = {data.descriptor.key: data for data in records}

    def discover(self):
        return tuple(data.descriptor for data in self.records.values())

    def read(self, key, *, variables=(), start=None, stop=None):
        return self.records[key]


class SurfaceDepthContractTests(unittest.TestCase):
    def test_legacy_scalar_batch_and_window_have_unknown_depth_on_same_axes(self):
        scalar = buoy.ScalarSeries(times(0, 1), [1.5, 1.6])
        self.assertEqual(scalar.depths.shape, (2,))
        self.assertTrue(np.isnan(scalar.depths).all())
        data = record("simba:ice", {"ice_thickness": scalar})
        batch = buoy.BuoyObservationDataset([MemorySource(data)]).at(times(0, 1))
        self.assertEqual(batch.depths.shape, batch.variables.shape)
        self.assertEqual(batch.depths.dtype, np.dtype("float32"))
        self.assertTrue(np.isnan(batch.depths).all())
        self.assertEqual(batch[0].depths.shape, (2, 2))
        self.assertTrue(np.isnan(replace(batch, depths=None).depths).all())
        self.assertTrue(np.isnan(replace(batch[0], depths=None).depths).all())
        with self.assertRaises(ValueError):
            buoy.ScalarSeries(times(0, 1), [1., 2.], depths=[.5])
        with self.assertRaises(ValueError):
            replace(batch, depths=np.zeros((1, 2)))

    def test_mixed_variables_keep_independent_depths_times_and_units(self):
        data = record("uptempo:shared", {
            "ice_thickness": buoy.ScalarSeries(times(0, 2), [1.5, 1.7]),
            "sst": buoy.ScalarSeries(times(0, 2), [-1.5, -1.], depths=[.5, 2.]),
            "sss": buoy.ScalarSeries(times(1), [0.], depths=[3.]),
        })
        ds = buoy.BuoyObservationDataset(
            [MemorySource(data)], variables=("sss", "ice_thickness", "sst"),
            time_method="nearest", tolerance="1h",
        )
        batch = ds.at(times(1, 2))
        self.assertEqual(batch.units, ("psu", "m", "degC"))
        np.testing.assert_allclose(batch.variables[0], [[0, 1.5, -1.5], [0, 1.7, -1.]])
        np.testing.assert_allclose(batch.depths[0], [[3, np.nan, .5], [3, np.nan, 2]], equal_nan=True)
        np.testing.assert_array_equal(batch.value_times[0], [times(1, 0, 0), times(1, 2, 2)])
        np.testing.assert_allclose(batch[0].depths, batch.depths[0], equal_nan=True)
        empty = ds.between(times(0)[0], times(0)[0])
        self.assertEqual(empty.depths.shape, (0, 0, 3))

    def test_native_depth_slices_are_half_open_and_do_not_mutate_cached_data(self):
        data = record("uptempo:one", {
            "sst": buoy.ScalarSeries(times(0, 1, 2), [-1., 0., 1.], depths=[.5, 1., 2.]),
        })
        ds = buoy.BuoyObservationDataset([MemorySource(data)], variables=("sst",))
        selected = ds.read_native("uptempo:one", start=times(1)[0], stop=times(2)[0])
        np.testing.assert_array_equal(selected.series["sst"].datetimes, times(1))
        np.testing.assert_allclose(selected.series["sst"].depths, [1.])
        selected.series["sst"].depths[0] = 999
        np.testing.assert_allclose(ds.read_native("uptempo:one").series["sst"].depths, [.5, 1., 2.])

    def test_conflicting_depths_invalidate_measurement_and_survive_repeated_merge(self):
        def fragment(depths):
            return record("uptempo:one", {
                "sst": buoy.ScalarSeries(times(0, 1), [-1., 0.], [.1, .2], depths),
            })

        first = fragment([.5, np.nan])
        second = fragment([1., 2.])
        merged = utils.merge_native_data([first, second])
        np.testing.assert_allclose(merged.series["sst"].values, [np.nan, 0], equal_nan=True)
        np.testing.assert_allclose(merged.series["sst"].depths, [np.nan, 2], equal_nan=True)
        np.testing.assert_allclose(merged.series["sst"].uncertainty, [np.nan, .2], equal_nan=True)
        conflicts = [item for item in merged.metadata["diagnostics"]
                     if item["kind"] == "conflicting_depths"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["variable"], "sst")
        merged_again = utils.merge_native_data([merged, first])
        self.assertTrue(np.isnan(merged_again.series["sst"].values[0]))
        self.assertTrue(np.isnan(merged_again.series["sst"].depths[0]))
        batch = buoy.BuoyObservationDataset(
            [MemorySource(merged), MemorySource(first)], variables=("sst",),
        ).at(times(0, 1))
        np.testing.assert_allclose(batch.depths[0, :, 0], [np.nan, 2], equal_nan=True)
        self.assertFalse(batch.valid[0, 0, 0])
        self.assertTrue(np.isnat(batch.value_times[0, 0, 0]))

    def test_profile_prefilter_skips_outside_queries_and_keeps_exact_nearest_boundaries(self):
        observations = [record(f"aotd:profile:{hour}", {
            "sst": buoy.ScalarSeries(times(hour), [-1.], depths=[2.]),
        }, position_times=times(hour)) for hour in (0, 10)]
        exact = buoy.BuoyObservationDataset([MemorySource(*observations)], variables=("sst",))
        with patch.object(exact, "_select", wraps=exact._select) as select:
            empty = exact.at(times(1))
            self.assertEqual(empty.depths.shape, (0, 1, 1))
            select.assert_not_called()
        batch = exact.at(times(0))
        self.assertEqual(batch.bids.tolist(), ["aotd:profile:0"])
        np.testing.assert_allclose(batch.depths, [[[2.]]])

        nearest = buoy.BuoyObservationDataset(
            [MemorySource(*observations)], variables=("sst",),
            time_method="nearest", tolerance="1h",
        )
        query = np.array([times(1)[0], times(1)[0] + np.timedelta64(1, "ns")])
        result = nearest.at(query)
        self.assertEqual(result.bids.tolist(), ["aotd:profile:0"])
        np.testing.assert_array_equal(result.valid[0, :, 0], [True, False])
        np.testing.assert_allclose(result.depths[0, :, 0], [2., np.nan], equal_nan=True)
        with patch.object(nearest, "_select", wraps=nearest._select) as select:
            empty = nearest.at(query[1:])
            self.assertEqual(empty.variables.shape, (0, 1, 1))
            select.assert_not_called()

    def test_profile_prefilter_does_not_confuse_coordinate_and_measurement_tolerances(self):
        data = record("uptempo:separate_clocks", {
            "sst": buoy.ScalarSeries(times(0), [-1.], depths=[.5]),
        }, position_times=times(5))
        broad = buoy.BuoyObservationDataset(
            [MemorySource(data)], variables=("sst",), time_method="nearest",
            tolerance="1h", coord_tolerance="4h",
        )
        batch = broad.at(times(1))
        self.assertEqual(batch.bids.tolist(), ["uptempo:separate_clocks"])
        np.testing.assert_array_equal(batch.coord_times, [times(5)])
        np.testing.assert_array_equal(batch.value_times[0, :, 0], times(0))
        narrow = buoy.BuoyObservationDataset(
            [MemorySource(data)], variables=("sst",), time_method="nearest",
            tolerance="1h", coord_tolerance="3h",
        )
        self.assertEqual(narrow.at(times(1)).variables.shape, (0, 1, 1))


class SourceFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def write_uptempo(self, rows, *, name="track_L2.dat", identity="300534062898720",
                      salinity=True, columns=None):
        if columns is None:
            columns = [
                "year", "month", "day", "hour (GMT)", "Latitude (N)", "Longitude (E)",
                "Temperature (C) at nominal depth 0.14 (m)",
                "Sea Surface Temperature", "Sea Surface Temperature Depth",
            ]
            if salinity:
                columns += ["Salinity (psu) at nominal depth 0.38 (m)",
                            "Sea Surface Salinity", "Sea Surface Salinity Depth"]
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ["%UpTempO 2023 #01", f"%Iridium ID: {identity}", "%WMO: 4802638",
                  "%DATA COLUMNS:"]
        header.extend(f"% {i} = {column}" for i, column in enumerate(columns))
        header += ["%LEVEL 2 Quality Control Modifications:",
                   "% Capped and Unphysical Values set to -999? YES",
                   "% NOTE: SST/SSS were selected from wet sensors by the source.", "END"]
        path.write_text("\n".join(header + [" ".join(map(str, row)) for row in rows]) + "\n",
                        encoding="utf-8")
        return path

    def write_aotd(self, *, name="profiles.nc", temp=None, salt=None, pres=None,
                   depth=None, clock=None, units="hours since 2023-01-01 00:00:00",
                   invalid_shape=False, units_by_variable=None):
        if depth is None:
            depth = [0., 2., 5., 20.]
        if temp is None:
            temp = [[np.nan, -1., -2., 3.], [np.nan, np.nan, 2., 3.],
                    [np.nan, -1., -2., 3.]]
        temp = np.asarray(temp, dtype=float)
        count = len(temp)
        if salt is None:
            salt = [[0., np.nan, 31., 32.], [0., np.nan, 0., 32.],
                    [0., np.nan, 30., 32.]]
        salt = np.asarray(salt, dtype=float)
        if invalid_shape:
            salt = salt[:-1]
        if pres is None:
            pres = np.tile(depth, (count, 1))
        if clock is None:
            clock = np.arange(count, dtype=float)
        time_attrs = {} if units is None else {"units": units, "calendar": "standard"}
        physical_units = {"temp": "degree_Celsius", "salt": "psu", "depth": "m"}
        physical_units.update(units_by_variable or {})

        def unit_attrs(name):
            return {} if physical_units[name] is None else {"units": physical_units[name]}

        # Real AOTD uses different dimensions for observation metadata and
        # profile matrices. Equal row positions, not xarray labels, join them.
        dataset = xr.Dataset({
            "lat": ("location", 70. + np.arange(count)),
            "lon": ("location", 200. + np.arange(count)),
            "time": ("clock", np.asarray(clock), time_attrs),
            "temp": (("profile", "depth"), temp, unit_attrs("temp")),
            "salt": (("salinity_row", "depth"), salt, unit_attrs("salt")),
            "pres": (("pressure_row", "depth"), np.asarray(pres, dtype=float), {"units": "dbar"}),
            "depth": ("depth", np.asarray(depth), {**unit_attrs("depth"), "positive": "down"}),
        }, coords={"location": 100 + np.arange(count), "clock": 200 + np.arange(count),
                   "profile": 300 + np.arange(count), "salinity_row": 400 + np.arange(len(salt)),
                   "pressure_row": 500 + np.arange(count)})
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(path, engine="h5netcdf")
        return path


class UpTempOSurfaceTests(SourceFixtureTests):
    def test_published_products_are_depth_filtered_independently_without_raw_fallback(self):
        path = self.write_uptempo([
            [2023, 1, 1, 0, 80, 200, -12, -1.5, .5, 3, 31, 5],
            [2023, 1, 1, 1, 81, 201, -13, -1.4, 6, 4, 0, .38],
            [2023, 1, 1, 2, 82, 202, -14, -999, .5, 30, -999, .38],
        ])
        source = buoy.UpTempOTabSource(self.folder, max_surface_depth=5)
        descriptor = source.discover()[0]
        self.assertEqual(descriptor.key, "uptempo:300534062898720")
        self.assertEqual(set(descriptor.variables), {"sst", "sss"})
        raw = source.read(descriptor.key, variables=("sst", "sss"))
        np.testing.assert_allclose(raw.series["sst"].values, [-1.5, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(raw.series["sss"].values, [31, 0, np.nan], equal_nan=True)
        np.testing.assert_allclose(raw.series["sst"].depths, [.5, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(raw.series["sss"].depths, [5, .38, np.nan], equal_nan=True)
        np.testing.assert_allclose(raw.positions.coords[:, 1], [-160, -159, -158])
        self.assertIn(str(path), raw.metadata["files"])
        self.assertEqual(raw.metadata["diagnostic_counts"]["missing_value"], 2)
        self.assertEqual(raw.metadata["diagnostic_counts"]["surface_depth_rejected"], 1)
        normalized_again = utils.merge_native_data([raw])
        self.assertEqual(normalized_again.metadata["diagnostic_counts"], raw.metadata["diagnostic_counts"])

    def test_header_column_positions_and_fractional_gmt_hours_are_respected(self):
        columns = ["Sea Surface Temperature Depth", "Longitude (E)", "year", "month",
                   "day", "hour (GMT)", "Latitude (N)", "Sea Surface Temperature"]
        self.write_uptempo([[.25, 20, 2023, 1, 1, "0.4833", 80, -1.25]], columns=columns)
        source = buoy.UpTempOTabSource(self.folder, max_surface_depth=1)
        raw = source.read(source.discover()[0].key, variables=("sst",))
        expected = np.array(["2023-01-01T00:28:59.880000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(raw.positions.datetimes, expected)
        np.testing.assert_array_equal(raw.series["sst"].datetimes, expected)
        np.testing.assert_allclose(raw.series["sst"].values, [-1.25])

    def test_split_files_merge_by_header_identity_and_optional_salinity_stays_absent(self):
        self.write_uptempo([[2023, 1, 1, 0, 80, 20, -3, -1, .5]],
                           name="first/anything_L2.dat", salinity=False)
        self.write_uptempo([[2023, 1, 1, 1, 80, 21, -4, 0, 1]],
                           name="second/different_L2.dat", salinity=False)
        source = buoy.UpTempOTabSource(self.folder, max_surface_depth=1)
        descriptors = source.discover()
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0].variables, ("sst",))
        self.assertEqual(len(descriptors[0].files), 2)
        raw = source.read(descriptors[0].key, variables=("sst",))
        np.testing.assert_array_equal(raw.series["sst"].datetimes, times(0, 1))
        np.testing.assert_allclose(raw.series["sst"].depths, [.5, 1])
        batch = buoy.BuoyObservationDataset([source], variables=("sst", "sss")).at(times(0, 1))
        self.assertTrue(np.isnan(batch.variables[:, :, 1]).all())
        self.assertTrue(np.isnan(batch.depths[:, :, 1]).all())

    def test_negative_or_missing_depth_rejects_value_but_zero_depth_is_valid(self):
        self.write_uptempo([
            [2023, 1, 1, 0, 80, 20, -3, -1., -1],
            [2023, 1, 1, 1, 80, 21, -3, -1., -999],
            [2023, 1, 1, 2, 80, 22, -3, -1., 0],
        ], salinity=False)
        source = buoy.UpTempOTabSource(self.folder, max_surface_depth=0)
        raw = source.read(source.discover()[0].key, variables=("sst",))
        np.testing.assert_allclose(raw.series["sst"].values, [np.nan, np.nan, -1], equal_nan=True)
        np.testing.assert_allclose(raw.series["sst"].depths, [np.nan, np.nan, 0], equal_nan=True)

    def test_surface_value_without_depth_column_is_a_contextual_schema_error(self):
        for label in ("Sea Surface Temperature", "Sea Surface Salinity"):
            columns = ["year", "month", "day", "hour (GMT)", "Latitude (N)",
                       "Longitude (E)", label]
            path = self.write_uptempo([[2023, 1, 1, 0, 80, 20, 1]], columns=columns,
                                      name=f"{label.replace(' ', '_')}/broken_L2.dat")
            with self.subTest(variable=label), self.assertRaisesRegex(ValueError, path.name):
                buoy.UpTempOTabSource(path.parent, max_surface_depth=5).discover()

    def test_invalid_time_is_diagnosed_and_invalid_position_does_not_erase_scalar(self):
        path = self.write_uptempo([
            [2023, 2, 30, 0, 80, 20, -5, -2, .5],
            [2023, 1, 1, 0, 91, 20, -5, -1, .5],
            [2023, 1, 1, 1, 80, 20, -5, -.5, .5],
        ], salinity=False)
        source = buoy.UpTempOTabSource(self.folder, max_surface_depth=1)
        raw = source.read(source.discover()[0].key, variables=("sst",))
        np.testing.assert_array_equal(raw.series["sst"].datetimes, times(0, 1))
        np.testing.assert_allclose(raw.series["sst"].values, [-1, -.5])
        self.assertTrue(np.isnan(raw.positions.coords[0]).all())
        first_data_line = path.read_text(encoding="utf-8").splitlines().index("END") + 2
        self.assertTrue(any(item.get("file") == str(path) and item.get("line") == first_data_line
                            for item in raw.metadata["diagnostics"]))
        batch = buoy.BuoyObservationDataset([source], variables=("sst",)).at(times(0, 1))
        self.assertTrue(batch.valid.all())
        np.testing.assert_array_equal(batch.coord_valid, [[False, True]])
        np.testing.assert_allclose(batch.depths, [[[.5], [.5]]])


class AotdSurfaceTests(SourceFixtureTests):
    def test_joint_sources_share_variable_axes_without_inventing_absent_values_or_depths(self):
        aotd = self.write_aotd()
        self.write_uptempo([
            [2023, 1, 1, 0, 80, 200, -12, -1.5, .5, 3, 31, 5],
            [2023, 1, 1, 1, 81, 201, -13, -1.4, 1, 4, 0, .38],
        ])
        legacy = record("simba:ice", {"ice_thickness": buoy.ScalarSeries(times(0, 2), [1., 2.])})
        ds = buoy.BuoyObservationDataset([
            buoy.UpTempOTabSource(self.folder, max_surface_depth=5),
            buoy.AotdNetCDFSource(aotd, max_surface_depth=5), MemorySource(legacy),
        ], variables=("sss", "ice_thickness", "sst"))
        batch = ds.at(times(0, 1, 2))
        self.assertEqual(batch.variables.shape, (5, 3, 3))
        self.assertEqual(batch.depths.shape, (5, 3, 3))
        self.assertEqual(batch.units, ("psu", "m", "degC"))
        np.testing.assert_allclose(batch["uptempo:300534062898720"].variables[:2],
                                   [[31, np.nan, -1.5], [0, np.nan, -1.4]], equal_nan=True)
        np.testing.assert_allclose(batch["uptempo:300534062898720"].depths[:2],
                                   [[5, np.nan, .5], [.38, np.nan, 1]], equal_nan=True)
        self.assertTrue(np.isnan(batch["simba:ice"].depths).all())
        self.assertTrue(np.isnan(batch["simba:ice"].variables[:, [0, 2]]).all())
        for profile in (bid for bid in batch.bids if bid.startswith("aotd:")):
            self.assertEqual(int(batch[profile].valid[:, 0].sum()), 1)
            self.assertTrue(np.isnan(batch[profile].variables[:, 1]).all())

    def test_independent_shallowest_levels_joint_fill_and_positional_profile_rows(self):
        path = self.write_aotd()
        source = buoy.AotdNetCDFSource(path, max_surface_depth=5)
        descriptors = source.discover()
        self.assertEqual(len(descriptors), 3)
        self.assertEqual(len({item.key for item in descriptors}), 3)
        for index, descriptor in enumerate(descriptors):
            self.assertTrue(descriptor.key.startswith("aotd:"))
            raw = source.read(descriptor.key, variables=("sst", "sss"))
            np.testing.assert_array_equal(raw.positions.datetimes, times(index))
            np.testing.assert_allclose(raw.positions.coords, [[70 + index, -160 + index]])
            np.testing.assert_allclose(raw.series["sst"].values, [[-1], [2], [-1]][index])
            np.testing.assert_allclose(raw.series["sst"].depths, [[2], [5], [2]][index])
            # salt=0 at 0m is the AOTD joint empty cell; salt=0 with a
            # finite temperature/pressure at 5m is a valid observation.
            np.testing.assert_allclose(raw.series["sss"].values, [[31], [0], [30]][index])
            np.testing.assert_allclose(raw.series["sss"].depths, [5])

    def test_surface_limit_is_inclusive_without_vertical_extrapolation(self):
        path = self.write_aotd()
        source = buoy.AotdNetCDFSource(path, max_surface_depth=2)
        raw = source.read(source.discover()[0].key, variables=("sst", "sss"))
        np.testing.assert_allclose(raw.series["sst"].values, [-1])
        self.assertTrue(np.isnan(raw.series["sss"].values).all())
        self.assertTrue(np.isnan(raw.series["sss"].depths).all())
        zero = buoy.AotdNetCDFSource(path, max_surface_depth=0)
        raw = zero.read(zero.discover()[0].key, variables=("sst", "sss"))
        self.assertTrue(all(np.isnan(s.values).all() for s in raw.series.values()))

    def test_time_units_override_is_explicit_and_original_units_are_honoured(self):
        path = self.write_aotd(clock=[0, 3600, 7200], units="seconds since 2023-01-01 00:00:00")
        source = buoy.AotdNetCDFSource(path, max_surface_depth=5)
        middle = source.read(source.discover()[1].key, variables=("sst",))
        np.testing.assert_array_equal(middle.positions.datetimes, times(1))
        missing = self.write_aotd(name="no_units.nc", units=None, clock=[0, 3600, 7200])
        with self.assertRaisesRegex(ValueError, missing.name):
            buoy.AotdNetCDFSource(missing, max_surface_depth=5).discover()
        overridden = buoy.AotdNetCDFSource(
            missing, max_surface_depth=5,
            time_units_override="seconds since 2023-01-01 00:00:00",
        )
        last = overridden.read(overridden.discover()[2].key, variables=("sst",))
        np.testing.assert_array_equal(last.positions.datetimes, times(2))
        conflicting = self.write_aotd(name="wrong_units.nc", clock=[0, 3600, 7200],
                                      units="days since 2023-01-01 00:00:00")
        explicit = buoy.AotdNetCDFSource(
            conflicting, max_surface_depth=5,
            time_units_override="seconds since 2023-01-01 00:00:00",
        )
        corrected = explicit.read(explicit.discover()[1].key, variables=("sst",))
        np.testing.assert_array_equal(corrected.positions.datetimes, times(1))

    def test_distinct_profiles_never_create_drift_or_merge_on_equal_time(self):
        path = self.write_aotd(clock=[0, 0, 1])
        source = buoy.AotdNetCDFSource(path, max_surface_depth=5)
        names = ("sst", "sss", "drift_speed")
        ds = buoy.BuoyObservationDataset([source], variables=names, drift_max_gap="2h")
        self.assertEqual(len(ds.buoy_ids), 3)
        batch = ds.at(times(0, 1))
        self.assertEqual(batch.variables.shape, (3, 2, 3))
        self.assertEqual(int(batch.valid[:, 0, 0].sum()), 2)
        self.assertFalse(batch.drift_support.valid.any())
        self.assertTrue(np.isnan(batch.variables[:, :, 2]).all())
        self.assertTrue(np.isnan(batch.depths[:, :, 2]).all())

    def test_profile_matrices_are_loaded_once_for_all_profile_reads_and_cached_as_copies(self):
        path = self.write_aotd()
        original = xr.DataArray.values.fget
        loaded = []

        def counted_values(array):
            if array.name in {"temp", "salt", "pres"}:
                self.assertLessEqual(array.shape[1], 3, "Reader materialized excluded deep levels")
                loaded.append(array.name)
            return original(array)

        with patch.object(xr.DataArray, "values", property(counted_values)):
            source = buoy.AotdNetCDFSource(path, max_surface_depth=5)
            descriptors = source.discover()
            self.assertEqual(loaded, [], "Discovery materialized profile matrices")
            for _ in range(2):
                for descriptor in descriptors:
                    source.read(descriptor.key, variables=("sst", "sss"))
        for name in ("temp", "salt", "pres"):
            self.assertEqual(loaded.count(name), 1, (name, loaded))
        raw = source.read(descriptors[0].key, variables=("sst",))
        raw.series["sst"].depths[0] = 999
        again = source.read(descriptors[0].key, variables=("sst",))
        np.testing.assert_allclose(again.series["sst"].depths, [2])

    def test_invalid_profile_dimensions_are_contextual_instead_of_silently_aligning(self):
        path = self.write_aotd(invalid_shape=True)
        with self.assertRaisesRegex(ValueError, path.name):
            buoy.AotdNetCDFSource(path, max_surface_depth=5).discover()

    def test_units_are_validated_before_values_are_labelled_as_celsius_psu_metres(self):
        cases = [("temp", "K"), ("depth", "cm"), ("temp", None),
                 ("salt", None), ("depth", None), ("salt", "kg/kg")]
        for index, (variable, units) in enumerate(cases):
            path = self.write_aotd(name=f"wrong_units_{index}.nc", units_by_variable={variable: units})
            with self.subTest(variable=variable, units=units), self.assertRaisesRegex(ValueError, path.name):
                buoy.AotdNetCDFSource(path, max_surface_depth=5).discover()
        alias = self.write_aotd(name="unit_aliases.nc", units_by_variable={
            "temp": "degrees_Celsius", "salt": "1", "depth": "metres",
        })
        source = buoy.AotdNetCDFSource(alias, max_surface_depth=5)
        batch = buoy.BuoyObservationDataset([source], variables=("sst", "sss")).at(times(0))
        self.assertEqual(batch.units, ("degC", "psu"))
        np.testing.assert_allclose(batch.variables, [[[-1, 31]]])
        np.testing.assert_allclose(batch.depths, [[[2, 5]]])

    def test_both_sources_require_an_explicit_finite_nonnegative_depth_cap(self):
        path = self.write_aotd()
        for source_type, location in ((buoy.UpTempOTabSource, self.folder),
                                      (buoy.AotdNetCDFSource, path)):
            with self.subTest(source=source_type.__name__), self.assertRaises((TypeError, ValueError)):
                source_type(location)
            for cap in (None, -1, np.nan, np.inf):
                with self.subTest(source=source_type.__name__, cap=cap):
                    with self.assertRaises((TypeError, ValueError)):
                        source_type(location, max_surface_depth=cap)


if __name__ == "__main__":
    unittest.main()
