"""Drift queries retain native segment geometry independently of query clocks."""

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


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "libs" / "validation" / "datasets" / "buoy"
ALIAS = "_buoy_drift_test_parent"
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

DRIFT = ("drift_speed", "drift_eastward", "drift_northward")


def tearDownModule():
    for name in tuple(sys.modules):
        if name == ALIAS or name.startswith(ALIAS + "."):
            sys.modules.pop(name)
    sys.modules.update(_previous_alias_modules)


def times(*offsets, unit="h"):
    unit = {"min": "m"}.get(unit, unit)
    return np.datetime64("2023-01-01", "ns") + np.asarray(offsets).astype(f"timedelta64[{unit}]")


def native(key, datetimes, coords, ice=None):
    series = {} if ice is None else {
        "ice_thickness": buoy.ScalarSeries(np.asarray(ice[0], dtype="datetime64[ns]"), np.asarray(ice[1])),
    }
    source, source_id = key.split(":", 1)
    return buoy.NativeBuoyData(
        buoy.BuoyDescriptor(key, source, source_id, (), tuple(series)),
        buoy.PositionSeries(np.asarray(datetimes, dtype="datetime64[ns]"), np.asarray(coords).reshape(-1, 2)),
        series,
    )


class MemorySource:
    def __init__(self, *records):
        self.records = {record.descriptor.key: record for record in records}
        self.requests = []

    def discover(self):
        return tuple(record.descriptor for record in self.records.values())

    def read(self, key, *, variables=(), start=None, stop=None):
        self.requests.append(tuple(variables))
        return self.records[key]


class BuoyDriftTests(unittest.TestCase):
    def dataset(self, *records, **kwargs):
        kwargs.setdefault("variables", DRIFT)
        kwargs.setdefault("drift_max_gap", "4h")
        return buoy.BuoyObservationDataset([MemorySource(*records)], **kwargs)

    def trajectory(self):
        return native("iabp:a", times(0, 1, 3), [[0, 0], [0, .01], [.01, .01]])

    def test_centimetre_per_second_values_start_anchors_and_pair_geometry(self):
        ds = self.dataset(self.trajectory())
        np.testing.assert_array_equal(ds.times, times(0, 1))
        batch = ds.at(times(0, 1, 3))
        self.assertEqual(batch.units, ("cm/s", "cm/s", "cm/s"))
        self.assertEqual(batch.metadata["iabp:a"]["drift"]["units"], "cm/s")
        self.assertEqual(batch.variables.shape, (1, 3, 3))
        # Independent WGS84 distances in metres, converted to cm/s.
        east = 1113.1949079327358 * 100 / 3600
        north = 1105.7427583286865 * 100 / 7200
        np.testing.assert_allclose(batch.variables[0, :2], [[east, east, 0], [north, 0, north]], atol=1e-7)
        np.testing.assert_array_equal(batch.valid[0], [[True] * 3, [True] * 3, [False] * 3])
        np.testing.assert_array_equal(batch.value_times[0, :2], np.repeat(times(0, 1)[:, None], 3, axis=1))
        self.assertTrue(np.isnan(batch.uncertainty).all())
        support = batch.drift_support
        self.assertEqual(support.coords.shape, (1, 3, 2, 2))
        self.assertEqual(support.datetimes.shape, (1, 3, 2))
        self.assertEqual(support.valid.shape, (1, 3))
        self.assertEqual(support.coords.dtype, np.dtype("float64"))
        self.assertEqual(support.datetimes.dtype, np.dtype("datetime64[ns]"))
        np.testing.assert_allclose(support.coords[0, :2], [[[0, 0], [0, .01]], [[0, .01], [.01, .01]]])
        np.testing.assert_array_equal(support.datetimes[0, :2], [times(0, 1), times(1, 3)])
        self.assertFalse(support.valid[0, 2])
        self.assertTrue(np.isnan(support.coords[0, 2]).all())
        self.assertTrue(np.isnat(support.datetimes[0, 2]).all())
        window = batch["iabp:a"]
        self.assertEqual(window.drift_support.coords.shape, (3, 2, 2))
        np.testing.assert_array_equal(window.drift_support.valid, support.valid[0])

    def test_returned_support_reconstructs_selected_segment_with_existing_kernel(self):
        coords = [[81.234567891, 179.912345678], [81.235876543, -179.978765432]]
        batch = self.dataset(native("iabp:a", times(0, 47, unit="min"), coords)).at(times(0))
        pair = batch.drift_support.coords[0, 0]
        pair_times = batch.drift_support.datetimes[0, 0]
        np.testing.assert_allclose(pair, coords, rtol=0, atol=1e-12)
        seconds = float((pair_times[1] - pair_times[0]) / np.timedelta64(1, "s"))
        kernel = importlib.import_module(ALIAS + ".iabp_utils").drift_uv_cm_s_from_latlon
        east, north, speed, _ = kernel(pair[:, 0], pair[:, 1], seconds)
        np.testing.assert_allclose(batch.variables[0, 0], np.array([speed[0], east[0], north[0]]), rtol=1e-6)
        self.assertGreater(batch.variables[0, 0, 1], 0)
        self.assertLess(batch.variables[0, 0, 0], 1000)

    def test_explicit_positive_gap_required_only_when_drift_requested(self):
        for value in (None, "0h", "-1h"):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.dataset(self.trajectory(), drift_max_gap=value)
        scalar = native("simba:a", times(0), [[80, 20]], ice=(times(0), [1.5]))
        batch = buoy.BuoyObservationDataset([MemorySource(scalar)]).at(times(0))
        self.assertIsNone(batch.drift_support)
        self.assertIsNone(batch[0].drift_support)

    def test_gap_limit_is_inclusive_and_uses_native_interval(self):
        clock = np.array([times(0)[0], times(1)[0], times(2)[0] + np.timedelta64(1, "ns")])
        ds = self.dataset(native("iabp:a", clock, [[0, 0], [0, .01], [0, .02]]), drift_max_gap="1h")
        np.testing.assert_array_equal(ds.times, times(0))
        batch = ds.at(clock)
        np.testing.assert_array_equal(batch.drift_support.valid, [[True, False, False]])
        self.assertTrue(np.isnan(batch.variables[0, 1:]).all())

    def test_pre_epoch_times_and_large_gap_do_not_overflow_to_valid_segment(self):
        clock = np.array(["1800-01-01", "1800-01-02", "2200-01-01"], dtype="datetime64[ns]")
        ds = self.dataset(native("iabp:a", clock, [[0, 0], [0, .01], [0, .02]]),
                          drift_max_gap="100000D")
        np.testing.assert_array_equal(ds.times, clock[:1])
        batch = ds.at(clock)
        np.testing.assert_array_equal(batch.drift_support.valid, [[True, False, False]])

    def test_reordered_component_subset_preserves_requested_variable_axis(self):
        data = native("simba:a", times(0, 1, 3), [[0, 0], [0, .01], [.01, .01]],
                      ice=(times(0, 1), [1.5, 1.75]))
        names = ("drift_northward", "ice_thickness", "drift_eastward")
        batch = self.dataset(data, variables=names).at(times(0, 1))
        full = self.dataset(data).at(times(0, 1))
        self.assertEqual(batch.var_names, names)
        self.assertEqual(batch.units, ("cm/s", "m", "cm/s"))
        np.testing.assert_allclose(batch.variables[:, :, 0], full.variables[:, :, 2])
        np.testing.assert_allclose(batch.variables[:, :, 1], [[1.5, 1.75]])
        np.testing.assert_allclose(batch.variables[:, :, 2], full.variables[:, :, 1])
        np.testing.assert_array_equal(batch.drift_support.datetimes, full.drift_support.datetimes)

    def test_support_rejects_invalid_pair_mask_and_nonpositive_duration(self):
        batch = self.dataset(self.trajectory()).at(times(0))
        support = batch.drift_support
        invalid_changes = (
            {"valid": np.zeros((1, 1), dtype=bool)},
            {"datetimes": support.datetimes[..., ::-1]},
            {"coords": support.coords[:, :, 0]},
        )
        for change in invalid_changes:
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                replace(support, **change)

    def test_invalid_or_conflicting_position_breaks_both_pairs_without_skipping(self):
        records = [
            [MemorySource(native("iabp:a", times(0, 1, 2, 3), [[0, 0], [np.nan, .01], [0, .02], [0, .03]]))],
            [MemorySource(native("iabp:a", times(0, 1, 2, 3), [[0, 0], [0, .01], [0, .02], [0, .03]])),
             MemorySource(native("iabp:a", times(1), [[1, .01]]))],
        ]
        for sources in records:
            with self.subTest(sources=len(sources)):
                ds = buoy.BuoyObservationDataset(sources, variables=DRIFT, drift_max_gap="4h")
                np.testing.assert_array_equal(ds.times, times(2))
                batch = ds.at(times(0, 1, 2, 3))
                np.testing.assert_array_equal(batch.drift_support.valid, [[False, False, True, False]])
                np.testing.assert_array_equal(batch.drift_support.datetimes[0, 2], times(2, 3))

    def test_fragments_sort_and_merge_before_deriving_cross_file_pair(self):
        first = native("iabp:a", times(1, 0), [[0, .01], [0, 0]])
        second = native("iabp:a", times(2, 1), [[0, .02], [0, .01]])
        ds = buoy.BuoyObservationDataset([MemorySource(first), MemorySource(second)],
                                         variables=DRIFT, drift_max_gap="1h")
        np.testing.assert_array_equal(ds.times, times(0, 1))
        batch = ds.at(times(0, 1))
        self.assertTrue(batch.valid.all())
        np.testing.assert_array_equal(batch.drift_support.datetimes[0], [times(0, 1), times(1, 2)])

    def test_nearest_ties_and_all_components_use_one_pair(self):
        ds = self.dataset(native("iabp:a", times(0, 2, 4), [[0, 0], [0, .01], [.02, .01]]),
                          time_method="nearest", tolerance="1h")
        batch = ds.at(times(1, 3))
        np.testing.assert_array_equal(batch.value_times[0], np.repeat(times(0, 2)[:, None], 3, axis=1))
        np.testing.assert_array_equal(batch.drift_support.datetimes[0], [times(0, 2), times(2, 4)])
        self.assertGreater(batch.variables[0, 0, 1], 0)
        self.assertAlmostEqual(float(batch.variables[0, 0, 2]), 0, places=7)
        self.assertAlmostEqual(float(batch.variables[0, 1, 1]), 0, places=7)
        self.assertGreater(batch.variables[0, 1, 2], 0)

    def test_mixed_request_preserves_shared_coords_and_independent_thickness_time(self):
        query = times(190, unit="min")
        data = native("simba:a", times(0, 2, 4), [[0, 0], [0, .01], [.02, .01]], ice=(query, [1.75]))
        batch = self.dataset(data, variables=("ice_thickness",) + DRIFT,
                             time_method="nearest", tolerance="90min").at(query)
        np.testing.assert_allclose(batch.coords[0, 0], [.02, .01])
        np.testing.assert_allclose(batch.drift_support.coords[0, 0], [[0, .01], [.02, .01]])
        self.assertEqual(batch.coord_times[0, 0], times(4)[0])
        np.testing.assert_array_equal(batch.value_times[0, 0], np.concatenate((query, times(2, 2, 2))))
        self.assertAlmostEqual(float(batch.variables[0, 0, 0]), 1.75)

    def test_window_step_cannot_change_native_drift_and_read_native_stays_raw(self):
        data = native("simba:a", times(0, 1, 3), [[0, 0], [0, .01], [.01, .01]],
                      ice=(times(1), [1.75]))
        ds = self.dataset(data, variables=("ice_thickness",) + DRIFT, T=2, step="3h")
        direct = ds.at(times(0))
        window = ds[0]
        np.testing.assert_allclose(window.variables[0, 0, 1:], direct.variables[0, 0, 1:])
        np.testing.assert_array_equal(window.drift_support.datetimes[0, 0], times(0, 1))
        raw = ds.read_native("simba:a", start=times(1)[0], stop=times(3)[0])
        self.assertEqual(set(raw.series), {"ice_thickness"})
        np.testing.assert_array_equal(raw.positions.datetimes, times(1))

    def test_coordinates_only_sources_are_not_asked_to_supply_derived_columns(self):
        source = MemorySource(self.trajectory())
        ds = buoy.BuoyObservationDataset([source], variables=("drift_speed",), drift_max_gap="4h")
        self.assertEqual(source.requests, [()])
        self.assertEqual(ds.at(times(0)).variables.shape, (1, 1, 1))
        self.assertEqual(ds.read_native("iabp:a").series, {})

    def test_mixed_complete_coverage_can_require_drift_while_thickness_is_optional(self):
        ds = self.dataset(self.trajectory(), variables=("ice_thickness",) + DRIFT,
                          required_variables=("drift_speed",), coverage="complete")
        batch = ds.at(times(0, 1))
        self.assertEqual(batch.bids.tolist(), ["iabp:a"])
        self.assertTrue(np.isnan(batch.variables[:, :, 0]).all())
        self.assertTrue(batch.valid[:, :, 1:].all())
        excluded = ds.at(times(0, 1, 3))
        self.assertEqual(excluded.variables.shape, (0, 3, 4))
        self.assertEqual(excluded.drift_support.valid.shape, (0, 3))

    def test_stationary_positions_are_valid_zero_and_empty_results_keep_support_axes(self):
        still = native("iabp:a", times(0, 1), [[80, 20], [80, 20]])
        batch = self.dataset(still).at(times(0))
        np.testing.assert_array_equal(batch.variables, np.zeros((1, 1, 3)))
        self.assertTrue(batch.drift_support.valid.all())
        for data in (native("iabp:one", times(0), [[80, 20]]),
                     native("iabp:bad", times(0, 1), [[np.nan, np.nan], [np.nan, np.nan]])):
            with self.subTest(key=data.descriptor.key):
                ds = self.dataset(data)
                self.assertEqual(len(ds), 0)
                result = ds.at(times(0, 1))
                self.assertEqual(result.variables.shape, (0, 2, 3))
                self.assertEqual(result.drift_support.coords.shape, (0, 2, 2, 2))
                empty = ds.between(times(0)[0], times(0)[0])
                self.assertEqual(empty.drift_support.coords.shape, (0, 0, 2, 2))

    def test_speed_limit_is_inclusive_and_masks_vector_magnitude_jointly(self):
        utils = importlib.import_module(ALIAS + ".buoy.buoy_utils")
        kernel = importlib.import_module(ALIAS + ".iabp_utils")
        data = native("iabp:a", times(0, 1, 2, 3),
                      [[80, 20], [80, 20.01], [80, 20.02], [80, 20.03]])
        east = np.array([60., 80., 30., np.nan])
        north = np.array([80., 80., 40., np.nan])
        speed = np.hypot(east, north)
        distance = speed * 3600 / 100
        with patch.object(kernel, "drift_uv_cm_s_from_latlon",
                          return_value=(east, north, speed, distance)):
            cache = utils.derive_drift(data.positions, "4h")
            ds = self.dataset(data)
        np.testing.assert_array_equal(cache.datetimes, times(0, 1, 2))
        np.testing.assert_array_equal(cache.support.valid, [True, False, True])
        np.testing.assert_allclose(cache.values[[0, 2]], [[100, 60, 80], [50, 30, 40]])
        self.assertTrue(np.isnan(cache.values[1]).all())
        np.testing.assert_array_equal(ds.times, times(0, 2))
        batch = ds.at(times(0, 1, 2))
        np.testing.assert_array_equal(batch.valid[0], [[True] * 3, [False] * 3, [True] * 3])
        self.assertTrue(np.isnan(batch.variables[0, 1]).all())
        self.assertTrue(np.isnat(batch.value_times[0, 1]).all())
        self.assertFalse(batch.drift_support.valid[0, 1])
        metadata = batch.metadata["iabp:a"]["drift"]
        self.assertEqual(metadata["drift_max_speed_cm_s"], 100.)
        self.assertEqual(metadata["speed_exceeded_segments"], 1)
        self.assertEqual(metadata["valid_segments"], 2)
        self.assertNotIn("rejected_speed_segments", metadata)
        self.assertEqual(len(cache.rejected_speed_segments), 1)

    def test_speed_limit_can_be_disabled_or_explicitly_increased(self):
        data = native("iabp:a", times(0, 1), [[0, 0], [0, .05]])
        # 5.566 km/hour is about 154.6 cm/s: physically excessive for the default.
        rejected = self.dataset(data)
        self.assertEqual(len(rejected.times), 0)
        self.assertIsNone(rejected._observation_bounds["iabp:a"])
        for limit in (None, 200., np.float64(200.)):
            with self.subTest(limit=limit):
                ds = self.dataset(data, drift_max_speed=limit)
                batch = ds.at(times(0))
                self.assertEqual(batch.bids.tolist(), ["iabp:a"])
                self.assertTrue(batch.valid.all())
                self.assertGreater(batch.variables[0, 0, 0], 100.)
                self.assertEqual(batch.metadata["iabp:a"]["drift"]["drift_max_speed_cm_s"], limit)
                self.assertEqual(batch.metadata["iabp:a"]["drift"]["speed_exceeded_segments"], 0)
        # Disabling speed QC must not disable the independent time-gap policy.
        too_long = self.dataset(data, drift_max_speed=None, drift_max_gap="30min")
        self.assertEqual(len(too_long.times), 0)

    def test_invalid_speed_limit_is_rejected_by_dataset_and_native_derivation(self):
        utils = importlib.import_module(ALIAS + ".buoy.buoy_utils")
        for value in (0, -1, np.nan, np.inf, -np.inf, True, False,
                      np.bool_(True), "100", [], np.array([100.]), 10**1000):
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    self.dataset(self.trajectory(), drift_max_speed=value)
                with self.assertRaises((TypeError, ValueError)):
                    utils.derive_drift(self.trajectory().positions, "4h", max_speed=value)

    def test_rejected_speed_anchor_cannot_borrow_a_nearby_valid_segment(self):
        data = native("iabp:a", times(0, 1, 2, 3, 4),
                      [[80, 20], [80, 20.01], [0, 0], [80, 20.02], [80, 20.03]])
        ds = self.dataset(data, time_method="nearest", tolerance="2h")
        np.testing.assert_array_equal(ds.times, times(0, 3))
        query = times(0, 60, 90, 120, 180, unit="min")
        batch = ds.at(query)
        np.testing.assert_array_equal(batch.drift_support.valid, [[True, False, False, False, True]])
        self.assertTrue(np.isnan(batch.variables[0, 1:4]).all())
        self.assertTrue(np.isnat(batch.value_times[0, 1:4]).all())
        np.testing.assert_array_equal(batch.drift_support.datetimes[0, 0], times(0, 1))
        np.testing.assert_array_equal(batch.drift_support.datetimes[0, 4], times(3, 4))
        # Native positions are retained, including the rejected position jump.
        np.testing.assert_allclose(ds.read_native("iabp:a").positions.coords,
                                   data.positions.coords, rtol=0, atol=1e-12)
        metadata = batch.metadata["iabp:a"]["drift"]
        self.assertEqual(metadata["speed_exceeded_segments"], 2)
        self.assertNotIn("rejected_speed_segments", metadata)
        rejected = ds.read_drift_diagnostics("iabp:a")
        self.assertEqual(len(rejected), 2)
        for entry, index in zip(rejected, [1, 2]):
            self.assertEqual(entry["buoy_id"], "iabp:a")
            self.assertEqual(entry["kind"], "drift_speed_exceeded")
            self.assertEqual(entry["limit_cm_s"], 100.)
            self.assertEqual(np.datetime64(entry["t0"]), times(index)[0])
            self.assertEqual(np.datetime64(entry["t1"]), times(index + 1)[0])
            np.testing.assert_allclose(entry["coords0"], data.positions.coords[index])
            np.testing.assert_allclose(entry["coords1"], data.positions.coords[index + 1])
            self.assertEqual(entry["dt_seconds"], 3600.)
            self.assertGreater(entry["speed_cm_s"], 100.)
            self.assertAlmostEqual(entry["distance_m"] * 100 / entry["dt_seconds"],
                                   entry["speed_cm_s"], places=7)

    def test_drift_diagnostics_accessor_returns_independent_copies_and_all_buoys(self):
        coords = [[80, 20], [0, 0], [80, 20.01]]
        ds = self.dataset(*(native(key, times(0, 1, 2), coords)
                            for key in ("iabp:a", "iabp:b")))
        diagnostics = ds.read_drift_diagnostics()
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual({entry["buoy_id"] for entry in diagnostics}, {"iabp:a", "iabp:b"})
        diagnostics[0]["speed_cm_s"] = -1
        original_coords = diagnostics[0]["coords0"]
        if isinstance(original_coords, (list, np.ndarray)):
            original_coords[0] = -123
        else:
            diagnostics[0]["coords0"] = [-123, -456]
        reread = ds.read_drift_diagnostics("iabp:a")
        self.assertEqual(len(reread), 2)
        self.assertGreater(reread[0]["speed_cm_s"], 100.)
        np.testing.assert_allclose(reread[0]["coords0"], coords[0])
        with self.assertRaises(KeyError):
            ds.read_drift_diagnostics("iabp:missing")
        still = self.dataset(native("iabp:still", times(0, 1), [[80, 20], [80, 20]]))
        self.assertEqual(still.read_drift_diagnostics(), [])
        scalar = native("simba:a", times(0), [[80, 20]], ice=(times(0), [1.5]))
        scalar_only = buoy.BuoyObservationDataset([MemorySource(scalar)])
        self.assertEqual(scalar_only.read_drift_diagnostics(), [])
        self.assertEqual(scalar_only.read_drift_diagnostics("simba:a"), [])
        with self.assertRaises(KeyError):
            scalar_only.read_drift_diagnostics("simba:missing")

    def test_rejected_coordinate_and_gap_anchors_do_not_bridge_nearest_queries(self):
        cases = (
            (times(0, 1, 2, 3, 4), [[0, 0], [0, .01], [np.nan, np.nan], [0, .02], [0, .03]],
             "1h", times(0, 1, 2, 3), [[True, False, False, True]]),
            (times(0, 1, 4, 5), [[0, 0], [0, .01], [0, .02], [0, .03]],
             "1h", times(0, 1, 4), [[True, False, True]]),
        )
        for clock, coords, gap, query, expected in cases:
            with self.subTest(clock=clock):
                ds = self.dataset(native("iabp:a", clock, coords), drift_max_gap=gap,
                                  time_method="nearest", tolerance="3h")
                batch = ds.at(query)
                np.testing.assert_array_equal(batch.drift_support.valid, expected)
                invalid = ~np.asarray(expected[0])
                self.assertTrue(np.isnan(batch.variables[0, invalid]).all())
                self.assertTrue(np.isnat(batch.value_times[0, invalid]).all())


class IabpTabSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def write(self, filename, header, rows):
        path = self.folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_position_clock_alias_and_invalid_position_retention(self):
        self.write("named_differently.dat", "BuoyID Year Hour Min DOY PosDOY Lat Lon BP", [
            "300534062898720 2023 23 59 1.5 1.0 80 200 1000",
            "300534062898720 2023 23 59 1.75 1.25 80.1 200.1 1001",
            "300534062898720 2023 23 59 2.0 1.5 -9999 200.2 1002",
        ])
        source = buoy.IabpTabSource(self.folder)
        descriptor = source.discover()[0]
        self.assertEqual(descriptor.key, "iabp:300534062898720")
        self.assertEqual(descriptor.variables, ())
        raw = source.read(descriptor.key)
        self.assertEqual(raw.series, {})
        np.testing.assert_array_equal(raw.positions.datetimes, times(0, 6, 12))
        np.testing.assert_allclose(raw.positions.coords[:2], [[80, -160], [80.1, -159.9]])
        self.assertTrue(np.isnan(raw.positions.coords[2]).all())
        ds = buoy.BuoyObservationDataset([source], variables=DRIFT, drift_max_gap="6h")
        np.testing.assert_array_equal(ds.times, times(0))
        np.testing.assert_array_equal(ds.at(times(0, 6)).drift_support.valid, [[True, False]])

    def test_ids_across_files_year_rollover_and_coordinate_only_drift(self):
        header = "BuoyID Year DOY POS_DOY Lat Lon"
        self.write("2022/multiple.dat", header, ["123 2023 1.0 365.5 80 10", "456 2023 1.0 1.0 70 20"])
        self.write("2023/another.dat", header, ["123 2023 1.5 1.5 80 10.1", "456 2023 1.5 1.5 70 20.1"])
        source = buoy.IabpTabSource(self.folder)
        self.assertEqual([d.key for d in source.discover()], ["iabp:123", "iabp:456"])
        raw = source.read("iabp:123")
        expected = np.array(["2022-12-31T12:00:00", "2023-01-01T12:00:00"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(raw.positions.datetimes, expected)
        ds = buoy.BuoyObservationDataset([source], variables=DRIFT, drift_max_gap="24h")
        batch = ds.at(expected[:1])
        self.assertEqual(batch.bids.tolist(), ["iabp:123"])
        np.testing.assert_array_equal(batch.drift_support.datetimes[0, 0], expected)

    def test_missing_position_clock_is_a_contextual_schema_error(self):
        path = self.write("broken.dat", "BuoyID Year DOY Lat Lon", ["123 2023 1.0 80 10"])
        with self.assertRaisesRegex(ValueError, path.name):
            buoy.IabpTabSource(self.folder).discover()

    def test_empty_folder_wrong_pattern_and_header_only_table_fail_with_context(self):
        header = "BuoyID Year DOY POS_DOY Lat Lon"
        for pattern in ("*.dat", "*.missing"):
            if pattern == "*.missing":
                self.write("valid.dat", header, ["123 2023 1.0 1.0 80 10"])
            with self.subTest(pattern=pattern):
                with self.assertRaises(FileNotFoundError) as caught:
                    buoy.IabpTabSource(self.folder, pattern=pattern).discover()
                self.assertIn(str(self.folder), str(caught.exception))
                self.assertIn(pattern, str(caught.exception))
        header_only = self.write("header_only.dat", header, [])
        with self.assertRaisesRegex(ValueError, header_only.name):
            buoy.IabpTabSource(self.folder).discover()

    def test_invalid_day_fields_are_excluded_with_row_diagnostics(self):
        path = self.write("invalid_times.dat", "BuoyID Year DOY POS_DOY Lat Lon", [
            "123 2023 1.0 1.0 80 10",
            "123 2023 1.5 0.5 81 11",
            "123 2023 366.0 365.0 82 12",
            "123 2023 2.0 2.0 80 10.1",
        ])
        raw = buoy.IabpTabSource(self.folder).read("iabp:123")
        np.testing.assert_array_equal(raw.positions.datetimes, times(0, 24))
        diagnostics = raw.metadata["diagnostics"]
        failures = {(entry.get("reason"), entry.get("line"))
                    for entry in diagnostics if entry.get("kind") == "invalid_position_time"}
        self.assertIn(("invalid_position_doy", 3), failures)
        self.assertIn(("invalid_report_doy", 4), failures)
        for entry in diagnostics:
            if entry.get("kind") == "invalid_position_time":
                self.assertEqual(entry["file"], str(path))

    def test_decimal_day_text_preserves_exact_nanosecond_clock(self):
        self.write("fractional.dat", "BuoyID Year DOY POS_DOY Lat Lon", [
            "800054 2023 308.77435 308.77435 80 10",
        ])
        raw = buoy.IabpTabSource(self.folder).read("iabp:800054")
        expected = np.array(["2023-11-04T18:35:03.840000000"], dtype="datetime64[ns]")
        np.testing.assert_array_equal(raw.positions.datetimes, expected)

    def test_merging_iabp_sources_preserves_row_counts_and_counts_new_conflict_once(self):
        header = "BuoyID Year DOY POS_DOY Lat Lon"
        self.write("first/track.dat", header, ["123 2023 1.0 1.0 80 10"])
        self.write("second/track.dat", header, ["123 2023 1.0 1.0 81 10"])
        ds = buoy.BuoyObservationDataset([
            buoy.IabpTabSource(self.folder / "first"),
            buoy.IabpTabSource(self.folder / "second"),
        ], variables=DRIFT, drift_max_gap="1h")
        raw = ds.read_native("iabp:123")
        np.testing.assert_array_equal(raw.positions.datetimes, times(0))
        self.assertTrue(np.isnan(raw.positions.coords).all())
        self.assertEqual(raw.metadata["row_counts"]["rows_read"], 2)
        self.assertEqual(raw.metadata["diagnostic_counts"]["conflicting_coordinates"], 1)
        conflicts = [entry for entry in raw.metadata["diagnostics"]
                     if entry["kind"] == "conflicting_coordinates"]
        self.assertEqual(len(conflicts), 1)


if __name__ == "__main__":
    unittest.main()
