"""Public buoy query behavior, using small native-format and in-memory fixtures."""

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
ALIAS = "_buoy_test_parent"
_previous_alias_modules = {
    name: module for name, module in sys.modules.items()
    if name == ALIAS or name.startswith(ALIAS + ".")
}
_parent_package = ModuleType(ALIAS)
_parent_package.__path__ = [str(PACKAGE.parent)]
sys.modules[ALIAS] = _parent_package
SPEC = importlib.util.spec_from_file_location(
    ALIAS + ".buoy", PACKAGE / "__init__.py",
    submodule_search_locations=[str(PACKAGE)],
)
buoy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = buoy
SPEC.loader.exec_module(buoy)


def tearDownModule():
    # Keep the private modules registered during tests for introspection and
    # pickling, then restore their prior state without touching project imports.
    for _module_name in tuple(sys.modules):
        if _module_name == ALIAS or _module_name.startswith(ALIAS + "."):
            sys.modules.pop(_module_name)
    sys.modules.update(_previous_alias_modules)


def times(*hours):
    return np.datetime64("2023-01-01", "ns") + np.asarray(hours).astype("timedelta64[h]")


def record(key, ice=None, snow=None, positions=None, uncertainty=None):
    """Use separate native clocks; tuples contain (timestamps, values)."""
    series = {}
    for name, observations in (("ice_thickness", ice), ("snow_thickness", snow)):
        if observations is not None:
            t, values = observations
            series[name] = buoy.ScalarSeries(
                datetimes=np.asarray(t, dtype="datetime64[ns]"),
                values=np.asarray(values, dtype=float),
                uncertainty=(None if uncertainty is None else uncertainty.get(name)),
            )
    if positions is None:
        positions = (times(0, 1, 2), [[80, 120], [81, 121], [82, 122]])
    descriptor = buoy.BuoyDescriptor(
        key=key, source=key.split(":")[0], source_buoy_id=key.split(":", 1)[1],
        files=(), variables=tuple(series),
    )
    return buoy.NativeBuoyData(
        descriptor=descriptor,
        positions=buoy.PositionSeries(
            datetimes=np.asarray(positions[0], dtype="datetime64[ns]"),
            coords=np.asarray(positions[1], dtype=float).reshape(-1, 2),
        ),
        series=series,
    )


class MemorySource:
    """Minimal source protocol implementation with no disk or optional imports."""

    def __init__(self, *records):
        self.records = {item.descriptor.key: item for item in records}

    def discover(self):
        return tuple(item.descriptor for item in self.records.values())

    def read(self, key, *, variables=("ice_thickness", "snow_thickness"), start=None, stop=None):
        return self.records[key]


class BuoyObservationQueryTests(unittest.TestCase):
    def dataset(self, *records, **kwargs):
        return buoy.BuoyObservationDataset([MemorySource(*records)], **kwargs)

    def assert_contract(self, batch, n, t, v=2):
        self.assertEqual(batch.bids.shape, (n,))
        self.assertEqual(batch.coords.shape, (n, t, 2))
        self.assertEqual(batch.datetimes.shape, (t,))
        self.assertEqual(batch.variables.shape, (n, t, v))
        self.assertEqual(batch.uncertainty.shape, (n, t, v))
        self.assertEqual(batch.valid.shape, (n, t, v))
        self.assertEqual(batch.coord_valid.shape, (n, t))
        self.assertEqual(batch.value_times.shape, (n, t, v))
        self.assertEqual(batch.coord_times.shape, (n, t))
        for name in ("coords", "variables", "uncertainty"):
            self.assertEqual(getattr(batch, name).dtype, np.dtype("float32"))
        for name in ("datetimes", "coord_times", "value_times"):
            self.assertEqual(getattr(batch, name).dtype, np.dtype("datetime64[ns]"))
        self.assertEqual(batch.valid.dtype, np.dtype(bool))
        self.assertEqual(batch.coord_valid.dtype, np.dtype(bool))

    def test_exact_values_masks_and_axes_when_buoy_count_equals_time_count(self):
        ds = self.dataset(
            record("simba:a", (times(0, 1), [11, 12]), (times(0, 1), [.11, .12])),
            record("crrel:a", (times(0, 1), [21, 22]), (times(0, 1), [.21, .22])),
        )
        batch = ds.at(times(0, 1))
        self.assert_contract(batch, 2, 2)
        self.assertEqual(batch.bids.tolist(), ["crrel:a", "simba:a"])
        self.assertEqual(batch.var_names, ("ice_thickness", "snow_thickness"))
        self.assertEqual(batch.units, ("m", "m"))
        np.testing.assert_allclose(batch.variables[:, :, 0], [[21, 22], [11, 12]])
        np.testing.assert_allclose(batch.variables[:, :, 1], [[.21, .22], [.11, .12]])
        self.assertTrue(batch.valid.all())
        np.testing.assert_allclose(batch.var("ice_thickness"), [[21, 22], [11, 12]])
        np.testing.assert_allclose(batch["simba:a"].var("snow_thickness"), [.11, .12])
        self.assertEqual(batch[0].bid, "crrel:a")
        self.assertEqual(batch[-1].bid, "simba:a")

    def test_batch_rejects_wrong_shapes_and_inconsistent_validity(self):
        batch = self.dataset(record("simba:a", (times(0, 1, 2), [4, 5, 6]))).at(times(0, 1, 2))
        invalid_fields = (
            {"coords": batch.coords.transpose(1, 0, 2)},
            {"variables": batch.variables.transpose(1, 0, 2)},
            {"valid": ~batch.valid},
            {"value_times": np.full(batch.value_times.shape, np.datetime64("NaT", "ns"))},
        )
        for fields in invalid_fields:
            with self.subTest(fields=tuple(fields)), self.assertRaises(ValueError):
                replace(batch, **fields)
        with self.assertRaises(KeyError):
            batch.var("temperature")
        with self.assertRaises(KeyError):
            batch["missing:buoy"]

    def test_unequal_axes_and_missing_optional_values_remain_nan_and_nat(self):
        ds = self.dataset(record("simba:a", (times(0, 2), [4, 8])))
        batch = ds.at(times(0, 1, 2))
        self.assert_contract(batch, 1, 3)
        np.testing.assert_allclose(batch.variables[0, :, 0], [4, np.nan, 8], equal_nan=True)
        self.assertTrue(np.isnan(batch.variables[0, :, 1]).all())
        self.assertTrue(np.isnat(batch.value_times[0, :, 1]).all())
        np.testing.assert_array_equal(batch.valid[0, :, 0], [True, False, True])

    def test_nearest_uses_independent_finite_clocks_and_earlier_tie(self):
        ds = self.dataset(
            record(
                "simba:a", (times(0, 1, 2), [7, np.nan, 9]),
                (times(1, 3), [.4, .8]),
                positions=(times(0, 2), [[75, 130], [77, 132]]),
                uncertainty={"ice_thickness": np.array([.07, .99, .09])},
            ), time_method="nearest", tolerance="1h", coord_tolerance="1h",
        )
        batch = ds.at(times(1))
        np.testing.assert_allclose(batch.variables[0, 0], [7, .4])
        np.testing.assert_allclose(batch.coords[0, 0], [75, 130])
        np.testing.assert_array_equal(batch.value_times[0, 0], times(0, 1))
        self.assertEqual(batch.coord_times[0, 0], times(0)[0])
        self.assertAlmostEqual(float(batch.uncertainty[0, 0, 0]), .07, places=6)

    def test_tolerance_is_inclusive_and_not_an_unbounded_fill(self):
        ds = self.dataset(
            record("simba:a", (times(0), [6]), (times(0), [.6]),
                   positions=(times(0), [[70, 100]])),
            time_method="nearest", tolerance="1h", coord_tolerance="1h",
        )
        query = np.array([times(1)[0], times(1)[0] + np.timedelta64(1, "ns")])
        batch = ds.at(query)
        np.testing.assert_array_equal(batch.valid[0, :, 0], [True, False])
        np.testing.assert_array_equal(batch.coord_valid[0], [True, False])
        self.assertTrue(np.isnan(batch.variables[0, 1]).all())
        self.assertTrue(np.isnan(batch.coords[0, 1]).all())
        self.assertTrue(np.isnat(batch.value_times[0, 1]).all())
        self.assertTrue(np.isnat(batch.coord_times[0, 1]))

    def test_position_tolerance_can_be_stricter_than_measurement_tolerance(self):
        ds = self.dataset(
            record("simba:a", (times(0, 1), [4, 5]), (times(0, 1), [.4, .5]),
                   positions=(times(0), [[70, 100]])),
            time_method="nearest", tolerance="1h", coord_tolerance="30min",
        )
        batch = ds.at(times(0, 1))
        self.assertTrue(batch.valid[0, 1].all())
        self.assertFalse(batch.coord_valid[0, 1])
        self.assertTrue(np.isnan(batch.coords[0, 1]).all())

    def test_candidates_use_finite_measurements_and_default_windows(self):
        ds = self.dataset(
            record("simba:a", (times(2, 0, 1), [8, 4, np.nan]),
                   positions=(times(0, 1, 2, 8), [[80, 10], [81, 11], [82, 12], [83, 13]])),
            T=2, step="1h",
        )
        np.testing.assert_array_equal(ds.times, times(0, 2))
        self.assertEqual(len(ds), 2)
        np.testing.assert_array_equal(ds[0].datetimes, times(0, 1))
        np.testing.assert_array_equal(ds[1].datetimes, times(2, 3))

    def test_explicit_start_times_and_between_use_half_open_intervals(self):
        ds = self.dataset(
            record("simba:a", (times(0, 1, 2), [1, 2, 3])),
            start_times=times(1), T=2, step="1h",
        )
        self.assertEqual(len(ds), 1)
        np.testing.assert_array_equal(ds[0].datetimes, times(1, 2))
        batch = ds.between(times(0)[0], times(2)[0], step="1h")
        np.testing.assert_array_equal(batch.datetimes, times(0, 1))
        np.testing.assert_allclose(batch.variables[0, :, 0], [1, 2])

    def test_coverage_and_required_variables_do_not_require_optional_snow(self):
        complete = record("simba:complete", (times(0, 1), [1, 2]))
        partial = record("simba:partial", (times(0), [3]))
        full = self.dataset(complete, partial, coverage="complete",
                            required_variables=("ice_thickness",)).at(times(0, 1))
        self.assertEqual(full.bids.tolist(), ["simba:complete"])
        self.assertTrue(np.isnan(full.variables[:, :, 1]).all())
        some = self.dataset(complete, partial, coverage="partial",
                            required_variables=("ice_thickness",)).at(times(0, 1))
        self.assertEqual(some.bids.tolist(), ["simba:complete", "simba:partial"])

    def test_missing_required_variable_excludes_complete_buoy(self):
        ds = self.dataset(record("simba:a", (times(0), [1])), coverage="complete",
                          required_variables=("ice_thickness", "snow_thickness"))
        self.assert_contract(ds.at(times(0)), 0, 1)

    def test_empty_sources_and_no_finite_candidates_return_well_shaped_results(self):
        for ds in (self.dataset(), self.dataset(record("simba:a", (times(0), [np.nan])))):
            with self.subTest(dataset=ds):
                self.assertEqual(len(ds), 0)
                self.assertEqual(ds.times.shape, (0,))
                self.assert_contract(ds.at(times(0, 1)), 0, 2)

    def test_unmatched_queries_and_empty_intervals_preserve_requested_variable_order(self):
        ds = self.dataset(record("simba:a", (times(0), [1])),
                          variables=("snow_thickness", "ice_thickness"))
        batch = ds.at(times(8, 9, 10))
        empty = ds.between(times(0)[0], times(0)[0], step="1h")
        self.assert_contract(batch, 0, 3)
        self.assert_contract(empty, 0, 0)
        for result in (batch, empty):
            self.assertEqual(result.var_names, ("snow_thickness", "ice_thickness"))
            self.assertEqual(result.units, ("m", "m"))
        np.testing.assert_array_equal(batch.datetimes, times(8, 9, 10))
        with self.assertRaises(ValueError):
            ds.at(times())

    def test_utc_offset_is_normalized_before_year_boundary_matching(self):
        ds = self.dataset(record("simba:a", (times(0, 1), [5, 6])))
        batch = ds.at(["2022-12-31T19:00:00-05:00", "2023-01-01T04:00:00+03:00"])
        np.testing.assert_array_equal(batch.datetimes, times(0, 1))
        np.testing.assert_allclose(batch.variables[0, :, 0], [5, 6])

    def test_query_rejects_duplicate_unsorted_and_nat_times(self):
        ds = self.dataset(record("simba:a", (times(0), [1])))
        for query in (times(0, 0), times(1, 0), np.array(["NaT"], dtype="datetime64[ns]")):
            with self.subTest(query=query), self.assertRaises((ValueError, TypeError)):
                ds.at(query)

    def test_invalid_window_and_nearest_options_are_rejected(self):
        for kwargs in ({"T": 0}, {"step": "0h"}, {"time_method": "nearest"},
                       {"time_method": "nearest", "tolerance": "-1h"},
                       {"time_method": "linear"}, {"coverage": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.dataset(record("simba:a", (times(0), [1])), **kwargs)

    def test_duplicate_parts_merge_complementary_values_and_invalidate_conflicts(self):
        first = record("simba:a", (times(0, 1, 2), [5, 7, 9]),
                       (times(0, 1, 2), [np.nan, .2, .3]),
                       positions=(times(0, 1, 2), [[80, 10], [81, 11], [82, 12]]))
        second = record("simba:a", (times(0, 1, 2), [np.nan, 8, 9]),
                        (times(0, 1, 2), [.1, .2, .3]),
                        positions=(times(0, 1, 2), [[np.nan, np.nan], [81, 11], [83, 12]]))
        ds = buoy.BuoyObservationDataset([MemorySource(first), MemorySource(second)])
        batch = ds.at(times(0, 1, 2))
        self.assertEqual(batch.bids.tolist(), ["simba:a"])
        np.testing.assert_allclose(batch.variables[0, :, 0], [5, np.nan, 9], equal_nan=True)
        np.testing.assert_allclose(batch.variables[0, :, 1], [.1, .2, .3])
        np.testing.assert_allclose(batch.coords[0, 0], [80, 10])
        self.assertTrue(np.isnan(batch.coords[0, 2]).all())
        self.assertFalse(batch.coord_valid[0, 2])
        self.assertTrue(batch.metadata["simba:a"]["diagnostics"])

    def test_conflicting_uncertainty_does_not_invalidate_agreeing_values(self):
        observations = record("simba:a", (times(0, 0, 1), [5, 5, 6]),
                              uncertainty={"ice_thickness": np.array([.1, .2, .3])})
        batch = self.dataset(observations).at(times(0, 1))
        np.testing.assert_allclose(batch.variables[0, :, 0], [5, 6])
        self.assertTrue(np.isnan(batch.uncertainty[0, 0, 0]))
        self.assertAlmostEqual(float(batch.uncertainty[0, 1, 0]), .3, places=6)
        self.assertTrue(batch.metadata["simba:a"]["diagnostics"])


class NativeBuoySourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def write_simba(self, name="2023T1_icethick.tab", rows=None):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows is None:
            rows = ["2023-01-01T00:00:00\t80\t200\t1.25\t\t0.04\t",
                    "2023-01-01T01:00:00\t81\t201\t1.50\t-0.08\t0.06\t0.02",
                    "2023-01-01T02:00:00\t82\t202\t1.75\t0.20\t0.08\t0.03"]
        path.write_text(
            "/* DATA DESCRIPTION:\nAbstract:\tThickness smoothed with a 3-day running mean.\n*/\n"
            "Date/Time\tLatitude\tLongitude\tEsEs [m]\tSnow thick [m]\tEsEs unc [m]\tSnow thick unc [m]\n"
            + "\n".join(rows) + "\n", encoding="utf-8",
        )
        return path

    def write_crrel(self, name, west=False, units="m", fill=False):
        data = {
            "lat": ("time", [74., 75., 76.]),
            "lon": ("time", [170., 171., 172.]),
            "hi": ("time", [125., -999. if fill else 150., 175.], {"units": units}),
            "hs": ("time", [10., 15., 20.], {"units": units}),
            "T": (("time", "depth"), np.arange(12).reshape(3, 4)),
        }
        if west:
            data["hi_west"] = ("time", [90., 91., 92.], {"units": units})
            data["hs_west"] = ("time", [50., 51., 52.], {"units": units})
        path = self.folder / name
        xr.Dataset(data, coords={
            "time": ("time", [0., 3600., 7200.],
                     {"units": "seconds since 2023-01-01 00:00:00", "calendar": "standard"}),
            "depth": np.arange(4),
        }).to_netcdf(path, engine="h5netcdf")
        return path

    def test_simba_header_blank_cells_uncertainty_and_negative_snow(self):
        self.write_simba("nested/2023T1_icethick.tab")
        source = buoy.SimbaTabSource(self.folder)
        descriptors = source.discover()
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0].key, "simba:2023T1")
        batch = buoy.BuoyObservationDataset([source]).at(times(0, 1, 2))
        np.testing.assert_allclose(batch.variables[0, :, 0], [1.25, 1.5, 1.75])
        np.testing.assert_allclose(batch.variables[0, :, 1], [np.nan, -.08, .2], equal_nan=True)
        np.testing.assert_allclose(batch.uncertainty[0, :, 0], [.04, .06, .08])
        np.testing.assert_allclose(batch.coords[0, :, 1], [-160, -159, -158])

    def test_native_read_preserves_native_clock_and_half_open_bounds(self):
        self.write_simba()
        ds = buoy.BuoyObservationDataset([buoy.SimbaTabSource(self.folder)])
        native = ds.read_native("simba:2023T1", start=times(1)[0], stop=times(2)[0])
        np.testing.assert_array_equal(native.positions.datetimes, times(1))
        np.testing.assert_array_equal(native.series["ice_thickness"].datetimes, times(1))
        np.testing.assert_allclose(native.series["ice_thickness"].values, [1.5])

    def test_joint_adapter_batch_uses_one_common_schema(self):
        self.write_simba()
        self.write_crrel("1997E_updated.nc", units="cm")
        ds = buoy.BuoyObservationDataset([
            buoy.SimbaTabSource(self.folder), buoy.CrrelNetCDFSource(self.folder),
        ])
        batch = ds.at(times(0, 1, 2))
        self.assertEqual(batch.bids.tolist(), ["crrel:1997E", "simba:2023T1"])
        self.assertEqual(batch.variables.shape, (2, 3, 2))
        np.testing.assert_allclose(batch.variables[:, :, 0], [[1.25, 1.5, 1.75], [1.25, 1.5, 1.75]])
        np.testing.assert_allclose(batch.variables[:, :, 1], [[.1, .15, .2], [np.nan, -.08, .2]], equal_nan=True)
        np.testing.assert_allclose(batch.coords[:, 0], [[74, 170], [80, -160]])
        self.assertTrue(np.isnan(batch.uncertainty[0]).all())
        np.testing.assert_allclose(batch.uncertainty[1, :, 0], [.04, .06, .08])

    def test_multiple_files_for_one_buoy_read_their_own_header_lengths(self):
        first = self.write_simba("first/2023T1_icethick.tab", rows=[
            "2023-01-01T00:00:00\t80\t100\t1\t.1\t.01\t.02",
            "2023-01-01T01:00:00\t81\t101\t2\t.2\t.01\t.02",
        ])
        second = self.write_simba("second/2023T1_icethick.tab", rows=[
            "2023-01-01T02:00:00\t82\t102\t3\t.3\t.01\t.02",
            "2023-01-01T03:00:00\t83\t103\t4\t.4\t.01\t.02",
        ])
        second.write_text(second.read_text(encoding="utf-8").replace(
            "*/", "Citation:\tSecond file\nAdditional metadata:\tLonger header\n*/"
        ), encoding="utf-8")
        source = buoy.SimbaTabSource(self.folder)
        self.assertEqual(len(source.discover()), 1)
        self.assertEqual(len(source.discover()[0].files), 2)
        batch = buoy.BuoyObservationDataset([source]).at(times(0, 1, 2, 3))
        self.assertEqual(batch.bids.tolist(), ["simba:2023T1"])
        np.testing.assert_allclose(batch.variables[0, :, 0], [1, 2, 3, 4])
        np.testing.assert_allclose(batch.coords[0, :, 0], [80, 81, 82, 83])
        metadata = batch.metadata["simba:2023T1"]
        self.assertEqual(set(metadata["file_metadata"]), {str(first), str(second)})
        self.assertEqual({tuple(part["files"]) for part in metadata["source_records"]},
                         {(str(first),), (str(second),)})

    def test_previously_normalized_conflict_cannot_be_filled_by_another_part(self):
        for directory, conflicting_value in (("first", 1), ("second", 2)):
            self.write_simba(f"native/{directory}/2023T1_icethick.tab", rows=[
                f"2023-01-01T00:00:00\t80\t100\t{conflicting_value}\t.1\t.01\t.02",
                "2023-01-01T01:00:00\t81\t101\t3\t.2\t.01\t.02",
            ])
        source = buoy.SimbaTabSource(self.folder / "native")
        normalized = source.read("simba:2023T1")
        self.assertTrue(np.isnan(normalized.series["ice_thickness"].values[0]))
        supplement = record("simba:2023T1", (times(0, 1), [1, 3]),
                            positions=(times(0, 1), [[80, 100], [81, 101]]))
        batch = buoy.BuoyObservationDataset([
            MemorySource(normalized), MemorySource(supplement),
        ]).at(times(0, 1))
        np.testing.assert_allclose(batch.variables[0, :, 0], [np.nan, 3], equal_nan=True)
        self.assertFalse(batch.valid[0, 0, 0])
        self.assertTrue(np.isnat(batch.value_times[0, 0, 0]))
        self.assertTrue(batch.metadata["simba:2023T1"]["diagnostics"])

    def test_missing_files_and_invalid_schema_errors_identify_the_asset(self):
        with self.assertRaisesRegex(OSError, "missing_source"):
            buoy.SimbaTabSource(self.folder / "missing_source")
        for source_type in (buoy.SimbaTabSource, buoy.CrrelNetCDFSource):
            with self.subTest(source=source_type.__name__):
                with self.assertRaisesRegex(FileNotFoundError, self.folder.name):
                    source_type(self.folder).discover()
        missing = self.write_simba()
        source = buoy.SimbaTabSource(self.folder)
        source.discover()
        missing.unlink()
        with self.assertRaisesRegex((ValueError, OSError), missing.name):
            source.read("simba:2023T1")
        malformed = self.write_simba("broken_icethick.tab")
        malformed.write_text(malformed.read_text(encoding="utf-8").replace(
            "Date/Time\tLatitude\tLongitude", "Date/Time\tLatitude\tUnknownPosition"
        ), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, malformed.name):
            buoy.SimbaTabSource(self.folder).discover()

    def test_both_crrel_scalar_schemas_use_cf_time_units_and_main_measurements(self):
        self.write_crrel("1997E_updated.nc", west=True, units="cm")
        self.write_crrel("SIMB3 2024W.nc", west=False, units="m")
        source = buoy.CrrelNetCDFSource(self.folder)
        self.assertEqual({d.key for d in source.discover()}, {"crrel:1997E", "crrel:SIMB3 2024W"})
        batch = buoy.BuoyObservationDataset([source]).at(times(0, 1, 2))
        np.testing.assert_allclose(batch.variables[0, :, 0], [1.25, 1.5, 1.75])
        np.testing.assert_allclose(batch.variables[0, :, 1], [.1, .15, .2])
        np.testing.assert_allclose(batch.variables[1, :, 0], [125, 150, 175])
        np.testing.assert_array_equal(batch.datetimes, times(0, 1, 2))

    def test_crrel_ignores_profile_data_without_loading_its_values(self):
        self.write_crrel("1997E_updated.nc")
        original = xr.DataArray.values.fget

        def scalar_values_only(array):
            if array.name == "T":
                raise AssertionError("Scalar reader materialized the temperature profile")
            return original(array)

        with patch.object(xr.DataArray, "values", property(scalar_values_only)):
            source = buoy.CrrelNetCDFSource(self.folder)
            native = source.read(source.discover()[0].key)
        self.assertEqual(set(native.series), {"ice_thickness", "snow_thickness"})
        np.testing.assert_allclose(native.series["ice_thickness"].values, [125, 150, 175])

    def test_crrel_fill_value_is_missing_and_cannot_become_nearest_neighbor(self):
        self.write_crrel("1997E_updated.nc", units="cm", fill=True)
        ds = buoy.BuoyObservationDataset([buoy.CrrelNetCDFSource(self.folder)],
                                         time_method="nearest", tolerance="1h")
        batch = ds.at(times(1))
        np.testing.assert_allclose(batch.variables[0, 0], [1.25, .15])
        self.assertEqual(batch.value_times[0, 0, 0], times(0)[0])

    def test_crrel_incompatible_units_fail_with_file_context(self):
        path = self.write_crrel("bad_units_updated.nc", units="kelvin")
        with self.assertRaisesRegex(ValueError, path.name):
            source = buoy.CrrelNetCDFSource(self.folder)
            source.read(source.discover()[0].key)


if __name__ == "__main__":
    unittest.main()
