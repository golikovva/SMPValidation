"""Source position sentinels are gaps, not geography or duplicate conflicts."""

import importlib
import importlib.util
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
ALIAS = "_buoy_coordinate_qc_test_parent"
_previous_modules = {
    name: module for name, module in sys.modules.items()
    if name == ALIAS or name.startswith(ALIAS + ".")
}
_parent = ModuleType(ALIAS)
_parent.__path__ = [str(PACKAGE.parent)]
sys.modules[ALIAS] = _parent
_spec = importlib.util.spec_from_file_location(
    ALIAS + ".buoy", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)],
)
buoy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = buoy
_spec.loader.exec_module(buoy)
adapters = importlib.import_module(ALIAS + ".buoy.buoy_adapters")
utils = importlib.import_module(ALIAS + ".buoy.buoy_utils")


def tearDownModule():
    for name in tuple(sys.modules):
        if name == ALIAS or name.startswith(ALIAS + "."):
            sys.modules.pop(name)
    sys.modules.update(_previous_modules)


class BuoyCoordinateQCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def read_iabp(self, rows):
        path = self.folder / "42.dat"
        path.write_text(
            "BuoyID Year DOY POS_DOY Lat Lon\n"
            + "".join(f"42 2023 {report} {position} {lat} {lon}\n"
                      for report, position, lat, lon in rows),
            encoding="utf-8",
        )
        return buoy.IabpTabSource(self.folder).read("iabp:42"), path

    def read_crrel(self, seconds, coords):
        coords = np.asarray(coords, dtype=float)
        path = self.folder / "42.nc"
        xr.Dataset({
            "lat": ("time", coords[:, 0]),
            "lon": ("time", coords[:, 1]),
            "hi": ("time", np.ones(len(seconds)), {"units": "m"}),
        }, coords={
            "time": ("time", np.asarray(seconds, dtype=float),
                     {"units": "seconds since 2023-01-01 00:00:00"}),
        }).to_netcdf(path, engine="scipy")
        open_dataset = xr.open_dataset

        def scipy_reader(*args, **kwargs):
            kwargs["engine"] = "scipy"
            return open_dataset(*args, **kwargs)

        # Exercise the real reader/schema/CF decoding with an actual NetCDF3
        # fixture; optional netCDF4/h5netcdf installations are not needed here.
        with patch.object(adapters.xr, "open_dataset", side_effect=scipy_reader):
            source = buoy.CrrelNetCDFSource(self.folder)
            data = source.read(source.discover()[0].key, variables=("ice_thickness",))
        return data, path

    def test_iabp_exact_pair_only_and_existing_invalid_coordinate_diagnostics(self):
        data, path = self.read_iabp([
            ("1", "1", "-90.00000", "-180.00000"),
            ("1.0417", "1.0417", "-90", "-180.0"),
            ("1.0833", "1.0833", -90, 0),
            ("1.1250", "1.1250", 80, -180),
            ("1.1667", "1.1667", 0, 0),
            ("1.2083", "1.2083", 91, 20),
        ])
        self.assertEqual(len(data.positions.datetimes), 6)
        self.assertTrue(np.isnan(data.positions.coords[[0, 1, 5]]).all())
        np.testing.assert_array_equal(data.positions.coords[2:5], [[-90, 0], [80, -180], [0, 0]])
        invalid = [d for d in data.metadata["diagnostics"] if d["kind"] == "invalid_coordinates"]
        self.assertEqual([d["reason"] for d in invalid], [
            "iabp_missing_position_sentinel", "iabp_missing_position_sentinel",
            "missing_or_out_of_range_position",
        ])
        self.assertEqual([d["line"] for d in invalid], [2, 3, 7])
        self.assertEqual(data.metadata["row_counts"]["invalid_coordinate_rows"], 3)
        for diagnostic in invalid:
            self.assertEqual(diagnostic["source"], "iabp")
            self.assertEqual(diagnostic["file"], str(path))
            self.assertIn("time", diagnostic)
            self.assertIn("coords", diagnostic)

    def test_iabp_sentinel_stays_a_gap_for_drift(self):
        data, _ = self.read_iabp([
            ("1", "1", 80, 20),
            ("1.0417", "1.0417", -90, -180),
            ("1.0833", "1.0833", 80, 20.002),
            ("1.1250", "1.1250", 80, 20.003),
        ])
        drift = utils.derive_drift(data.positions, "6h")
        np.testing.assert_array_equal(drift.datetimes, data.positions.datetimes[:-1])
        np.testing.assert_array_equal(drift.support.valid, [False, False, True])
        self.assertTrue(np.isnan(drift.values[:2]).all())
        np.testing.assert_array_equal(drift.support.datetimes[2], data.positions.datetimes[2:])
        self.assertEqual(drift.metadata["invalid_position_segments"], 2)

    def test_iabp_raw_2023_extreme_pair_is_not_a_valid_drift_segment(self):
        data, _ = self.read_iabp([
            ("14.7528", "14.7528", 75.928, 8.836),
            ("14.7944", "14.7527", -90, -180),
        ])
        np.testing.assert_array_equal(data.positions.datetimes, np.array([
            "2023-01-14T18:03:53.280", "2023-01-14T18:04:01.920",
        ], dtype="datetime64[ns]"))
        self.assertTrue(np.isnan(data.positions.coords[0]).all())
        drift = utils.derive_drift(data.positions, "6h")
        np.testing.assert_array_equal(drift.datetimes, data.positions.datetimes[:1])
        self.assertFalse(drift.support.valid.any())
        self.assertTrue(np.isnan(drift.values).all())

    def test_iabp_valid_duplicate_wins_over_sentinel_in_either_row_order(self):
        for sentinel_first in (False, True):
            with self.subTest(sentinel_first=sentinel_first):
                middle = [("1.0417", "1.0417", 80, 20.001),
                          ("1.0417", "1.0417", -90, -180)]
                if sentinel_first:
                    middle.reverse()
                data, _ = self.read_iabp([
                    ("1", "1", 80, 20), *middle, ("1.0833", "1.0833", 80, 20.002),
                ])
                np.testing.assert_allclose(data.positions.coords, [[80, 20], [80, 20.001], [80, 20.002]])
                self.assertEqual(len(utils.derive_drift(data.positions, "6h").datetimes), 2)
                self.assertFalse(any(d["kind"] == "conflicting_coordinates"
                                     for d in data.metadata["diagnostics"]))

    def test_crrel_sentinel_gap_preserves_scalar_values_and_diagnostics(self):
        data, path = self.read_crrel([0, 3600, 7200, 10800], [
            [80, 20], [0, 0], [80, 20.002], [80, 20.003],
        ])
        self.assertEqual(len(data.positions.datetimes), 4)
        self.assertTrue(np.isnan(data.positions.coords[1]).all())
        np.testing.assert_array_equal(data.series["ice_thickness"].values, np.ones(4))
        drift = utils.derive_drift(data.positions, "6h")
        np.testing.assert_array_equal(drift.datetimes, data.positions.datetimes[:-1])
        np.testing.assert_array_equal(drift.support.valid, [False, False, True])
        self.assertTrue(np.isnan(drift.values[:2]).all())
        self.assertEqual(drift.metadata["invalid_position_segments"], 2)
        self.assertEqual(data.metadata["diagnostics"], [{
            "kind": "invalid_coordinates", "source": "crrel", "file": str(path),
            "index": 1, "time": "2023-01-01T01:00:00.000000000",
            "reason": "crrel_missing_position_sentinel", "coords": [0.0, 0.0],
        }])
        self.assertEqual(data.metadata["diagnostic_counts"]["invalid_coordinates"], 1)

    def test_crrel_valid_duplicate_wins_and_single_zero_coordinates_stay_valid(self):
        for sentinel_first in (False, True):
            with self.subTest(sentinel_first=sentinel_first):
                pair = [[80, 20.001], [0, 0]]
                if sentinel_first:
                    pair.reverse()
                data, _ = self.read_crrel(
                    [0, 3600, 3600, 7200, 10800, 14400],
                    [[80, 20], *pair, [0, 20], [80, 0], [-90, -180]],
                )
                np.testing.assert_allclose(data.positions.coords, [
                    [80, 20], [80, 20.001], [0, 20], [80, 0], [-90, -180],
                ])
                self.assertFalse(any(d["kind"] == "conflicting_coordinates"
                                     for d in data.metadata["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
