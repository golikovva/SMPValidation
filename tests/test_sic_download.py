"""Offline selection, NetCDF validation, and safe-resume checks for SIC downloads."""

import contextlib
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import sic_download as downloader

try:
    import h5netcdf  # noqa: F401
    import numpy as np
    import xarray as xr
except ImportError:
    xr = None


def parse(*options):
    return downloader.build_parser().parse_args(list(options))


class SICDateSelectionTests(unittest.TestCase):
    def test_year_ranges_and_months_include_leap_day_without_duplicates(self):
        args = parse("--years", "2020-2021", "2021:2022", "--months", "2")
        days = downloader.selected_dates(args)
        self.assertEqual(len(days), 29 + 28 + 28)
        self.assertEqual(days[0], date(2020, 2, 1))
        self.assertEqual(days[-1], date(2022, 2, 28))
        self.assertIn(date(2020, 2, 29), days)
        self.assertEqual(days, sorted(set(days)))

    def test_dates_are_sorted_and_deduplicated(self):
        args = parse("--dates", "2024-10-02", "2024-02-29", "2024-10-02")
        self.assertEqual(
            downloader.selected_dates(args),
            [date(2024, 2, 29), date(2024, 10, 2)],
        )

    def test_date_range_includes_both_endpoints_across_years(self):
        args = parse("--start-date", "2023-12-31", "--end-date", "2024-01-02")
        self.assertEqual(
            downloader.selected_dates(args),
            [date(2023, 12, 31), date(2024, 1, 1), date(2024, 1, 2)],
        )

    def test_invalid_selections_are_rejected(self):
        invalid = [
            ("--dates", "2023-02-29"),
            ("--years", "2022-2020"),
            ("--years", "2024", "--months", "0"),
            ("--years", "2024", "--months", "13"),
            ("--dates", "2024-10-02", "--months", "10"),
            ("--start-date", "2024-10-02"),
            ("--start-date", "2024-10-02", "--end-date", "2024-10-01"),
            ("--dates", "2024-10-02", "--end-date", "2024-10-03"),
        ]
        for options in invalid:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError, SystemExit)):
                    downloader.selected_dates(parse(*options))

    def test_selection_is_required_and_modes_are_mutually_exclusive(self):
        for options in [(), ("--years", "2024", "--dates", "2024-10-02")]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse(*options)

    def test_request_selects_all_four_variables_and_whole_day(self):
        day = date(2024, 10, 2)
        request = downloader.request_for_day(parse("--dates", day.isoformat()), day, "202411")
        self.assertEqual(request["dataset_id"], "cmems_obs-si_arc_phy_my_l3_P1D")
        self.assertEqual(request["dataset_part"], "lowResolution")
        self.assertEqual(request["dataset_version"], "202411")
        self.assertEqual(
            set(request["variables"]),
            {"acq_time", "sic", "status_flag", "standard_deviation_sic"},
        )
        self.assertEqual(request["minimum_longitude"], -179.994)
        self.assertEqual(request["maximum_longitude"], 179.994)
        self.assertAlmostEqual(request["minimum_latitude"], 31.002)
        self.assertEqual(request["maximum_latitude"], 89.994)
        start = datetime.fromisoformat(str(request["start_datetime"])).replace(tzinfo=None)
        end = datetime.fromisoformat(str(request["end_datetime"])).replace(tzinfo=None)
        self.assertEqual(start, datetime(2024, 10, 2))
        self.assertEqual(end.date(), day)
        self.assertGreaterEqual(end, datetime(2024, 10, 2, 23, 59, 59))


@unittest.skipIf(xr is None, "numpy, xarray, and h5netcdf are needed for NetCDF checks")
class SICDownloadFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.day = date(2024, 10, 2)
        self.args = parse("--dates", self.day.isoformat(), "--attempts", "2", "--retry-wait", "0")

    def write_dataset(self, path, day=None, variables=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        variables = downloader.VARIABLES if variables is None else variables
        dataset = xr.Dataset(
            {name: (("time", "y", "x"), np.ones((1, 2, 2))) for name in variables},
            coords={
                "time": [np.datetime64(f"{day or self.day}T23:06:40")],
                "y": [0, 1],
                "x": [0, 1],
            },
        )
        dataset.to_netcdf(path, engine="h5netcdf")

    def successful_subset(self, **kwargs):
        path = Path(kwargs["output_directory"]) / kwargs["output_filename"]
        self.write_dataset(path)
        return SimpleNamespace(file_path=path)

    def test_verification_accepts_valid_file_and_rejects_wrong_day_or_missing_variable(self):
        path = self.root / "sample.nc"
        self.write_dataset(path)
        downloader.verify_file(path, self.day)
        self.write_dataset(path, day=date(2024, 10, 3))
        with self.assertRaises(ValueError):
            downloader.verify_file(path, self.day)
        self.write_dataset(path, variables=["sic"])
        with self.assertRaises(ValueError):
            downloader.verify_file(path, self.day)
        path.write_bytes(b"interrupted download")
        with self.assertRaises((ValueError, OSError)):
            downloader.verify_file(path, self.day)

    def test_matching_file_and_request_are_resumed_without_another_download(self):
        cm = SimpleNamespace(subset=Mock(side_effect=self.successful_subset))
        self.assertEqual(downloader.download_day(cm, self.args, self.day, "202411", self.root), "downloaded")
        self.assertEqual(len(list(self.root.rglob("*.nc"))), 1)
        self.assertEqual(len(list(self.root.rglob("*.json"))), 1)
        self.assertEqual(downloader.download_day(cm, self.args, self.day, "202411", self.root), "skipped")
        self.assertEqual(cm.subset.call_count, 1)

    def test_changed_request_requires_overwrite(self):
        cm = SimpleNamespace(subset=Mock(side_effect=self.successful_subset))
        downloader.download_day(cm, self.args, self.day, "202411", self.root)
        with self.assertRaises((ValueError, RuntimeError)):
            downloader.download_day(cm, self.args, self.day, "202501", self.root)
        self.assertEqual(cm.subset.call_count, 1)
        self.args.overwrite = True
        self.assertEqual(downloader.download_day(cm, self.args, self.day, "202501", self.root), "downloaded")
        self.assertEqual(cm.subset.call_count, 2)

    def test_failed_overwrite_preserves_existing_download_and_metadata(self):
        cm = SimpleNamespace(subset=Mock(side_effect=self.successful_subset))
        downloader.download_day(cm, self.args, self.day, "202411", self.root)
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.args.overwrite = True

        def partial_download(**kwargs):
            path = Path(kwargs["output_directory"]) / kwargs["output_filename"]
            path.write_bytes(b"incomplete")
            raise OSError("connection interrupted")

        cm.subset = Mock(side_effect=partial_download)
        with self.assertRaises((OSError, RuntimeError)):
            downloader.download_day(cm, self.args, self.day, "202501", self.root)
        self.assertEqual(cm.subset.call_count, self.args.attempts)
        remaining = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(remaining, original)

    def test_wrong_day_response_never_becomes_completed_output(self):
        def wrong_day_subset(**kwargs):
            path = Path(kwargs["output_directory"]) / kwargs["output_filename"]
            self.write_dataset(path, day=date(2024, 10, 3))
            return SimpleNamespace(file_path=path)

        cm = SimpleNamespace(subset=Mock(side_effect=wrong_day_subset))
        with self.assertRaises((ValueError, RuntimeError)):
            downloader.download_day(cm, self.args, self.day, "202411", self.root)
        self.assertFalse(any(path.is_file() for path in self.root.rglob("*")))


class SICCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = parse("--start-date", "2024-10-01", "--end-date", "2024-10-03")
        self.days = downloader.selected_dates(self.args)
        self.coordinate = SimpleNamespace(
            coordinate_id="time",
            coordinate_unit="milliseconds since 1970-01-01 00:00:00 (no leap seconds)",
            values=[
                datetime(2024, 10, day, 23, 6, 40, tzinfo=timezone.utc).timestamp() * 1000
                for day in [1, 3]
            ],
        )

    def catalogue(self, *versions):
        service = SimpleNamespace(
            service_name="arco-geo-series",
            variables=[
                SimpleNamespace(short_name=name, coordinates=[self.coordinate])
                for name in downloader.VARIABLES
            ],
        )
        dataset = SimpleNamespace(
            dataset_id=downloader.DATASET_ID,
            versions=[
                SimpleNamespace(label=version, parts=[SimpleNamespace(name=downloader.DATASET_PART, services=[service])])
                for version in versions
            ],
        )
        return SimpleNamespace(products=[SimpleNamespace(datasets=[dataset])])

    def test_irregular_millisecond_values_keep_gaps(self):
        self.assertEqual(
            downloader.catalogue_dates(self.coordinate),
            {date(2024, 10, 1), date(2024, 10, 3)},
        )

    def test_missing_catalogue_day_stops_before_any_subset(self):
        cm = SimpleNamespace(describe=Mock(return_value=self.catalogue("202411")), subset=Mock())
        with patch.dict(sys.modules, {"copernicusmarine": cm, "xarray": SimpleNamespace(), "h5netcdf": SimpleNamespace()}):
            result = downloader.main([
                "--start-date", "2024-10-01", "--end-date", "2024-10-03", "--output", str(self.root),
            ])
        self.assertEqual(result, 1)
        cm.subset.assert_not_called()
        report = json.loads((self.root / "availability.json").read_text(encoding="utf-8"))
        self.assertEqual(report["unavailable_dates"], ["2024-10-02"])

    def test_allow_incomplete_returns_only_available_dates_and_pins_version(self):
        cm = SimpleNamespace(describe=Mock(return_value=self.catalogue("202411")))
        self.args.allow_incomplete = True
        version, selected, missing = downloader.resolve_dataset(cm, self.args, self.root, self.days)
        self.assertEqual(version, "202411")
        self.assertEqual(selected, [date(2024, 10, 1), date(2024, 10, 3)])
        self.assertEqual(missing, [date(2024, 10, 2)])
        lock = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["dataset_version"], "202411")
        cm.describe.return_value = self.catalogue("202501", "202411")
        version, selected, missing = downloader.resolve_dataset(cm, self.args, self.root, self.days)
        self.assertEqual(version, "202411")
        self.assertTrue(cm.describe.call_args.kwargs["show_all_versions"])

    def test_explicit_version_cannot_change_an_existing_output_dataset(self):
        cm = SimpleNamespace(describe=Mock(return_value=self.catalogue("202411")))
        downloader.resolve_dataset(cm, self.args, self.root, [self.days[0]])
        self.args.dataset_version = "202501"
        with self.assertRaises(downloader.ConfigurationError):
            downloader.resolve_dataset(cm, self.args, self.root, [self.days[0]])
        self.assertEqual(cm.describe.call_count, 1)

    def test_dry_run_requires_no_remote_library_or_output_directory(self):
        output = self.root / "not_created"
        printed = io.StringIO()
        with patch.dict(sys.modules, {"copernicusmarine": None, "xarray": None, "h5netcdf": None}), contextlib.redirect_stdout(printed):
            result = downloader.main(["--dates", "2024-10-02", "--dry-run", "--output", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(printed.getvalue())["dates"], ["2024-10-02"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
