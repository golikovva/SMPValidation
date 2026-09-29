"""CLI selection and backwards-compatibility checks for the IABP downloader."""

from contextlib import redirect_stderr
from datetime import date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch

import pandas as pd

import iabp_buoy_download as downloader


class IABPDownloadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({"BuoyID": ["300234", "300235"]})
        self.fetch = self._patch("fetch_arctic_table", return_value=self.table)
        self.filter_recent = self._patch("filter_recent_buoys", return_value=self.table)
        self.download_current = self._patch("download_buoy_data")
        self.download_historical = self._patch("download_historical_data")
        request_patch = patch.object(
            downloader.requests,
            "get",
            side_effect=AssertionError("CLI tests must not make network requests"),
        )
        self.request = request_patch.start()
        self.addCleanup(request_patch.stop)

    def _patch(self, name, **kwargs):
        patcher = patch.object(downloader, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def assert_historical_selection(self, argv, years, start_date=None, end_date=None):
        downloader.main(argv)

        self.download_historical.assert_called_once_with(
            years,
            "selected-data",
            start_date=start_date,
            end_date=end_date,
        )
        self.fetch.assert_not_called()
        self.filter_recent.assert_not_called()
        self.download_current.assert_not_called()
        self.request.assert_not_called()

    def test_years_are_sorted_and_deduplicated(self):
        self.assert_historical_selection(
            ["--years", "2024", "2022", "2024", "--output-dir", "selected-data"],
            [2022, 2024],
        )

    def test_cross_year_date_range_selects_every_touched_year(self):
        self.assert_historical_selection(
            [
                "--start-date", "2022-12-31",
                "--end-date", "2024-01-01",
                "--output-dir", "selected-data",
            ],
            [2022, 2023, 2024],
            start_date=date(2022, 12, 31),
            end_date=date(2024, 1, 1),
        )

    def test_underscore_date_aliases_accept_single_leap_day(self):
        self.assert_historical_selection(
            [
                "--start_date", "2024-02-29",
                "--end_date", "2024-02-29",
                "--output-dir", "selected-data",
            ],
            [2024],
            start_date=date(2024, 2, 29),
            end_date=date(2024, 2, 29),
        )

    def test_default_mode_keeps_last_seven_days_and_default_output(self):
        downloader.main([])

        self.fetch.assert_called_once_with()
        self.filter_recent.assert_called_once_with(self.table, days=7)
        self.download_current.assert_called_once_with(
            ["300234", "300235"], "./itp_data"
        )
        self.download_historical.assert_not_called()
        self.request.assert_not_called()

    def test_days_mode_keeps_custom_window_and_output(self):
        downloader.main(["--days", "30", "--output-dir", "recent-data"])

        self.fetch.assert_called_once_with()
        self.filter_recent.assert_called_once_with(self.table, days=30)
        self.download_current.assert_called_once_with(
            ["300234", "300235"], "recent-data"
        )
        self.download_historical.assert_not_called()
        self.request.assert_not_called()

    def test_no_recent_buoys_does_not_download(self):
        self.filter_recent.return_value = pd.DataFrame({"BuoyID": []})

        downloader.main([])

        self.fetch.assert_called_once_with()
        self.filter_recent.assert_called_once_with(self.table, days=7)
        self.download_current.assert_not_called()
        self.download_historical.assert_not_called()
        self.request.assert_not_called()

    def test_invalid_selection_fails_before_fetching_data(self):
        cases = [
            ["--years"],
            ["--years", "0"],
            ["--years", "-1"],
            ["--years", "10000"],
            ["--years", "2024.5"],
            ["--years", "not-a-year"],
            ["--start-date", "2024-01-01"],
            ["--end-date", "2024-01-01"],
            ["--start-date", "2024-02-30", "--end-date", "2024-03-01"],
            ["--start-date", "2023-02-29", "--end-date", "2023-03-01"],
            ["--start-date", "2024-01-01", "--end-date", "not-a-date"],
            ["--start-date", "2024-1-1", "--end-date", "2024-01-02"],
            ["--start-date", "20240101", "--end-date", "2024-01-02"],
            ["--start-date", "2024-02-01", "--end-date", "2024-01-31"],
            [
                "--years", "2024",
                "--start-date", "2024-01-01", "--end-date", "2024-12-31",
            ],
            ["--years", "2024", "--start-date", "2024-01-01"],
            ["--years", "2024", "--days", "7"],
            [
                "--start-date", "2024-01-01", "--end-date", "2024-12-31",
                "--days", "7",
            ],
            ["--days", "0"],
            ["--days", "-1"],
            ["--days", "1.5"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    downloader.main(argv)

                self.assertEqual(error.exception.code, 2)
                self.assertIn("error:", stderr.getvalue())
                self.fetch.assert_not_called()
                self.filter_recent.assert_not_called()
                self.download_current.assert_not_called()
                self.download_historical.assert_not_called()
                self.request.assert_not_called()


class IABPDownloadDataTests(unittest.TestCase):
    def test_table_parser_keeps_every_buoy_and_pads_historical_columns(self):
        response = Mock(text=(
            "101; 48001; 2020; SVP; IABP; Polarstern; 02/29/2024 23:30:00; 80.1; -120.2\n"
            "102; 48002; 2024; SVP; IABP; Polarstern; 03/01/2024 00:00:00; 80.2; -120.3; 1013.2; -1.8; -15.0\n"
            "\n"
            "103; 48003; 2020; SVP; IABP; Polarstern; unknown; 80.3; -120.4\n"
        ))
        url = f"{downloader.TABLES_BASE}ArcticTable_2020.txt"
        with patch.object(downloader.requests, "get", return_value=response):
            table = downloader.fetch_arctic_table(url)

        self.assertEqual(table["BuoyID"].tolist(), ["101", "102", "103"])
        self.assertEqual(table.loc[0, ["BP", "Ts", "Ta"]].tolist(), ["", "", ""])
        self.assertEqual(table.loc[1, ["BP", "Ts", "Ta"]].tolist(), ["1013.2", "-1.8", "-15.0"])
        self.assertEqual(table.loc[0, "LastReport_dt"], pd.Timestamp("2024-02-29 23:30:00"))
        self.assertTrue(pd.isna(table.loc[2, "LastReport_dt"]))

    def test_selected_years_preserve_header_and_raw_rows(self):
        header = b"BuoyID\tYear  DOY POS_DOY Lat\r\n"
        first = b"123\t2022  365.9999 365.5 81.000\r\n"
        skipped = b"123 2023 1.0 1.0 80.1\r\n"
        last = b"123 2024 366.75 366.5 79.00"

        result = downloader.select_buoy_records(
            header + first + skipped + last, [2022, 2024]
        )

        self.assertEqual(result, {2022: header + first, 2024: header + last})

    def test_leap_day_range_includes_full_day_and_uses_observation_doy(self):
        header = b"Year POS_DOY DOY\n"
        before = b"2024 60.0 59.9999\n"
        first = b"2024 59.0 60.0\n"
        last = b"2024 59.5 60.9999\n"
        after = b"2024 60.0 61.0\n"

        result = downloader.select_buoy_records(
            header + before + first + last + after,
            [2024],
            start_date=date(2024, 2, 29),
            end_date=date(2024, 2, 29),
        )

        self.assertEqual(result, {2024: header + first + last})

    def test_empty_and_out_of_range_data_have_no_selected_years(self):
        for content in (b"", b" \n", b"Year DOY\n", b"Year DOY\n2023 1.0\n"):
            with self.subTest(content=content):
                self.assertEqual(downloader.select_buoy_records(content, [2024]), {})

    def test_invalid_header_or_observation_date_is_reported(self):
        cases = [
            b"Year POS_DOY\n2024 60\n",
            b"BuoyID DOY\n123 60\n",
            b"Year DOY\nnot-a-year 1\n",
            b"Year DOY\n2024\n",
            b"Year DOY\n2024 NaN\n",
            b"Year DOY\n2024 inf\n",
            b"Year DOY\n2024 0.99\n",
            b"Year DOY\n2024 367\n",
            b"Year DOY\n2023 366\n",
        ]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                downloader.select_buoy_records(content, [2023, 2024])

    @staticmethod
    def _table(rows):
        table = pd.DataFrame(rows, columns=["BuoyID", "Year", "LastReport_dt"])
        table["LastReport_dt"] = pd.to_datetime(table["LastReport_dt"])
        return table

    def test_candidates_include_older_deployments_and_later_last_reports(self):
        listing = Mock(text="".join(
            f'<a href="ArcticTable_{year}.txt">{year}</a>'
            for year in [2010, 2023, 2024, 2025]
        ))
        tables = [
            self._table([
                ("101", 2010, "2025-06-01"),
                ("102", 2010, "2024-02-29"),
                ("103", 2010, None),
            ]),
            self._table([("104", 2023, "2024-03-01")]),
            self._table([("105", 2024, "2024-04-01")]),
            self._table([
                ("101", 2010, "2025-06-01"),
                ("106", 2025, "2025-06-01"),
            ]),
        ]
        with patch.object(downloader.requests, "get", return_value=listing) as get, \
                patch.object(downloader, "fetch_arctic_table", side_effect=tables) as fetch:
            result = downloader.historical_buoy_ids(
                [2024], start_date=date(2024, 3, 1)
            )

        self.assertEqual(result, ["101", "103", "104", "105"])
        self.assertEqual(get.call_args.args, (downloader.TABLES_BASE,))
        listing.raise_for_status.assert_called_once_with()
        self.assertEqual(fetch.call_args_list, [
            call(f"{downloader.TABLES_BASE}ArcticTable_2010.txt"),
            call(f"{downloader.TABLES_BASE}ArcticTable_2023.txt"),
            call(f"{downloader.TABLES_BASE}ArcticTable_2024.txt"),
            call(),
        ])

    def test_missing_archive_coverage_fails_before_table_downloads(self):
        listing = '<a href="ArcticTable_2023.txt">2023</a>'
        for years, text in (([2022], listing), ([2024], listing), ([2023], "")):
            with self.subTest(years=years, listing=text), \
                    patch.object(downloader.requests, "get", return_value=Mock(text=text)), \
                    patch.object(downloader, "fetch_arctic_table") as fetch:
                with self.assertRaises(ValueError):
                    downloader.historical_buoy_ids(years)
                fetch.assert_not_called()

    def test_historical_download_writes_only_selected_years_and_nonempty_files(self):
        header = b"Year DOY\n"
        first = b"2022 365.5\n"
        last = b"2024 1.5\n"
        responses = [
            Mock(content=header + first + b"2023 100\n" + last),
            Mock(content=header + b"2023 1\n"),
        ]
        with TemporaryDirectory() as directory, \
                patch.object(downloader, "historical_buoy_ids", return_value=["101", "102"]), \
                patch.object(downloader.requests, "get", side_effect=responses) as get:
            downloader.download_historical_data([2022, 2024], directory)

            root = Path(directory)
            self.assertEqual((root / "2022" / "101.dat").read_bytes(), header + first)
            self.assertEqual((root / "2024" / "101.dat").read_bytes(), header + last)
            self.assertEqual(len(list(root.rglob("*.dat"))), 2)
            self.assertFalse((root / "2023").exists())
            self.assertEqual(
                [request.args[0] for request in get.call_args_list],
                [f"{downloader.WEBDATA_BASE}{buoy_id}.dat" for buoy_id in ["101", "102"]],
            )

    def test_historical_download_applies_date_bounds_before_writing(self):
        header = b"Year DOY\n"
        selected = b"2024 60.9999\n"
        response = Mock(content=header + b"2024 59\n" + selected + b"2024 61\n")
        with TemporaryDirectory() as directory, \
                patch.object(downloader, "historical_buoy_ids", return_value=["101"]) as history, \
                patch.object(downloader.requests, "get", return_value=response):
            downloader.download_historical_data(
                [2024], directory, start_date=date(2024, 2, 29), end_date=date(2024, 2, 29)
            )

            history.assert_called_once_with([2024], start_date=date(2024, 2, 29))
            self.assertEqual(
                (Path(directory) / "2024" / "101.dat").read_bytes(), header + selected
            )

    def test_failed_download_is_reported_after_remaining_buoys_are_processed(self):
        failures = [
            downloader.requests.HTTPError("archive unavailable"),
            Mock(content=b"invalid data\n"),
        ]
        content = b"Year DOY\n2024 60\n"
        for failure in failures:
            with self.subTest(failure=failure), TemporaryDirectory() as directory, \
                    patch.object(downloader, "historical_buoy_ids", return_value=["101", "102"]), \
                    patch.object(downloader.requests, "get", side_effect=[failure, Mock(content=content)]), \
                    self.assertLogs(level="ERROR") as logs:
                with self.assertRaisesRegex(RuntimeError, "1 buoys failed"):
                    downloader.download_historical_data([2024], directory)

                self.assertIn("101.dat", "\n".join(logs.output))
                self.assertFalse((Path(directory) / "2024" / "101.dat").exists())
                self.assertEqual((Path(directory) / "2024" / "102.dat").read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
