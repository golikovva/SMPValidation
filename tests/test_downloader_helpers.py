from datetime import datetime
from pathlib import Path
import unittest

import pandas as pd

from meteostat_download import (
    MANIFEST_COLUMNS,
    RP5_MAP_COLUMNS,
    _load_table,
    canonical_output_path,
    has_mapped_rp5_url,
    hourly_inventory_overlaps,
    migrate_loaded_tables,
    rp5_map_row,
    sanitize_station_filename_component,
    should_skip_manifest_record,
    unresolved_rp5_report_dataframe,
    unresolved_rp5_station_ids,
)
from rp5_download import (
    CANONICAL_COLUMNS,
    RP5CallBudget,
    RP5CallBudgetExceeded,
    _extract_download_link,
    _to_rp5_date,
    is_blocking_response,
    normalize_rp5_dataframe,
    resolve_rp5_station_page_from_meta,
    rp5_search_result_url,
)


class DownloaderHelperTests(unittest.TestCase):
    def test_rp5_date_and_download_link_helpers(self):
        self.assertEqual(_to_rp5_date("2024-01-31"), "31.01.2024")
        self.assertEqual(_to_rp5_date("31.01.2024"), "31.01.2024")
        self.assertEqual(
            _extract_download_link('<a href="https://rp5.ru/archive/example.csv.gz">csv</a>'),
            "https://rp5.ru/archive/example.csv.gz",
        )

    def test_rp5_call_budget_counts_and_stops_without_sleep(self):
        budget = RP5CallBudget(max_calls=2, min_delay=0, max_delay=0)
        budget.consume("first")
        budget.consume("second")

        self.assertEqual(budget.calls_made, 2)
        self.assertEqual(budget.remaining, 0)
        with self.assertRaises(RP5CallBudgetExceeded):
            budget.consume("third")

    def test_rp5_block_detection(self):
        self.assertTrue(is_blocking_response(429, ""))
        self.assertTrue(is_blocking_response(403, ""))
        self.assertTrue(is_blocking_response(200, "Error #FS000"))
        self.assertFalse(is_blocking_response(200, "archive is ready"))

    def test_rp5_json_search_matches_wmo_without_leading_zero(self):
        payload = [
            {
                "name": "2789 Rautavaara Yla-luosta in Northern Savonia",
                "id": "2789",
                "namealt": "Weather_archive_in_Rautavaara_Yla-luosta",
            }
        ]
        self.assertEqual(
            rp5_search_result_url(payload, "02789"),
            "https://rp5.ru/Weather_archive_in_Rautavaara_Yla-luosta",
        )

    def test_rp5_json_search_is_first_resolution_strategy(self):
        class FakeResponse:
            status_code = 200
            text = "json response"

            @staticmethod
            def json():
                return [
                    {
                        "id": "2789",
                        "namealt": "Weather_archive_in_Rautavaara_Yla-luosta",
                    }
                ]

            @staticmethod
            def raise_for_status():
                return None

        class FakeSession:
            def __init__(self):
                self.calls = []

            def request(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                return FakeResponse()

        session = FakeSession()
        budget = RP5CallBudget(max_calls=1, min_delay=0, max_delay=0)
        url, page_bootstrapped = resolve_rp5_station_page_from_meta(
            {
                "id": "02789",
                "identifiers": {"wmo": "02789"},
                "name": {"en": "Unhelpful Meteostat Name"},
            },
            session,
            budget=budget,
        )

        self.assertEqual(
            url,
            "https://rp5.ru/Weather_archive_in_Rautavaara_Yla-luosta",
        )
        self.assertFalse(page_bootstrapped)
        self.assertEqual(budget.calls_made, 1)
        self.assertEqual(session.calls[0][2]["params"]["q"], "02789")

    def test_canonical_output_path_preserves_leading_zero_station_id(self):
        path = canonical_output_path(
            Path("stations_meteostat/nestp"),
            "01006",
            "Edgeoya",
            datetime(2024, 1, 1),
            datetime(2024, 12, 31, 23, 59),
        )
        self.assertEqual(
            path,
            Path(
                "stations_meteostat/nestp/2024/"
                "01006_Edgeoya_2024-01-01_2024-12-31.csv"
            ),
        )

    def test_station_name_is_safe_for_windows_filename(self):
        self.assertEqual(
            sanitize_station_filename_component('Cape / Test: "North"'),
            "Cape-Test-North",
        )

    def test_hourly_inventory_overlap(self):
        meta = {"inventory": {"hourly": {"start": "2020-01-01", "end": "2024-01-31"}}}
        self.assertTrue(
            hourly_inventory_overlaps(meta, datetime(2024, 1, 1), datetime(2024, 1, 2))
        )
        self.assertFalse(
            hourly_inventory_overlaps(meta, datetime(2025, 1, 1), datetime(2025, 1, 2))
        )

    def test_normalize_rp5_dataframe_to_canonical_columns(self):
        raw = pd.DataFrame(
            {
                "Local time in Test": ["01.01.2024 03:00"],
                "T": ["-1,5"],
                "Td": ["-3"],
                "U": ["80"],
                "RRR": ["Trace of precipitation"],
                "DD": ["Wind blowing from the north-east"],
                "Ff": ["2"],
                "ff10": ["4"],
                "P": ["1001,2"],
                "WW": ["10"],
            }
        )

        out = normalize_rp5_dataframe(
            raw,
            timezone="Europe/Moscow",
            start=datetime(2024, 1, 1, 0),
            end=datetime(2024, 1, 1, 0),
        )

        self.assertEqual(list(out.columns), CANONICAL_COLUMNS)
        self.assertEqual(out.loc[0, "time"], pd.Timestamp("2024-01-01 00:00:00"))
        self.assertAlmostEqual(out.loc[0, "temp"], -1.5)
        self.assertAlmostEqual(out.loc[0, "dwpt"], -3.0)
        self.assertAlmostEqual(out.loc[0, "rhum"], 80.0)
        self.assertAlmostEqual(out.loc[0, "prcp"], 0.0)
        self.assertAlmostEqual(out.loc[0, "wdir"], 45.0)
        self.assertAlmostEqual(out.loc[0, "wspd"], 7.2)
        self.assertAlmostEqual(out.loc[0, "wpgt"], 14.4)
        self.assertAlmostEqual(out.loc[0, "pres"], 1001.2)
        self.assertEqual(out.loc[0, "temp_source"], "rp5")

    def test_normalize_repairs_rp5_time_index_and_shifted_columns(self):
        local_time_name = (
            "\u041c\u0435\u0441\u0442\u043d\u043e\u0435 "
            "\u0432\u0440\u0435\u043c\u044f"
        )
        raw = pd.DataFrame(
            [
                [
                    -2.1,
                    756.4,
                    757.6,
                    None,
                    88,
                    "\u0412\u0435\u0442\u0435\u0440, "
                    "\u0434\u0443\u044e\u0449\u0438\u0439 \u0441 "
                    "\u044e\u0433\u0430",
                    4,
                    7,
                    5,
                    -3.2,
                    None,
                ]
            ],
            index=["30.04.2026 02:00"],
            columns=[
                local_time_name,
                "T",
                "Po",
                "P",
                "Pa",
                "U",
                "DD",
                "Ff",
                "ff10",
                "Td",
                "RRR",
            ],
        )

        out = normalize_rp5_dataframe(raw, timezone="Europe/Oslo")

        self.assertEqual(out.loc[0, "time"], pd.Timestamp("2026-04-30 00:00:00"))
        self.assertAlmostEqual(out.loc[0, "temp"], -2.1)
        self.assertAlmostEqual(out.loc[0, "pres"], 757.6)
        self.assertAlmostEqual(out.loc[0, "rhum"], 88)
        self.assertAlmostEqual(out.loc[0, "wdir"], 180.0)
        self.assertAlmostEqual(out.loc[0, "wspd"], 14.4)

    def test_rp5_map_rows_include_manual_context_columns(self):
        meta = {
            "id": "01011",
            "name": {"en": "Kvitoya"},
            "country": "NO",
            "region": "",
            "identifiers": {"wmo": "01011"},
            "location": {"latitude": 80.1, "longitude": 31.4},
        }
        table = pd.DataFrame(
            [
                rp5_map_row(
                    station_meta=meta,
                    rp5_url="https://rp5.ru/Weather_archive_in_Kvitoya",
                    status="mapped",
                )
            ],
            columns=RP5_MAP_COLUMNS,
        )

        self.assertEqual(list(table.columns), RP5_MAP_COLUMNS)
        self.assertEqual(table.loc[0, "name"], "Kvitoya")
        self.assertEqual(table.loc[0, "country"], "NO")
        self.assertEqual(table.loc[0, "latitude"], "80.1")

    def test_existing_map_migrates_without_losing_manual_url(self):
        meta = {
            "id": "01011",
            "name": {"en": "Kvitoya"},
            "country": "NO",
            "region": "",
            "identifiers": {"wmo": "01011"},
            "location": {"latitude": 80.1, "longitude": 31.4},
        }
        manifest_path = Path("does-not-exist-download_manifest.csv")
        map_path = Path("does-not-exist-rp5_station_map.csv")
        manifest = _load_table(manifest_path, MANIFEST_COLUMNS)
        rp5_map = pd.DataFrame(
            [
                {
                    "station_id": "01011",
                    "wmo_id": "01011",
                    "rp5_url": "https://rp5.ru/Weather_archive_in_Kvitoya",
                    "status": "mapped",
                    "last_error": "",
                    "updated_at": "2026-04-30T00:00:00+00:00",
                }
            ]
        )
        for column in RP5_MAP_COLUMNS:
            if column not in rp5_map.columns:
                rp5_map[column] = ""
        rp5_map = rp5_map[RP5_MAP_COLUMNS]
        _, migrated_map = migrate_loaded_tables(
            manifest_path=manifest_path,
            manifest=manifest,
            rp5_map_path=map_path,
            rp5_map=rp5_map,
            metadata=[meta],
        )

        self.assertEqual(
            migrated_map.loc[0, "rp5_url"],
            "https://rp5.ru/Weather_archive_in_Kvitoya",
        )
        self.assertEqual(migrated_map.loc[0, "name"], "Kvitoya")
        self.assertEqual(migrated_map.loc[0, "country"], "NO")

    def test_unresolved_manifest_reopens_only_when_map_is_manually_mapped(self):
        record = pd.Series(
            {
                "station_id": "01011",
                "status": "unresolved_rp5",
                "path": "",
            }
        )
        unresolved_map = pd.DataFrame(
            [
                {
                    "station_id": "01011",
                    "status": "unresolved",
                    "rp5_url": "",
                }
            ]
        )
        mapped_map = pd.DataFrame(
            [
                {
                    "station_id": "01011",
                    "status": "mapped",
                    "rp5_url": "https://rp5.ru/Weather_archive_in_Kvitoya",
                }
            ]
        )

        self.assertTrue(should_skip_manifest_record(record))
        self.assertFalse(has_mapped_rp5_url(unresolved_map, "01011"))
        self.assertTrue(has_mapped_rp5_url(mapped_map, "01011"))

    def test_unresolved_retry_selection_excludes_successful_stations(self):
        manifest = pd.DataFrame(
            [
                {"station_id": "01011", "status": "unresolved_rp5"},
                {"station_id": "01006", "status": "rp5_saved"},
            ]
        )
        rp5_map = pd.DataFrame(
            [
                {"station_id": "02789", "status": "unresolved"},
                {"station_id": "01006", "status": "mapped"},
            ]
        )
        self.assertEqual(
            unresolved_rp5_station_ids(manifest, rp5_map),
            {"01011", "02789"},
        )

    def test_unresolved_report_contains_manual_fill_columns(self):
        meta = {
            "id": "01011",
            "name": {"en": "Kvitoya"},
            "country": "NO",
            "region": "",
            "identifiers": {"wmo": "01011"},
            "location": {"latitude": 80.1, "longitude": 31.4},
        }
        manifest = pd.DataFrame(
            [{"station_id": "01011", "status": "unresolved_rp5"}]
        )
        rp5_map = pd.DataFrame(
            [{"station_id": "01011", "status": "unresolved", "rp5_url": ""}]
        )

        report = unresolved_rp5_report_dataframe(
            metadata=[meta],
            manifest=manifest,
            rp5_map=rp5_map,
        ).fillna("")

        self.assertEqual(report.loc[0, "station_id"], "01011")
        self.assertEqual(report.loc[0, "name"], "Kvitoya")
        self.assertEqual(report.loc[0, "country"], "NO")
        self.assertEqual(report.loc[0, "rp5_url"], "")


if __name__ == "__main__":
    unittest.main()
