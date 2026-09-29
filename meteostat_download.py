from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from contextlib import nullcontext
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rp5_download import (
    CANONICAL_COLUMNS,
    RP5BlockedError,
    RP5CallBudget,
    RP5CallBudgetExceeded,
    _make_session,
    download_rp5_synop_by_id,
    normalize_rp5_csv_gz,
    resolve_rp5_station_page_from_meta,
)


DEFAULT_METADATA = Path("metadata/nestp_stations_metadata.json")
DEFAULT_OUT_DIR = Path("stations_meteostat/nestp")

MANIFEST_COLUMNS = [
    "station_id",
    "wmo_id",
    "name",
    "country",
    "year",
    "start",
    "end",
    "status",
    "source",
    "path",
    "rows",
    "attempts",
    "last_error",
    "updated_at",
]

RP5_MAP_COLUMNS = [
    "station_id",
    "wmo_id",
    "name",
    "country",
    "region",
    "latitude",
    "longitude",
    "rp5_url",
    "status",
    "last_error",
    "updated_at",
]

TERMINAL_STATUSES = {
    "meteostat_saved",
    "rp5_saved",
    "existing_saved",
    "no_data",
    "unresolved_rp5",
}

NON_TERMINAL_RP5_STATUSES = {
    "pending_rp5_budget",
    "pending_rp5_blocked",
}


def utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def station_id(station_meta: dict[str, Any]) -> str:
    return str(station_meta["id"])


def station_wmo(station_meta: dict[str, Any]) -> str:
    identifiers = station_meta.get("identifiers", {})
    wmo = identifiers.get("wmo")
    if wmo:
        return str(wmo)
    sid = station_id(station_meta)
    return sid if sid.isdigit() else ""


def station_name(station_meta: dict[str, Any]) -> str:
    name = station_meta.get("name", {})
    if isinstance(name, dict):
        return str(name.get("en") or next(iter(name.values()), station_id(station_meta)))
    if name:
        return str(name)
    return station_id(station_meta)


def station_country(station_meta: dict[str, Any]) -> str:
    return str(station_meta.get("country") or "")


def station_region(station_meta: dict[str, Any]) -> str:
    return str(station_meta.get("region") or "")


def station_location_value(station_meta: dict[str, Any], key: str) -> str:
    value = station_meta.get("location", {}).get(key)
    return "" if value is None else str(value)


def parse_date_arg(value: str, *, end_of_day: bool = False) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt == "%Y-%m-%d" and end_of_day:
                return datetime.combine(parsed.date(), datetime_time(23, 59))
            return parsed
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Unsupported date format {value!r}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM."
    )


def resolve_period(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.start:
        start = parse_date_arg(args.start)
    else:
        start = datetime(args.year, 1, 1)

    if args.end:
        end = parse_date_arg(args.end, end_of_day=True)
    else:
        end = datetime(args.year, 12, 31, 23, 59)

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    end = min(end, now_utc)
    if end < start:
        raise ValueError(f"End date {end} is earlier than start date {start}")
    return start, end


def canonical_output_path(
    out_dir: Path,
    station_id_value: str,
    station_name_value: str,
    start: datetime | date,
    end: datetime | date,
) -> Path:
    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end
    safe_name = sanitize_station_filename_component(station_name_value)
    return (
        out_dir
        / str(start_date.year)
        / f"{station_id_value}_{safe_name}_{start_date}_{end_date}.csv"
    )


def sanitize_station_filename_component(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", value)
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip(" .-_")
    return value or "unknown-station"


def legacy_output_path(
    out_dir: Path,
    station_id_value: str,
    start: datetime | date,
    end: datetime | date,
) -> Path:
    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end
    return out_dir / str(start_date.year) / f"{station_id_value}_{start_date}_{end_date}.csv"


def migrate_legacy_output_file(legacy_path: Path, named_path: Path) -> bool:
    if named_path.exists() or not legacy_path.exists():
        return False
    named_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.replace(named_path)
    return True


def _load_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        table = pd.read_csv(path, dtype=str).fillna("")
        for column in columns:
            if column not in table.columns:
                table[column] = ""
        return table[columns]
    return pd.DataFrame(columns=columns)


def metadata_by_station_id(metadata: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {station_id(item): item for item in metadata}


def _apply_station_metadata_columns(
    table: pd.DataFrame,
    metadata_lookup: dict[str, dict[str, Any]],
    *,
    include_region_location: bool,
) -> pd.DataFrame:
    if table.empty or "station_id" not in table.columns:
        return table

    table = table.copy()
    for index, row in table.iterrows():
        meta = metadata_lookup.get(str(row.get("station_id", "")))
        if not meta:
            continue
        table.at[index, "wmo_id"] = station_wmo(meta)
        table.at[index, "name"] = station_name(meta)
        table.at[index, "country"] = station_country(meta)
        if include_region_location:
            table.at[index, "region"] = station_region(meta)
            table.at[index, "latitude"] = station_location_value(meta, "latitude")
            table.at[index, "longitude"] = station_location_value(meta, "longitude")
    return table


def migrate_loaded_tables(
    *,
    manifest_path: Path,
    manifest: pd.DataFrame,
    rp5_map_path: Path,
    rp5_map: pd.DataFrame,
    metadata: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = metadata_by_station_id(metadata)
    manifest = _apply_station_metadata_columns(
        manifest,
        lookup,
        include_region_location=False,
    )
    rp5_map = _apply_station_metadata_columns(
        rp5_map,
        lookup,
        include_region_location=True,
    )

    if manifest_path.exists():
        manifest.to_csv(manifest_path, index=False)
    if rp5_map_path.exists():
        rp5_map.to_csv(rp5_map_path, index=False)
    return manifest, rp5_map


def _upsert_table(
    path: Path,
    table: pd.DataFrame,
    row: dict[str, Any],
    *,
    key_columns: list[str],
    columns: list[str],
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {column: str(row.get(column, "")) for column in columns}

    if table.empty:
        table = pd.DataFrame([normalized], columns=columns)
    else:
        mask = pd.Series(True, index=table.index)
        for column in key_columns:
            mask &= table[column].astype(str) == normalized[column]
        if mask.any():
            for column in columns:
                table.loc[mask, column] = normalized[column]
        else:
            table = pd.concat(
                [table, pd.DataFrame([normalized], columns=columns)],
                ignore_index=True,
            )

    table.to_csv(path, index=False)
    return table


def manifest_record(
    table: pd.DataFrame,
    *,
    sid: str,
    start: datetime,
    end: datetime,
) -> pd.Series | None:
    if table.empty:
        return None
    mask = (
        (table["station_id"].astype(str) == sid)
        & (table["start"].astype(str) == start.date().isoformat())
        & (table["end"].astype(str) == end.date().isoformat())
    )
    if not mask.any():
        return None
    return table.loc[mask].iloc[-1]


def should_skip_manifest_record(record: pd.Series | None) -> bool:
    if record is None:
        return False
    status = str(record.get("status", ""))
    if status in {"meteostat_saved", "rp5_saved", "existing_saved"}:
        path_text = str(record.get("path", ""))
        if not path_text:
            return False
        path = Path(path_text)
        return path.exists()
    return status in TERMINAL_STATUSES


def upsert_manifest(
    manifest_path: Path,
    manifest: pd.DataFrame,
    *,
    station_meta: dict[str, Any],
    start: datetime,
    end: datetime,
    status: str,
    source: str = "",
    path: Path | str = "",
    rows: int | str = "",
    attempts: int | str = "",
    error: str = "",
) -> pd.DataFrame:
    return _upsert_table(
        manifest_path,
        manifest,
        {
            "station_id": station_id(station_meta),
            "wmo_id": station_wmo(station_meta),
            "name": station_name(station_meta),
            "country": station_country(station_meta),
            "year": start.year,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "status": status,
            "source": source,
            "path": str(path),
            "rows": rows,
            "attempts": attempts,
            "last_error": error[:800],
            "updated_at": utc_now_text(),
        },
        key_columns=["station_id", "start", "end"],
        columns=MANIFEST_COLUMNS,
    )


def rp5_map_record(table: pd.DataFrame, sid: str) -> pd.Series | None:
    if table.empty:
        return None
    mask = table["station_id"].astype(str) == sid
    if not mask.any():
        return None
    return table.loc[mask].iloc[-1]


def has_mapped_rp5_url(table: pd.DataFrame, sid: str) -> bool:
    record = rp5_map_record(table, sid)
    if record is None:
        return False
    return str(record.get("status", "")).strip().lower() == "mapped" and bool(
        str(record.get("rp5_url", "")).strip()
    )


def unresolved_rp5_station_ids(
    manifest: pd.DataFrame,
    rp5_map: pd.DataFrame,
) -> set[str]:
    station_ids: set[str] = set()
    if not manifest.empty:
        mask = manifest["status"].astype(str) == "unresolved_rp5"
        station_ids.update(manifest.loc[mask, "station_id"].astype(str))
    if not rp5_map.empty:
        mask = rp5_map["status"].astype(str) == "unresolved"
        station_ids.update(rp5_map.loc[mask, "station_id"].astype(str))
    return station_ids


def upsert_rp5_map(
    map_path: Path,
    table: pd.DataFrame,
    *,
    station_meta: dict[str, Any],
    rp5_url: str = "",
    status: str,
    error: str = "",
) -> pd.DataFrame:
    return _upsert_table(
        map_path,
        table,
        rp5_map_row(
            station_meta=station_meta,
            rp5_url=rp5_url,
            status=status,
            error=error,
        ),
        key_columns=["station_id"],
        columns=RP5_MAP_COLUMNS,
    )


def rp5_map_row(
    *,
    station_meta: dict[str, Any],
    rp5_url: str = "",
    status: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "station_id": station_id(station_meta),
        "wmo_id": station_wmo(station_meta),
        "name": station_name(station_meta),
        "country": station_country(station_meta),
        "region": station_region(station_meta),
        "latitude": station_location_value(station_meta, "latitude"),
        "longitude": station_location_value(station_meta, "longitude"),
        "rp5_url": rp5_url,
        "status": status,
        "last_error": error[:800],
        "updated_at": utc_now_text(),
    }


def unresolved_rp5_report_dataframe(
    metadata: list[dict[str, Any]],
    manifest: pd.DataFrame,
    rp5_map: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest_status_by_station = {}
    if not manifest.empty:
        for _, row in manifest.iterrows():
            manifest_status_by_station[str(row.get("station_id", ""))] = str(row.get("status", ""))

    for meta in metadata:
        sid = station_id(meta)
        map_record = rp5_map_record(rp5_map, sid)
        map_status = str(map_record.get("status", "")) if map_record is not None else ""
        rp5_url = str(map_record.get("rp5_url", "")) if map_record is not None else ""
        last_error = str(map_record.get("last_error", "")) if map_record is not None else ""
        manifest_status = manifest_status_by_station.get(sid, "")

        unresolved_by_map = map_status == "unresolved"
        unresolved_by_manifest = manifest_status == "unresolved_rp5"
        if not (unresolved_by_map or unresolved_by_manifest):
            continue
        if map_status == "mapped" and rp5_url:
            continue

        rows.append(
            {
                "station_id": sid,
                "wmo_id": station_wmo(meta),
                "name": station_name(meta),
                "country": station_country(meta),
                "region": station_region(meta),
                "latitude": station_location_value(meta, "latitude"),
                "longitude": station_location_value(meta, "longitude"),
                "rp5_url": rp5_url,
                "status": map_status or "unresolved",
                "manifest_status": manifest_status,
                "last_error": last_error,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "station_id",
            "wmo_id",
            "name",
            "country",
            "region",
            "latitude",
            "longitude",
            "rp5_url",
            "status",
            "manifest_status",
            "last_error",
        ],
    )


def export_unresolved_rp5_report(
    *,
    report_path: Path,
    metadata: list[dict[str, Any]],
    manifest: pd.DataFrame,
    rp5_map: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_rp5_report_dataframe(
        metadata=metadata,
        manifest=manifest,
        rp5_map=rp5_map,
    ).to_csv(report_path, index=False)


def parse_inventory_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def hourly_inventory_overlaps(
    station_meta: dict[str, Any],
    start: datetime,
    end: datetime,
) -> bool:
    hourly = station_meta.get("inventory", {}).get("hourly", {})
    inventory_start = parse_inventory_date(hourly.get("start"))
    inventory_end = parse_inventory_date(hourly.get("end"))
    if inventory_start is None or inventory_end is None:
        return False
    return inventory_start <= end.date() and inventory_end >= start.date()


def sleep_fixed(seconds: float, *, label: str) -> None:
    if seconds <= 0:
        return
    jitter = random.uniform(0, min(5.0, seconds * 0.25))
    delay = seconds + jitter
    print(f"{label} cooldown {delay:.1f}s")
    time.sleep(delay)


def configure_meteostat(cache_dir: Path) -> Any:
    import meteostat as ms

    cache_dir.mkdir(parents=True, exist_ok=True)
    config = getattr(ms, "config", None)
    if config is not None:
        for key, value in {
            "include_model_data": False,
            "cache_enable": True,
            "cache_directory": str(cache_dir),
        }.items():
            try:
                setattr(config, key, value)
            except Exception:
                pass
    return ms


def normalize_meteostat_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    elif "time" not in out.columns:
        out = out.reset_index()

    if "time" not in out.columns:
        first_column = out.columns[0]
        out = out.rename(columns={first_column: "time"})
    if "snow" in out.columns and "snwd" not in out.columns:
        out = out.rename(columns={"snow": "snwd"})
    if "snow_source" in out.columns and "snwd_source" not in out.columns:
        out = out.rename(columns={"snow_source": "snwd_source"})

    out["time"] = pd.to_datetime(out["time"], errors="coerce").dt.floor("h")
    out = out.dropna(subset=["time"])

    for column in CANONICAL_COLUMNS:
        if column == "time":
            continue
        if column not in out.columns:
            out[column] = pd.NA

    for column in [c for c in CANONICAL_COLUMNS if not c.endswith("_source") and c != "time"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        source_column = f"{column}_source"
        if source_column in out.columns:
            out[source_column] = out[source_column].where(out[source_column].notna(), pd.NA)
        else:
            out[source_column] = pd.NA
        out.loc[out[column].notna() & out[source_column].isna(), source_column] = "meteostat"

    out = out.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def fetch_meteostat_hourly(
    station_meta: dict[str, Any],
    start: datetime,
    end: datetime,
    *,
    cache_dir: Path,
) -> pd.DataFrame:
    ms = configure_meteostat(cache_dir)
    ts = ms.hourly(station_id(station_meta), start, end)
    return ts.fetch(clean=True, sources=True)


def fetch_meteostat_with_retries(
    station_meta: dict[str, Any],
    start: datetime,
    end: datetime,
    *,
    cache_dir: Path,
    max_retries: int,
    base_delay: float,
) -> tuple[pd.DataFrame, int]:
    last_error: Exception | None = None
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            return fetch_meteostat_hourly(
                station_meta,
                start,
                end,
                cache_dir=cache_dir,
            ), attempt
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = max(base_delay, 1.0) * (2 ** (attempt - 1)) + random.uniform(0, 5)
            print(
                f"meteostat retry {attempt}/{attempts} for "
                f"{station_id(station_meta)} after {delay:.1f}s: {type(exc).__name__}"
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def save_canonical_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def resolve_rp5_url(
    station_meta: dict[str, Any],
    *,
    rp5_map_path: Path,
    rp5_map: pd.DataFrame,
    session: Any,
    budget: RP5CallBudget,
    timeout: int,
    force: bool,
) -> tuple[str | None, bool, pd.DataFrame, str]:
    sid = station_id(station_meta)
    record = rp5_map_record(rp5_map, sid)
    if record is not None:
        status = str(record.get("status", ""))
        url = str(record.get("rp5_url", ""))
        if status == "mapped" and url:
            return url, False, rp5_map, ""
        if status == "unresolved" and not force:
            return None, False, rp5_map, str(record.get("last_error", "unresolved"))

    if not station_wmo(station_meta):
        error = "Station has no WMO id for RP5"
        rp5_map = upsert_rp5_map(
            rp5_map_path,
            rp5_map,
            station_meta=station_meta,
            status="unresolved",
            error=error,
        )
        return None, False, rp5_map, error

    try:
        url, page_bootstrapped = resolve_rp5_station_page_from_meta(
            station_meta,
            session,
            budget=budget,
            timeout=timeout,
        )
    except (RP5BlockedError, RP5CallBudgetExceeded):
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rp5_map = upsert_rp5_map(
            rp5_map_path,
            rp5_map,
            station_meta=station_meta,
            status="unresolved",
            error=error,
        )
        return None, False, rp5_map, error

    rp5_map = upsert_rp5_map(
        rp5_map_path,
        rp5_map,
        station_meta=station_meta,
        rp5_url=url,
        status="mapped",
    )
    return url, page_bootstrapped, rp5_map, ""


def try_rp5_fallback(
    station_meta: dict[str, Any],
    start: datetime,
    end: datetime,
    *,
    output_path: Path,
    raw_dir: Path,
    rp5_map_path: Path,
    rp5_map: pd.DataFrame,
    session: Any,
    budget: RP5CallBudget,
    timeout: int,
    keep_raw: bool,
    force: bool,
) -> tuple[str, int, Path | str, pd.DataFrame, str]:
    wmo = station_wmo(station_meta)
    if not wmo:
        return "unresolved_rp5", 0, "", rp5_map, "Station has no WMO id for RP5"

    url, page_bootstrapped, rp5_map, resolve_error = resolve_rp5_url(
        station_meta,
        rp5_map_path=rp5_map_path,
        rp5_map=rp5_map,
        session=session,
        budget=budget,
        timeout=timeout,
        force=force,
    )
    if not url:
        return "unresolved_rp5", 0, "", rp5_map, resolve_error

    raw_filename = f"{station_id(station_meta)}_{start.date()}_{end.date()}.csv.gz"
    raw_path = download_rp5_synop_by_id(
        wmo,
        start,
        end,
        station_page_url=url,
        out_dir=raw_dir,
        filename=raw_filename,
        timeout=timeout,
        budget=budget,
        session=session,
        skip_bootstrap=page_bootstrapped,
    )
    normalized = normalize_rp5_csv_gz(
        raw_path,
        timezone=station_meta.get("timezone"),
        start=start,
        end=end,
    )
    if normalized.empty:
        if not keep_raw:
            try:
                raw_path.unlink()
            except OSError:
                pass
        return "no_data", 0, raw_path if keep_raw else "", rp5_map, "RP5 archive contained no usable rows"

    save_canonical_csv(normalized, output_path)
    if not keep_raw:
        try:
            raw_path.unlink()
        except OSError:
            pass
    return "rp5_saved", len(normalized), output_path, rp5_map, ""


def load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Metadata file must contain a list of stations: {path}")
    return data


def process_station(
    station_meta: dict[str, Any],
    *,
    args: argparse.Namespace,
    start: datetime,
    end: datetime,
    output_path: Path,
    manifest_path: Path,
    manifest: pd.DataFrame,
    rp5_map_path: Path,
    rp5_map: pd.DataFrame,
    rp5_session: Any,
    rp5_budget: RP5CallBudget,
    rp5_state: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sid = station_id(station_meta)
    print(f"station {sid} {station_name(station_meta)}")

    if output_path.exists() and not args.force:
        rows = sum(1 for _ in output_path.open("r", encoding="utf-8", errors="ignore")) - 1
        manifest = upsert_manifest(
            manifest_path,
            manifest,
            station_meta=station_meta,
            start=start,
            end=end,
            status="existing_saved",
            source="existing",
            path=output_path,
            rows=max(rows, 0),
        )
        print(f"  skip existing file {output_path}")
        return manifest, rp5_map

    record = manifest_record(manifest, sid=sid, start=start, end=end)
    if not args.force and should_skip_manifest_record(record):
        retry_unresolved = (
            str(record.get("status", "")) == "unresolved_rp5"
            and (
                has_mapped_rp5_url(rp5_map, sid)
                or args.retry_unresolved_rp5
            )
        )
        if retry_unresolved:
            print("  retrying unresolved station with RP5 lookup")
        else:
            print(f"  skip manifest status {record.get('status')}")
            return manifest, rp5_map

    if hourly_inventory_overlaps(station_meta, start, end):
        try:
            df, attempts = fetch_meteostat_with_retries(
                station_meta,
                start,
                end,
                cache_dir=args.out_dir / ".meteostat_cache",
                max_retries=args.max_retries,
                base_delay=args.meteostat_delay,
            )
            normalized = normalize_meteostat_dataframe(df)
            sleep_fixed(args.meteostat_delay, label="meteostat")
            if not normalized.empty:
                save_canonical_csv(normalized, output_path)
                manifest = upsert_manifest(
                    manifest_path,
                    manifest,
                    station_meta=station_meta,
                    start=start,
                    end=end,
                    status="meteostat_saved",
                    source="meteostat",
                    path=output_path,
                    rows=len(normalized),
                    attempts=attempts,
                )
                print(f"  saved meteostat rows={len(normalized)}")
                return manifest, rp5_map
            print("  meteostat empty, trying RP5 fallback")
        except Exception as exc:
            sleep_fixed(args.meteostat_delay, label="meteostat")
            manifest = upsert_manifest(
                manifest_path,
                manifest,
                station_meta=station_meta,
                start=start,
                end=end,
                status="failed",
                source="meteostat",
                attempts=args.max_retries,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"  meteostat failed: {type(exc).__name__}: {exc}")
            return manifest, rp5_map
    else:
        print("  no overlapping Meteostat hourly inventory, trying RP5 fallback")

    if not args.rp5_enabled:
        manifest = upsert_manifest(
            manifest_path,
            manifest,
            station_meta=station_meta,
            start=start,
            end=end,
            status="pending_rp5_disabled",
            source="none",
            error="RP5 fallback disabled",
        )
        return manifest, rp5_map

    if rp5_state.get("disabled"):
        manifest = upsert_manifest(
            manifest_path,
            manifest,
            station_meta=station_meta,
            start=start,
            end=end,
            status="pending_rp5_blocked",
            source="rp5",
            error=str(rp5_state.get("disabled_reason", "RP5 disabled for this run")),
        )
        print("  RP5 disabled for this run")
        return manifest, rp5_map

    raw_dir = (
        args.out_dir / "_rp5_raw" / str(start.year)
        if args.rp5_keep_raw
        else args.out_dir / "_rp5_tmp" / str(start.year)
    )
    try:
        status, rows, saved_path, rp5_map, error = try_rp5_fallback(
            station_meta,
            start,
            end,
            output_path=output_path,
            raw_dir=raw_dir,
            rp5_map_path=rp5_map_path,
            rp5_map=rp5_map,
            session=rp5_session,
            budget=rp5_budget,
            timeout=args.rp5_timeout,
            keep_raw=args.rp5_keep_raw,
            force=args.force or args.retry_unresolved_rp5,
        )
        manifest = upsert_manifest(
            manifest_path,
            manifest,
            station_meta=station_meta,
            start=start,
            end=end,
            status=status,
            source="rp5" if status == "rp5_saved" else "none",
            path=saved_path,
            rows=rows,
            error=error,
        )
        print(f"  {status} rows={rows} rp5_calls={rp5_budget.calls_made}")
    except RP5CallBudgetExceeded as exc:
        rp5_state["disabled"] = True
        rp5_state["disabled_reason"] = str(exc)
        manifest = upsert_manifest(
            manifest_path,
            manifest,
            station_meta=station_meta,
            start=start,
            end=end,
            status="pending_rp5_budget",
            source="rp5",
            error=str(exc),
        )
        print(f"  RP5 budget exhausted: {exc}")
    except RP5BlockedError as exc:
        rp5_state["block_failures"] = int(rp5_state.get("block_failures", 0)) + 1
        if rp5_state["block_failures"] >= args.rp5_stop_after_blocks:
            rp5_state["disabled"] = True
            rp5_state["disabled_reason"] = (
                f"RP5 stopped after {rp5_state['block_failures']} blocking-like failures"
            )
        manifest = upsert_manifest(
            manifest_path,
            manifest,
            station_meta=station_meta,
            start=start,
            end=end,
            status="pending_rp5_blocked",
            source="rp5",
            error=str(exc),
        )
        print(f"  RP5 blocked-like failure: {exc}")
    except Exception as exc:
        manifest = upsert_manifest(
            manifest_path,
            manifest,
            station_meta=station_meta,
            start=start,
            end=end,
            status="failed",
            source="rp5",
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"  RP5 failed: {type(exc).__name__}: {exc}")

    return manifest, rp5_map


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NESTP stations from Meteostat first, then RP5 fallback."
    )
    parser.add_argument("--year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--meteostat-delay", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--station-id", action="append", default=[])
    parser.add_argument(
        "--export-unresolved-rp5",
        action="store_true",
        help="Write unresolved RP5 stations to a manual-fill CSV and exit.",
    )
    parser.add_argument(
        "--unresolved-report",
        type=Path,
        default=None,
        help="Output path for --export-unresolved-rp5.",
    )
    parser.add_argument(
        "--retry-unresolved-rp5",
        action="store_true",
        help="Retry only stations previously marked unresolved by RP5 lookup.",
    )

    rp5_group = parser.add_mutually_exclusive_group()
    rp5_group.add_argument("--rp5-enabled", dest="rp5_enabled", action="store_true", default=True)
    rp5_group.add_argument("--no-rp5", dest="rp5_enabled", action="store_false")
    parser.add_argument("--rp5-max-calls-per-run", type=int, default=10000)
    parser.add_argument("--rp5-min-delay", type=float, default=3.0)
    parser.add_argument("--rp5-max-delay", type=float, default=12.0)
    parser.add_argument("--rp5-stop-after-blocks", type=int, default=3)
    parser.add_argument("--rp5-keep-raw", action="store_true")
    parser.add_argument("--rp5-timeout", type=int, default=60)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start, end = resolve_period(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    if args.station_id:
        selected = {str(value) for value in args.station_id}
        metadata = [item for item in metadata if station_id(item) in selected]

    manifest_path = args.out_dir / "download_manifest.csv"
    rp5_map_path = args.out_dir / "rp5_station_map.csv"
    manifest = _load_table(manifest_path, MANIFEST_COLUMNS)
    rp5_map = _load_table(rp5_map_path, RP5_MAP_COLUMNS)
    manifest, rp5_map = migrate_loaded_tables(
        manifest_path=manifest_path,
        manifest=manifest,
        rp5_map_path=rp5_map_path,
        rp5_map=rp5_map,
        metadata=metadata,
    )

    if args.retry_unresolved_rp5:
        unresolved_ids = unresolved_rp5_station_ids(manifest, rp5_map)
        metadata = [item for item in metadata if station_id(item) in unresolved_ids]

    if args.export_unresolved_rp5:
        report_path = args.unresolved_report or args.out_dir / "rp5_unresolved_stations.csv"
        export_unresolved_rp5_report(
            report_path=report_path,
            metadata=metadata,
            manifest=manifest,
            rp5_map=rp5_map,
        )
        print(f"wrote unresolved RP5 report: {report_path}")
        return 0

    rp5_budget = RP5CallBudget(
        max_calls=args.rp5_max_calls_per_run,
        min_delay=args.rp5_min_delay,
        max_delay=args.rp5_max_delay,
    )
    rp5_state: dict[str, Any] = {"disabled": False, "block_failures": 0}

    print(f"period {start} -> {end}")
    print(f"metadata stations={len(metadata)} out_dir={args.out_dir}")
    print(
        "rp5 "
        f"enabled={args.rp5_enabled} max_calls={args.rp5_max_calls_per_run} "
        f"delay={args.rp5_min_delay}-{args.rp5_max_delay}s"
    )

    rp5_session_context = _make_session(trust_env=False) if args.rp5_enabled else nullcontext(None)
    with rp5_session_context as rp5_session:
        for station_meta in metadata:
            sid = station_id(station_meta)
            output_path = canonical_output_path(
                args.out_dir,
                sid,
                station_name(station_meta),
                start,
                end,
            )
            old_output_path = legacy_output_path(args.out_dir, sid, start, end)
            if migrate_legacy_output_file(old_output_path, output_path):
                print(f"renamed legacy station file to {output_path.name}")
            manifest, rp5_map = process_station(
                station_meta,
                args=args,
                start=start,
                end=end,
                output_path=output_path,
                manifest_path=manifest_path,
                manifest=manifest,
                rp5_map_path=rp5_map_path,
                rp5_map=rp5_map,
                rp5_session=rp5_session,
                rp5_budget=rp5_budget,
                rp5_state=rp5_state,
            )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
