from __future__ import annotations

import io
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Union
from urllib.parse import quote, urljoin

import pandas as pd

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ModuleNotFoundError:
    requests = None
    HTTPAdapter = None
    Retry = None

RP5_POST_URL = "https://rp5.ru/responses/reFileSynop.php"
RP5_SEARCH_URL = "https://rp5.ru/responses/reJsonSynop.php"
RP5_BASE_URL = "https://rp5.ru/"

DateLike = Union[str, date, datetime]

CANONICAL_COLUMNS = [
    "time",
    "temp",
    "temp_source",
    "dwpt",
    "dwpt_source",
    "rhum",
    "rhum_source",
    "prcp",
    "prcp_source",
    "snwd",
    "snwd_source",
    "wdir",
    "wdir_source",
    "wspd",
    "wspd_source",
    "wpgt",
    "wpgt_source",
    "pres",
    "pres_source",
    "coco",
    "coco_source",
]

RP5_BLOCK_PATTERNS = (
    "captcha",
    "too many requests",
    "access denied",
    "temporarily unavailable",
    "error #fs000",
    "\u043f\u0440\u043e\u0432\u0435\u0440\u043e\u0447\u043d\u044b\u0439 \u043a\u043e\u0434",
    "\u043a\u0430\u043f\u0447\u0430",
    "\u0441\u043b\u0438\u0448\u043a\u043e\u043c \u043c\u043d\u043e\u0433\u043e \u0437\u0430\u043f\u0440\u043e\u0441\u043e\u0432",
    "\u0434\u043e\u0441\u0442\u0443\u043f \u0437\u0430\u043f\u0440\u0435\u0449",
)


class RP5BlockedError(RuntimeError):
    """Raised when RP5 appears to block or reject automated requests."""


class RP5CallBudgetExceeded(RuntimeError):
    """Raised when the per-run RP5 call budget is exhausted."""


@dataclass
class RP5CallBudget:
    """Small rate limiter for RP5 page and archive-generation requests."""

    max_calls: int | None = 30
    min_delay: float = 60.0
    max_delay: float = 180.0
    calls_made: int = 0

    @property
    def remaining(self) -> int | None:
        if self.max_calls is None:
            return None
        return max(0, self.max_calls - self.calls_made)

    def consume(self, label: str = "rp5") -> None:
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise RP5CallBudgetExceeded(
                f"RP5 call budget exhausted before {label}: "
                f"{self.calls_made}/{self.max_calls} calls used"
            )

        if self.calls_made > 0:
            delay = random.uniform(self.min_delay, self.max_delay)
            if delay > 0:
                print(f"rp5 cooldown {delay:.1f}s before {label}")
                time.sleep(delay)

        self.calls_made += 1


def _to_rp5_date(x: DateLike) -> str:
    if isinstance(x, str):
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(x, fmt).strftime("%d.%m.%Y")
            except ValueError:
                pass
        raise ValueError(f"Unsupported date string format: {x!r}")
    if isinstance(x, datetime):
        return x.strftime("%d.%m.%Y")
    if isinstance(x, date):
        return x.strftime("%d.%m.%Y")
    raise TypeError(f"Unsupported date type: {type(x)}")


def _make_session(*, trust_env: bool = False, retry_total: int = 0) -> requests.Session:
    if requests is None or HTTPAdapter is None or Retry is None:
        raise RuntimeError("The 'requests' package is required for RP5 downloads")

    session = requests.Session()
    session.trust_env = trust_env

    retry = Retry(
        total=retry_total,
        connect=retry_total,
        read=retry_total,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
            "Connection": "close",
        }
    )
    return session


def is_blocking_response(status_code: int | None = None, text: str | None = None) -> bool:
    if status_code in {403, 429}:
        return True
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in RP5_BLOCK_PATTERNS)


def _request_with_budget(
    session: requests.Session,
    method: str,
    url: str,
    *,
    budget: RP5CallBudget | None,
    label: str,
    timeout: int,
    **kwargs,
) -> requests.Response:
    if budget is not None:
        budget.consume(label)
    try:
        response = session.request(method, url, timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        raise RP5BlockedError(f"RP5 timeout during {label}") from exc
    if is_blocking_response(response.status_code, response.text):
        raise RP5BlockedError(
            f"RP5 blocking-like response during {label}: HTTP {response.status_code}"
        )
    response.raise_for_status()
    return response


def _extract_download_link(text: str) -> str:
    patterns = [
        r'<a\s+href=(https?://[^ >]+\.csv\.gz)',
        r'<a\s+href="(https?://[^"]+\.csv\.gz)"',
        r"<a\s+href='(https?://[^']+\.csv\.gz)'",
        r'(https?://[^\s"\']+\.csv\.gz)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise RuntimeError(f"Could not find .csv.gz link in rp5 response:\n{text[:800]}")


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = re.sub(r"\s*\([^)]*\)", "", value)
    value = value.replace("\xa0", " ").replace("-", " ").replace(",", " ").strip()
    value = re.sub(r"\s{2,}", " ", value)
    return value


def name_variants(name: str) -> list[str]:
    name = normalize_name(name)
    variants = {name}
    swaps = [
        ("Mys ", "Cape "),
        ("Cape ", "Mys "),
        ("Ostrov ", "Island "),
        ("Island ", "Ostrov "),
        ("Bukhta ", "Bay "),
        ("Bay ", "Bukhta "),
        ("Gora ", "Mount "),
        ("Mount ", "Gora "),
    ]
    for base in list(variants):
        for source, target in swaps:
            if base.startswith(source):
                variants.add(target + base[len(source) :])

    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    if ascii_name:
        variants.add(ascii_name)
    return sorted({variant.strip() for variant in variants if variant.strip()})


def rp5_archive_url(title: str) -> str:
    slug = re.sub(r"\s+", "_", normalize_name(title))
    return f"https://rp5.ru/Weather_archive_in_{quote(slug, safe='_')}"


def normalize_rp5_station_id(value: int | str) -> str:
    text = str(value).strip()
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def rp5_search_result_url(
    payload: object,
    station_id: int | str,
) -> str | None:
    if not isinstance(payload, list):
        return None

    target = normalize_rp5_station_id(station_id)
    for item in payload:
        if not isinstance(item, dict):
            continue
        if normalize_rp5_station_id(item.get("id", "")) != target:
            continue

        namealt = str(item.get("namealt") or "").strip()
        if not namealt:
            continue
        encoded_path = quote(namealt.lstrip("/"), safe="/_-%")
        return urljoin(RP5_BASE_URL, encoded_path)
    return None


def resolve_rp5_station_page_via_search(
    station_id: int | str,
    session: requests.Session,
    *,
    budget: RP5CallBudget | None = None,
    timeout: int = 60,
    lang_id: int = 1,
    limit: int = 500,
) -> str | None:
    target = str(station_id)
    response = _request_with_budget(
        session,
        "GET",
        RP5_SEARCH_URL,
        budget=budget,
        label=f"rp5-search-{target}",
        timeout=timeout,
        params={
            "langid": str(lang_id),
            "q": target,
            "limit": str(limit),
            "timestamp": str(int(time.time() * 1000)),
        },
        headers={
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": RP5_BASE_URL,
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    try:
        payload = response.json()
    except ValueError:
        return None
    return rp5_search_result_url(payload, target)


def extract_wmo_id(html: str) -> str | None:
    input_match = re.search(
        r"<input\b[^>]*(?:id|name)=['\"]wmo_id['\"][^>]*>",
        html,
        flags=re.IGNORECASE,
    )
    if not input_match:
        return None
    value_match = re.search(
        r"\bvalue=['\"]?([^'\"\s>]+)",
        input_match.group(0),
        flags=re.IGNORECASE,
    )
    return value_match.group(1) if value_match else None


def station_name_hints(station_meta: dict) -> list[str]:
    names = station_meta.get("name", {})
    if isinstance(names, dict):
        hints = [str(value) for value in names.values() if value]
    elif names:
        hints = [str(names)]
    else:
        hints = []
    return hints


def build_rp5_candidate_urls(
    station_meta: dict,
    *,
    max_candidates: int = 8,
) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for hint in station_name_hints(station_meta):
        for variant in name_variants(hint):
            url = rp5_archive_url(variant)
            if url not in seen:
                urls.append(url)
                seen.add(url)
            if len(urls) >= max_candidates:
                return urls
    return urls


def resolve_rp5_station_page(
    station_id: int | str,
    session: requests.Session,
    name_hints: Iterable[str],
    *,
    budget: RP5CallBudget | None = None,
    timeout: int = 60,
    max_candidates: int = 8,
) -> str:
    target = str(station_id)
    candidates: list[str] = []
    seen: set[str] = set()

    for hint in name_hints:
        for variant in name_variants(hint):
            url = rp5_archive_url(variant)
            if url not in seen:
                candidates.append(url)
                seen.add(url)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    for url in candidates:
        try:
            response = _request_with_budget(
                session,
                "GET",
                url,
                budget=budget,
                label=f"rp5-discover-{target}",
                timeout=timeout,
            )
        except requests.RequestException:
            continue

        page_station_id = extract_wmo_id(response.text)
        if page_station_id and (
            normalize_rp5_station_id(page_station_id)
            == normalize_rp5_station_id(target)
        ):
            return url

    raise RuntimeError(f"Failed to resolve RP5 page for station {target}")


def resolve_rp5_station_page_from_meta(
    station_meta: dict,
    session: requests.Session,
    *,
    budget: RP5CallBudget | None = None,
    timeout: int = 60,
    max_candidates: int = 8,
) -> tuple[str, bool]:
    identifiers = station_meta.get("identifiers", {})
    station_id = identifiers.get("wmo") or station_meta.get("id")
    if not station_id:
        raise RuntimeError("Station has no WMO or Meteostat id for RP5 lookup")

    try:
        search_url = resolve_rp5_station_page_via_search(
            station_id,
            session,
            budget=budget,
            timeout=timeout,
        )
    except requests.RequestException:
        search_url = None
    if search_url:
        return search_url, False

    guessed_url = resolve_rp5_station_page(
        station_id,
        session,
        station_name_hints(station_meta),
        budget=budget,
        timeout=timeout,
        max_candidates=max_candidates,
    )
    return guessed_url, True


def download_rp5_synop_by_id(
    station_id: int | str,
    start: DateLike,
    end: DateLike,
    *,
    station_page_url: str,
    out_dir: Union[str, Path] = "rp5_data",
    filename: str | None = None,
    timeout: int = 60,
    trust_env: bool = False,
    budget: RP5CallBudget | None = None,
    session: requests.Session | None = None,
    skip_bootstrap: bool = False,
    f_ed3: int = 3,
    f_ed4: int = 3,
    f_ed5: int = 27,
    f_pe: int = 1,
    f_pe1: int = 2,
    lng_id: int = 2,
) -> Path:
    """
    Download a RP5 SYNOP archive by WMO station id.

    The page bootstrap GET and archive-generation POST count against
    ``budget``. The final generated .csv.gz download does not.
    """
    start_str = _to_rp5_date(start)
    end_str = _to_rp5_date(end)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"rp5_{station_id}_{start_str.replace('.', '')}_{end_str.replace('.', '')}.csv.gz"
    out_path = out_dir / filename

    close_session = session is None
    active_session = session or _make_session(trust_env=trust_env)

    try:
        if not skip_bootstrap:
            _request_with_budget(
                active_session,
                "GET",
                station_page_url,
                budget=budget,
                label=f"rp5-bootstrap-{station_id}",
                timeout=timeout,
            )

        headers = {
            "Origin": "https://rp5.ru",
            "Referer": station_page_url,
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Accept": "text/html, */*; q=0.01",
        }

        payload = {
            "wmo_id": str(station_id),
            "a_date1": start_str,
            "a_date2": end_str,
            "f_ed3": str(f_ed3),
            "f_ed4": str(f_ed4),
            "f_ed5": str(f_ed5),
            "f_pe": str(f_pe),
            "f_pe1": str(f_pe1),
            "lng_id": str(lng_id),
        }

        response = _request_with_budget(
            active_session,
            "POST",
            RP5_POST_URL,
            data=payload,
            headers=headers,
            budget=budget,
            label=f"rp5-archive-post-{station_id}",
            timeout=timeout,
        )
        link = _extract_download_link(response.text)

        try:
            archive = active_session.get(
                link,
                headers={"Referer": station_page_url},
                timeout=timeout,
                stream=True,
            )
        except requests.Timeout as exc:
            raise RP5BlockedError(f"RP5 timeout during archive download for {station_id}") from exc
        if is_blocking_response(archive.status_code):
            raise RP5BlockedError(
                f"RP5 blocking-like response during archive download: HTTP {archive.status_code}"
            )
        archive.raise_for_status()

        with out_path.open("wb") as file:
            for chunk in archive.iter_content(chunk_size=1024 * 128):
                if chunk:
                    file.write(chunk)
    finally:
        if close_session:
            active_session.close()

    return out_path


def read_rp5_csv_gz(
    file: Union[str, Path, bytes],
    *,
    skiprows: int = 6,
    sep: str = ";",
    encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1251", "latin1"),
) -> pd.DataFrame:
    last_error = None
    for encoding in encodings:
        try:
            if isinstance(file, (str, Path)):
                return pd.read_csv(
                    file,
                    compression="gzip",
                    sep=sep,
                    skiprows=skiprows,
                    encoding=encoding,
                )
            return pd.read_csv(
                io.BytesIO(file),
                compression="gzip",
                sep=sep,
                skiprows=skiprows,
                encoding=encoding,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Could not parse rp5 csv.gz with the provided encodings") from last_error


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in df.columns}


def _find_column(df: pd.DataFrame, *names: str, contains: str | None = None) -> str | None:
    lookup = _column_lookup(df)
    for name in names:
        key = name.strip().lower()
        if key in lookup:
            return lookup[key]
    if contains is not None:
        needle = contains.lower()
        for key, column in lookup.items():
            if needle in key:
                return column
    return None


def _numeric_series(series: pd.Series, *, zero_words: bool = False) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    extracted = text.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    out = pd.to_numeric(extracted, errors="coerce")
    if zero_words:
        lowered = text.str.lower()
        zero_mask = lowered.str.contains(
            r"no precipitation|trace|calm|\u0448\u0442\u0438\u043b\u044c|"
            r"\u0441\u043b\u0435\u0434|\u043e\u0441\u0430\u0434\u043a\u043e\u0432 "
            r"\u043d\u0435\u0442",
            regex=True,
            na=False,
        )
        out = out.mask(zero_mask & out.isna(), 0.0)
    return out


def _parse_wind_direction(series: pd.Series) -> pd.Series:
    numeric = _numeric_series(series)
    parsed = numeric.where((numeric >= 0) & (numeric <= 360))
    text = (
        series.astype(str)
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace("_", " ", regex=False)
    )

    patterns = [
        (r"north\s+north\s+east|\bnne\b", 22.5),
        (r"north\s+east|\bnortheast\b|\bne\b|\u0441\u0435\u0432\u0435\u0440\u043e\s*\u0432\u043e\u0441\u0442", 45.0),
        (r"east\s+north\s+east|\bene\b", 67.5),
        (r"east\s+south\s+east|\bese\b", 112.5),
        (r"south\s+east|\bsoutheast\b|\bse\b|\u044e\u0433\u043e\s*\u0432\u043e\u0441\u0442", 135.0),
        (r"south\s+south\s+east|\bsse\b", 157.5),
        (r"south\s+south\s+west|\bssw\b", 202.5),
        (r"south\s+west|\bsouthwest\b|\bsw\b|\u044e\u0433\u043e\s*\u0437\u0430\u043f", 225.0),
        (r"west\s+south\s+west|\bwsw\b", 247.5),
        (r"west\s+north\s+west|\bwnw\b", 292.5),
        (r"north\s+west|\bnorthwest\b|\bnw\b|\u0441\u0435\u0432\u0435\u0440\u043e\s*\u0437\u0430\u043f", 315.0),
        (r"north\s+north\s+west|\bnnw\b", 337.5),
        (r"\bnorth\b|\bn\b|\u0441\u0435\u0432\u0435\u0440", 0.0),
        (r"\beast\b|\be\b|\u0432\u043e\u0441\u0442", 90.0),
        (r"\bsouth\b|\bs\b|\u044e\u0436|\u044e\u0433", 180.0),
        (r"\bwest\b|\bw\b|\u0437\u0430\u043f", 270.0),
    ]
    for pattern, degrees in patterns:
        mask = parsed.isna() & text.str.contains(pattern, regex=True, na=False)
        parsed = parsed.mask(mask, degrees)

    calm_or_variable = text.str.contains(
        r"calm|variable|\u0448\u0442\u0438\u043b\u044c|\u043f\u0435\u0440\u0435\u043c\u0435\u043d",
        regex=True,
        na=False,
    )
    return parsed.mask(calm_or_variable)


def _parse_datetime_values(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    formats = (
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H:%M:%S",
    )
    for fmt in formats:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format=fmt, errors="coerce")

    missing = parsed.isna()
    if missing.any():
        try:
            parsed.loc[missing] = pd.to_datetime(
                text.loc[missing],
                dayfirst=True,
                errors="coerce",
                format="mixed",
            )
        except TypeError:
            parsed.loc[missing] = pd.to_datetime(
                text.loc[missing],
                dayfirst=True,
                errors="coerce",
            )
    return parsed


def _looks_like_time_column_name(column: object) -> bool:
    name = str(column).strip().lower()
    return (
        name == "time"
        or "local time" in name
        or "\u043c\u0435\u0441\u0442\u043d\u043e\u0435" in name
        or "\u0432\u0440\u0435\u043c\u044f" in name
    )


def _repair_rp5_indexed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    RP5 sometimes has one more data field than header field, so pandas promotes
    the timestamp field to the index and shifts the remaining columns left.
    """
    if isinstance(df.index, pd.RangeIndex) or df.empty:
        return df

    parsed_index = _parse_datetime_values(pd.Series(df.index.astype(str), index=df.index))
    valid_ratio = float(parsed_index.notna().mean()) if len(parsed_index) else 0.0
    if valid_ratio < 0.5:
        return df

    repaired = df.copy()
    columns = list(repaired.columns)
    if columns and _looks_like_time_column_name(columns[0]):
        repaired.columns = columns[1:] + ["_rp5_extra"]

    if "time" in repaired.columns:
        repaired = repaired.rename(columns={"time": "_rp5_original_time"})
    repaired.insert(0, "time", parsed_index.to_numpy())
    return repaired.reset_index(drop=True)


def _parse_time_column(
    df: pd.DataFrame,
    *,
    timezone: str | None = None,
) -> pd.Series:
    if {"year", "month", "day", "hour"}.issubset({str(c).lower() for c in df.columns}):
        lookup = _column_lookup(df)
        return pd.to_datetime(
            {
                "year": pd.to_numeric(df[lookup["year"]], errors="coerce"),
                "month": pd.to_numeric(df[lookup["month"]], errors="coerce"),
                "day": pd.to_numeric(df[lookup["day"]], errors="coerce"),
                "hour": pd.to_numeric(df[lookup["hour"]], errors="coerce"),
            },
            errors="coerce",
        )

    time_col = (
        _find_column(df, "time")
        or _find_column(df, contains="local time")
        or _find_column(df, contains="\u043c\u0435\u0441\u0442\u043d\u043e\u0435")
        or df.columns[0]
    )
    parsed = _parse_datetime_values(df[time_col])

    if timezone and parsed.notna().any():
        try:
            parsed = (
                parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )
        except (TypeError, ValueError):
            pass
    return parsed


def _assign_with_source(
    out: pd.DataFrame,
    column: str,
    values: pd.Series | None,
    *,
    source: str,
) -> None:
    if values is None:
        out[column] = pd.NA
        out[f"{column}_source"] = pd.NA
        return
    out[column] = values
    out[f"{column}_source"] = pd.NA
    out.loc[out[column].notna(), f"{column}_source"] = source


def normalize_rp5_dataframe(
    df: pd.DataFrame,
    *,
    timezone: str | None = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    source: str = "rp5",
) -> pd.DataFrame:
    """Convert a parsed RP5 archive dataframe to the canonical station CSV shape."""
    df = _repair_rp5_indexed_frame(df)
    out = pd.DataFrame()
    out["time"] = _parse_time_column(df, timezone=timezone).dt.floor("h")
    temp_col = _find_column(df, "T")
    dwpt_col = _find_column(df, "Td")
    rhum_col = _find_column(df, "U")
    prcp_col = _find_column(df, "RRR")
    snwd_col = _find_column(df, "sss")
    wdir_col = _find_column(df, "DD")
    wspd_col = _find_column(df, "Ff")
    wpgt_col = _find_column(df, "ff10") or _find_column(df, "ff3")
    pres_col = _find_column(df, "P") or _find_column(df, "Po")
    coco_col = _find_column(df, "WW")

    _assign_with_source(
        out,
        "temp",
        _numeric_series(df[temp_col]) if temp_col is not None else None,
        source=source,
    )
    _assign_with_source(
        out,
        "dwpt",
        _numeric_series(df[dwpt_col]) if dwpt_col is not None else None,
        source=source,
    )
    _assign_with_source(
        out,
        "rhum",
        _numeric_series(df[rhum_col]) if rhum_col is not None else None,
        source=source,
    )
    _assign_with_source(
        out,
        "prcp",
        _numeric_series(df[prcp_col], zero_words=True) if prcp_col is not None else None,
        source=source,
    )
    _assign_with_source(
        out,
        "snwd",
        _numeric_series(df[snwd_col]) if snwd_col is not None else None,
        source=source,
    )
    _assign_with_source(
        out,
        "wdir",
        _parse_wind_direction(df[wdir_col]) if wdir_col is not None else None,
        source=source,
    )
    _assign_with_source(
        out,
        "wspd",
        (_numeric_series(df[wspd_col], zero_words=True) * 3.6)
        if wspd_col is not None
        else None,
        source=source,
    )
    _assign_with_source(
        out,
        "wpgt",
        (_numeric_series(df[wpgt_col], zero_words=True) * 3.6)
        if wpgt_col is not None
        else None,
        source=source,
    )
    _assign_with_source(
        out,
        "pres",
        _numeric_series(df[pres_col]) if pres_col is not None else None,
        source=source,
    )
    _assign_with_source(
        out,
        "coco",
        _numeric_series(df[coco_col]) if coco_col is not None else None,
        source=source,
    )

    out = out.dropna(subset=["time"])
    if start is not None:
        out = out[out["time"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["time"] <= pd.Timestamp(end)]

    out = out.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    for column in CANONICAL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def normalize_rp5_csv_gz(
    file: Union[str, Path, bytes],
    *,
    timezone: str | None = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    source: str = "rp5",
) -> pd.DataFrame:
    return normalize_rp5_dataframe(
        read_rp5_csv_gz(file),
        timezone=timezone,
        start=start,
        end=end,
        source=source,
    )


if __name__ == "__main__":
    raise SystemExit(
        "rp5_download.py is a helper module. Run meteostat_download.py for the "
        "unified Meteostat/RP5 downloader."
    )
