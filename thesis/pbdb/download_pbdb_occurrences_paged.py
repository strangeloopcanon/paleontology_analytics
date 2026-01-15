from __future__ import annotations

import argparse
import csv
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PBDB_OCCS_CSV = "https://paleobiodb.org/data1.2/occs/list.csv"


@dataclass
class DownloadState:
    url: str
    params: dict[str, Any]
    page_size: int
    next_offset: int
    started_at_unix: float
    updated_at_unix: float
    pages_downloaded: int
    records_downloaded: int
    complete: bool


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_state(path: Path) -> DownloadState | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return DownloadState(**data)


def _write_state(path: Path, state: DownloadState) -> None:
    state.updated_at_unix = time.time()
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")


def _request_with_retries(
    session: requests.Session,
    *,
    url: str,
    params: dict[str, Any],
    timeout_s: float,
    max_retries: int,
    backoff_s: float,
) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(1, int(max_retries) + 1):
        try:
            r = session.get(url, params=params, timeout=float(timeout_s))
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            sleep_s = float(backoff_s) * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 90.0)
            print(f"Request failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < int(max_retries):
                time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def _count_csv_records(text: str) -> int:
    # PBDB CSV occasionally contains embedded newlines in quoted fields (e.g., references),
    # so line-based counting can miscount and cause offset drift. Use a CSV parser.
    buf = io.StringIO(text)
    reader = csv.reader(buf)
    try:
        next(reader)  # header
    except StopIteration:
        return 0
    n = 0
    for _ in reader:
        n += 1
    return int(n)


def download_paged_csv(
    *,
    interval: str,
    out_path: Path,
    show: str,
    vocab: str,
    base_name: str | None,
    page_size: int,
    sleep_s: float,
    timeout_s: float,
    max_retries: int,
    backoff_s: float,
    resume: bool,
) -> Path:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    state_path = out_path.with_suffix(out_path.suffix + ".state.json")

    params_base: dict[str, Any] = {
        "interval": interval,
        "show": show,
        "vocab": vocab,
    }
    if base_name:
        params_base["base_name"] = base_name

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "paleontology_analytics/1.0 (paged PBDB downloader; contact: local)",
            "Accept": "text/csv",
        }
    )

    state = _read_state(state_path) if resume else None
    if state is not None and state.complete:
        print(f"State indicates complete; skipping download: {state_path}")
        return out_path

    if state is not None:
        if state.url != PBDB_OCCS_CSV or state.params != params_base or state.page_size != int(page_size):
            raise SystemExit(
                "Existing state does not match requested download settings.\n"
                f"- state: {state_path}\n"
                f"- requested interval={interval!r} page_size={page_size} show={show!r} vocab={vocab!r}\n"
                "Delete the state+csv files if you want to restart."
            )
    else:
        now = time.time()
        state = DownloadState(
            url=PBDB_OCCS_CSV,
            params=params_base,
            page_size=int(page_size),
            next_offset=0,
            started_at_unix=now,
            updated_at_unix=now,
            pages_downloaded=0,
            records_downloaded=0,
            complete=False,
        )
        _write_state(state_path, state)

    has_output = out_path.exists() and out_path.stat().st_size > 0
    mode = "ab" if has_output else "wb"
    with open(out_path, mode) as f:
        while True:
            params = dict(params_base)
            params["limit"] = int(page_size)
            params["offset"] = int(state.next_offset)

            print(f"Fetching offset={state.next_offset} limit={page_size} …")
            r = _request_with_retries(
                session,
                url=PBDB_OCCS_CSV,
                params=params,
                timeout_s=float(timeout_s),
                max_retries=int(max_retries),
                backoff_s=float(backoff_s),
            )

            text = r.text
            n_records = _count_csv_records(text)
            if n_records == 0:
                print("No more records; download complete.")
                state.complete = True
                _write_state(state_path, state)
                break

            content = r.content
            nl = content.find(b"\n")
            if nl == -1:
                raise RuntimeError("PBDB response did not contain a newline; cannot split header from body")

            if has_output:
                payload = content[nl + 1 :]
            else:
                payload = content
                has_output = True

            if payload:
                if not payload.endswith(b"\n"):
                    payload = payload + b"\n"
                f.write(payload)
                f.flush()

            state.pages_downloaded += 1
            state.records_downloaded += int(n_records)
            # Advance by page size to avoid any ambiguity about record counting / embedded newlines.
            # The PBDB API pagination is record-indexed; a final short page will be handled on the next request.
            state.next_offset += int(page_size)
            _write_state(state_path, state)

            if n_records < int(page_size):
                print("Final partial page received; download complete.")
                state.complete = True
                _write_state(state_path, state)
                break

            if sleep_s > 0:
                time.sleep(float(sleep_s))

    return out_path


def _validate_csv(path: Path, *, usecols: list[str] | None = None) -> None:
    print(f"Validating CSV parse: {path}")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    print(f"Parsed OK: {len(df):,} rows; columns={len(df.columns)}")
    if "max_ma" in df.columns and "min_ma" in df.columns:
        max_ma = pd.to_numeric(df["max_ma"], errors="coerce")
        min_ma = pd.to_numeric(df["min_ma"], errors="coerce")
        mid_ma = (max_ma + min_ma) / 2.0
        print(f"mid_ma range: {mid_ma.min():.4g} – {mid_ma.max():.4g} Ma")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="Paleogene,Holocene")
    p.add_argument("--base-name", default=None, help="Optional PBDB taxonomic filter (e.g., Mammalia).")
    p.add_argument("--out", default="data/raw/pbdb_occurrences_paleogene_holocene_paged.csv")
    p.add_argument("--show", default="coords,class,paleoloc,strat,time,env,ref")
    p.add_argument("--vocab", default="pbdb")
    p.add_argument("--page-size", type=int, default=50_000)
    p.add_argument("--sleep-s", type=float, default=0.2)
    p.add_argument("--timeout-s", type=float, default=240.0)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--backoff-s", type=float, default=2.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    out_path = download_paged_csv(
        interval=str(args.interval),
        base_name=str(args.base_name) if args.base_name else None,
        out_path=Path(args.out),
        show=str(args.show),
        vocab=str(args.vocab),
        page_size=int(args.page_size),
        sleep_s=float(args.sleep_s),
        timeout_s=float(args.timeout_s),
        max_retries=int(args.max_retries),
        backoff_s=float(args.backoff_s),
        resume=not bool(args.no_resume),
    )

    if args.validate:
        _validate_csv(out_path, usecols=["occurrence_no", "collection_no", "max_ma", "min_ma"])


if __name__ == "__main__":
    main()
