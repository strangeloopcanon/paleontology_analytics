from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


MACROSTRAT_SECTIONS = "https://macrostrat.org/api/v2/sections"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_sections(*, out_path: Path, timeout_s: float, force: bool) -> Path:
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    if out_path.exists() and not force:
        print(f"Exists, skipping: {out_path}")
        return out_path

    params = {"all": 1, "format": "csv"}
    headers = {"User-Agent": "paleontology_analytics/1.0 (Macrostrat fetch)"}
    t0 = time.time()
    r = requests.get(MACROSTRAT_SECTIONS, params=params, headers=headers, timeout=float(timeout_s))
    r.raise_for_status()
    out_path.write_bytes(r.content)
    dt = time.time() - t0
    print(f"Wrote {out_path} ({len(r.content)/1e6:.1f} MB) in {dt:.1f}s")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/external/macrostrat/sections_all.csv")
    p.add_argument("--timeout-s", type=float, default=240.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    download_sections(out_path=Path(args.out), timeout_s=float(args.timeout_s), force=bool(args.force))


if __name__ == "__main__":
    main()

