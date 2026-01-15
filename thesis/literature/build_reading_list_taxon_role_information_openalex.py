from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


OPENALEX = "https://api.openalex.org"


@dataclass(frozen=True)
class Query:
    name: str
    search: str
    openalex_filter: str | None = None


DEFAULT_QUERIES: list[Query] = [
    Query(
        "taxon_role_mutual_information",
        "mutual information taxonomic functional roles fossil record",
    ),
    Query(
        "taxon_role_modularity",
        "modularity taxon functional roles ecospace Paleobiology Database",
    ),
    Query(
        "ecospace_nestedness_roles",
        "nestedness ecospace roles fossil record",
    ),
    Query(
        "functional_interchangeability",
        "taxonomic interchangeability functional roles paleontology",
    ),
    Query(
        "function_taxonomy_decoupling",
        "functional taxonomic decoupling fossil record functional diversity",
    ),
]


def _fetch_all(search: str, *, openalex_filter: str | None, max_results: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = "*"
    per_page = 200
    while cursor and len(out) < max_results:
        params = {"search": search, "per-page": per_page, "cursor": cursor}
        if openalex_filter:
            params["filter"] = openalex_filter
        r = requests.get(f"{OPENALEX}/works", params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        out.extend(payload.get("results") or [])
        cursor = payload.get("meta", {}).get("next_cursor")
        if not payload.get("results"):
            break
    return out[:max_results]


def _doi(work: dict[str, Any]) -> str | None:
    doi = work.get("doi")
    if not doi:
        return None
    return str(doi).removeprefix("https://doi.org/")


def _venue(work: dict[str, Any]) -> str:
    primary = (work.get("primary_location") or {}).get("source") or {}
    return primary.get("display_name") or "Unknown venue"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="thesis/literature/reading_lists")
    p.add_argument("--max-per-query", type=int, default=250)
    p.add_argument("--max-md-per-query", type=int, default=60)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, dict[str, Any]] = {}
    by_query: dict[str, list[str]] = {}

    for q in DEFAULT_QUERIES:
        works = _fetch_all(q.search, openalex_filter=q.openalex_filter, max_results=int(args.max_per_query))
        ids = []
        for w in works:
            wid = w.get("id")
            if not wid:
                continue
            ids.append(str(wid))
            if wid not in combined:
                combined[wid] = w
        by_query[q.name] = ids
        print(f"{q.name}: {len(works)} works")

    raw = {
        "queries": [q.__dict__ for q in DEFAULT_QUERIES],
        "by_query": by_query,
        "works": list(combined.values()),
    }
    raw_path = out_dir / "reading_list_taxon_role_information_openalex.json"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")

    rows = [
        "# OpenAlex reading list (auto-built): taxon↔role information / modularity / interchangeability",
        "",
        "Auto-built from OpenAlex keyword searches (broad; needs manual triage).",
        "",
        f"- Max fetched per query: {int(args.max_per_query)}",
        f"- Max rows written per query in this markdown: {int(args.max_md_per_query)}",
        "",
    ]

    for q in DEFAULT_QUERIES:
        rows.extend(
            [
                "",
                f"## {q.name}",
                "",
                f"- search: `{q.search}`",
                f"- filter: `{q.openalex_filter}`",
                "",
                "| Cited by | Year | DOI | Title | Venue |",
                "|---:|---:|---|---|---|",
            ]
        )
        works = [combined[wid] for wid in by_query.get(q.name, []) if wid in combined]
        works.sort(key=lambda w: ((w.get("cited_by_count") or 0), (w.get("publication_year") or 0)), reverse=True)
        for w in works[: int(args.max_md_per_query)]:
            title = (w.get("display_name") or w.get("title") or "").strip() or "Untitled"
            year = w.get("publication_year") or ""
            cited = w.get("cited_by_count") or 0
            doi = _doi(w) or ""
            doi_cell = f"[{doi}](https://doi.org/{doi})" if doi else ""
            venue = _venue(w)
            rows.append(f"| {cited} | {year} | {doi_cell} | {title} | {venue} |")

    md_path = out_dir / "reading_list_taxon_role_information_openalex.md"
    md_path.write_text("\n".join(rows) + "\n")

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
