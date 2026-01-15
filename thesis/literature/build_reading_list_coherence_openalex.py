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


PALEO_FILTER = "concept.id:C88385403|C151730666|C167570900"


DEFAULT_QUERIES: list[Query] = [
    Query(
        "spatial_coherence_climate_biotic_homogenization",
        "spatial coherence climate change biotic homogenization fossil",
        openalex_filter=PALEO_FILTER,
    ),
    Query(
        "synchrony_refugia_portfolio_extinction",
        "synchronous climate forcing refugia geographic range extinction selectivity",
        openalex_filter=PALEO_FILTER,
    ),
    Query(
        "provinciality_drivers_climate_vs_tectonics",
        "marine provinciality climate change plate tectonics 250 million years",
        openalex_filter=PALEO_FILTER,
    ),
    Query(
        "post_extinction_similarity_physiology_climate",
        "post extinction similarity physiology climate oxygen metabolic demand",
        openalex_filter=PALEO_FILTER,
    ),
    Query(
        "functional_beta_diversity_deep_time",
        "functional beta diversity fossil record",
        openalex_filter=PALEO_FILTER,
    ),
    Query(
        "EOF_paleoclimate_patterns_biogeography",
        "EOF principal component paleoclimate biogeography",
        openalex_filter=PALEO_FILTER,
    ),
    Query(
        "spatial_scaling_beta_diversity_fossil",
        "spatial scaling beta diversity shallow-marine fossil record",
        openalex_filter=PALEO_FILTER,
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


def _oa_url(work: dict[str, Any]) -> str | None:
    oa = work.get("open_access") or {}
    url = oa.get("oa_url") or None
    return str(url) if url else None


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
    raw_path = out_dir / "reading_list_coherence_openalex.json"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")

    rows = [
        "# OpenAlex reading list (targeted): coherence / synchrony of forcing × homogenization × portfolio selectivity",
        "",
        "Targeted OpenAlex keyword searches with a paleobiology/paleontology/Phanerozoic concept filter.",
        "",
        f"- concept filter: `{PALEO_FILTER}`",
        f"- max fetched per query: {int(args.max_per_query)}",
        f"- max rows written per query: {int(args.max_md_per_query)}",
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
                "| Cited by | Year | DOI | Title | Venue | OA URL |",
                "|---:|---:|---|---|---|---|",
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
            oa_url = _oa_url(w) or ""
            oa_cell = f"[link]({oa_url})" if oa_url else ""
            rows.append(f"| {cited} | {year} | {doi_cell} | {title} | {venue} | {oa_cell} |")

    md_path = out_dir / "reading_list_coherence_openalex.md"
    md_path.write_text("\n".join(rows) + "\n")

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
