from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CuratedSection:
    name: str
    title_any: tuple[str, ...]
    title_all: tuple[str, ...] = ()
    title_exclude: tuple[str, ...] = ()
    concept_any: tuple[str, ...] = ()


SECTIONS: list[CuratedSection] = [
    CuratedSection(
        "Dinosaur body size (distribution / bimodality)",
        title_any=("dinosaur", "dinosauria", "theropod", "sauropod", "ornith", "mesozoic"),
        title_all=(),
        concept_any=("Paleontology", "Paleobiology", "Evolutionary biology"),
    ),
    CuratedSection(
        "Ecospace / functional diversity (marine fossils)",
        title_any=("ecospace", "functional diversity", "functional composition", "functional disparity", "ecospace dynamics"),
        title_all=(),
        concept_any=("Paleontology", "Paleobiology", "Ecology"),
    ),
    CuratedSection(
        "Convergence / homogenization (deep time)",
        title_any=("convergence", "homogenization", "biotic homogenization", "functional redundancy"),
        title_all=(),
        concept_any=("Paleontology", "Paleobiology", "Ecology"),
    ),
    CuratedSection(
        "Provinciality / plate tectonics / biogeography",
        title_any=("provincial", "biogeograph", "plate tect", "paleogeograph"),
        concept_any=("Paleontology", "Paleobiology", "Biogeography"),
    ),
    CuratedSection(
        "Climate volatility forcing (deep time)",
        title_any=("climate simulation", "climate", "paleoclimate", "cesm", "540 million"),
        title_exclude=("children", "auditory", "organ", "genome"),
        concept_any=("Climatology", "Paleontology", "Earth science"),
    ),
    CuratedSection(
        "PBDB sampling bias / rock record",
        title_any=("paleobiology database", "pbdb", "sampling", "rock record", "macrostrat", "spatial bias"),
        concept_any=("Paleontology", "Paleobiology", "Geology"),
    ),
]


def _title(work: dict[str, Any]) -> str:
    return (work.get("display_name") or work.get("title") or "").strip()


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


def _concept_names(work: dict[str, Any]) -> list[str]:
    out = []
    for c in (work.get("concepts") or []):
        name = c.get("display_name")
        if name:
            out.append(str(name))
    return out


def _match(work: dict[str, Any], sec: CuratedSection) -> bool:
    title = _title(work).lower()
    if not title:
        return False
    if sec.title_any and not any(t in title for t in sec.title_any):
        return False
    if sec.title_all and not all(t in title for t in sec.title_all):
        return False
    if sec.title_exclude and any(t in title for t in sec.title_exclude):
        return False
    if sec.concept_any:
        concepts = [c.lower() for c in _concept_names(work)]
        if not any(any(k.lower() in c for c in concepts) for k in sec.concept_any):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", default="thesis/literature/reading_lists/reading_list_openalex.json")
    p.add_argument("--out-md", default="thesis/literature/shortlist.md")
    p.add_argument("--max-per-section", type=int, default=40)
    args = p.parse_args()

    payload = json.loads(Path(args.in_json).read_text())
    works = payload.get("works") or []

    rows = [
        "# Curated shortlist (from OpenAlex dump)",
        "",
        "This is a heuristic, title/concept-filtered shortlist to reduce noise in the raw OpenAlex reading list.",
        "",
        f"- Source: `{Path(args.in_json)}`",
        f"- Max per section: {int(args.max_per_section)}",
        "",
    ]

    for sec in SECTIONS:
        matched = [w for w in works if _match(w, sec)]
        matched.sort(key=lambda w: ((w.get("cited_by_count") or 0), (w.get("publication_year") or 0)), reverse=True)
        rows.extend(
            [
                "",
                f"## {sec.name}",
                "",
                "| Cited by | Year | DOI | Title | Venue | OA URL |",
                "|---:|---:|---|---|---|---|",
            ]
        )
        for w in matched[: int(args.max_per_section)]:
            title = _title(w) or "Untitled"
            year = w.get("publication_year") or ""
            cited = w.get("cited_by_count") or 0
            doi = _doi(w) or ""
            doi_cell = f"[{doi}](https://doi.org/{doi})" if doi else ""
            venue = _venue(w)
            oa_url = _oa_url(w) or ""
            oa_cell = f"[link]({oa_url})" if oa_url else ""
            rows.append(f"| {cited} | {year} | {doi_cell} | {title} | {venue} | {oa_cell} |")

        if not matched:
            rows.append("|  |  |  | (no matches under current filters) |  |  |")

    Path(args.out_md).write_text("\n".join(rows) + "\n")
    print(f"Wrote: {Path(args.out_md)}")


if __name__ == "__main__":
    main()
