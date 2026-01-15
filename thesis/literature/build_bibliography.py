from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


OPENALEX = "https://api.openalex.org"


def _clean_doi(doi: str) -> str:
    doi = doi.strip()
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return doi


def _safe_bib_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    return text[:40] if text else "key"


def _escape_bibtex(value: str) -> str:
    # Minimal escaping for BibTeX.
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _get_primary_source(work: dict[str, Any]) -> str:
    primary = (work.get("primary_location") or {}).get("source") or {}
    name = primary.get("display_name")
    return name or "Unknown venue"


def _first_author_last_name(work: dict[str, Any]) -> str:
    authorships = work.get("authorships") or []
    if not authorships:
        return "Unknown"
    raw = (authorships[0].get("author") or {}).get("display_name") or "Unknown"
    return raw.split()[-1]


def _format_authors(work: dict[str, Any]) -> str:
    parts = []
    for auth in work.get("authorships") or []:
        name = (auth.get("author") or {}).get("display_name")
        if name:
            parts.append(name)
    if not parts:
        return "Unknown"
    return " and ".join(parts)


@dataclass(frozen=True)
class Work:
    doi: str
    title: str
    year: int | None
    venue: str
    url: str | None
    cited_by: int | None
    authors_bibtex: str


def _fetch_openalex_work_by_doi(doi: str) -> dict[str, Any] | None:
    doi_url = f"https://doi.org/{doi}"
    endpoint = f"{OPENALEX}/works/{quote(doi_url, safe=':/')}"
    r = requests.get(endpoint, timeout=30)
    if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
        return r.json()
    return None


def _search_openalex_work(needle: str) -> dict[str, Any] | None:
    r = requests.get(f"{OPENALEX}/works", params={"search": needle, "per-page": 10}, timeout=30)
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    return results[0]


def _fetch_openalex_work_by_id(openalex_id: str) -> dict[str, Any] | None:
    openalex_id = openalex_id.strip()
    if not openalex_id:
        return None
    openalex_id = openalex_id.removeprefix("https://openalex.org/").removeprefix("openalex.org/")
    r = requests.get(f"{OPENALEX}/works/{openalex_id}", timeout=30)
    if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
        return r.json()
    return None


def _resolve_work(doi: str, *, override: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if override:
        if override.get("openalex_id"):
            work = _fetch_openalex_work_by_id(str(override["openalex_id"]))
            if work is not None:
                return work
        if override.get("search"):
            # Use a custom query string, then try to find an exact DOI match in the results.
            r = requests.get(f"{OPENALEX}/works", params={"search": str(override["search"]), "per-page": 20}, timeout=30)
            r.raise_for_status()
            results = r.json().get("results") or []
            for cand in results:
                cand_doi = (cand.get("doi") or "").removeprefix("https://doi.org/").lower()
                if cand_doi == doi.lower():
                    return cand
            if results:
                return results[0]

    work = _fetch_openalex_work_by_doi(doi)
    if work is not None:
        return work

    # Fallback: search (OpenAlex sometimes struggles with complex Paleobiology-style DOIs).
    r = requests.get(f"{OPENALEX}/works", params={"search": doi, "per-page": 20}, timeout=30)
    r.raise_for_status()
    results = r.json().get("results") or []
    for cand in results:
        cand_doi = (cand.get("doi") or "").removeprefix("https://doi.org/").lower()
        if cand_doi == doi.lower():
            return cand
    return results[0] if results else None


def _to_work(doi: str, work: dict[str, Any]) -> Work:
    title = (work.get("display_name") or work.get("title") or "Untitled").strip()
    year = work.get("publication_year")
    venue = _get_primary_source(work)
    url = work.get("id")
    cited_by = work.get("cited_by_count")
    authors = _format_authors(work)
    return Work(doi=doi, title=title, year=year, venue=venue, url=url, cited_by=cited_by, authors_bibtex=authors)


def _bibtex_entry(work: Work) -> str:
    key_parts = [_safe_bib_key(_first_token(work.title)), str(work.year or "nd"), _safe_bib_key(_first_token(work.venue))]
    key = _safe_bib_key("".join(key_parts)) or "work"
    fields = {
        "title": work.title,
        "author": work.authors_bibtex,
        "year": str(work.year) if work.year else None,
        "journal": work.venue if work.venue != "Unknown venue" else None,
        "doi": work.doi,
        "url": work.url,
    }

    lines = [f"@article{{{key},"]
    for k, v in fields.items():
        if not v:
            continue
        lines.append(f"  {k} = {{{_escape_bibtex(str(v))}}},")
    lines.append("}")
    return "\n".join(lines)


def _first_token(text: str) -> str:
    return re.split(r"\s+", text.strip(), maxsplit=1)[0] if text.strip() else ""


def main() -> None:
    here = Path(__file__).resolve().parent
    dois_path = here / "core_dois.txt"
    out_bib = here / "references.bib"
    out_md = here / "bibliography.md"
    missing_path = here / "missing_dois.txt"
    overrides_path = here / "doi_overrides.json"

    overrides: dict[str, Any] = {}
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text())

    dois = [_clean_doi(line) for line in dois_path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]

    works: list[Work] = []
    missing: list[str] = []
    for doi in dois:
        resolved = _resolve_work(doi, override=overrides.get(doi))
        if resolved is None:
            missing.append(doi)
            continue
        works.append(_to_work(doi, resolved))

    works_sorted = sorted(works, key=lambda w: (w.year or 10_000, w.cited_by or -1), reverse=False)

    out_bib.write_text("\n\n".join(_bibtex_entry(w) for w in works_sorted) + "\n")

    rows = ["| Year | DOI | Title | Venue | Cited by |", "|---:|---|---|---|---:|"]
    for w in sorted(works, key=lambda x: (x.year or 0, x.cited_by or 0), reverse=True):
        doi_link = f"https://doi.org/{w.doi}"
        rows.append(f"| {w.year or ''} | [{w.doi}]({doi_link}) | {w.title} | {w.venue} | {w.cited_by or ''} |")
    out_md.write_text("\n".join(rows) + "\n")

    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    print(f"Wrote: {out_bib}")
    print(f"Wrote: {out_md}")
    if missing:
        print(f"Missing {len(missing)} DOIs (see {missing_path})")


if __name__ == "__main__":
    main()
