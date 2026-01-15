from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class DedupGroup:
    sha256: str
    bytes_each: int
    canonical: Path
    duplicates: list[Path]


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> list[DedupGroup]:
    obj = json.loads(path.read_text())
    groups: list[DedupGroup] = []
    for g in obj.get("groups") or []:
        groups.append(
            DedupGroup(
                sha256=str(g["sha256"]),
                bytes_each=int(g["bytes_each"]),
                canonical=Path(str(g["canonical"])),
                duplicates=[Path(str(p)) for p in (g.get("duplicates") or [])],
            )
        )
    return groups


def _ensure_same_blob(group: DedupGroup) -> None:
    canonical_hash = _sha256(group.canonical)
    if canonical_hash != group.sha256:
        raise ValueError(
            f"Canonical hash mismatch for {group.canonical}: expected {group.sha256}, got {canonical_hash}"
        )
    for dup in group.duplicates:
        dup_hash = _sha256(dup)
        if dup_hash != group.sha256:
            raise ValueError(f"Duplicate hash mismatch for {dup}: expected {group.sha256}, got {dup_hash}")


def _same_inode(a: Path, b: Path) -> bool:
    sa = a.stat()
    sb = b.stat()
    return (sa.st_ino == sb.st_ino) and (sa.st_dev == sb.st_dev)


def _hardlink_replace(*, src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if not dst.exists():
        raise FileNotFoundError(dst)
    if _same_inode(src, dst):
        return

    dst.unlink()
    os.link(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Deduplicate identical blobs by replacing duplicates with hardlinks.")
    p.add_argument(
        "--manifest",
        default="thesis/_meta/dedup_manifest.json",
        help="Path to a dedup manifest JSON file.",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate and report only; do not modify files.")
    p.add_argument("--report-out", default="", help="Optional JSON report output path.")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    groups = _load_manifest(manifest_path)

    started_at = time.time()
    links_created = 0
    already_linked = 0
    bytes_saved = 0
    actions: list[dict[str, str]] = []

    for group in groups:
        if not group.canonical.exists():
            raise FileNotFoundError(group.canonical)
        for dup in group.duplicates:
            if not dup.exists():
                raise FileNotFoundError(dup)

        _ensure_same_blob(group)

        for dup in group.duplicates:
            if _same_inode(group.canonical, dup):
                already_linked += 1
                actions.append({"action": "skip_already_linked", "canonical": str(group.canonical), "path": str(dup)})
                continue

            if args.dry_run:
                actions.append({"action": "would_hardlink", "canonical": str(group.canonical), "path": str(dup)})
                continue

            _hardlink_replace(src=group.canonical, dst=dup)
            links_created += 1
            bytes_saved += group.bytes_each
            actions.append({"action": "hardlinked", "canonical": str(group.canonical), "path": str(dup)})

    report = {
        "manifest": str(manifest_path),
        "dry_run": bool(args.dry_run),
        "links_created": links_created,
        "already_linked": already_linked,
        "approx_bytes_saved": bytes_saved,
        "seconds": round(time.time() - started_at, 3),
        "actions": actions,
    }

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
