#!/usr/bin/env python3
"""Verify the canonical multi-campaign artifact inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmark-output" / "manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema_version") != "2.0.0":
        raise SystemExit("unsupported manifest schema")
    for item in manifest["artifacts"]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise SystemExit(f"missing artifact: {item['path']}")
        if digest(path) != item["sha256"]:
            raise SystemExit(f"checksum mismatch: {item['path']}")
        if path.suffix == ".csv":
            rows = sum(1 for _ in path.open("rb")) - 1
            if rows != item["rows"]:
                raise SystemExit(f"row-count mismatch: {item['path']}")

    freeze_path = ROOT / manifest["source_freeze"]["manifest"]
    if not freeze_path.is_file():
        raise SystemExit(f"missing source-freeze manifest: {freeze_path}")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema_version") != "1.0.0":
        raise SystemExit("unsupported source-freeze schema")
    verified = 0
    for campaign in freeze["campaigns"]:
        archive = ROOT / campaign["archive"]
        if not archive.is_file():
            raise SystemExit(f"missing source archive: {campaign['archive']}")
        if digest(archive) != campaign["archive_sha256"]:
            raise SystemExit(f"source-archive checksum mismatch: {campaign['archive']}")
        verified += 1
    print(
        f"Verified {len(manifest['artifacts'])} canonical artifacts and "
        f"{verified} frozen source archives."
    )


if __name__ == "__main__":
    main()
