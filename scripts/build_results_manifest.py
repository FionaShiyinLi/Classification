#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for curated public results."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST = RESULTS_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    artifacts = []
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "results",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    public_paths = sorted(REPO_ROOT / item for item in listed.stdout.splitlines())
    for path in public_paths:
        if not path.is_file() or path in {MANIFEST, RESULTS_DIR / "README.md"}:
            continue
        artifacts.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(REPO_ROOT)} with {len(artifacts)} artifacts")


if __name__ == "__main__":
    main()
