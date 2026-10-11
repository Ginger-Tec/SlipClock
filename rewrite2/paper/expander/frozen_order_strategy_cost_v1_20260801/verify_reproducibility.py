#!/usr/bin/env python3
"""Rerun the experiment and verify byte-identical deterministic outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MANIFEST = HERE / "manifest.json"
REPORT = HERE / "results/reproducibility_verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths() -> list[Path]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = [HERE / relative for relative in payload["outputs"]]
    return sorted([MANIFEST, *paths])


def snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(HERE)): sha256_file(path) for path in paths}


def main() -> int:
    if not MANIFEST.exists():
        raise FileNotFoundError("run_experiment.py must create manifest.json first")

    before_paths = tracked_paths()
    before = snapshot(before_paths)
    command = [sys.executable, str(HERE / "run_experiment.py"), "--skip-extract"]
    completed = subprocess.run(command, cwd=REPO, check=False)

    after_paths = tracked_paths() if completed.returncode == 0 else []
    after = snapshot(after_paths) if after_paths else {}
    all_names = sorted(set(before) | set(after))
    mismatches = [
        {
            "path": name,
            "before_sha256": before.get(name),
            "after_sha256": after.get(name),
        }
        for name in all_names
        if before.get(name) != after.get(name)
    ]
    status = "REPRODUCIBLE" if completed.returncode == 0 and not mismatches else "MISMATCH"
    report = {
        "status": status,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "file_count": len(all_names),
        "mismatches": mismatches,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if status == "REPRODUCIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
