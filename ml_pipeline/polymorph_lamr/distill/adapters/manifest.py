"""Build MANIFEST.json for staged and referenced text corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_samples(path: Path, n: int = 3) -> list[str]:
    samples: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped:
                samples.append(stripped)
            if len(samples) >= n:
                break
    return samples


def _glob_samples(glob_path: Path, n: int = 3) -> list[str]:
    samples: list[str] = []
    for path in sorted(glob_path.parent.glob(glob_path.name)):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if stripped:
                    samples.append(stripped)
                if len(samples) >= n:
                    return samples
    return samples


def _glob_stats(glob_path: Path) -> tuple[int, int]:
    line_count = 0
    total_bytes = 0
    for path in sorted(glob_path.parent.glob(glob_path.name)):
        total_bytes += path.stat().st_size
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.rstrip("\n"):
                    line_count += 1
    return line_count, total_bytes


def _file_stats(path: Path) -> tuple[int, int]:
    total_bytes = path.stat().st_size
    line_count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.rstrip("\n"):
                line_count += 1
    return line_count, total_bytes


def build_manifest(repo_root: Path, staged_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    for entry in staged_entries:
        staged_path = repo_root / entry["staged_path"]
        line_count, total_bytes = _file_stats(staged_path)
        manifest.append(
            {
                "name": entry["name"],
                "format": "log_line",
                "source": entry["source"],
                "staged_path": entry["staged_path"],
                "line_count": line_count,
                "bytes": total_bytes,
                "samples": _read_samples(staged_path),
                "skipped_rows": entry.get("skipped_rows", 0),
            }
        )

    trainticket_glob = repo_root / "data/bench/trainticket_logs/*.txt"
    tt_lines, tt_bytes = _glob_stats(trainticket_glob)
    manifest.append(
        {
            "name": "trainticket_traces",
            "format": "log_line",
            "source": "data/bench/trainticket_logs/",
            "source_glob": "data/bench/trainticket_logs/*.txt",
            "line_count": tt_lines,
            "bytes": tt_bytes,
            "samples": _glob_samples(trainticket_glob),
        }
    )

    apache_path = repo_root / "data/raw/server_logs/logfiles.log"
    ap_lines, ap_bytes = _file_stats(apache_path)
    manifest.append(
        {
            "name": "apache_access",
            "format": "log_line",
            "source": "data/raw/server_logs/logfiles.log",
            "line_count": ap_lines,
            "bytes": ap_bytes,
            "samples": _read_samples(apache_path),
        }
    )

    return manifest


def write_manifest(repo_root: Path, manifest: list[dict[str, Any]]) -> Path:
    out_path = repo_root / "data/staged/MANIFEST.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return out_path
