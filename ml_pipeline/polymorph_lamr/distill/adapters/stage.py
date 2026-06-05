"""Stage CSV corpora to uniform log-line text files and write MANIFEST.json."""

from __future__ import annotations

import sys
from pathlib import Path

from . import (
    alibaba_gpu,
    api_failures,
    cicd_failures,
    cloudtrail_flaws,
    distsys_synth,
    python_tracebacks,
    security_synth,
    servicenow_itsm,
    syslog_cremev2,
    win_events,
)
from .manifest import build_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[4]

_SINGLE_ADAPTERS = (
    ("distsys_synth", distsys_synth),
    ("api_failures", api_failures),
    ("alibaba_gpu", alibaba_gpu),
    ("cicd_failures", cicd_failures),
    ("win_events", win_events),
    ("syslog_cremev2", syslog_cremev2),
    ("servicenow_itsm", servicenow_itsm),
    ("cloudtrail_flaws", cloudtrail_flaws),
    ("python_tracebacks", python_tracebacks),
)


def _append_single_adapter(
    staged_entries: list[dict[str, object]], name: str, module: object, repo_root: Path
) -> None:
    written, skipped = module.stage(repo_root)
    source = getattr(module, "SOURCE_CSV", None) or getattr(module, "SOURCE_JSON", None)
    staged_entries.append(
        {
            "name": name,
            "source": source,
            "staged_path": module.STAGED_TXT,
            "written_rows": written,
            "skipped_rows": skipped,
        }
    )


def stage_all(repo_root: Path = REPO_ROOT) -> list[dict[str, object]]:
    staged_entries: list[dict[str, object]] = []
    for name, module in _SINGLE_ADAPTERS:
        _append_single_adapter(staged_entries, name, module, repo_root)
    staged_entries.extend(security_synth.stage_corpora(repo_root))
    return staged_entries


def main(argv: list[str] | None = None) -> int:
    repo_root = REPO_ROOT
    if argv and len(argv) > 1:
        repo_root = Path(argv[1]).resolve()

    staged_entries = stage_all(repo_root)
    manifest = build_manifest(repo_root, staged_entries)
    manifest_path = write_manifest(repo_root, manifest)

    total_staged_bytes = 0
    print("Corpus staging summary")
    print("=" * 60)
    for entry in manifest:
        if "staged_path" in entry and entry["staged_path"].startswith("data/staged/"):
            total_staged_bytes += entry["bytes"]
        skipped = entry.get("skipped_rows")
        skip_note = f" (skipped {skipped})" if skipped else ""
        print(
            f"  {entry['name']:20s}  lines={entry['line_count']:>8,d}  "
            f"bytes={entry['bytes']:>12,d}{skip_note}"
        )
    print("-" * 60)
    print(f"  {'staged txt total':20s}  bytes={total_staged_bytes:>12,d}")
    print(f"  MANIFEST written to {manifest_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
