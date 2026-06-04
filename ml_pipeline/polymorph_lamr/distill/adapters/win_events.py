"""Adapter: win_events (pt-BR Windows Event Viewer CSV) → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    sanitize_field,
    stream_csv_to_txt,
)

_REQUIRED = (
    "Nível",
    "Data e Hora",
    "Fonte",
    "Identificação do Evento",
    "Categoria da Tarefa",
    "Log",
    "Computador",
    "Description1",
    "Description2",
    "Description3",
)

SOURCE_CSV = "data/raw/win_events/eventos.csv"
STAGED_TXT = "data/staged/win_events.txt"


def _descriptions(row: dict[str, str | None]) -> str:
    parts = [
        sanitize_field(row["Description1"]),
        sanitize_field(row["Description2"]),
        sanitize_field(row["Description3"]),
    ]
    return " ".join(p for p in parts if p)


def render_row(row: dict[str, str | None]) -> str | None:
    desc = _descriptions(row)
    parts = [
        sanitize_field(row["Data e Hora"]),
        sanitize_field(row["Nível"]),
        sanitize_field(row["Fonte"]),
        f"event_id={sanitize_field(row['Identificação do Evento'])}",
        f"task={sanitize_field(row['Categoria da Tarefa'])}",
        f"log={sanitize_field(row['Log'])}",
        f"computer={sanitize_field(row['Computador'])}",
    ]
    if desc:
        parts.append(desc)
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    csv_path = repo_root / SOURCE_CSV
    out_path = repo_root / STAGED_TXT
    return stream_csv_to_txt(
        csv_path, out_path, render_row=render_row, required_columns=_REQUIRED
    )
