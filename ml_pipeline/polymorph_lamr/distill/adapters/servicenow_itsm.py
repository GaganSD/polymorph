"""Adapter: ServiceNow-style IT incident CSV → uniform log lines."""

from __future__ import annotations

from pathlib import Path

from polymorph_lamr.distill.adapters._common import (
    collapse_whitespace,
    row_field,
    stream_csv_to_txt,
)

_REQUIRED = ("number", "incident_state", "sys_updated_at")

SOURCE_CSV = "data/raw/it_incident/incident_event_log.csv"
STAGED_TXT = "data/staged/servicenow_itsm.txt"


def render_row(row: dict[str, str | None]) -> str | None:
    parts = [
        row_field(row, "sys_updated_at"),
        f"incident={row_field(row, 'number')}",
        f"state={row_field(row, 'incident_state')}",
        f"active={row_field(row, 'active')}",
        f"category={row_field(row, 'category')}",
        f"subcategory={row_field(row, 'subcategory')}",
        f"symptom={row_field(row, 'u_symptom')}",
        f"priority={row_field(row, 'priority')}",
        f"impact={row_field(row, 'impact')}",
        f"urgency={row_field(row, 'urgency')}",
        f"contact={row_field(row, 'contact_type')}",
        f"group={row_field(row, 'assignment_group')}",
        f"reassignments={row_field(row, 'reassignment_count')}",
        f"reopens={row_field(row, 'reopen_count')}",
        f"mods={row_field(row, 'sys_mod_count')}",
        f"sla={row_field(row, 'made_sla')}",
        f"notify={row_field(row, 'notify')}",
        f"closed_code={row_field(row, 'closed_code')}",
        f"location={row_field(row, 'location')}",
        f"ci={row_field(row, 'cmdb_ci')}",
    ]
    return collapse_whitespace(" ".join(parts))


def stage(repo_root: Path) -> tuple[int, int]:
    return stream_csv_to_txt(
        repo_root / SOURCE_CSV,
        repo_root / STAGED_TXT,
        render_row=render_row,
        required_columns=_REQUIRED,
    )
