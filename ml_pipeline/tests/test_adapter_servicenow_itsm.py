"""Tests for servicenow_itsm CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import servicenow_itsm
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
number,incident_state,active,category,subcategory,u_symptom,priority,impact,urgency,contact_type,assignment_group,reassignment_count,reopen_count,sys_mod_count,made_sla,notify,closed_code,location,cmdb_ci,sys_updated_at
INC0000001,New,true,Category 1,Sub 1,Symptom 1,3,2,2,Phone,Group A,0,0,1,true,false,,Location 1,CI-1,2016-02-29 01:23:00
"""

EXPECTED = (
    "2016-02-29 01:23:00 incident=INC0000001 state=New active=true category=Category 1 "
    "subcategory=Sub 1 symptom=Symptom 1 priority=3 impact=2 urgency=2 contact=Phone "
    "group=Group A reassignments=0 reopens=0 mods=1 sla=true notify=false closed_code= "
    "location=Location 1 ci=CI-1"
)


def test_render_row_exact_line():
    row = {
        "number": "INC0000001",
        "incident_state": "New",
        "sys_updated_at": "2016-02-29 01:23:00",
        "active": "true",
        "category": "Category 1",
        "subcategory": "Sub 1",
        "u_symptom": "Symptom 1",
        "priority": "3",
        "impact": "2",
        "urgency": "2",
        "contact_type": "Phone",
        "assignment_group": "Group A",
        "reassignment_count": "0",
        "reopen_count": "0",
        "sys_mod_count": "1",
        "made_sla": "true",
        "notify": "false",
        "closed_code": "",
        "location": "Location 1",
        "cmdb_ci": "CI-1",
    }
    assert servicenow_itsm.render_row(row) == EXPECTED


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=servicenow_itsm.render_row,
        required_columns=servicenow_itsm._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED
