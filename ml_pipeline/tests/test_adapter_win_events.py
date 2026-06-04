"""Tests for win_events CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import win_events
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
Nível,Data e Hora,Fonte,Identificação do Evento,Categoria da Tarefa,Identificação do Processo,Identificação de Thread,Identificação do processador,Identificação da Sessão,Nome de Origem do Evento,Usuário,Palavras-chave,Código Operacional,Log,Computador,Tempo do Kernel,Tempo do Usuário,Tempo do Processador,ID de Correlação,ID de Correlação Relativa,Description1,Description2,Description3
Informações,6/28/2020 10:57:55 PM,MSSQLSERVER,17890,Server,0,0,,,,,Clássico,,Application,DESKTOP-UJI8L8D,,,,00000000-0000-0000-0000-000000000000,00000000-0000-0000-0000-000000000000,"A significant part of sql server process memory has been paged out. Duration: 1512 seconds. Working set (KB): 93448, committed (KB): 234032, memory utilization: 39%%.",,
"""

EXPECTED = (
    "6/28/2020 10:57:55 PM Informações MSSQLSERVER event_id=17890 task=Server "
    "log=Application computer=DESKTOP-UJI8L8D A significant part of sql server "
    "process memory has been paged out. Duration: 1512 seconds. Working set (KB): "
    "93448, committed (KB): 234032, memory utilization: 39%%."
)


def test_render_row_exact_line():
    row = {
        "Nível": "Informações",
        "Data e Hora": "6/28/2020 10:57:55 PM",
        "Fonte": "MSSQLSERVER",
        "Identificação do Evento": "17890",
        "Categoria da Tarefa": "Server",
        "Log": "Application",
        "Computador": "DESKTOP-UJI8L8D",
        "Description1": (
            "A significant part of sql server process memory has been paged out. "
            "Duration: 1512 seconds. Working set (KB): 93448, committed (KB): 234032, "
            "memory utilization: 39%%."
        ),
        "Description2": "",
        "Description3": "",
    }
    assert win_events.render_row(row) == EXPECTED


def test_render_row_joins_multiple_descriptions():
    row = {
        "Nível": "Erro",
        "Data e Hora": "1/1/2020 12:00:00 AM",
        "Fonte": "Service",
        "Identificação do Evento": "1",
        "Categoria da Tarefa": "Task",
        "Log": "System",
        "Computador": "HOST",
        "Description1": "part one",
        "Description2": "part two",
        "Description3": "part three",
    }
    line = win_events.render_row(row)
    assert line is not None
    assert line.endswith("part one part two part three")


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=win_events.render_row,
        required_columns=win_events._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED


def test_stage_integration(tmp_path: Path):
    raw = tmp_path / "data/raw/win_events"
    raw.mkdir(parents=True)
    (raw / "eventos.csv").write_text(FIXTURE_CSV, encoding="utf-8")
    written, skipped = win_events.stage(tmp_path)
    assert written == 1
    assert skipped == 0
