"""Tests for corpus staging orchestration."""

import json
from pathlib import Path

from polymorph_lamr.distill.adapters import distsys_synth
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt
from polymorph_lamr.distill.adapters.stage import main, stage_all


def _seed_minimal_corpora(repo_root: Path) -> None:
    raw = repo_root / "data/raw/distsys_synth"
    raw.mkdir(parents=True)
    (raw / "logdata.csv").write_text(
        ",Timestamp,LogLevel,Service,Message,RequestID,User,ClientIP,TimeTaken\n"
        "0,2023-11-20T08:40:50.664842,WARNING,ServiceA,Performance Warnings,"
        "6743,User96,192.168.1.102,28ms\n",
        encoding="utf-8",
    )

    for name, fname, header, row in [
        (
            "api_failures",
            "api_error_logs_with_root_causes_220k_rows.csv",
            "timestamp,api_name,service_owner,environment,http_method,endpoint,status_code,error_type,root_cause,latency_ms,request_size_bytes,response_size_bytes,retry_count,is_retry_successful,client_ip,region,container_id,host_id,thread_id,log_level,error_message,resolution_action",
            "2024-01-01 00:00:00,inventory-api,team-beta,dev,DELETE,/v1/x,503,Timeout,cause,1,1,1,0,False,127.0.0.1,us-east-1,c,h,t,INFO,msg,res",
        ),
        (
            "alibaba_gpu",
            "job_info_df.csv",
            "job_name,organization,gpu_model,cpu_request,gpu_request,worker_num,submit_time,duration,job_type",
            "1,2,A10,1.0,1.0,1,0.0,1.0,HP",
        ),
        (
            "cicd_failures",
            "ci_cd_pipeline_failure_logs_dataset.csv",
            "pipeline_id,run_id,timestamp,ci_tool,repository,branch,commit_hash,author,language,os,cloud_provider,build_duration_sec,test_duration_sec,deploy_duration_sec,failure_stage,failure_type,error_code,error_message,severity,cpu_usage_pct,memory_usage_mb,retry_count,is_flaky_test,rollback_triggered,incident_created",
            "p,r,2025-01-01T00:00:00,J,r,b,c,a,Py,linux,AWS,1,1,1,s,t,e,m,M,1.0,1,0,False,False,False",
        ),
        (
            "win_events",
            "eventos.csv",
            "Nível,Data e Hora,Fonte,Identificação do Evento,Categoria da Tarefa,Identificação do Processo,Identificação de Thread,Identificação do processador,Identificação da Sessão,Nome de Origem do Evento,Usuário,Palavras-chave,Código Operacional,Log,Computador,Tempo do Kernel,Tempo do Usuário,Tempo do Processador,ID de Correlação,ID de Correlação Relativa,Description1,Description2,Description3",
            "Informações,1/1/2020,Src,1,Task,0,0,,,,,,,App,HOST,,,,,,,desc,,",
        ),
    ]:
        d = repo_root / "data/raw" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(f"{header}\n{row}\n", encoding="utf-8")

    bench = repo_root / "data/bench/trainticket_logs"
    bench.mkdir(parents=True)
    (bench / "a.txt").write_text("tt line 1\ntt line 2\n", encoding="utf-8")

    apache = repo_root / "data/raw/server_logs"
    apache.mkdir(parents=True)
    (apache / "logfiles.log").write_text("apache 1\napache 2\napache 3\n", encoding="utf-8")


def test_stream_skips_when_render_returns_none(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a\n1\n", encoding="utf-8")

    def render(_row):
        return None

    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path, out_path, render_row=render, required_columns=["a"]
    )
    assert written == 0
    assert skipped == 1


def test_stage_all_writes_manifest_inputs(tmp_path: Path):
    _seed_minimal_corpora(tmp_path)
    entries = stage_all(tmp_path)
    assert len(entries) == 5
    assert (tmp_path / distsys_synth.STAGED_TXT).exists()


def test_main_writes_manifest_and_prints_summary(tmp_path: Path, capsys):
    _seed_minimal_corpora(tmp_path)
    rc = main(["lamr-stage-corpora", str(tmp_path)])
    assert rc == 0

    manifest_path = tmp_path / "data/staged/MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == 7
    assert manifest[-1]["name"] == "apache_access"
    assert "staged_path" not in manifest[-1]

    captured = capsys.readouterr().out
    assert "Corpus staging summary" in captured
    assert "MANIFEST written to data/staged/MANIFEST.json" in captured
