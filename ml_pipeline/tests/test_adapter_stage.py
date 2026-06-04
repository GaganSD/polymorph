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

    (repo_root / "data/raw/cremev2").mkdir(parents=True)
    (repo_root / "data/raw/cremev2/original_label_syslog.csv").write_text(
        "Time,HostName,Component,PID_or_IP,Content,Tactic,Technique\n"
        "2023-01-01T00:00:00Z,host,svc,1,msg,Normal,Normal\n",
        encoding="utf-8",
    )

    (repo_root / "data/raw/it_incident").mkdir(parents=True)
    (repo_root / "data/raw/it_incident/incident_event_log.csv").write_text(
        "number,incident_state,sys_updated_at,active,category,subcategory,u_symptom,priority,impact,urgency,contact_type,assignment_group,reassignment_count,reopen_count,sys_mod_count,made_sla,notify,closed_code,location,cmdb_ci\n"
        "INC1,New,2016-01-01,true,C1,S1,S1,1,1,1,Phone,G1,0,0,1,true,false,,L1,CI1\n",
        encoding="utf-8",
    )

    (repo_root / "data/raw/aws_cloudtrail").mkdir(parents=True)
    (repo_root / "data/raw/aws_cloudtrail/nineteenFeaturesDf.csv").write_text(
        "eventTime,eventName,eventSource,awsRegion,userIdentitytype,eventType,userIdentityarn,userIdentityuserName,sourceIPAddress,userAgent,userIdentityprincipalId,errorCode,errorMessage,requestParametersinstanceType\n"
        "2017-01-01T00:00:00Z,GetObject,s3.amazonaws.com,us-east-1,Root,AwsApiCall,arn::root,,1.2.3.4,agent,p1,,,\n",
        encoding="utf-8",
    )

    (repo_root / "data/raw/python_tracebacks").mkdir(parents=True)
    (repo_root / "data/raw/python_tracebacks/stacktraces.json").write_text(
        '{"infile": "x", "data": [[1, "u", "cpython", "Traceback (most recent call last):\\r  File \\"a.py\\"\\rError: x"]]}',
        encoding="utf-8",
    )

    sec = repo_root / "data/raw/security_synth"
    sec.mkdir(parents=True)
    for fname, header_row in [
        (
            "okta_system_log.csv",
            "published,severity,eventType,displayMessage,actor.displayName,actor.alternateId,outcome.result,outcome.reason,client.ipAddress,client.geographicalContext.city,client.userAgent.rawUserAgent\n"
            "2024-01-01T00:00:00Z,INFO,evt,msg,actor,aid,OK,reason,1.2.3.4,city,ua\n",
        ),
        (
            "cisco_umbrella_dns.csv",
            "Timestamp,Identities,InternalIp,ExternalIp,Action,QueryType,ResponseCode,Domain,Categories,Verdict,URL\n"
            "2024-01-01T00:00:00Z,id,10.0.0.1,8.8.8.8,Allow,A,NOERROR,ex.com,Cat,OK,http://ex/\n",
        ),
        (
            "crowdstrike_network.csv",
            "timestamp,EventType,ConnectionDirection,LocalAddressIP4,LocalPort,RemoteAddressIP4,RemotePort,Protocol,Status,ImageFileName,CommandLine,UserName,ComputerName,Technique\n"
            "2024-01-01T00:00:00Z,Net,Out,10.0.0.1,1,8.8.8.8,2,TCP,OK,i.exe,c,u,h,T\n",
        ),
        (
            "crowdstrike_process.csv",
            "ProcessStartTime,EventType,ImageFileName,CommandLine,ParentBaseFileName,UserName,ComputerName,Technique\n"
            "2024-01-01T00:00:00Z,Proc,i.exe,c,p,u,h,T\n",
        ),
        (
            "crowdstrike_registry.csv",
            "timestamp,EventType,RegObjectName,RegValueName,RegStringValue,RegOperationType,ImageFileName,UserName,ComputerName,Technique\n"
            "2024-01-01T00:00:00Z,Reg,obj,val,str,Set,i.exe,u,h,T\n",
        ),
        (
            "proofpoint_email.csv",
            "time,sender,recipient,subject,headerFrom,spamScore,phishScore,mlxLabel,action,malwareName,sendingIp\n"
            "2024-01-01T00:00:00Z,a@x.com,b@y.com,sub,a@x.com,0,0,clean,Delivered,,1.2.3.4\n",
        ),
        (
            "zscaler_proxy.csv",
            "timestamp,user,department,location,clientip,serverip,hostname,url,urlcategory,action,requestmethod,responsecode,contenttype,useragent,threatname,rulelabel,appname\n"
            "2024-01-01T00:00:00Z,u,d,l,10.0.0.1,10.0.0.2,h,http://x/,News,Allow,GET,200,text/html,UA,,r,a\n",
        ),
        (
            "aws_cloudtrail.csv",
            "eventTime,eventName,eventSource,awsRegion,sourceIPAddress,userAgent,userIdentity.type,userIdentity.principalId,userIdentity.arn,userIdentity.userName,errorCode,requestParameters.bucketName\n"
            "2024-01-01T00:00:00Z,GetObject,s3.amazonaws.com,us-east-1,10.0.0.1,agent,Role,P,A,U,,b\n",
        ),
    ]:
        (sec / fname).write_text(header_row, encoding="utf-8")


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
    assert len(entries) == 17
    assert (tmp_path / distsys_synth.STAGED_TXT).exists()


def test_main_writes_manifest_and_prints_summary(tmp_path: Path, capsys):
    _seed_minimal_corpora(tmp_path)
    rc = main(["lamr-stage-corpora", str(tmp_path)])
    assert rc == 0

    manifest_path = tmp_path / "data/staged/MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == 19
    assert manifest[-1]["name"] == "apache_access"
    assert "staged_path" not in manifest[-1]

    captured = capsys.readouterr().out
    assert "Corpus staging summary" in captured
    assert "MANIFEST written to data/staged/MANIFEST.json" in captured
