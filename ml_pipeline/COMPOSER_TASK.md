# Composer task brief — corpus staging for Polymorph distillation

You are **Composer**, working as a supervised long-running agent on the
`lulu-polymorph` repo. Claude Code is working in parallel on the ML code
(stratified sampler, distillation orchestration, training glue). **This file is
your goal.** It is self-contained — you do not need the prior conversation.

## Context (1 paragraph)
Polymorph is an OSS, deterministic, reversible compressor for **audit logs +
production traces**, with a neural token-pruner trained by distilling an
extractive teacher (DeepSeek V3.2 on AWS Bedrock). To make the pruner
state-of-the-art we need **log-format diversity**. Seven corpora are staged on
disk; five are CSV and must be converted to uniform log-line text so the
distillation pipeline can chunk them like the others.

## YOUR GOAL (bounded, mechanical, parallel-safe)
Produce **`ml_pipeline/../data/staged/`** — every corpus as uniform,
line-oriented `.txt` (one realistic log line per record) — plus a
**`data/staged/MANIFEST.json`**. That's it. Do NOT dedup, sample, or call any
paid API (Claude owns sampling + the Bedrock build + all spend).

### Guardrails (hard rules)
- **Local + free only.** No network/API calls. No secrets. Never read/modify `.env`.
- **Never modify or delete anything under `data/raw/`** — read-only inputs.
- Write ONLY under `data/staged/` and your adapter code under
  `ml_pipeline/polymorph_lamr/distill/adapters/`.
- All outputs are byte-for-byte reproducible (no timestamps/randomness in code).
- Stream the CSVs (220k–467k rows) — do not load whole files into memory.
- Add a `pytest` test per adapter (≥90% coverage on the adapter module) with a
  tiny inline CSV fixture asserting the exact rendered line.

## The work

### 1. Five CSV→logline adapters
Create `ml_pipeline/polymorph_lamr/distill/adapters/` with one function per
dataset. Each reads the CSV (streaming, `csv.DictReader`, `encoding="utf-8"`) and
writes one log line per row to the staged `.txt`. Render EXACTLY these formats:

**a) `distsys_synth`** — `data/raw/distsys_synth/logdata.csv`
Columns: `(index),Timestamp,LogLevel,Service,Message,RequestID,User,ClientIP,TimeTaken`
Line: `{Timestamp} {LogLevel} [{Service}] {Message} request_id={RequestID} user={User} client_ip={ClientIP} time_taken={TimeTaken}`
→ `data/staged/distsys_synth.txt`

**b) `api_failures`** — `data/raw/api_failures/api_error_logs_with_root_causes_220k_rows.csv`
Columns: `timestamp,api_name,service_owner,environment,http_method,endpoint,status_code,error_type,root_cause,latency_ms,request_size_bytes,response_size_bytes,retry_count,is_retry_successful,client_ip,region,container_id,host_id,thread_id,log_level,error_message,resolution_action`
Line: `{timestamp} {log_level} {api_name} {http_method} {endpoint} status={status_code} error_type={error_type} latency_ms={latency_ms} retry={retry_count} retry_ok={is_retry_successful} env={environment} region={region} container={container_id} host={host_id} thread={thread_id} client_ip={client_ip} owner={service_owner} req_bytes={request_size_bytes} resp_bytes={response_size_bytes} msg="{error_message}" root_cause="{root_cause}" resolution="{resolution_action}"`
→ `data/staged/api_failures.txt`

**c) `alibaba_gpu`** — `data/raw/alibaba_gpu/job_info_df.csv` (weakest fit; render anyway)
Columns: `job_name,organization,gpu_model,cpu_request,gpu_request,worker_num,submit_time,duration,job_type`
Line: `submit_time={submit_time} job={job_name} org={organization} type={job_type} gpu_model={gpu_model} gpu_request={gpu_request} cpu_request={cpu_request} workers={worker_num} duration={duration}`
→ `data/staged/alibaba_gpu.txt`
(Ignore `node_info_df.csv` — infra metadata, not log lines.)

**d) `cicd_failures`** — `data/raw/cicd_failures/ci_cd_pipeline_failure_logs_dataset.csv`
Columns: `pipeline_id,run_id,timestamp,ci_tool,repository,branch,commit_hash,author,language,os,cloud_provider,build_duration_sec,test_duration_sec,deploy_duration_sec,failure_stage,failure_type,error_code,error_message,severity,cpu_usage_pct,memory_usage_mb,retry_count,is_flaky_test,rollback_triggered,incident_created`
Line: `{timestamp} {severity} {ci_tool} pipeline={pipeline_id} run={run_id} repo={repository} branch={branch} commit={commit_hash} lang={language} os={os} cloud={cloud_provider} stage={failure_stage} failure_type={failure_type} error_code={error_code} build_s={build_duration_sec} test_s={test_duration_sec} deploy_s={deploy_duration_sec} cpu_pct={cpu_usage_pct} mem_mb={memory_usage_mb} retry={retry_count} flaky={is_flaky_test} rollback={rollback_triggered} incident={incident_created} author={author} msg="{error_message}"`
→ `data/staged/cicd_failures.txt`

**e) `win_events`** — `data/raw/win_events/eventos.csv` (Windows Event Viewer; **pt-BR
column headers AND level values**, e.g. `Nível`=`Informações`/`Erro`/`Aviso` — keep
them as-is, multilingual metadata is intentional diversity. Many columns are empty.)
Columns (pt-BR): `Nível,Data e Hora,Fonte,Identificação do Evento,Categoria da Tarefa,Identificação do Processo,Identificação de Thread,Identificação do processador,Identificação da Sessão,Nome de Origem do Evento,Usuário,Palavras-chave,Código Operacional,Log,Computador,Tempo do Kernel,Tempo do Usuário,Tempo do Processador,ID de Correlação,ID de Correlação Relativa,Description1,Description2,Description3`
Line: `{Data e Hora} {Nível} {Fonte} event_id={Identificação do Evento} task={Categoria da Tarefa} log={Log} computer={Computador} {desc}`
where `{desc}` = the non-empty values of `Description1`, `Description2`, `Description3`
joined by a single space. Note the descriptions contain literal `%%` — leave as-is.
→ `data/staged/win_events.txt`

Edge cases (ALL adapters): skip rows missing required columns; replace embedded
newlines in any field with a space; collapse runs of whitespace produced by empty
fields to a single space and strip the line; leave scalar empty fields as empty
(`field=`); do not quote-escape beyond the literal `"..."` shown.

### 2. Stage the two existing text corpora (copy, don't convert)
These are already line-oriented; just reference them in the manifest (no copy
needed, record their paths):
- `trainticket_traces` → `data/bench/trainticket_logs/*.txt` (Spring/Jaeger traces)
- `apache_access`      → `data/raw/server_logs/logfiles.log` (Apache access)

### 3. `data/staged/MANIFEST.json`
A JSON list; one object per corpus:
`{ "name", "format", "source", "staged_path"|"source_glob", "line_count", "bytes", "samples": [<first 3 lines>] }`

## Acceptance criteria (your definition of done)
- `data/staged/{distsys_synth,api_failures,alibaba_gpu,cicd_failures,win_events}.txt`
  exist; line counts match source row counts (minus skipped rows — log how many skipped).
- `MANIFEST.json` validates and lists all 7 corpora with 3 real sample lines each.
- `pytest ml_pipeline/tests/` passes; new adapter tests ≥90% coverage.
- Print a summary: per-corpus line count + total staged bytes.

## Handing back
When done, leave a short note (a `data/staged/STAGED.md` or a PR comment) with the
manifest summary. Claude's stratified sampler then reads `MANIFEST.json` + the
staged `.txt` files to build the distillation input. Do not run distillation.
