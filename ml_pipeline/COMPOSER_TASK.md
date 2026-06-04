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

---

# BATCH 2 — expand REAL-log diversity (2026-06-05)

Batch 1 (above) is **merged** (PR #2). Batch 2 adds **12 new corpora** from 5 new
sources. Claude already downloaded + assessed everything; your job is purely the
mechanical adapter + staging work, same pattern as Batch 1 (`adapters/_common.py`,
`stream_csv_to_txt`, one staged `.txt` + one MANIFEST entry per corpus, ≥90%
test coverage, identical guardrails: local/free only, never touch `data/raw/`,
stream large files, byte-reproducible).

## DO NOT TOUCH these (assessed and dropped — leave the raw files, add no adapter)
- `data/raw/k8s_structured/`, `data/raw/k8s_anomaly/` — CICFlowMeter *network-flow numeric features*, not k8s logs.
- `data/raw/servicenow_incident/` — a 119,999-row subset of `it_incident` (use it_incident instead).
- `data/raw/asterisk_otel/` — a 319-byte metadata stub, not a corpus.
- `data/raw/cremev2/{label_accounting,label_traffic,label_syslog}.csv` — numeric/atop/flow/one-hot-featurized. Use ONLY `original_label_syslog.csv`.
- `data/raw/aws_cloudtrail/dec12_18features.csv` — mangled (truncated timestamps/IPs). Use ONLY `nineteenFeaturesDf.csv`.

## New single-output adapters (same contract as Batch 1: `SOURCE`, `STAGED_TXT`, `stage(repo_root)->(written,skipped)`)

### a) `syslog_cremev2` — `data/raw/cremev2/original_label_syslog.csv`
Cols: `Time,HostName,Component,PID_or_IP,Content,EventId,EventTemplate,ParameterList,Timestamp,Label,Tactic,Technique,SubTechnique,Label_lifecycle`
Required: `Time,HostName,Component,Content`
Line: `{Time} {HostName} {Component}[{PID_or_IP}]: {Content} tactic={Tactic} technique={Technique}`
→ `data/staged/syslog_cremev2.txt`

### b) `servicenow_itsm` — `data/raw/it_incident/incident_event_log.csv`  (NOT servicenow_incident)
Required: `number,incident_state,sys_updated_at`
Line: `{sys_updated_at} incident={number} state={incident_state} active={active} category={category} subcategory={subcategory} symptom={u_symptom} priority={priority} impact={impact} urgency={urgency} contact={contact_type} group={assignment_group} reassignments={reassignment_count} reopens={reopen_count} mods={sys_mod_count} sla={made_sla} notify={notify} closed_code={closed_code} location={location} ci={cmdb_ci}`
→ `data/staged/servicenow_itsm.txt`   (values are anonymized like `Category 55` — keep as-is; `?` = missing, keep as-is.)

### c) `cloudtrail_flaws` — `data/raw/aws_cloudtrail/nineteenFeaturesDf.csv`  (the 19-feature file ONLY)
Required: `eventTime,eventName,eventSource`
Line: `{eventTime} {eventName} {eventSource} region={awsRegion} identity_type={userIdentitytype} event_type={eventType} arn={userIdentityarn} user={userIdentityuserName} src_ip={sourceIPAddress} agent="{userAgent}" principal={userIdentityprincipalId} err={errorCode} err_msg="{errorMessage}" req_instance_type={requestParametersinstanceType}`
→ `data/staged/cloudtrail_flaws.txt`

### d) `python_tracebacks` — `data/raw/python_tracebacks/stacktraces.json`  (JSON, 418MB — DO NOT json.load the whole file)
Structure: `{"infile": "...", "data": [[id, github_url, "cpython", "<traceback string>"], ...]}`. The traceback is element **index 3** of each `data` item; it uses `\r` / `\n` between frames.
- Add a streaming helper to `_common.py`: `stream_json_array_to_txt(json_path, out_path, *, array_key, render_item)` that scans for `"{array_key}": [` then yields each top-level `[...]` element via **bracket-depth scanning that respects JSON string state** (track in-string + backslash-escape so `[`/`]`/`"` inside the traceback text don't break framing), `json.loads`-ing each element. Memory-bounded; never loads the whole file.
- `render_item(item)`: take `item[3]`; skip (return None) if it's falsy or lacks any of `Traceback`/`File "`/`Error`; replace `\r`/`\n` with a single space and `collapse_whitespace` → one line per traceback.
→ `data/staged/python_tracebacks.txt`   (~382k tracebacks expected)

## New MULTI-output adapter: `security_synth` (8 vendor CSVs → 8 staged files + 8 MANIFEST entries)
One module exposing `stage_corpora(repo_root) -> list[dict]` (one dict per vendor:
`{"name","source","staged_path","written","skipped"}`). `stage.py` must call this and
extend the staged-entries list (generalize the single-output loop, or special-case it).
Each vendor = its own corpus name so the sampler balances them independently. Required
columns = the first 3 listed in each render. Skip columns absent from the real header.

- `okta_auth` ← `okta_system_log.csv`:
  `{published} {severity} okta event={eventType} msg="{displayMessage}" actor={actor.displayName} actor_id={actor.alternateId} result={outcome.result} reason="{outcome.reason}" ip={client.ipAddress} city={client.geographicalContext.city} ua="{client.userAgent.rawUserAgent}"`
  → `data/staged/okta_auth.txt`
- `cisco_dns` ← `cisco_umbrella_dns.csv`:
  `{Timestamp} cisco_umbrella identity={Identities} int_ip={InternalIp} ext_ip={ExternalIp} action={Action} qtype={QueryType} rcode={ResponseCode} domain={Domain} categories={Categories} verdict={Verdict} url={URL}`
  → `data/staged/cisco_dns.txt`
- `crowdstrike_network` ← `crowdstrike_network.csv`:
  `{timestamp} crowdstrike {EventType} dir={ConnectionDirection} local={LocalAddressIP4}:{LocalPort} remote={RemoteAddressIP4}:{RemotePort} proto={Protocol} status={Status} image={ImageFileName} cmd="{CommandLine}" user={UserName} host={ComputerName} technique={Technique}`
  → `data/staged/crowdstrike_network.txt`
- `crowdstrike_process` ← `crowdstrike_process.csv`:
  `{ProcessStartTime} crowdstrike {EventType} image={ImageFileName} cmd="{CommandLine}" parent={ParentBaseFileName} user={UserName} host={ComputerName} technique={Technique}`
  → `data/staged/crowdstrike_process.txt`
- `crowdstrike_registry` ← `crowdstrike_registry.csv`:
  `{timestamp} crowdstrike {EventType} reg_obj={RegObjectName} reg_val={RegValueName} reg_str="{RegStringValue}" op={RegOperationType} image={ImageFileName} user={UserName} host={ComputerName} technique={Technique}`
  → `data/staged/crowdstrike_registry.txt`
- `proofpoint_email` ← `proofpoint_email.csv`:
  `{time} proofpoint sender={sender} recipient={recipient} subject="{subject}" from={headerFrom} spam={spamScore} phish={phishScore} mlx_label={mlxLabel} action={action} malware={malwareName} sending_ip={sendingIp}`
  → `data/staged/proofpoint_email.txt`
- `zscaler_proxy` ← `zscaler_proxy.csv`:
  `{timestamp} zscaler user={user} dept={department} loc={location} client_ip={clientip} server_ip={serverip} host={hostname} url={url} cat={urlcategory} action={action} method={requestmethod} status={responsecode} ctype={contenttype} ua="{useragent}" threat={threatname} rule={rulelabel} app={appname}`
  → `data/staged/zscaler_proxy.txt`
- `cloudtrail_synth` ← `aws_cloudtrail.csv`:
  `{eventTime} {eventName} {eventSource} region={awsRegion} src_ip={sourceIPAddress} identity_type={userIdentity.type} principal={userIdentity.principalId} arn={userIdentity.arn} user={userIdentity.userName} agent="{userAgent}" err={errorCode} bucket={requestParameters.bucketName}`
  → `data/staged/cloudtrail_synth.txt`

## Wiring + acceptance (Batch 2)
- Register all new corpora in `stage.py` and `manifest.py` so `MANIFEST.json` lists **all 19 corpora** (7 from Batch 1 + 12 here). `lamr-stage-corpora` prints the per-corpus line/skip/byte summary.
- One `pytest` test per new adapter (tiny inline CSV/JSON fixture asserting the exact rendered line), ≥90% coverage on the adapters package, full suite green.
- Same edge-case rules as Batch 1 (sanitize embedded newlines, collapse whitespace, utf-8, skip rows missing required columns, stream). The JSON helper additionally must not load the whole file.
