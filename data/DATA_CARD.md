# Polymorph — Dataset Card

Corpora used for the deterministic-compression benchmark (the evidence gate) and
as raw input for OpenRouter teacher distillation. Fetched by
`scripts/fetch_datasets.sh`. The `data/` tree is git-ignored (large).

> Licenses verified 2026-06-04 (the dataset research report guessed Apache-2.0 for
> TrainTicket; the actual Zenodo record is **CC-BY-4.0** — attribution is required).

| Corpus | Path | License | Auth | Role |
|---|---|---|---|---|
| TrainTicket microservice logs + Jaeger traces + monitoring | `data/raw/trainticket/` | **CC-BY-4.0** (attribution required) | none | benchmark + distillation; `potentialAnomalies_*.txt` seed REVIEW/answer-token labels |
| server-logs (Apache access logs, 1M lines) | `data/raw/server_logs/logfiles.log` | **CC0-1.0** (public domain) | public | messy real-world benchmark corpus |
| logs-dataset (Kaggle kernel) | `data/raw/kaggle_logs/` | per kernel author | **Kaggle auth required** | additional log samples |

## Attribution (required for TrainTicket, CC-BY-4.0)

> "Anomalies in Microservice Architecture (train-ticket) based on version
> configurations." Zenodo record 6979726. https://doi.org/10.5281/zenodo.6979726
> Used under CC-BY-4.0.

Carry this attribution into any published benchmark results or redistributed
artifacts derived from TrainTicket.

## Notes & caveats

- **TrainTicket is synthetic integration-test traffic** (clean, uniform, highly
  templated). It tends to *overstate* deterministic-dedup ratios versus messy
  production logs. Always report benchmark numbers on `server_logs` (messy) too,
  and treat TrainTicket as the optimistic end of the range. (Outside-voice finding,
  2026-06-04 eng review.)
- **server-logs** are Apache access logs: high-cardinality, interleaved (every line
  a different IP/URL/user-agent). Consecutive run-length dedup finds little here by
  design — this is the honest hard case.
- **Kaggle kernels need auth**; `kaggle datasets download` for public CC0 datasets
  often works anonymously. See the script header for auth setup.
- **Answer-token survival proxy:** TrainTicket `potentialAnomalies_*.txt` files seed
  the benchmark's must-survive strings (a cheap stand-in for downstream answer
  accuracy until the full trace-QA eval set lands — TODOS.md, P1).

## Provenance

```
scripts/fetch_datasets.sh
  ├─ Zenodo 6979726  -> data/raw/trainticket/anomalies_microservice_trainticket_version_configurations/
  ├─ kaggle vishnu0399/server-logs -> data/raw/server_logs/logfiles.log
  └─ kaggle adepvenugopal/logs-dataset (auth) -> data/raw/kaggle_logs/
data/bench/trainticket_logs/   # curated LOGS_*.txt subset for quick benches
```
