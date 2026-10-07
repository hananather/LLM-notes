# Offline result analysis

Snapshot: 2026-10-07T18:26:50.806684+00:00

Counts below use complete saved runs and reconstructed task validation. The full paired cases, source-label references and deterministic examples are in the JSON files beside this report.

| Task | Split | Baseline correct | Model correct | Corrections | Regressions |
|---|---|---:|---:|---:|---:|
| Affiliation exact target sets | val | 74/100 | 89/100 | 16 | 1 |
| Affiliation exact target sets | test | 456/644 | 593/644 | 152 | 15 |
| NAICS synthetic decisions | dev | 10/16 | 13/16 | 4 | 1 |
| NOC synthetic decisions | dev | 12/16 | 14/16 | 2 | 0 |
| NAICS synthetic decisions | test | 18/32 | 28/32 | 11 | 1 |
| NOC synthetic decisions | test | 23/32 | 27/32 | 4 | 0 |
| Statistics one_shot | test | 29/32 | 29/32 | 3 | 3 |
| Statistics agent | test | 29/32 | 29/32 | 1 | 1 |
| Statistics discovery | test | 9/24 | 15/24 | 6 | 0 |

Statistics model-run status: **complete**.

NCES school continuity stopped before model calls: 300/300 probability-sample matches were correct. The separate changed-name-and-address panel was 72/100, with 96/100 targets retrieved; these populations are not pooled.

Affiliation scores include review-flagged selections. Automatic acceptance, review flags and domain-invalid responses are reported separately. Synthetic coding families and authored statistical questions do not estimate national workload accuracy. Exact source quotes verify text presence, not semantic entailment. API price is not human effort or operational savings.

Rebuild without model calls: `.venv/bin/python LLM_basics/nso-semantic-workflows/analyze.py`.
