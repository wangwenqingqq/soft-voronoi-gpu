# Experiment ledger

| Experiment | Hypothesis | Contract | Comparator / candidate | Added costs | Invariants | Evidence | Decision |
|---|---|---|---|---|---|---|---|
| softvoronoi_20261029_multivote_batch1 | Avoiding exact variable-length distances may repay GPU routing, posting aggregation, ordering and real gather | CONTRACT.md v1 | Same-binary Myers full scan / multi-assignment CSR | Dense vote reset, atomics, radix order, gather, launches | Same metric, query, tie order, top-k and full timing scope | runs/raw/dev-v1; runs/raw/test-v1; docs/INTEGRITY-AUDIT.md | Correct bounded prototype; rejected by feasibility gate: all 18 configurations lose query latency |

Implementation status, measured mechanism benefit, and thesis impact are separate decisions. A cheap scan is expected to be a difficult comparator; clustered synthetic success alone will not rescue the research thesis. No embedding baseline has been measured in v1.
