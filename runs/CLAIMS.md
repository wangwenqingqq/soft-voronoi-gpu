# Claim-evidence ledger

Experiment: `softvoronoi_20261029_multivote_batch1`.
Frozen source checkpoint: `7dbd3de2304efce3dc49a98599d9278af9669615`.

| Claim | Status | Exact scope and evidence | Allowed wording / counterevidence |
|---|---|---|---|
| Correct exact distance and GPU candidate path on tested fixtures | measured | gates-final; independent CPU checks in runs/audit; integrity audit | Tests passed; no claim of proof for all byte strings or production certification |
| Full-query speedup at N100k | rejected | all 6912 pairs, 18 configs, test-v1; matched same-binary scan | This implementation loses on every fixed row; not a universal index impossibility |
| High recall at <=5% candidates | measured, workload-specific | clustered DNA kb16/C5%:98.83%; words maximum93.67%; random DNA maximum18.59% | High recall is synthetic-locality-specific; preserve real/hard-load failures |
| 5% candidate cap implies 5% total index work | rejected | kb16 final union about45–53% on average; posting visits separately recorded | Only exact candidate verification is capped; voting and dense ordering are not |
| Single-CTA routing causes underutilization relative to scan | inferred | source ownership and measured route stages; no Nsight counters | Plausible code-visible bottleneck, not isolated hardware-counter proof |
| Complete offline setup cost and amortization | unvalidated | partial build/layout timers only; allocation/parsing gaps in audit | Do not report all-in cost or exact amortization threshold |
| Better than embeddings or all GPU scans | unknown | no external embedding/optimized-library baseline | No evidence |
| New soft Voronoi index contribution | rejected as stated | NAPP/GENIE and related prior-art routing | Prototype only; a distinct mechanism and decisive same-contract advantage would be needed |
