# Experiment: data preparation and result aggregation

## Changes made

- Added deterministic, hash-manifested real dictionary and two contrasting synthetic DNA workloads.
- Added disjoint database/development/test splitting, exact duplicate accounting and no-overwrite protections.
- Added complete-configuration result aggregation with separate GPU-event and host-completion timing scopes.
- Added CPU unit tests for generation, preservation, quality consistency and summary arithmetic.

## Why

The feasibility decision requires both real data and favorable/adversarial synthetic locality regimes, without query leakage or favorable-case filtering. Timing and quality denominators must remain explicit.

## Owned files

- `scripts/prepare_data.py`
- `scripts/summarize.py`
- `scripts/test_data.py`
- `docs/data.md`

## Correctness

`python3 scripts/test_data.py` passed 10 CPU tests on 2026-10-29. No GPU tests or performance measurements are performed by these scripts. Full data generation completion and hashes are recorded in `data/manifest.json` rather than asserted here before generation.

## How to run

```bash
python3 scripts/test_data.py
python3 scripts/prepare_data.py --output-dir data --direct-download
python3 scripts/summarize.py runs/outputs/final/*.jsonl --output-dir runs/summary/final
```

## Pre-measurement contract alignment

Before any benchmark measurement, a review found that the word shuffle used a derived seed instead of the contract's literal seed 1729. The original prepared files were preserved privately; the generator was corrected to direct seed 1729 and regenerated. All six synthetic split hashes remained identical. The authoritative manifest is the corrected `data/manifest.json`; no performance result informed this correction.

## Pre-final summary audit improvements

Before final evaluation, promoted host-completion latency to the first Markdown results table, separated paired-geometric speedup from marginal-median ratios, added output-transfer and stage-sum/residual statistics, and rejected duplicate input paths and duplicate process/configuration/query/repeat pairs. The expanded CPU suite passes 13 tests. The data generator and every dataset file remain unchanged.
