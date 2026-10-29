# Experiment v1: exact edit-distance CUDA soft Voronoi

## Changes
- Added a standalone CUDA/CUB executable for bounded exact byte Levenshtein.
- Added carry-correct multiword Myers, shared by scan, routing, setup and verify.
- Added GPU nearest-anchor memberships and compact CSR posting construction.
- Added GPU vote pooling, deterministic candidate ranking, physical-row ordering,
  actual candidate-byte gathering, and hierarchical exact top-10.
- Added direction-balanced paired complete-query timing and raw JSONL records.
- Added independent scalar DP tests and complete small-fixture pipeline checks.

## Files
- `soft_voronoi_bench.cu`
- `docs/core-design.md`

## Build
Use the actual toolkit binary if a relocated `nvcc` symlink fails to locate
headers. The tested build command is:

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo \
  -Xptxas=-v soft_voronoi_bench.cu -o build/soft_voronoi_bench
```

The initial compilation passes. The common distance/top-10 kernel uses 48
registers and 9,280 bytes shared memory with no compiler-reported spills. This
is resource evidence, not a correctness or performance result.

## Smoke and correctness
Run under the campaign GPU lock on the assigned otherwise-idle GPU:

```sh
./build/soft_voronoi_bench --self-test
./build/soft_voronoi_bench --db data/smoke_db.txt --queries data/smoke_queries.txt \
  --n 1024 --m 32 --kb 4,8,16 --candidate-frac .02,.05 \
  --repeats 1 --warmup 1 --cpu-check --output smoke.jsonl
```

The self-test checks 4,864 boundary/random byte-string pairs eight times and 25
full pipeline cases. Pipeline reference checks cover memberships, complete CSR
contents, query routing, votes, candidate membership/order, statistics and exact
top-10. GPU execution and sanitizer evidence must be recorded separately before
promoting any result.

A deterministic singleton-candidate fixture exercises fewer-than-10 candidates and
asserts all remaining output slots are INVALID. `--sample-check 64` independently
checks actual full-corpus query/database pairs outside online timing. Requested
allocation accounting includes resident/query scratch and temporary build peaks,
excluding CUDA runtime/library-internal allocations.
