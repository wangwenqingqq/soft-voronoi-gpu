# GPU soft Voronoi / multi-assignment ANN feasibility prototype

**Result: no query-latency win in the frozen 100,000-object experiment.** All
18 configurations lost to a same-binary exact CUDA scan. This is a reproducible
negative feasibility result, not a new-index claim or proof that all GPU metric
indexes fail. [Full results](runs/summary/test-v1/summary.md) ·
[Independent audit](docs/INTEGRITY-AUDIT.md) · [Contract](runs/CONTRACT.md).

## Final held-out snapshot

Same configuration in all three rows: N=100,000, m=512, kb=16, candidate cap=5%,
128 held-out queries, three paired repetitions. All other configurations and
losing rows are retained in the linked full table. Device: one NVIDIA RTX PRO
6000 Blackwell Server Edition; CUDA 13.1.115, driver 590.48.01.

| Workload | Recall@10 | Scan ms/query | Indexed ms/query | Median-ratio speedup |
|---|---:|---:|---:|---:|
| Real dictionary words | 93.67% | 0.056320 | 0.213639 | 0.264x |
| Clustered synthetic DNA | 98.83% | 0.157816 | 0.501222 | 0.315x |
| Independent random DNA | 18.59% | 0.215598 | 0.630155 | 0.342x |

Latencies are **synchronized host-complete medians**, including query H2D,
GPU pattern construction, routing, pooling, candidate sorting/gathering,
verification, top-10 and result D2H. Speedup here is the ratio of those medians;
the full report separately retains paired-geometric speedup and CUDA-event
stage timings. Database/index/scratch are resident, not reloaded per query.

The implementation supports exact global **unit-cost byte Levenshtein up to
256 bytes**, not L2 and not arbitrary host callbacks. Dataset words have lengths
1–29; synthetic DNA is variable-length and bounded to 128–256. Real queries are
held-out dictionary words, not a measured user-typo distribution. No embedding,
batched-query or million-object baseline has been measured.

## Implemented pipeline

1. **GPU build:** sample database anchors, exact object-anchor distances, nearest
   kb memberships, flat CSR posting arrays. One N*kb ID array plus m+1 offsets
   avoids separately allocated variable-length lists.
2. **GPU routing/pooling:** exact anchor distances, nearest kb cells, posting
   votes/deduplication, top-C vote selection. No CPU candidate processing.
3. **GPU verification:** sort candidates by physical/length order, actually
   gather their bytes into contiguous SoA blocks, exact Myers distances and
   hierarchical top-10. All ordering and copying costs remain in the timer.

Matched scan uses the **same Myers primitive and top-10 kernel**, with the same
length-sorted SoA advantage. Source and design:
[`soft_voronoi_bench.cu`](soft_voronoi_bench.cu), [design card](docs/core-design.md).

## Build and reproduce

Requirements: Linux, NVIDIA CUDA toolkit with CUB, C++17, Python 3.9+ standard
library, `flock`, `taskset`, and a user-approved idle GPU. No package install is
required. Adjust architecture to the target card. Never commandeer a busy GPU.

```sh
make NVCC=/usr/local/cuda-13.1/bin/nvcc ARCH=sm_120
python3 scripts/test_data.py
python3 scripts/prepare_data.py --output-dir data
# Add --direct-download only if the configured proxy is unsuitable.
export CUDA_VISIBLE_DEVICES=0  # choose an approved idle GPU, not blindly GPU 0
./build/soft_voronoi_bench --self-test
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/soft_voronoi_bench --self-test
compute-sanitizer --tool initcheck --error-exitcode 99 ./build/soft_voronoi_bench --self-test
compute-sanitizer --tool synccheck --error-exitcode 99 ./build/soft_voronoi_bench --self-test
bash scripts/run_campaign.sh dev runs/raw/my-dev
# Freeze implementation and parameters before reading the held-out evaluation.
bash scripts/run_campaign.sh test runs/raw/my-test
python3 scripts/summarize.py runs/raw/my-test/*.jsonl --output-dir runs/summary/my-test
```

The runner serializes its own work, checks for active compute processes on the
target GPU before each workload, pins eight CPU cores, and refuses to overwrite
an output directory. Coordinate allocation with other users; a file lock is not
a cluster reservation. Use the same lock for manual sanitizer/profiler runs.

The core is one independently compilable source file:

```sh
nvcc -O3 -std=c++17 -arch=sm_120 soft_voronoi_bench.cu -o soft_voronoi_bench
./soft_voronoi_bench --db data/words/db.txt --queries data/words/dev.txt \
  --workload words --n 100000 --m 512 --kb 4,8,16 \
  --candidate-frac .02,.05 --sample-check 64 --output words.jsonl
```

The pinned public word source and synthetic generation rules are in
[docs/data.md](docs/data.md). No raw corpus is redistributed. Source, workload,
compiler and binary hashes accompany raw measurements. To verify the shipped
matrix without a GPU:

```sh
python3 scripts/verify_results.py
python3 scripts/summarize.py runs/raw/test-v1/*.jsonl --output-dir /tmp/soft-voronoi-summary
```

Missing local datasets/binary are reported as unavailable, not as verified.
[Provenance aliases and the independent CPU oracle](runs/provenance/README.md)
explain how to reproduce the stronger checks.

## What passed, what did not

- Passed: 4,864 CPU-DP/GPU-Myers pairs repeated eight times; 25 complete pipeline
  fixtures, including a singleton candidate pool; memcheck/initcheck/synccheck;
  64 additional pairs per full workload; independent exhaustive CPU scans of
  query 0 for each 100,000-object database; 258 returned-pair DP checks.
- Failed feasibility screen: zero configurations combine >=95% canonical
  Recall@10 with >=1.5x host speedup and <=5% candidates. All 18 speedups are <1.
- Important implementation limits: routing uses one CTA; pooling sorts N vote
  keys; a small final candidate set does not mean a small posting union. These
  are measured/code-visible costs, not fundamental GPU lower bounds.
- Offline build timers are **partial component costs**, excluding some buffer
  allocation, parsing and scratch initialization. They are not all-in setup or
  a defensible exact amortization threshold.
- Not established: novel index, external state-of-the-art scan superiority,
  embedding superiority, high-batch throughput, arbitrary lengths/metrics,
  multi-seed confidence, or production readiness.

[Research decision and reopen conditions](runs/METHODOLOGY.md).
