# Frozen feasibility contract v1 — 2026-10-29

## Question and novelty boundary
Does a GPU NAPP-like multi-assignment inverted index save end-to-end latency over a competent matched CUDA exact scan at useful original-metric recall? The nearest-anchor / overlap-vote / exact-rerank mechanism is prior art, not a novelty claim. This experiment does not establish superiority to embeddings, commercial search libraries, or all metric spaces.

## Semantics and workloads
- Exact unit-cost global byte Levenshtein distance. The implementation bound is 0..256 bytes; benchmark records contain nonempty lowercase ASCII words or DNA strings. This is not an arbitrary opaque CPU callback and not an unbounded-string implementation.
- N=100,000 database records per workload; m=512 distinct database anchors sampled independently of queries with seed 42; kb in {4,8,16}; candidate budgets C in {2,000,5,000}. Database and query assignments use the same kb.
- Words: public English word list, unique lowercase ASCII words; shuffled with seed 1729; disjoint database/dev/test partitions.
- Clustered DNA: 256 random length-192 centers; independent sequences with approximately 5% substitutions and 2% insertion and deletion probabilities; lengths restricted to 128..256. Intended locality-positive synthetic control.
- Random DNA: independent uniform bases and lengths 128..256; intended concentration/poor-routing synthetic control. No result-dependent data selection.
- Each workload has 32 development and 128 final held-out queries. Exact generation, immutable upstream revision, file hashes and length statistics are recorded by scripts/prepare_data.py in data/manifest.json. No test query content or test answers may inform implementation or parameters.
- Data preparation may materialize test files, but development runs read only dev files. The final evaluation is a separate pass after implementation and parameters are frozen. All six configurations and all three workloads are reported, not only the best row.

## Algorithm and comparators
- Index build: GPU computes object-anchor distances and nearest memberships, GPU constructs compact CSR postings (offsets m+1, row IDs N*kb); no CPU membership or posting computation.
- Query: GPU anchor routing, GPU posting vote aggregation/deduplication and candidate selection. Zero-vote records are not candidates. Budget is an upper bound, not forced padding. Vote ties break deterministically by original input ID.
- GPU sorting/bucketing and real candidate-byte gathering into a regular SoA are part of online cost. Exact GPU verification and final top-10 are included.
- Matched scan evaluates every database record with the SAME exact Myers primitive and SAME hierarchical top-10 machinery as verification. Both paths receive length-sorted, coalesced database storage. No CPU scan timing is used as the speedup denominator.
- Canonical top-10 uses ascending (distance, original input ID). Ground truth is matched full GPU scan, independently checked against scalar CPU dynamic programming on small fixtures and sampled full-query pairs. GPU memberships and query candidate path must pass small-fixture reference checks before final measurement.

## Measurement denominator
- Single-query, database/index/scratch resident on GPU. Setup, file parsing, host database sorting, allocation and index build are excluded online and reported separately. They are not hidden in the speedup denominator.
- Timed request begins before query input H2D and GPU query-mask construction; ends after top-10 and counters D2H complete. Each request has a CUDA-event total and a host steady-clock total with completion synchronization. Host-complete latency is primary; CUDA-event totals and stage breakdowns are complementary.
- Includes all launches, dense vote reset, posting traversal, candidate ordering, length ordering, gathering, exact distance, top-k, transfers and host submission. No precomputed query distance, routing, candidates, or answer replay.
- Per configuration: 3 warmup requests per path, then 3 repetitions of every query. Alternate matched scan/ANN and ANN/matched scan order; retain every raw pair and repetition. No outlier deletion. No CUDA Graphs in v1.
- Report p10/median/p90 host and device latency, ratio of marginal medians, geometric mean of paired scan/ANN ratios, order splits, mean canonical Recall@10 and per-query recall distribution, actual candidate %, posting visits, union %, and per-stage times. Repeated quality counts must not multiply query sample size.
- Build time and storage/workspace bytes are reported. Amortization calculations, if any, use measured per-query savings and clearly state their scope.

## Hardware and provenance
- Single NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, approximately 96 GB; driver 590.48.01; CUDA 13.1.115. GPU clocks and power are not changed; current values/state are captured per run. Other GPUs share the host, so CPU contention remains a caveat.
- Serialize all own GPU work using flock; pin host CPU work to cores 0..7. Check target GPU users, processes, memory and utilization immediately before launch. Do not use another user's occupied GPU or kill any process.
- Expected peak device memory <2 GiB and host RAM <2 GiB, excluding CUDA runtime overhead. Start with a small fixture and use bounded timeouts.
- Record source, evaluator, generator, data and binary SHA256, compiler flags/resource output, environment and GPU state in each run manifest BEFORE measurements. Baseline and candidate are functions in the same frozen binary, preventing divergent distance-kernel versions. No pre-existing baseline commit exists for this new prototype.

## Gates and interpretation
1. Build; CPU DP versus GPU Myers boundary/random correctness; complete small-data scan and index routing/candidate reference checks.
2. Compute Sanitizer memcheck, initcheck and synccheck on the small full pipeline; repeated-query stress. Any unresolved error means unvalidated timings, not a performance claim.
3. Development smoke only; fix implementation bugs, not dataset/quality definitions. Keep rejected/raw runs.
4. Freeze source/binary manifests, run the entire held-out matrix in a separate process per workload, then have a different agent perform a read-only integrity/code/results audit.
5. Feasibility screen (not publication acceptance): a useful positive row has mean canonical Recall@10 >=0.95, <=5% candidates, median host speedup >=1.5 and favorable geometric paired speedup in both ordering directions. Also report the >=0.90 frontier. A synthetic-only positive is not evidence of general real-workload benefit.
6. Report failures symmetrically. Failure of this implementation does not prove all inverted indexing impossible; a pass does not resolve NAPP/GENIE/misi novelty or defeat embedding methods. No graduation/mainline recommendation based on kernel-only or low-recall wins.

## Immutable attachments
This file is not edited after creation. Data/source identities are sealed in separately timestamped run manifests once generation/build complete. Corrections require a new explicitly versioned contract and retained prior evidence, not silent changes here.
