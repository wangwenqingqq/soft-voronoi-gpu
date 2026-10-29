# Independent prototype integrity audit

Date: 2026-10-29.

**Verdict: PASS for the frozen single-query recall and query-latency evidence;
LIMITED for complete offline-cost accounting and broader research claims.**
No evidence of fabricated results, omitted losing configurations, answer replay,
or deliberately weakened scan was found in the inspected material. This is a
bounded artifact audit, not an exhaustive full-history misconduct clearance.

## Independence and scope

The reviewer did not implement the core, generate the datasets, or launch GPU
measurements. The review inspected the frozen contract, CUDA source, generator,
runner, aggregator, compiler records, gate logs, development/final manifests,
all final query records, and relevant current-turn action records. It independently
reran 13 CPU unit tests, recomputed result checks, checked the remote binary hash
read-only, and performed additional CPU dynamic-programming oracle checks after
the final evaluation completed. No held-out query contents were used for design
or parameter selection. The reviewer did not inspect every historical session
record or internal reasoning and did not independently rerun GPU timing.

Public evidence paths are repository-relative. Raw conversations, machine
configuration, and private source-corpus files are not audit deliverables.

## Frozen identity

Baseline and candidate execute from the same binary. The source checkpoint is
`7dbd3de2304efce3dc49a98599d9278af9669615`.

| Artifact | SHA-256 |
|---|---|
| `runs/CONTRACT.md` | `16699eb1b56df19c827ee440442abdbb1c35eb5addc3cabda9c07cb8de5469b5` |
| `soft_voronoi_bench.cu` | `6bd003a60220565ef27d3c5d97a1b4d17166c9e1826c579e1a3e23fd30676973` |
| Benchmark binary | `b4debd6522fbf827fa5ddf0d552ffb255f2e517672e70f125631fa96fac26eb3` |
| `data/manifest.json` | `221d403a2add95f22827646681473fe7169901abaefe1f9c92fd2a2cffa70344` |
| `runs/raw/test-v1/manifest.json` | `8f388191f177a3e9c3eb35e22c2f1d8362297d2e04b13dca41f27791426b2aec` |

All 14 locally available files named in the final run manifest matched their
hashes; the fifteenth, the executable retained on the GPU host, matched via an
independent read-only remote SHA-256 check. Compiler command/resource records
and all three raw result files also matched across local and remote copies.
Every shared development/final manifest entry was identical: no implementation,
parameter, evaluator, generator or database change followed the development
results. Final query file hashes match the pre-generated held-out manifest.

The final run completed at 09:25:37 UTC on 2026-10-29. These are initial
single-process-per-workload measurements, not independent multi-process
confidence estimates.

## Correctness and sanitizer gates

- Gate logs in `runs/raw/gates-final/` show 4,864 boundary/random string pairs,
  eight repetitions, and 25 complete pipeline fixtures passing.
- The pipeline reference checks cover memberships, complete CSR contents,
  query routing, votes, candidate membership/order, statistics and exact top-10.
- The explicit singleton fixture asserts one candidate and nine invalid result
  slots, exercising the fewer-than-ten-candidates path.
- Compute Sanitizer memcheck, initcheck and synccheck each report zero errors
  on the final binary. These logs are correctness evidence, not timing evidence.
- Each final workload records 64 deterministic full-database/query pair checks
  against scalar CPU dynamic programming outside the timed section.
- The independent reviewer additionally computed an exhaustive **100,000-object
  scalar CPU dynamic-programming scan for held-out query 0 in each workload**.
  All three canonical top-10 lists exactly matched the GPU scan, including ties.
- A second independent Python DP checker verified 258 distinct returned
  query/object distances across query indices 0, 31, 64 and 127: words 60,
  clustered DNA 41, random DNA 157. All scan/ANN distances matched.
- `scripts/test_data.py`: 13 tests passed independently during the final review.

The extra CPU oracle uses the ordinary recurrence
`D(i,j)=min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+[a_i != b_j])`, initialized with
`D(0,j)=j` and `D(i,0)=i`, then sorts `(distance, original input ID)`. It does not
reuse the bit-parallel Myers implementation. The compiled checker's source
SHA-256 was `3ec1052f88a82041361b76d30bbfdc09d052a99b3dd6a9a1b7695bdc68e192d9`.
These extra checks strengthen, but do not constitute a proof of, correctness for
every possible 256-byte input.

## Completeness and independent recomputation

Each of words, clustered DNA and random DNA has exactly six configurations
(kb=4/8/16, C cap=2%/5%), 128 query IDs and three repetitions. All **6,912 paired
query records** are present, with no duplicated or missing configuration/query/
repeat key. Every configuration has 192 scan-then-ANN and 192 ANN-then-scan pairs.

Independent checks confirmed:

- original-metric canonical Recall@10 and supplemental tie-aware recall;
- sorted, unique valid result IDs and deterministic answers across repetitions;
- identical scan answers across all configurations for a given query;
- candidate count <= union count and candidate capacity, with posting visits
  recorded separately;
- positive host/device timings and correct AB/BA order labels;
- routing + pooling + gathering + verification + output equals the CUDA-event
  total, with maximum rounding residual below `5.04e-8` milliseconds;
- all host paired-geometric and marginal-median speedup estimators in the
  summary agree with recomputation from raw records;
- summary input hashes match the raw files and all 18 rows are retained.

| Workload | Maximum canonical Recall@10 among six rows | Host paired-geometric speedup range |
|---|---:|---:|
| Real dictionary words | 93.67% | 0.268–0.316x |
| Clustered synthetic DNA | 98.83% | 0.330–0.354x |
| Independent random DNA | 18.59% | 0.327–0.343x |

These are all negative speed results. High recall on clustered synthetic data
is not presented as a real-workload or universal metric-space success.

## Fairness and timing review

`soft_voronoi_bench.cu` uses the same multiword Myers primitive and hierarchical
exact top-10 for scan and indexed verification. Both get length-sorted,
character-position-major database storage. The scan is not a naive CPU or
quadratic-GPU-DP comparator.

The index build computes distances, memberships, posting sort and prefix offsets
on GPU. CSR uses one contiguous posting array plus m+1 offsets, not separately
allocated linked lists. Query routing, posting voting, candidate selection,
physical-row ordering and actual candidate-byte gathering stay on GPU. Zero-vote
entries remain invalid and are not credited as candidates.

The query timers (source lines 517–575 at the frozen checkpoint) include H2D,
mask preparation, launches, sorting/gathering, exact verification, top-k and D2H.
CUDA-event synchronization precedes recorded host completion. No CPU-enqueue-only
latency or omitted gather was found. Host-complete values are the primary table;
CUDA event totals and stages remain separate. Repeated queries do not inflate
the distinct quality sample count.

## Resolved initial findings

The following issues were found and repaired before development/final timing:

1. Runner now passes the workload name; unrelated datasets are not merged.
2. Runner enables the 64-pair correctness check for every workload.
3. The singleton candidate fixture explicitly validates invalid top-10 tails.
4. The headline summary exposes both host-complete latencies and distinguishes
   paired geometric speedup from the marginal-median ratio.
5. Run manifests include compiler command and compiler resource-log hashes.

The word shuffle initially used a derived seed rather than literal seed 1729.
It was corrected before any benchmark measurement; the original data was
preserved privately and all six DNA split hashes stayed unchanged. No measured
performance or held-out answer informed that correction.

## Remaining limitations and material caveats

### WARNING: offline timings are not all-in setup costs

`AnchorSetup::build_ms` and `Index::build_ms` start inside constructor bodies,
after persistent member-buffer allocations. File parsing and Engine scratch
initialization are not independently timed. Therefore the logged offline figures
are component timings, not complete setup time. This falls short of the
contract's intent to separately report all excluded setup costs. Query timings
and recall remain valid; an exact all-in amortization threshold is unsupported.
Do not silently rename these partial timers as total build cost.

### WARNING: implementation cost is not a universal lower bound

Routing launches one CTA and assigns four sequential anchor distances per lane;
with m=512, only 128 lanes own valid anchors. The scan launches 391 CTAs for
100,000 objects. The measured DNA routing stage alone averages about 0.240 ms
(clustered) or 0.311 ms (random), exposing a substantial low-parallelism boundary
of this implementation. Pooling also sorts N vote keys, even when C is small.
These facts do not prove that every GPU inverted index must lose. No Nsight
counter evidence isolates atomic contention, divergence, or random-memory cost,
so stage timing must not be over-interpreted as a specific microarchitectural
cause.

### INFO: claim boundaries

The matched scan is a reasonable same-kernel comparator, not a demonstrated
state-of-the-art implementation across external GPU edit-distance libraries.
No embedding baseline, large query batch, alternative routing schedule,
multiple independent anchor seeds, or million-object workload was measured.
Length ordering reduces variation but no numerical divergence reduction is
claimed. This bounded implementation handles Levenshtein strings up to 256 bytes,
not arbitrary CPU callbacks or every metric space. Sanitizer coverage is a small
fixture gate, not production certification.

## Final conclusion

The retained evidence supports: **this particular batch-1, N=100,000, m=512
GPU multi-assignment prototype does not beat its matched exact CUDA scan on any
of the 18 frozen configurations.** It also supports recall collapse on the
specified independent-random-DNA load. It does not support a universal
impossibility claim about GPU metric indexes, an embedding comparison, or a
novel-index claim. The source-level engineering weaknesses and incomplete
offline-cost accounting remain explicit rather than being hidden by the
negative outcome.
