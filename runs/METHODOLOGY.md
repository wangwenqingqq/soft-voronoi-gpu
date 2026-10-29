# Mechanism, evidence, and research decision

## Prior art and claim boundary

The nearest-reference / posting overlap / exact-rerank idea is established by
[NAPP](https://users.dcc.uchile.cl/~gnavarro/ps/is12.pdf). GPU inverted candidate
processing also has prior art, including
[GENIE](https://arxiv.org/abs/1603.08390). The related
[misi preprint](https://arxiv.org/abs/2608.27422) further prevents treating a
multi-stage GPU pipeline alone as a novel contribution. This repository is a
feasibility implementation, not a claim to invent soft Voronoi or metric ANN.

## Concrete GPU mechanisms

The single CUDA file implements unit-cost global Levenshtein with a multiword
Myers bit vector. Carry and shift propagation spans up to four 64-bit words.
The same function serves indexing, routing, exhaustive scan, and reranking.
Independent scalar DP provides the correctness oracle.

Both scan and verification use character-position-major storage, sorted by
string length. Candidate physical-row sorting and real byte gathering recreate
that layout on GPU and are timed. There is no unsupported claim that divergence
falls by any particular percentage. A generic distance interface still needs a
device implementation and memory representation; this prototype does not turn an
arbitrary host callback into a GPU kernel.

The index stores one flat array of N*kb 32-bit posting IDs plus m+1 offsets.
CSR avoids separately allocated variable-length chains. For m=512, postings
occupy approximately 1.60, 3.20 and 6.40 MB at kb=4,8,16, including offsets.
GPU sorting and prefix sums construct it. Random pivot sampling on CPU is
offline; no distance, membership, posting construction, vote aggregation or
candidate selection is offloaded to CPU.

Query voting uses atomic increments into an N-entry array, then sorts N keys by
(-vote, original ID). It is intentionally a straightforward, fully measured
baseline, not an optimal sparse aggregator. Routing uses one CTA; at m=512,
128 lanes each own four sequential anchor distances. This underuses the GPU for
costly patterns compared with 391 CTAs in a 100,000-object scan. These are concrete
implementation limits, not fundamental lower bounds.

## Evaluation semantics

Primary quality is canonical Recall@10 against the exact original metric,
breaking distance ties by input ID. Tie-aware recall is supplementary because
edit distances have many ties; it never replaces the primary acceptance gate.
The latter credits all recovered strictly closer neighbors and up to the
remaining boundary-distance slots. All 18 configurations are retained.

Primary latency is a synchronized host request, from query packing/H2D through
GPU mask construction, all kernels and layout operations, to completed result
D2H. CUDA events retain stage and total device-timeline measurements. Database,
index, and scratch are resident; this is not cold-start service latency. The
scan is a matched, reasonably parallel bit-vector implementation, not a claim
to be the fastest possible external GPU edit-distance library.

## Symmetric decision

The hypothesis predicted that evaluating only 2–5% of objects might repay the
indexing overhead. It failed for this N=100,000, m=512, batch-1 implementation on
all three fixed workloads. Clustered synthetic DNA reaches high recall, but
still loses latency; independent random DNA loses recall as well. Dictionary
words are real corpus data, but the query distribution is held-out dictionary
words, not measured user typo traffic.

The recorded stage measurements diagnose where the concrete implementation
spends time. They do not isolate atomic contention, L2/DRAM behavior, or warp
stalls; no Nsight counter claims are made. A 5% final candidate fraction does
not imply 5% index work: overlapping posting lists can visit nearly N entries
and their union can cover roughly half the database.

## Reopen condition, not a promised next project

Do not promote this implementation or the known index idea into a standalone
research thesis. Reopening would require a non-incremental mechanism and a new,
predeclared test: genuinely parallel routing and sparse candidate aggregation,
a strong matched scan plus an original-metric embedding/rerank baseline, and
high recall on real workloads under the same full timing contract. Larger N,
longer strings, query batching, and alternative metrics remain unmeasured, not
proven failures. A few micro-optimizations or synthetic-only wins would not
resolve novelty.
