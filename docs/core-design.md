# CUDA core design card

## Contract and hypothesis
Exact unit-cost global Levenshtein for byte strings of length 0..256. File input
is nonempty ASCII lines; empty and full-byte inputs belong to the correctness
suite. This bounded implementation is not an arbitrary CPU callback offload.
Target: SM120, CUDA 13.1. Both scan and indexed verification use the same
multiword Myers implementation and hierarchical exact top-10 selection.
Hypothesis: saved expensive edit-distance evaluations may exceed GPU routing,
voting, candidate sorting, gathering, and launch overhead; this is not assumed.

## Layout and ownership
The database is sorted by (length, input ID) once. Its characters use a
character-position-major SoA, with physical row as the contiguous dimension.
Lengths and original IDs are separate contiguous arrays. Matched scan receives
exactly this layout advantage. Query patterns use 256 character masks times four
64-bit words. Each lane owns one distance and its four positive/negative bit
vectors; a warp handles adjacent, therefore similar-length, text rows.

Offline anchor distances use grid (database tile, anchor), and a GPU block radix
sort per database row retains the nearest 16 anchor IDs. Each kb index consists
of a single sorted posting array of N*kb 32-bit physical IDs and m+1 offsets.
There is no per-list allocation and no linked-list fragmentation. Build may use
large temporary arrays, released after construction. Query buffers are allocated
once outside timing.

## Pipeline and ready graph
1. H2D query bytes -> GPU pattern masks -> GPU anchor distances/top-kb.
2. Zero dense votes -> selected posting reads/atomic increments -> GPU vote keys
   -> radix sort by (-votes, original ID), with zero-vote entries marked invalid.
3. Top-C physical IDs -> GPU physical-row sort (length order) -> actual SoA gather
   -> exact common distance kernel -> hierarchical block top-10 -> result D2H.
All edges are same-stream kernel/event ordering. No CPU candidate decisions,
no custom async barriers, no cluster communication, no persistent kernel state.
Scratch reuse is allowed only after the preceding same-stream consumer ends.

## CTA, live state, and tails
Distance CTAs have 256 threads. Each thread owns Pv/Mv (at most eight uint64),
Eq/Xv/Xh/Ph/Mh temporaries, addition/shift carry, score, and text address. Native
work is scalar integer bit operations, not tensor-core operations. Block sorts
use CUB shared temporary storage; all threads participate including invalid tails.
Invalid items carry UINT64_MAX and cannot displace valid top-k entries. Query
length chooses 1/2/3/4-word template dispatch; the 0-length path returns text length.
Exact per-thread register counts and spills are recorded from ptxas, not assumed.

## Verification and rejection
Independent scalar CPU DP checks all bit-word boundaries and randomized pairs.
Small end-to-end fixtures compare GPU scan with CPU exact sorted neighbors.
Sanitizer and stress are required before performance claims. Rejection includes
wrong answers, CPU pooling fallback, omitted gather costs, or unfair scan layout.
A measured narrow win does not establish novelty, universal metrics, arbitrary
length support, embedding superiority, or production readiness.
