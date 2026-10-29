# Reproducible edit-distance workloads

The benchmark uses unit-cost global Levenshtein distance over ASCII bytes, not vector embeddings. The default dataset consists of three databases with 100,000 strings each, 32 development queries and 128 held-out queries per workload. Queries are exact-string-disjoint from their database and from each other across splits. The generator records all file hashes and split intersections in `data/manifest.json`. Generation and hash checks do not evaluate held-out nearest neighbors.

## Real dictionary words

Source: [dwyl/english-words](https://github.com/dwyl/english-words/tree/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c), pinned to commit `20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c`. Input is [`words_alpha.txt`](https://github.com/dwyl/english-words/blob/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c/words_alpha.txt). The repository's [`LICENSE.md`](https://github.com/dwyl/english-words/blob/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c/LICENSE.md) grants the Unlicense.

The loader strips line whitespace, retains only nonempty lowercase ASCII alphabetic strings of length at most 256, deduplicates, sorts, then shuffles deterministically. The first 100,000 words form the database; the next 32 and 128 are development and test queries. Queries are held-out real words, not generated mutations of database words. Consequently, this is a dictionary retrieval workload rather than evidence about user-generated typo frequency or natural query traffic. Raw corpus contents are not tracked in Git. The pinned raw input SHA-256 is `3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48`; both the raw-content and GitHub API transport endpoints must match it.

## Clustered synthetic DNA

Generate 256 independent uniform DNA centers, each of length 192. Every database or query string independently selects a center. For each center position:

1. Delete the base with probability 0.02.
2. If not deleted, substitute with a different base with probability 0.05.
3. Independently append an inserted uniform base with probability 0.02.

Reject a generated string if its length is outside 128–256; do not truncate. Each split uses an independent deterministic RNG stream. The database retains any duplicates and reports their count; query samples are unique and reject exact matches to preceding splits. This deliberately clustered workload tests a favorable locality regime. It is not a biological sequence model.

## Independent random synthetic DNA

Each string independently draws its length uniformly from the integers 128–256 and its bases uniformly from `ACGT`. The same duplicate and split-disjointness policy applies. This is an intentionally difficult concentration/locality counterexample, not a biological workload. Both successful and unsuccessful configurations must be retained.

## Seeds, files, and reproduction

The master seed is 1729. The word shuffle uses RNG seed 1729 directly. Each synthetic RNG seed is the little-endian integer represented by the first eight SHA-256 bytes of `"<master>/<workload>/<stream>"`. These derived seeds, corpus commit, corpus SHA-256, generator SHA-256, row counts, byte sizes, length statistics and exact duplicate counts are in the manifest.

```bash
python3 scripts/test_data.py
python3 scripts/prepare_data.py --output-dir data
# Only if the configured network proxy is unsuitable:
python3 scripts/prepare_data.py --output-dir data --direct-download
# Independent small fixtures, never confused with the 100,000-object result:
python3 scripts/prepare_data.py --output-dir data-smoke --smoke-size 64 --dev 8 --test 8 --workloads dna_clustered,dna_random
```

Each workload directory contains `db.txt`, `dev.txt`, and `test.txt`. Every line is one nonempty ASCII string. The generator refuses to overwrite an existing split. A completed existing dataset is reused only after configuration, generator and file hash verification. An interrupted output should be preserved and generation retried in a new directory.

## Result aggregation

```bash
python3 scripts/summarize.py runs/outputs/final/*.jsonl --output-dir runs/summary/final
```

The summarizer accepts the core's metadata, build and query JSONL records. It includes every recorded workload/configuration, preserves input SHA-256 hashes, reports query-level quality once per query/process, and refuses nondeterministic quality across repeated timings. It reports timing p10/median/p90, paired-ratio arithmetic and geometric means, marginal-median ratios, order splits, process-unit ratios/wins, actual candidate and union fractions, posting volume, and per-stage timing. It does not infer a confidence interval from pseudo-independent repeated queries, does not remove slow outliers, and does not drop low-recall or losing configurations. Both CUDA-event and host-completion denominators are retained separately.

The Markdown report leads with host-completion scan and ANN medians, explicitly labels paired-geometric speedup, and separately reports the ratio of marginal medians. CUDA-event totals and a five-stage breakdown follow. The output-transfer stage, sum of all stages, and signed event-total residual are explicit. Repeated input paths and duplicate `(process file, workload, configuration, query, repeat)` observations are rejected rather than silently double-counted.
