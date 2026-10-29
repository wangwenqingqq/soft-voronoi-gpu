# Frozen artifact provenance

Source checkpoint: `7dbd3de2304efce3dc49a98599d9278af9669615`.
Source and binary identities are in each run manifest and the integrity audit.

The manifest uses runtime paths. These exact-byte copies are published without
shipping databases, executables, or local machine configuration:

| Runtime path in manifest | Published exact-byte copy |
|---|---|
| `data/manifest.json` | `runs/provenance/data-manifest.json` |
| `build/compile.command` | `runs/provenance/compile.command` |
| `build/compile.stderr` | `runs/provenance/compile.stderr` |

The recorded build compiled a temporary `soft_voronoi_bench.next.cu` source into
a `.next` binary, then copied both to their canonical names before validation.
The temporary and canonical source files were byte-identical. The published
single source file builds normally with `make`; binary hashes can change with
compiler version, source filename/debug metadata, and build flags. Do not claim
that a new local build has the archived binary identity without hashing it.

`runs/audit/oracle.cpp` is the independent scalar-DP oracle used for three full
100,000-object query checks. Its source hash is recorded in the audit. Example:

```sh
c++ -O3 -std=c++17 runs/audit/oracle.cpp -o /tmp/soft-voronoi-cpu-oracle
/tmp/soft-voronoi-cpu-oracle data/words/db.txt data/words/test.txt 0
```

The output columns are distance and original input ID. Compare with the query-0
`scan_distances` and `scan_ids` in `runs/raw/test-v1/words.jsonl`; repeat for the
two DNA workloads. The oracle is a fixed-fixture audit tool (N>=10), not a
production search API. Its CPU wall times are not the GPU speedup baseline.
