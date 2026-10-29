#!/usr/bin/env python3
"""Summarize every paired query record; no quality or speed filtering is applied."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STAGES = ("route_ms", "pool_ms", "gather_ms", "verify_ms", "output_ms")
TIMINGS = ("scan_gpu_ms", "ann_gpu_ms", "scan_host_ms", "ann_host_ms", "route_ms", "pool_ms", "gather_ms", "verify_ms", "output_ms")


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("Cannot compute a quantile of an empty set")
    position = (len(values) - 1) * probability
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "p10": quantile(values, .1), "median": quantile(values, .5), "p90": quantile(values, .9)}


def load_records(paths: list[Path]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    queries, builds, metadata, provenance = [], [], [], []
    seen_paths: set[Path] = set()
    for path in paths:
        canonical_path = path.resolve()
        if canonical_path in seen_paths:
            raise ValueError(f"Duplicate input file: {path}")
        seen_paths.add(canonical_path)
        data = path.read_bytes()
        provenance.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        context = {}
        for line_no, raw in enumerate(data.decode("utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            row["_file"] = str(path)
            row["_line"] = line_no
            row_type = row.get("type", "query" if "scan_gpu_ms" in row else "unknown")
            if row_type == "metadata":
                context.update(row)
                metadata.append(row)
            elif row_type in ("build", "query"):
                for key in ("workload", "n", "m"):
                    if key not in row and key in context:
                        row[key] = context[key]
                row.setdefault("workload", path.stem)
                (builds if row_type == "build" else queries).append(row)
            else:
                # Preserve non-measurement records instead of silently counting them as observations.
                metadata.append(row)
    if not queries:
        raise ValueError("No measured query records found")
    return queries, builds, metadata, provenance


def summarize_records(queries: list[dict], builds: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    seen_observations: set[tuple] = set()
    required = ("n", "m", "kb", "candidate_frac", "query", "repeat", "order", "recall10", "candidate_count", "union_count", "posting_count") + TIMINGS
    for row in queries:
        for key in required:
            if key not in row:
                raise ValueError(f"Missing {key}: {row.get('_file')}:{row.get('_line')}")
        for key in TIMINGS:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError(f"Invalid timing {key}={row[key]}")
        for key in ("scan_gpu_ms", "ann_gpu_ms", "scan_host_ms", "ann_host_ms"):
            if row[key] <= 0:
                raise ValueError(f"Nonpositive full timing {key}={row[key]}")
        if row["n"] <= 0 or not (0 <= row["candidate_count"] <= row["union_count"] <= row["n"]):
            raise ValueError("Candidate/union/database counts violate their ordering")
        if not 0 <= row["recall10"] <= 1 or ("tie_recall10" in row and not 0 <= row["tie_recall10"] <= 1):
            raise ValueError("Recall must be a fraction in [0, 1]")
        key = (row["workload"], row["n"], row["m"], row["kb"], row["candidate_frac"])
        observation_key = (row["_file"], *key, row["query"], row["repeat"])
        if observation_key in seen_observations:
            raise ValueError(f"Duplicate query/repeat pair: {observation_key}")
        seen_observations.add(observation_key)
        groups[key].append(row)
    output = []
    for (workload, n, m, kb, frac), rows in sorted(groups.items()):
        # Repeated timings must not silently change the query's quality result.
        per_query: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            per_query[(row["_file"], row["query"])].append(row)
        quality = []
        for key, repeated in per_query.items():
            for field in ("recall10", "tie_recall10", "candidate_count", "union_count", "posting_count"):
                if field in repeated[0] and any(r.get(field) != repeated[0][field] for r in repeated):
                    raise ValueError(f"Nondeterministic {field} for repeated query {key}")
            quality.append(repeated[0])
        report = {"workload": workload, "n": n, "m": m, "kb": kb, "candidate_frac": frac,
                  "observations": len(rows), "query_process_pairs": len(quality),
                  "distinct_query_indices": len({r["query"] for r in rows}),
                  "process_files": len({r["_file"] for r in rows}),
                  "order_counts": dict(sorted((order, sum(r["order"] == order for r in rows)) for order in {r["order"] for r in rows})),
                  "recall10": stats([r["recall10"] for r in quality]),
                  "candidate_pct": stats([100 * r["candidate_count"] / n for r in quality]),
                  "union_pct": stats([100 * r["union_count"] / n for r in quality]),
                  "postings_pct_of_n": stats([100 * r["posting_count"] / n for r in quality]),
                  "timings_ms": {field: stats([r[field] for r in rows]) for field in TIMINGS}}
        stage_sums = [sum(r[field] for field in STAGES) for r in rows]
        report["timings_ms"]["stage_sum_ms"] = stats(stage_sums)
        report["timings_ms"]["stage_residual_ms"] = stats([r["ann_gpu_ms"] - total for r, total in zip(rows, stage_sums)])
        if all("tie_recall10" in r for r in quality):
            report["tie_recall10"] = stats([r["tie_recall10"] for r in quality])
        for scope in ("gpu", "host"):
            scan = f"scan_{scope}_ms"
            ann = f"ann_{scope}_ms"
            ratios = [r[scan] / r[ann] for r in rows]
            per_process = defaultdict(list)
            for row in rows:
                per_process[row["_file"]].append(math.log(row[scan] / row[ann]))
            process_ratios = [math.exp(statistics.mean(v)) for v in per_process.values()]
            report[f"speedup_{scope}"] = {**stats(ratios), "geomean": math.exp(statistics.mean(map(math.log, ratios))),
                                          "marginal_median_ratio": statistics.median(r[scan] for r in rows) / statistics.median(r[ann] for r in rows),
                                          "process_geomean_ratios": process_ratios,
                                          "process_wins": sum(r > 1 for r in process_ratios),
                                          "by_order_geomean": {o: math.exp(statistics.mean(math.log(r[scan] / r[ann]) for r in rows if r["order"] == o))
                                                               for o in sorted({r["order"] for r in rows})}}
        relevant_builds = [b for b in builds if b["workload"] == workload and b.get("n") == n and b.get("m") == m and b.get("kb", kb) == kb]
        report["build_records"] = relevant_builds
        output.append(report)
    return output


def markdown(reports: list[dict]) -> str:
    lines = ["# GPU soft Voronoi prototype results", "",
             "All recorded configurations are included. **Host-completion latency is the primary denominator.** Latencies below are median milliseconds per completed query; paired speedup is the geometric mean of matched scan/ANN ratios. The marginal-median ratio is reported separately and is not the same estimator. Recall and candidate fractions are query means. No process-level confidence claim is made from repeated queries alone.", "",
             "## Primary: completed host request", "",
             "| Workload | N | m | kb | C cap | Recall@10 | Tie-aware | Candidate % | Union % | Scan host ms | ANN host ms | Host paired geomean speedup | Host marginal-median ratio |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        t = r["timings_ms"]
        tie = f'{100 * r["tie_recall10"]["mean"]:.2f}%' if "tie_recall10" in r else "not recorded"
        lines.append(f'| {r["workload"]} | {r["n"]} | {r["m"]} | {r["kb"]} | {100*r["candidate_frac"]:g}% | {100*r["recall10"]["mean"]:.2f}% | {tie} | {r["candidate_pct"]["mean"]:.3f} | {r["union_pct"]["mean"]:.3f} | {t["scan_host_ms"]["median"]:.6f} | {t["ann_host_ms"]["median"]:.6f} | {r["speedup_host"]["geomean"]:.3f}x | {r["speedup_host"]["marginal_median_ratio"]:.3f}x |')
    lines += ["", "## Complementary: CUDA-event totals", "",
              "These device-timeline totals do not replace synchronized host-completion latency.", "",
              "| Workload | m | kb | C cap | Scan GPU ms | ANN GPU ms | GPU paired geomean speedup | GPU marginal-median ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        t = r["timings_ms"]
        lines.append(f'| {r["workload"]} | {r["m"]} | {r["kb"]} | {100*r["candidate_frac"]:g}% | {t["scan_gpu_ms"]["median"]:.6f} | {t["ann_gpu_ms"]["median"]:.6f} | {r["speedup_gpu"]["geomean"]:.3f}x | {r["speedup_gpu"]["marginal_median_ratio"]:.3f}x |')
    lines += ["", "## CUDA-event stage breakdown", "",
              "Stage means include sorting and real candidate gathering in their respective stages; output includes result transfers. Stage sum is computed per observation from route + pool + gather + verify + output. Residual is event total minus this sum (small signed rounding residuals are retained). Host submission/completion overhead is not inferred by summing device stages. Full p10/median/p90 statistics for both scopes are retained in summary.json.", "",
              "| Workload | m | kb | C cap | Route ms | Pool ms | Gather ms | Verify ms | Output ms | Stage sum ms | Residual ms | ANN GPU total ms | P10/P90 GPU total ms | Samples |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        t = r["timings_ms"]
        stages = " | ".join(f'{t[field]["mean"]:.6f}' for field in STAGES)
        lines.append(f'| {r["workload"]} | {r["m"]} | {r["kb"]} | {100*r["candidate_frac"]:g}% | {stages} | {t["stage_sum_ms"]["mean"]:.6f} | {t["stage_residual_ms"]["mean"]:.6f} | {t["ann_gpu_ms"]["mean"]:.6f} | {t["ann_gpu_ms"]["p10"]:.6f}/{t["ann_gpu_ms"]["p90"]:.6f} | {r["observations"]} |')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/summary"))
    args = parser.parse_args()
    queries, builds, metadata, provenance = load_records(args.inputs)
    reports = summarize_records(queries, builds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, "inputs": provenance, "metadata": metadata, "configurations": reports,
                "estimator": "Paired observation geometric mean; process-unit ratios retained; no confidence interval from pseudo-independent query repeats"}
    (args.output_dir / "summary.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    rendered = markdown(reports)
    (args.output_dir / "summary.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
