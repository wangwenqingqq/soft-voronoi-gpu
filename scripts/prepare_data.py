#!/usr/bin/env python3
"""Build pinned real-word and synthetic Levenshtein workloads without third-party dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c"
SOURCE_URL = f"https://raw.githubusercontent.com/dwyl/english-words/{SOURCE_COMMIT}/words_alpha.txt"
SOURCE_SHA256 = "3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48"
LICENSE_URL = f"https://raw.githubusercontent.com/dwyl/english-words/{SOURCE_COMMIT}/LICENSE.md"
WORKLOADS = ("words", "dna_clustered", "dna_random")
ALPHABET = "ACGT"
LOG = logging.getLogger("prepare_data")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}/{label}".encode()).digest()[:8], "little")


def load_words(path: Path) -> list[str]:
    # Reject rather than normalize case or punctuation: source words remain real strings.
    with path.open("r", encoding="ascii") as stream:
        return sorted({word for line in stream if (word := line.strip())
                       and 1 <= len(word) <= 256 and all("a" <= c <= "z" for c in word)})


def get_source(output: Path, direct: bool) -> Path:
    path = output / "source" / "words_alpha.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        LOG.info("Downloading pinned word source %s", SOURCE_COMMIT)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
        temporary = path.with_suffix(".download")
        urls = (SOURCE_URL, f"https://api.github.com/repos/dwyl/english-words/contents/words_alpha.txt?ref={SOURCE_COMMIT}")
        last_error = None
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw+json", "User-Agent": "soft-voronoi-data-prep"})
                with opener.open(request, timeout=120) as response, temporary.open("wb") as stream:
                    while block := response.read(1 << 20):
                        stream.write(block)
                if sha256(temporary) != SOURCE_SHA256:
                    raise ValueError("Downloaded source hash mismatch")
                temporary.replace(path)
                break
            except (OSError, ValueError) as exc:
                last_error = exc
                LOG.warning("Source transport failed (%s); trying the next pinned endpoint", type(exc).__name__)
            finally:
                if temporary.exists():
                    temporary.unlink()
        else:
            raise RuntimeError("Unable to retrieve the pinned source from either endpoint") from last_error
    if sha256(path) != SOURCE_SHA256:
        raise ValueError(f"Pinned source hash mismatch: {path}")
    return path


def random_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choices(ALPHABET, k=length))


def mutate_dna(center: str, rng: random.Random) -> str:
    result: list[str] = []
    for base in center:
        if rng.random() >= 0.02:
            if rng.random() < 0.05:
                base = rng.choice(ALPHABET.replace(base, ""))
            result.append(base)
        if rng.random() < 0.02:
            result.append(rng.choice(ALPHABET))
    return "".join(result)


def generate_dna(kind: str, n: int, dev: int, test: int, seed: int) -> tuple[dict[str, list[str]], dict]:
    centers_rng = random.Random(seed_for(seed, f"{kind}/centers"))
    centers = [random_dna(centers_rng, 192) for _ in range(256)] if kind == "dna_clustered" else []
    samples: dict[str, list[str]] = {}
    seen_previous: set[str] = set()
    rejected: dict[str, int] = {}
    for split, count in (("db", n), ("dev", dev), ("test", test)):
        rng = random.Random(seed_for(seed, f"{kind}/{split}"))
        rows: list[str] = []
        rejects = 0
        # Queries are unique and disjoint from all previous splits. DB duplicates are retained and counted.
        seen_split: set[str] = set()
        while len(rows) < count:
            if kind == "dna_clustered":
                value = mutate_dna(rng.choice(centers), rng)
            else:
                value = random_dna(rng, rng.randint(128, 256))
            if not 128 <= len(value) <= 256 or (split != "db" and (value in seen_previous or value in seen_split)):
                rejects += 1
                continue
            rows.append(value)
            seen_split.add(value)
        samples[split] = rows
        seen_previous.update(seen_split)
        rejected[split] = rejects
    return samples, {"kind": kind, "center_count": len(centers), "center_length": 192 if centers else None,
                     "substitution_probability": 0.05 if centers else None,
                     "deletion_probability": 0.02 if centers else None,
                     "insertion_probability": 0.02 if centers else None,
                     "length_range": [128, 256], "rejected_samples": rejected,
                     "seeds": {name: seed_for(seed, f"{kind}/{name}") for name in ("centers", "db", "dev", "test")}}


def write_split(path: Path, rows: list[str]) -> dict:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.writelines(value + "\n" for value in rows)
    lengths = list(map(len, rows))
    return {"path": str(path.name), "sha256": sha256(path), "count": len(rows), "unique_count": len(set(rows)),
            "duplicates": len(rows) - len(set(rows)), "bytes": path.stat().st_size,
            "length_min": min(lengths), "length_max": max(lengths), "length_mean": statistics.mean(lengths),
            "length_median": statistics.median(lengths)}


def prepare(output: Path, n: int, dev: int, test: int, seed: int, workloads: list[str], direct: bool = False) -> dict:
    if min(n, dev, test) < 1:
        raise ValueError("Database and both query split counts must be positive")
    if not workloads or len(set(workloads)) != len(workloads) or any(w not in WORKLOADS for w in workloads):
        raise ValueError(f"Choose distinct workloads from {WORKLOADS}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    config = {"n": n, "dev": dev, "test": test, "seed": seed, "workloads": workloads}
    generator_hash = sha256(Path(__file__))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != config or manifest["generator_sha256"] != generator_hash:
            raise ValueError("Existing manifest belongs to a different configuration/generator; use a new output directory")
        for workload, info in manifest["workloads"].items():
            for split, record in info["files"].items():
                if sha256(output / workload / record["path"]) != record["sha256"]:
                    raise ValueError(f"Existing split hash mismatch: {workload}/{split}")
        LOG.info("Existing complete dataset verified; no files changed")
        return manifest
    # Preflight all destinations before creating any split, preserving interrupted or unrelated output.
    for workload in workloads:
        for split in ("db", "dev", "test"):
            if (output / workload / f"{split}.txt").exists():
                raise FileExistsError(f"Existing split without complete manifest: {workload}/{split}.txt")
    manifest = {"schema_version": 1, "config": config, "generator_sha256": generator_hash,
                "metric": "unit-cost global Levenshtein distance on ASCII bytes", "workloads": {}}
    for workload in workloads:
        LOG.info("Preparing %s: database=%d dev=%d test=%d", workload, n, dev, test)
        if workload == "words":
            source = get_source(output, direct)
            words = load_words(source)
            if len(words) < n + dev + test:
                raise ValueError(f"Only {len(words)} valid unique words; insufficient for requested disjoint splits")
            source_count = len(words)
            shuffle_seed = seed
            random.Random(shuffle_seed).shuffle(words)
            samples = {"db": words[:n], "dev": words[n:n + dev], "test": words[n + dev:n + dev + test]}
            info = {"kind": "held-out real English dictionary words", "source_url": SOURCE_URL,
                    "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
                    "license": "Unlicense", "license_url": LICENSE_URL,
                    "valid_source_count": source_count, "shuffle_seed": shuffle_seed}
        else:
            samples, info = generate_dna(workload, n, dev, test, seed)
        split_sets = {key: set(value) for key, value in samples.items()}
        intersections = {f"{a}_{b}": len(split_sets[a] & split_sets[b])
                         for a, b in (("db", "dev"), ("db", "test"), ("dev", "test"))}
        if any(intersections.values()):
            raise AssertionError("Cross-split exact duplicates")
        info["split_intersections"] = intersections
        info["files"] = {split: write_split(output / workload / f"{split}.txt", rows) for split, rows in samples.items()}
        manifest["workloads"][workload] = info
        LOG.info("Finished %s; cross-split intersections=%s", workload, intersections)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOG.info("Manifest written: %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--dev", type=int, default=32)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--smoke-size", type=int, help="Override database size for an explicitly small smoke fixture")
    parser.add_argument("--direct-download", action="store_true", help="Ignore proxy environment for this download only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    prepare(args.output_dir, args.smoke_size if args.smoke_size is not None else args.n,
            args.dev, args.test, args.seed, args.workloads.split(","), args.direct_download)


if __name__ == "__main__":
    main()
