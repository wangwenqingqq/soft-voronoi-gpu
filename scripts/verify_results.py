#!/usr/bin/env python3
"""Check the frozen final matrix without evaluating or selecting new variants."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(directory):
    total = 0
    max_residual = 0.0
    for work in ('words', 'dna_clustered', 'dna_random'):
        records = [json.loads(line) for line in (directory / f'{work}.jsonl').read_text().splitlines()]
        meta = [r for r in records if r['type'] == 'metadata']
        assert len(meta) == 1 and meta[0]['queries'] == 128 and meta[0]['n'] == 100000
        assert meta[0]['m'] == 512 and meta[0]['repeats'] == 3
        assert any(r['type'] == 'sample_check' and r['status'] == 'pass' and r['pairs'] == 64 for r in records)
        rows = [r for r in records if r['type'] == 'query']
        expected = {(kb, frac, q, rep) for kb in (4, 8, 16) for frac in (.02, .05) for q in range(128) for rep in range(3)}
        observed = set()
        answers = {}
        scan_by_query = {}
        for r in rows:
            key = (r['kb'], r['candidate_frac'], r['query'], r['repeat'])
            assert key not in observed, ('duplicate', work, key)
            observed.add(key)
            assert r['workload'] == work
            assert r['order'] == ('scan_ann' if (r['query'] + r['repeat']) % 2 == 0 else 'ann_scan')
            assert 0 <= r['candidate_count'] <= min(r['candidate_cap'], r['union_count']) <= 100000
            assert r['candidate_count'] == min(r['candidate_cap'], r['union_count'])
            scan_ids, ann_ids = r['scan_ids'], r['ann_ids']
            for tag in ('scan', 'ann'):
                pairs = [(d, i) for d, i in zip(r[f'{tag}_distances'], r[f'{tag}_ids']) if i >= 0]
                assert pairs == sorted(pairs) and len({i for d, i in pairs}) == len(pairs)
                assert all(0 <= i < 100000 and 0 <= d <= 256 for d, i in pairs)
            recall = len(set(scan_ids) & {i for i in ann_ids if i >= 0}) / 10
            assert abs(recall - r['recall10']) < 1e-8
            cutoff = r['scan_distances'][9]
            strict = sum(d < cutoff for d in r['scan_distances'])
            got_strict = sum(d < cutoff and i in scan_ids for d, i in zip(r['ann_distances'], ann_ids) if i >= 0)
            boundary = sum(d == cutoff for d, i in zip(r['ann_distances'], ann_ids) if i >= 0)
            assert abs((got_strict + min(boundary, 10 - strict)) / 10 - r['tie_recall10']) < 1e-8
            answer = tuple(scan_ids), tuple(r['scan_distances']), tuple(ann_ids), tuple(r['ann_distances'])
            assert answers.setdefault(key[:3], answer) == answer
            assert scan_by_query.setdefault(r['query'], answer[:2]) == answer[:2]
            for field in ('scan_host_ms', 'ann_host_ms', 'scan_gpu_ms', 'ann_gpu_ms'):
                assert math.isfinite(r[field]) and r[field] > 0
            residual = abs(r['ann_gpu_ms'] - sum(r[f'{s}_ms'] for s in ('route', 'pool', 'gather', 'verify', 'output')))
            max_residual = max(max_residual, residual)
            assert residual < 1e-6
        assert observed == expected, (work, len(expected - observed), len(observed - expected))
        total += len(rows)
    manifest = json.loads((directory / 'manifest.json').read_text())
    aliases = {'data/manifest.json': 'runs/provenance/data-manifest.json',
               'build/compile.command': 'runs/provenance/compile.command',
               'build/compile.stderr': 'runs/provenance/compile.stderr'}
    checked, absent = [], []
    for name, expected_hash in manifest['files'].items():
        path = Path(name)
        if not path.exists() and name in aliases:
            path = Path(aliases[name])
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, name
            checked.append(name)
        else:
            absent.append(name)
    return {'status': 'pass', 'paired_records': total, 'configurations': 18,
            'max_stage_residual_ms': max_residual, 'hashes_checked': checked,
            'unavailable_local_artifacts': absent}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', type=Path, default=Path('runs/raw/test-v1'))
    args = p.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
