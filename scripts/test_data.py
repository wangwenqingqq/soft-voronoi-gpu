#!/usr/bin/env python3
"""CPU tests for deterministic data generation and honest result aggregation."""
import json
import tempfile
import unittest
from pathlib import Path

import prepare_data
import summarize


class DataTests(unittest.TestCase):
    def test_seed_domains(self):
        self.assertEqual(prepare_data.seed_for(1729, "a"), prepare_data.seed_for(1729, "a"))
        self.assertNotEqual(prepare_data.seed_for(1729, "a"), prepare_data.seed_for(1729, "b"))

    def test_word_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "words.txt"
            path.write_text("cat\ncat\nDog\nhello-world\n\n" + "a" * 256 + "\n" + "b" * 257 + "\nzebra\n")
            self.assertEqual(prepare_data.load_words(path), ["a" * 256, "cat", "zebra"])

    def test_dna_deterministic_and_disjoint(self):
        for kind in ("dna_clustered", "dna_random"):
            samples, metadata = prepare_data.generate_dna(kind, 64, 8, 8, 1729)
            repeated, repeated_metadata = prepare_data.generate_dna(kind, 64, 8, 8, 1729)
            self.assertEqual(samples, repeated)
            self.assertEqual(metadata, repeated_metadata)
            self.assertEqual(list(map(len, samples.values())), [64, 8, 8])
            for rows in samples.values():
                self.assertTrue(all(128 <= len(s) <= 256 and set(s) <= set("ACGT") for s in rows))
            self.assertFalse(set(samples["db"]) & set(samples["dev"]))
            self.assertFalse(set(samples["db"]) & set(samples["test"]))
            self.assertFalse(set(samples["dev"]) & set(samples["test"]))

    def test_smoke_generation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            first = prepare_data.prepare(output, 64, 8, 8, 1729, ["dna_clustered", "dna_random"])
            again = prepare_data.prepare(output, 64, 8, 8, 1729, ["dna_clustered", "dna_random"])
            self.assertEqual(first, again)
            with self.assertRaises(ValueError):
                prepare_data.prepare(output, 65, 8, 8, 1729, ["dna_clustered", "dna_random"])
            (output / "dna_random" / "db.txt").write_text("changed\n")
            with self.assertRaises(ValueError):
                prepare_data.prepare(output, 64, 8, 8, 1729, ["dna_clustered", "dna_random"])

    def test_incomplete_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dna_random" / "db.txt"
            path.parent.mkdir()
            path.write_text("ACGT\n")
            with self.assertRaises(FileExistsError):
                prepare_data.prepare(Path(folder), 16, 2, 2, 1729, ["dna_random"])
            self.assertEqual(path.read_text(), "ACGT\n")


class SummaryTests(unittest.TestCase):
    @staticmethod
    def row(**updates):
        row = {"type": "query", "workload": "fixture", "n": 100, "m": 16, "kb": 4, "candidate_frac": .05,
               "query": 0, "repeat": 0, "order": "scan_ann", "scan_gpu_ms": 2., "ann_gpu_ms": 1.,
               "scan_host_ms": 3., "ann_host_ms": 2., "route_ms": .1, "pool_ms": .3, "gather_ms": .1,
               "verify_ms": .4, "output_ms": .1, "recall10": .7, "tie_recall10": .8, "candidate_count": 5,
               "union_count": 20, "posting_count": 24, "_file": "fixture.jsonl", "_line": 1}
        row.update(updates)
        return row

    def test_quantiles(self):
        self.assertEqual(summarize.quantile([2., 4.], .5), 3.)
        self.assertEqual(summarize.quantile([2.], .9), 2.)

    def test_all_configurations_preserved(self):
        records = [self.row(), self.row(repeat=1, order="ann_scan"), self.row(kb=8, ann_gpu_ms=4., recall10=.2)]
        result = summarize.summarize_records(records, [])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["observations"], 2)
        self.assertAlmostEqual(result[0]["speedup_gpu"]["geomean"], 2.)
        self.assertAlmostEqual(result[1]["speedup_gpu"]["geomean"], .5)
        self.assertAlmostEqual(result[0]["candidate_pct"]["mean"], 5.)
        self.assertIn("0.500x", summarize.markdown(result))

    def test_stage_sum_and_primary_host_table(self):
        result = summarize.summarize_records([self.row()], [])
        self.assertAlmostEqual(result[0]["timings_ms"]["stage_sum_ms"]["mean"], 1.)
        self.assertAlmostEqual(result[0]["timings_ms"]["stage_residual_ms"]["mean"], 0.)
        rendered = summarize.markdown(result)
        self.assertLess(rendered.index("Scan host ms"), rendered.index("Scan GPU ms"))
        self.assertIn("Host paired geomean speedup", rendered)
        self.assertIn("Output ms", rendered)

    def test_reject_duplicate_observation(self):
        with self.assertRaisesRegex(ValueError, "Duplicate query/repeat pair"):
            summarize.summarize_records([self.row(), self.row(order="ann_scan")], [])
        # The same query and repeat in a separate process file is a distinct observation.
        result = summarize.summarize_records([self.row(), self.row(_file="process2.jsonl")], [])
        self.assertEqual(result[0]["process_files"], 2)

    def test_reject_duplicate_input_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.jsonl"
            path.write_text(json.dumps(self.row()) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate input file"):
                summarize.load_records([path, path.parent / "." / path.name])

    def test_reject_nondeterministic_quality(self):
        with self.assertRaises(ValueError):
            summarize.summarize_records([self.row(), self.row(repeat=1, recall10=.9)], [])

    def test_reject_invalid_measurements(self):
        for change in ({"ann_gpu_ms": 0.}, {"ann_host_ms": float("nan")}, {"recall10": 70.}, {"union_count": 4}):
            with self.assertRaises(ValueError):
                summarize.summarize_records([self.row(**change)], [])

    def test_metadata_inheritance_and_raw_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.jsonl"
            row = self.row()
            del row["workload"]
            path.write_text(json.dumps({"type": "metadata", "workload": "inherited"}) + "\n" + json.dumps(row) + "\n")
            queries, builds, meta, provenance = summarize.load_records([path])
            self.assertEqual(queries[0]["workload"], "inherited")
            self.assertEqual(len(provenance[0]["sha256"]), 64)
            self.assertEqual(len(meta), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
