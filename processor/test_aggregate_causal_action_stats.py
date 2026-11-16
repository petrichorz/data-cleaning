"""用已知原始样本验证合并结果，覆盖 mask、四元数和输入口径。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aggregate_causal_action_stats import (
    aggregate_statistics,
    discover_dataset_roots,
    main,
    save_statistics,
)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, name, values, masks, **overrides):
        values, masks = np.asarray(values), np.asarray(masks, dtype=bool)
        stats = {key: [] for key in ("min", "max", "mean", "std", "q01", "q99")}
        for d in range(values.shape[1]):
            x = values[masks[:, d], d]
            for key, fn in (
                ("min", np.min),
                ("max", np.max),
                ("mean", np.mean),
                ("std", np.std),
                ("q01", lambda x: np.quantile(x, 0.01)),
                ("q99", lambda x: np.quantile(x, 0.99)),
            ):
                stats[key].append(float(fn(x)) if len(x) else 0.0)
        stats.update(
            valid=masks.any(0).tolist(),
            valid_counts=masks.sum(0).tolist(),
            count=[len(values)],
        )
        stats.update(low=stats["q01"], high=stats["q99"])
        path = self.root / name / "meta/causal_action_stats.json"
        path.parent.mkdir(parents=True)
        (path.parent / "info.json").write_text("{}")
        provenance = dict(
            kind="fitted",
            method="exact",
            std_ddof=0,
            template_id="test",
            geometry={},
            population="all_valid_single_block_starts",
            start_stride=1,
            rotation_groups=[],
            columns={"target": "action_unified", "target_offset": 0},
            dataset_root=str(path.parent.parent),
            partial=False,
            blocks=len(values),
            available_blocks=len(values),
            episodes=[0],
        )
        provenance.update(overrides)
        path.write_text(json.dumps(dict(state=stats, action=stats, provenance=provenance)))
        return path

    def test_moments_masks_envelope_and_order(self):
        rng = np.random.default_rng(72)
        paths, population = [], [[] for _ in range(5)]
        for i in range(40):
            values = rng.normal(size=(int(rng.integers(1, 40)), 5)) + i * 10
            masks = rng.random(values.shape) > 0.4
            masks[:, -1] = False
            paths.append(self.source(str(i), values, masks))
            for d in range(5):
                population[d].extend(values[masks[:, d], d])
        for inputs in (paths, paths[::-1]):
            result = aggregate_statistics(inputs)
            for kind in ("state", "action"):
                stats = result[kind]
                for key, fn in (
                    ("mean", np.mean),
                    ("std", np.std),
                    ("min", np.min),
                    ("max", np.max),
                ):
                    np.testing.assert_allclose(stats[key], [fn(x) if x else 0 for x in population], atol=1e-12)
                self.assertEqual(stats["valid_counts"], [len(x) for x in population])
                self.assertFalse(stats["valid"][-1])
            self.assertEqual(result["provenance"]["method"], "source_quantile_envelope")

    def test_explicit_envelope_partial_and_bounds(self):
        a = self.source("a", [[1, 0], [3, 0]], [[1, 0]] * 2)
        b = self.source("b", [[100, 5], [200, 7], [300, 9]], [[1, 1]] * 3, partial=True)
        for bounds in ("q01_q99", "min_max"):
            result = aggregate_statistics([a, b], bounds=bounds)
            np.testing.assert_allclose(result["state"]["q01"], [1.02, 5.04])
            np.testing.assert_allclose(result["state"]["q99"], [298, 8.96])
            np.testing.assert_allclose(result["state"]["low"], [1, 5] if bounds == "min_max" else [1.02, 5.04])
            self.assertTrue(result["provenance"]["partial"])

    def test_quaternion_and_invalid_coverage(self):
        a = self.source("a", [[0, 0, 0, 1]], [[1] * 4], rotation_groups=[[0, 1, 2, 3]])
        result = aggregate_statistics([a])
        self.assertEqual(result["state"]["low"], [-1] * 4)
        self.assertEqual(result["state"]["high"], [1] * 4)
        self.assertEqual(result["state"]["min"], [0, 0, 0, 1])
        b = self.source("b", [[0, 0, 0, 1]], [[1, 1, 0, 0]], rotation_groups=[[0, 1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "quaternion"):
            aggregate_statistics([a, b])

    def test_incompatible_duplicate_and_malformed(self):
        a = self.source("a", [[1], [2]], [[1], [1]])
        b = self.source("b", [[3]], [[1]], columns={"target": "state_unified", "target_offset": 1})
        with self.assertRaisesRegex(ValueError, "columns"):
            aggregate_statistics([a, b])
        with self.assertRaisesRegex(ValueError, "distinct"):
            aggregate_statistics([a, a])
        original = json.loads(a.read_text())
        for key, value in (
            ("valid_counts", [-1]),
            ("count", [1.5]),
            ("q01", [3]),
            ("mean", [float("nan")]),
        ):
            malformed = copy.deepcopy(original)
            malformed["state"][key] = value
            a.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                aggregate_statistics([a])
        a.write_text(json.dumps(original))
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(a.read_text())
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            aggregate_statistics([a, duplicate])

    def test_discovery_cli_and_preserve_sources(self):
        a = self.source("robot_a", [[1], [2]], [[1], [1]])
        b = self.source("other_b", [[3]], [[1]])
        before = a.read_bytes(), b.read_bytes()
        output = self.root / "group.json"
        self.assertEqual(discover_dataset_roots([self.root], ["robot_"]), [a.parent.parent])
        main(
            [
                "--dataset-root",
                str(self.root),
                "--prefix",
                "robot_",
                "--output",
                str(output),
            ]
        )
        self.assertEqual(json.loads(output.read_text())["state"]["count"], [2])
        for destination in (a, b):
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--dataset-root",
                        str(self.root),
                        "--prefix",
                        "robot_",
                        "--output",
                        str(destination),
                    ]
                )
        self.assertEqual(before, (a.read_bytes(), b.read_bytes()))
        b.unlink()
        previous = output.read_bytes()
        with self.assertRaises(SystemExit):
            main(["--dataset-root", str(self.root), "--output", str(output)])
        self.assertEqual(output.read_bytes(), previous)
        with self.assertRaises(ValueError):
            save_statistics({"bad": float("nan")}, output)
        self.assertEqual(output.read_bytes(), previous)
        self.assertFalse(list(self.root.glob(".aggregate_stats_*")))

    def test_original_cosmos_provenance(self):
        a = self.source("a", [[1]], [[1]])
        data = json.loads(a.read_text())
        provenance = data["provenance"]
        provenance["dataset_roots"] = [provenance.pop("dataset_root")]
        provenance["read_options"] = provenance.pop("columns")
        provenance.update(split="train", split_seed=0, split_val_ratio=0.0)
        a.write_text(json.dumps(data))
        self.assertEqual(aggregate_statistics([a])["state"]["mean"], [1.0])


if __name__ == "__main__":
    unittest.main()
