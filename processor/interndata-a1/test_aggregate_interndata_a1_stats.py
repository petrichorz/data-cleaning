"""验证真实数值合并、单臂 mask、型号分组和命令行失败时的文件保留。"""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import aggregate_interndata_a1_stats as aggregator


class InternDataAggregateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def source(self, name, *, robot_type="Franka", left=(0.2, 0.4, 0.6), right=None,
               partial=False, action_left_valid=True):
        """从已知原始值计算子集统计；无效槽故意填大值，检查是否被排除。"""
        path = self.root / "datasets" / name / "meta/causal_action_stats.json"
        path.parent.mkdir(parents=True)
        (path.parent / "info.json").write_text(json.dumps({"robot_type": robot_type}))
        n = len(left)
        records = {}
        for kind, shift in (("state", 0.0), ("action", 0.1)):
            values = np.full((n, 55), 1e9)
            masks = np.zeros((n, 55), dtype=bool)
            values[:, 0] = np.arange(n)
            values[:, 17:21] = [0.0, 0.0, 0.0, 1.0]
            values[:, 28] = np.asarray(left) + shift
            masks[:, [0, 17, 18, 19, 20, 28]] = True
            if kind == "action" and not action_left_valid:
                masks[:, 28] = False
            if right is not None:
                values[:, 24:28] = [0.0, 0.0, 0.0, 1.0]
                values[:, 29] = np.asarray(right) + shift
                masks[:, [24, 25, 26, 27, 29]] = True
            stats = {}
            for key, fn in (
                ("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std),
                ("q01", lambda x: np.quantile(x, 0.01)),
                ("q99", lambda x: np.quantile(x, 0.99)),
            ):
                stats[key] = [float(fn(values[masks[:, d], d])) if masks[:, d].any() else 0.0
                              for d in range(55)]
            stats.update(valid=masks.any(0).tolist(), valid_counts=masks.sum(0).tolist(), count=[n],
                         low=stats["q01"].copy(), high=stats["q99"].copy())
            records[kind] = stats
        records["provenance"] = dict(
            kind="fitted", method="exact", std_ddof=0, template_id=aggregator.TEMPLATE_ID,
            geometry={"action_chunk_size": 32}, population="all_valid_single_block_starts",
            start_stride=1, rotation_groups=copy.deepcopy(aggregator.ROTATION_GROUPS),
            columns={"target": "action_unified", "target_offset": 0},
            dataset_root=str(path.parent.parent), partial=partial, blocks=n, available_blocks=n,
        )
        path.write_text(json.dumps(records))
        return path

    def run_main(self, *args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return aggregator.main(list(map(str, args)))

    def test_measured_statistics_survive_fixed_gripper_bounds(self):
        a = self.source("articulation_tasks/franka/task_a", left=(-0.15, 0.2, 1.136))
        b = self.source("long_horizon_tasks/franka/task_b", left=(0.3, 0.8), partial=True)
        inputs = [a.read_bytes(), b.read_bytes()]
        result = aggregator.aggregate_statistics([a, b])
        for kind, shift in (("state", 0.0), ("action", 0.1)):
            stats = result[kind]
            population = np.array([-0.15, 0.2, 1.136, 0.3, 0.8]) + shift
            self.assertEqual(stats["low"][28], 0.0)
            self.assertEqual(stats["high"][28], 1.0)
            self.assertTrue(stats["valid"][28])
            self.assertEqual(stats["valid_counts"][28], 5)
            for key, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)):
                self.assertAlmostEqual(stats[key][28], float(fn(population)))
            # 子集 A 包含两端极值，q01/q99 必须保留其实际分位数，而不是被改成 0/1。
            self.assertAlmostEqual(stats["q01"][28], np.quantile(population[:3], 0.01))
            self.assertAlmostEqual(stats["q99"][28], np.quantile(population[:3], 0.99))
            self.assertEqual(stats["low"][17:21], [-1.0] * 4)
            self.assertEqual(stats["high"][17:21], [1.0] * 4)
            self.assertFalse(stats["valid"][29])
            for key in ("low", "high", "min", "max", "q01", "q99", "valid_counts"):
                self.assertEqual(stats[key][29], 0)
        self.assertTrue(result["provenance"]["partial"])
        self.assertEqual(result["provenance"]["gripper_bounds"], [0, 1])
        self.assertEqual(inputs, [a.read_bytes(), b.read_bytes()])

    def test_bimanual_bounds_and_state_action_masks_are_independent(self):
        path = self.source("basic_tasks/lift2/task", robot_type="ARX Lift-2", right=(0.3, 0.5, 1.136),
                           action_left_valid=False)
        for bounds in ("q01_q99", "min_max"):
            result = aggregator.aggregate_statistics([path], bounds=bounds)
            self.assertEqual(result["state"]["low"][28:30], [0.0, 0.0])
            self.assertEqual(result["state"]["high"][28:30], [1.0, 1.0])
            self.assertEqual(result["action"]["valid"][28:30], [False, True])
            self.assertEqual(result["action"]["high"][28:30], [0.0, 1.0])
            self.assertAlmostEqual(result["state"]["high"][0], 2.0 if bounds == "min_max" else 1.98)

    def test_nested_discovery_uses_metadata_and_deduplicates_roots(self):
        a = self.source("articulation_tasks/franka/InternData-A1/articulation_tasks/franka/task")
        b = self.source("basic_tasks/franka/misleading_directory", robot_type="ARX Lift-2")
        groups = aggregator.discover_robot_groups([self.root / "datasets", a.parent.parent])
        self.assertEqual(groups["franka"], [a.parent.parent])
        self.assertEqual(groups["lift2"], [b.parent.parent])
        self.assertEqual(groups["genie1"], [])
        selected = aggregator.discover_robot_groups([self.root / "datasets"], ["lift2"])
        self.assertEqual(selected, {"lift2": [b.parent.parent]})

    def test_cli_writes_separate_files_for_four_robots(self):
        inputs = []
        for name, robot in (("Franka", "franka"), ("ARX Lift-2", "lift2"),
                            ("Genie-1", "genie1"), ("AgileX Split Aloha", "split_aloha")):
            inputs.append(self.source(f"category/{robot}/task", robot_type=name))
        before = [p.read_bytes() for p in inputs]
        output = self.root / "statistics"
        self.assertEqual(self.run_main("--dataset-root", self.root / "datasets", "--output-dir", output), 0)
        self.assertEqual(len(list(output.glob("*.json"))), 4)
        for robot in ("franka", "lift2", "genie1", "split_aloha"):
            result = json.loads((output / f"stats_interndata_a1_{robot}.json").read_text())
            self.assertEqual(result["provenance"]["embodiment"], robot)
            self.assertEqual(result["state"]["high"][28], 1)
        self.assertEqual(before, [p.read_bytes() for p in inputs])

    def test_missing_statistics_preserve_failed_output_and_process_other_robot(self):
        self.source("category/franka/task")
        missing = self.source("category/lift2/task", robot_type="ARX Lift-2")
        missing.unlink()
        output = self.root / "statistics"
        output.mkdir()
        old_output = output / "stats_interndata_a1_lift2.json"
        old_output.write_bytes(b"previous complete statistics")
        self.assertEqual(self.run_main("--dataset-root", self.root / "datasets", "--output-dir", output), 1)
        self.assertEqual(old_output.read_bytes(), b"previous complete statistics")
        self.assertTrue((output / "stats_interndata_a1_franka.json").is_file())

    def test_dry_run_validates_without_creating_output(self):
        path = self.source("category/franka/task")
        output = self.root / "does not exist"
        args = ("--dataset-root", self.root / "datasets", "--output-dir", output, "--dry-run")
        self.assertEqual(self.run_main(*args), 0)
        self.assertFalse(output.exists())
        path.unlink()
        self.assertEqual(self.run_main(*args), 1)
        self.assertFalse(output.exists())

    def test_rejects_wrong_template_width_and_reading_contract(self):
        a = self.source("category/franka/a")
        b = self.source("category/franka/b")
        original = json.loads(a.read_text())
        mutations = (
            ("template_id", "unified55-legacy"),
            ("rotation_groups", [[0, 1, 2, 3]]),
        )
        for key, value in mutations:
            changed = copy.deepcopy(original)
            changed["provenance"][key] = value
            a.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                aggregator.aggregate_statistics([a])
        changed = copy.deepcopy(original)
        for kind in ("state", "action"):
            for key, value in changed[kind].items():
                if len(value) == 55:
                    changed[kind][key] = value[:-1]
        a.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "55 dimensions"):
            aggregator.aggregate_statistics([a])
        changed = copy.deepcopy(original)
        changed["provenance"]["columns"]["target_offset"] = 1
        a.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "columns"):
            aggregator.aggregate_statistics([a, b])

    def test_unknown_robot_or_empty_selection_fails(self):
        path = self.source("category/franka/task")
        output = self.root / "statistics"
        self.assertEqual(self.run_main("--dataset-root", self.root / "datasets", "--output-dir", output,
                                       "--robots", "genie1"), 1)
        (path.parent / "info.json").write_text('{"robot_type": "unknown"}')
        with self.assertRaises(SystemExit) as context:
            self.run_main("--dataset-root", self.root / "datasets", "--output-dir", output)
        self.assertEqual(context.exception.code, 2)
        self.assertFalse(output.exists())

    def test_bash_entry_from_other_working_directory_with_spaces(self):
        self.source("category/franka/task with spaces")
        script = Path(aggregator.__file__).with_suffix(".sh")
        output = self.root / "statistics with spaces"
        result = subprocess.run(
            ["bash", str(script), str(self.root / "datasets"), str(output), "--robots", "franka"],
            cwd=self.root, env={**os.environ, "PYTHON_BIN": sys.executable},
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((output / "stats_interndata_a1_franka.json").is_file())


if __name__ == "__main__":
    unittest.main()
