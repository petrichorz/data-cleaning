"""检查单臂语义、完整 episode、原列保留、错误隔离及失败恢复。"""

from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import convert_interndata_a1_55d as converter
from interndata_a1_mapping import OUTPUT_COLUMNS, POSE_NAMES, build_mapping, choose_scale, convert_episode


def fixture(single=True, count=4, scale=0.08, episode=3, joints=None):
    joints = joints or (7 if single else 6)
    info = {"robot_type": "Franka" if single else "ARX Lift-2", "features": {}, "fps": 30,
            "total_frames": count, "total_episodes": 1, "original_metadata": {"keep": True}}
    sides = [""] if single else ["left_", "right_"]
    for stream in ("states", "actions"):
        for side in sides:
            info["features"][f"{stream}.{side}joint.position"] = {
                "shape": [joints], "names": [f"{side}joint_{i}" for i in range(joints)], "dtype": "float64"}
            info["features"][f"{stream}.{side}ee_to_robot_pose"] = {
                "shape": [7], "names": list(POSE_NAMES), "dtype": "float64"}
            info["features"][f"{stream}.{side}gripper.position"] = {"shape": [1], "names": ["gripper_0"], "dtype": "float64"}
    rows = []
    for t in range(count):
        row = {"episode_index": episode, "frame_index": t, "index": t, "timestamp": t / 30,
               "extra": f"keep-{episode}-{t}", "master_actions.unmapped": [100. + t]}
        for side in sides:
            row[f"states.{side}joint.position"] = [t * .1 + j for j in range(joints)]
            row[f"states.{side}ee_to_robot_pose"] = [float(t), 2., 3., 2 ** -.5, 0., 0., 2 ** -.5]
            row[f"states.{side}gripper.position"] = scale * t / max(1, count - 1)
        rows.append(row)
    for t, row in enumerate(rows):
        following = rows[min(t + 1, count - 1)]
        for key, value in list(following.items()):
            if key.startswith("states."):
                row["actions." + key[len("states."):]] = copy.deepcopy(value)
    stats = {f"states.{side}gripper.position": {"max": [scale], "mean": [scale / 2]} for side in sides}
    return info, rows, stats


def write_bucket(root, info, rows, stats, shards=None):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "videos/keep.mp4").write_bytes(b"untouched video")
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(json.dumps(stats))
    lengths = Counter(row["episode_index"] for row in rows)
    pq.write_table(pa.Table.from_pylist([{"episode_index": k, "length": v} for k, v in lengths.items()]),
                   root / "meta/episodes/chunk-000/file-000.parquet")
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"source": b"fixture metadata"})
    for i, indexes in enumerate(shards or [list(range(len(rows)))]):
        pq.write_table(table.take(pa.array(indexes)), root / f"data/chunk-000/file-{i:03d}.parquet", row_group_size=2)
    return table


class MappingTests(unittest.TestCase):
    def mapped(self, single=True, **kwargs):
        info, rows, stats = fixture(single=single, **kwargs)
        mapping = build_mapping(info, stats)
        result = convert_episode(pa.Table.from_pylist(rows), mapping)
        return info, rows, stats, mapping, result

    def test_single_arm_only_left_has_fifteen_valid_components(self):
        _, rows, _, _, (a, errors, _) = self.mapped()
        self.assertFalse(errors)
        np.testing.assert_array_equal(a[:, 2:].sum(axis=2), 15)
        np.testing.assert_array_equal(a[0, 0, :7], rows[0]["states.joint.position"])
        np.testing.assert_allclose(a[0, 0, 17:21], [0, 0, 2 ** -.5, 2 ** -.5])
        self.assertEqual(a[0, 2, 0], 1)  # 真正的零值仍然有效。
        self.assertTrue((a[:, :, np.r_[7:14, 21:28, 29:55]] == 0).all())
        self.assertEqual(a[0, 1, 14], 1.)  # 使用源动作，既不复制同帧也不再移位。
        np.testing.assert_allclose(a[:, 0, 28], np.linspace(0, 1, len(rows)))

    def test_bimanual_six_joints_pad_seventh_and_twenty_eight_valid(self):
        _, _, _, _, (a, errors, _) = self.mapped(single=False, scale=.088)
        self.assertFalse(errors)
        self.assertTrue((a[:, 2:].sum(axis=2) == 28).all())
        self.assertTrue((a[:, :, [6, 13]] == 0).all())
        self.assertTrue((a[:, :, 30:] == 0).all())

    def test_names_define_joint_and_pose_order_independently(self):
        info, rows, stats = fixture()
        for stream in ("states", "actions"):
            for suffix in ("joint.position", "ee_to_robot_pose"):
                column = stream + "." + suffix
                info["features"][column]["names"].reverse()
                for row in rows:
                    row[column].reverse()
        a, errors, raw = convert_episode(pa.Table.from_pylist(rows), build_mapping(info, stats))
        self.assertFalse(errors)
        np.testing.assert_array_equal(a[0, 0, :7], np.arange(7))
        np.testing.assert_allclose(a[0, 0, 17:21], [0, 0, 2 ** -.5, 2 ** -.5])
        np.testing.assert_array_equal(raw["action"][:-1], raw["state"][1:])

    def test_scalar_and_list_grippers_produce_identical_results(self):
        info, rows, stats = fixture()
        a, _, _ = convert_episode(pa.Table.from_pylist(rows), build_mapping(info, stats))
        for row in rows:
            for stream in ("states", "actions"):
                key = stream + ".gripper.position"
                row[key] = [row[key]]
        b, _, _ = convert_episode(pa.Table.from_pylist(rows), build_mapping(info, stats))
        np.testing.assert_array_equal(a, b)

    def test_fixed_size_vectors_and_null_numeric_components(self):
        info, rows, stats = fixture()
        rows[1]["states.joint.position"][2] = None
        table = pa.Table.from_pylist(rows)
        for name, feature in info["features"].items():
            if "gripper.position" not in name:
                index = table.schema.get_field_index(name)
                table = table.set_column(index, name, table[name].cast(pa.list_(pa.float64(), feature["shape"][0])))
        a, errors, _ = convert_episode(table.slice(1, 2), build_mapping(info, stats))
        self.assertEqual(a[0, 2, 2], 0)
        self.assertEqual(a[1, 2].sum(), 15)
        self.assertTrue(errors)

    def test_openwam_scale_selection_and_fallback(self):
        column = "states.gripper.position"
        for maximum, expected in ((.04, .08), (.08, .08), (1., 1.)):
            self.assertEqual(choose_scale("franka", {column: {"max": [maximum]}}, column)["value"], expected)
        self.assertEqual(choose_scale("franka", {}, column)["value"], .08)
        self.assertEqual(choose_scale("lift2", {column: {"max": [1.]}}, column)["value"], .088)
        self.assertEqual(choose_scale("genie1", {}, column)["value"], 5.74)
        _, _, _, _, (a, _, _) = self.mapped(scale=1.)
        np.testing.assert_allclose(a[:, 0, 28], np.linspace(0, 1, 4))

    def test_genie_bimanual_mapping_uses_574_for_both_streams(self):
        info, rows, stats = fixture(single=False, scale=5.74)
        info["robot_type"] = "Genie-1"
        mapping = build_mapping(info, stats)
        a, errors, _ = convert_episode(pa.Table.from_pylist(rows), mapping)
        self.assertFalse(errors)
        self.assertEqual([arm["gripper_scale"]["value"] for arm in mapping["arms"]], [5.74, 5.74])
        np.testing.assert_allclose(a[:, 0, 28], np.linspace(0, 1, 4))
        np.testing.assert_allclose(a[:-1, 1, 29], np.linspace(0, 1, 4)[1:])

    def test_nonfinite_and_bad_quaternion_are_locally_masked(self):
        info, rows, stats = fixture()
        rows[1]["states.joint.position"][2] = float("nan")
        rows[1]["actions.ee_to_robot_pose"][4] = float("inf")
        rows[2]["states.ee_to_robot_pose"][0] = 1e100
        a, errors, _ = convert_episode(pa.Table.from_pylist(rows), build_mapping(info, stats))
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a[1, 2, 2], 0)
        self.assertTrue((a[1, 3, 17:21] == 0).all())
        self.assertEqual(a[1, 3, 14], 1)
        self.assertEqual(a[2, 2, 14], 0)
        self.assertTrue((a[2, 2, 17:21] == 1).all())
        self.assertIn("float32_overflow", [e["error_type"] for e in errors])

    def test_structural_row_error_zeroes_four_outputs_only_at_that_row(self):
        info, rows, stats = fixture()
        rows[1]["states.joint.position"] = [1., 2.]
        a, errors, _ = convert_episode(pa.Table.from_pylist(rows), build_mapping(info, stats))
        self.assertTrue((a[1] == 0).all())
        self.assertEqual(a[0, 2].sum(), 15)
        self.assertTrue(any(e["result"] == "row_zeroed" for e in errors))

    def test_finite_gripper_outlier_is_preserved_and_roundoff_is_tolerated(self):
        info, rows, stats = fixture(scale=1.)
        rows[0]["states.gripper.position"] = 1.25
        rows[1]["states.gripper.position"] = 1.000000015
        a, errors, _ = convert_episode(pa.Table.from_pylist(rows), build_mapping(info, stats))
        self.assertEqual(a[0, 0, 28], 1.25)
        self.assertEqual(a[0, 2, 28], 1)
        self.assertEqual(len([e for e in errors if e["error_type"] == "gripper_out_of_range"]), 1)

    def test_invalid_joint_schema_and_partial_bimanual_are_rejected(self):
        info, _, stats = fixture()
        info["features"]["states.joint.position"]["names"][1] = "joint_8"
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_mapping(info, stats)
        info, _, stats = fixture(single=False, scale=.088)
        del info["features"]["actions.right_gripper.position"]
        with self.assertRaisesRegex(ValueError, "Missing feature"):
            build_mapping(info, stats)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_episode_across_files_storage_order_and_rerun(self):
        info, rows, stats = fixture()
        _, more, _ = fixture(count=2, episode=9)
        rows += more
        info.update(total_frames=6, total_episodes=2)
        source = self.root / "family/cat/franka/task"
        original = write_bucket(source, info, rows, stats, shards=[[2, 0, 4], [3, 1, 5]])
        disk_schema = pq.read_schema(next((source / "data").rglob("*.parquet")))
        metadata_before = (source / "meta/stats.json").read_bytes()
        result = converter.convert_dataset(source, batch_size=1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["next_state_exact_pairs"], 4)
        self.assertEqual(result["terminal_rows_masked"], 2)
        files = sorted((source / "data").rglob("*.parquet"))
        combined = pa.concat_tables([pq.read_table(p) for p in files])
        self.assertTrue(combined.select(original.column_names).schema.equals(disk_schema, check_metadata=True))
        expected = original.take(pa.array([2, 0, 4, 3, 1, 5]))
        self.assertTrue(combined.select(original.column_names).equals(expected))
        arrays = {name: np.asarray(combined[name].to_pylist()) for name in OUTPUT_COLUMNS}
        for i, row in enumerate(combined.select(["episode_index", "frame_index"]).to_pylist()):
            terminal = row["frame_index"] == (3 if row["episode_index"] == 3 else 1)
            self.assertEqual(arrays["mask_action"][i].sum(), 0 if terminal else 15)
            self.assertEqual(arrays["mask_state"][i].sum(), 15)
            if terminal:
                self.assertTrue((arrays["action_unified"][i] == 0).all())
            else:
                self.assertEqual(arrays["action_unified"][i, 14], row["frame_index"] + 1)
        self.assertEqual((source / "videos/keep.mp4").read_bytes(), b"untouched video")
        self.assertEqual((source / "meta/stats.json").read_bytes(), metadata_before)
        updated = json.loads((source / "meta/info.json").read_text())
        for key in info:
            if key != "features":
                self.assertEqual(updated[key], info[key])
        for key, value in info["features"].items():
            self.assertEqual(updated["features"][key], value)
        again = converter.convert_dataset(source, batch_size=2)
        self.assertEqual(again["status"], "complete")
        rerun = pa.concat_tables([pq.read_table(p) for p in files])
        self.assertTrue(rerun.equals(combined))
        self.assertEqual(len(rerun.column_names), len(set(rerun.column_names)))
        self.assertFalse(list(source.rglob("*.tmp")))

    def test_single_frame_episode_has_valid_state_but_no_action(self):
        info, rows, stats = fixture(count=1)
        source = self.root / "single"
        write_bucket(source, info, rows, stats)
        summary = converter.convert_dataset(source)
        self.assertEqual(summary["status"], "complete")
        table = pq.read_table(next((source / "data").rglob("*.parquet")))
        self.assertEqual(sum(table["mask_state"][0].as_py()), 15)
        self.assertEqual(sum(table["mask_action"][0].as_py()), 0)

    def test_mismatched_source_action_is_retained_and_reported_without_shift(self):
        info, rows, stats = fixture()
        rows[0]["actions.joint.position"][0] += .123
        source = self.root / "mismatch"
        write_bucket(source, info, rows, stats)
        summary = converter.convert_dataset(source)
        self.assertEqual(summary["status"], "complete_with_errors")
        self.assertEqual(summary["next_state_mismatched_pairs"], 1)
        table = pq.read_table(next((source / "data").rglob("*.parquet")))
        self.assertAlmostEqual(table["action_unified"][0].as_py()[0], .223, places=6)
        self.assertEqual(table["mask_action"][0].as_py()[0], 1)

    def test_invalid_frames_or_incomplete_manifest_leave_data_unmodified(self):
        for mode in ("duplicate", "missing_tail"):
            with self.subTest(mode=mode):
                info, rows, stats = fixture()
                if mode == "duplicate":
                    rows[1]["frame_index"] = 0
                source = self.root / mode
                write_bucket(source, info, rows, stats)
                path = next((source / "data").rglob("*.parquet"))
                if mode == "missing_tail":
                    pq.write_table(pq.read_table(path).slice(0, 3), path)
                before = path.read_bytes()
                summary = converter.convert_dataset(source)
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(before, path.read_bytes())
                self.assertNotIn("state_unified", json.loads((source / "meta/info.json").read_text())["features"])

    def test_write_failure_keeps_original_and_rerun_recovers(self):
        info, rows, stats = fixture()
        source = self.root / "failure"
        write_bucket(source, info, rows, stats)
        path = next((source / "data").rglob("*.parquet"))
        before = path.read_bytes()
        with patch.object(converter.pq, "ParquetWriter", side_effect=OSError("simulated write failure")):
            summary = converter.convert_dataset(source)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(source.rglob("*.tmp")))
        self.assertEqual(converter.convert_dataset(source)["status"], "complete")

    def test_info_write_failure_marks_incomplete_and_rerun_recovers(self):
        info, rows, stats = fixture()
        source = self.root / "info_failure"
        write_bucket(source, info, rows, stats)
        write_json = converter.write_json

        def fail_info(path, value):
            if path == source / "meta/info.json":
                raise OSError("simulated metadata failure")
            return write_json(path, value)

        with patch.object(converter, "write_json", side_effect=fail_info):
            summary = converter.convert_dataset(source)
        self.assertEqual(summary["status"], "incomplete")
        self.assertNotIn("state_unified", json.loads((source / "meta/info.json").read_text())["features"])
        self.assertEqual(converter.convert_dataset(source)["status"], "complete")

    def test_lock_rejection_does_not_overwrite_previous_reports(self):
        info, rows, stats = fixture()
        source = self.root / "lock"
        write_bucket(source, info, rows, stats)
        (source / "qc").mkdir()
        report = source / "qc/conversion_summary.json"
        report.write_text('"original summary"')
        with (source / "qc/.conversion.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "已有转换进程"):
                converter.convert_dataset(source)
        self.assertEqual(report.read_text(), '"original summary"')

    def test_recursive_discovery_dry_run_and_parallel_cli(self):
        for path, single, scale in (("art/franka/same_name", True, .08), ("basic/lift2/same_name", False, .088)):
            info, rows, stats = fixture(single=single, scale=scale)
            write_bucket(self.root / path, info, rows, stats)
        script = Path(converter.__file__)
        command = [sys.executable, "-B", str(script), str(self.root), "--workers", "2"]
        preview = subprocess.run([*command, "--dry-run"], capture_output=True, text=True)
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertFalse(list(self.root.rglob("qc")))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.root / "qc/conversion_batch_summary.json").read_text())
        self.assertEqual(summary["status_counts"], {"complete": 2})
        self.assertEqual(summary["rows"], 8)
        self.assertTrue(all((p / "qc/conversion.log").is_file() for p in converter.discover_datasets(self.root)))


if __name__ == "__main__":
    unittest.main()
