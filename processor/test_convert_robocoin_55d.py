"""CPU checks for mapping semantics, row isolation, and lossless source columns."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_robocoin_55d import OUTPUT_COLUMNS, convert_batch, convert_dataset, joint_mappings, write_json


def fixture(hand_count=6):
    # Right arm first and reversed; action has an entirely different source order.
    names = [f"right_arm_joint_{j}_rad" for j in range(6, 0, -1)]
    names += [f"left_arm_joint_{j}_rad" for j in range(1, 8)]
    names += [f"{side}_hand_joint_{j}_rad" for side in ("left", "right") for j in range(hand_count, 0, -1)]
    action_names = list(reversed(names))
    values = {name: float(i + 10) for i, name in enumerate(names)}
    info = {"features": {c: {"names": ns, "shape": [len(ns) + 2], "dtype": "float64"}
                         for c, ns in (("observation.state", names), ("action", action_names))}}
    # Unnamed tail is intentionally not repaired. Distinct values catch accidental action shifts/copies.
    row = {"observation.state": [values[n] for n in names] + [91., 92.],
           "action": [values[n] + 100 for n in action_names] + [93., 94.],
           "eef_sim_pose_state": [1., 2., 3., 0., 0., np.pi / 2, 4., 5., 6., 0., 0., 0.],
           "eef_sim_pose_action": [11., 12., 13., 0., 0., 0., 14., 15., 16., 0., 0., np.pi],
           "gripper_open_scale_state": [0., .25], "gripper_open_scale_action": [.75, 2.],
           "episode_index": 7, "frame_index": 8, "index": 9, "timestamp": .8, "extra": "keep me"}
    for column in ("eef_sim_pose_state", "eef_sim_pose_action", "gripper_open_scale_state", "gripper_open_scale_action"):
        info["features"][column] = {"shape": [len(row[column])], "names": None, "dtype": "float64"}
    return info, row, values


class ConvertTests(unittest.TestCase):
    def convert(self, info, rows):
        logs = []
        mappings, _ = joint_mappings(info)
        grips = {n for n in info["features"] if n.startswith("gripper_open_scale_")}
        result, bad = convert_batch(pa.Table.from_pylist(rows), mappings, grips, logs.append, "data/example.parquet", 10)
        arrays = {c: np.array(result[c].to_pylist()) for c in OUTPUT_COLUMNS}
        return result, arrays, bad, logs

    def test_mapping_quaternion_and_no_value_changes(self):
        info, row, values = fixture()
        result, a, bad, logs = self.convert(info, [row])
        self.assertEqual((bad, logs), (0, []))
        s, t = a["state_unified"][0], a["action_unified"][0]
        np.testing.assert_array_equal(s[:7], [values[f"left_arm_joint_{j}_rad"] for j in range(1, 8)])
        np.testing.assert_array_equal(s[7:13], [values[f"right_arm_joint_{j}_rad"] for j in range(1, 7)])
        np.testing.assert_array_equal(t[:7], s[:7] + 100)
        np.testing.assert_allclose(s[17:21], [0, 0, 2 ** -.5, 2 ** -.5], atol=1e-7)
        np.testing.assert_array_equal(t[14:17], [11, 12, 13])
        np.testing.assert_array_equal(s[30:36], [values[f"left_hand_joint_{j}_rad"] for j in range(1, 7)])
        self.assertEqual(t[29], 2.)  # A finite out-of-range gripper value is not clipped.
        self.assertEqual(a["mask_state"][0, 28], 1)  # A real zero remains valid.
        for c in OUTPUT_COLUMNS:
            self.assertEqual(a[c][0, 13], 0)
            self.assertTrue((a[c][0, 42:] == 0).all())
        original = pa.Table.from_pylist([row])
        self.assertTrue(result.select(original.column_names).equals(original))

    def test_nonfinite_any_source_zeros_whole_row_and_continues(self):
        info, good, _ = fixture(hand_count=12)
        cases = [("observation.state", -1, np.nan),  # unnamed component
                 ("action", 0, np.inf),  # excluded >6-channel hand
                 ("eef_sim_pose_state", 3, -np.inf),
                 ("eef_sim_pose_action", 6, np.nan),
                 ("gripper_open_scale_state", 1, np.inf),
                 ("gripper_open_scale_action", 0, np.nan)]
        for column, index, value in cases:
            with self.subTest(column=column):
                bad_row = copy.deepcopy(good)
                bad_row[column][index] = value
                _, arrays, bad, logs = self.convert(info, [good, bad_row, good])
                self.assertEqual(bad, 1)
                self.assertEqual(logs[0]["source_row"], 11)
                self.assertEqual(logs[0]["field"], column)
                for a in arrays.values():
                    self.assertTrue((a[1] == 0).all())
                    np.testing.assert_array_equal(a[0], a[2])
                self.assertTrue((arrays["mask_state"][:, 30:42] == 0).all())

    def test_missing_gripper_and_mismatched_hand_names(self):
        info, row, _ = fixture()
        for column in ("gripper_open_scale_state", "gripper_open_scale_action"):
            del info["features"][column]
            del row[column]
        ns = info["features"]["action"]["names"]
        ns[ns.index("left_hand_joint_1_rad")] = "left_hand_joint_9_rad"
        _, arrays, bad, _ = self.convert(info, [row])
        self.assertEqual(bad, 0)
        self.assertTrue((arrays["mask_state"][0, 28:36] == 0).all())
        self.assertTrue((arrays["mask_state"][0, 36:42] == 1).all())
        self.assertTrue((arrays["mask_state"][0, 14:28] == 1).all())

    def test_bad_row_shape_null_and_float32_overflow(self):
        info, good, _ = fixture()
        for column, value in (("eef_sim_pose_action", [0.] * 11),
                              ("eef_sim_pose_state", None),
                              ("action", [None] * len(good["action"])),
                              ("gripper_open_scale_action", [1e100, 0.])):
            with self.subTest(column=column, value=str(value)[:20]):
                row = copy.deepcopy(good)
                row[column] = value
                _, arrays, bad, logs = self.convert(info, [row, good])
                self.assertEqual(bad, 1)
                self.assertTrue(logs)
                for a in arrays.values():
                    self.assertTrue((a[0] == 0).all())
                self.assertGreater(arrays["mask_action"][1].sum(), 0)

    def test_arm_schema_failure(self):
        info, _, _ = fixture()
        names = info["features"]["observation.state"]["names"]
        names[names.index("left_arm_joint_2_rad")] = "left_arm_joint_8_rad"
        with self.assertRaisesRegex(ValueError, "contiguous"):
            joint_mappings(info)

    def test_disk_roundtrip_metadata_and_error_log(self):
        info, row, _ = fixture()
        bad = copy.deepcopy(row)
        bad["action"][-1] = np.nan
        original = pa.Table.from_pylist([row, bad, row])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            (source / "meta").mkdir(parents=True)
            (source / "videos").mkdir()
            (source / "videos/keep.mp4").write_bytes(b"video unchanged")
            data = source / "data/chunk-000/file-000.parquet"
            data.parent.mkdir(parents=True)
            info_bytes = json.dumps(info).encode()
            (source / "meta/info.json").write_bytes(info_bytes)
            (source / "meta/keep.txt").write_text("original metadata")
            pq.write_table(original, data)
            original_schema = pq.read_schema(data)
            summary = convert_dataset(source, batch_size=2)
            self.assertEqual(summary["status"], "complete_with_errors")
            self.assertEqual(summary["zeroed_rows"], 1)
            result = pq.read_table(data)
            self.assertEqual(list((source / "data").rglob("*.parquet")), [data])
            self.assertFalse(list(source.rglob("*.tmp")))
            self.assertTrue(result.select(original.column_names).schema.equals(original_schema, check_metadata=True))
            # Arrow equality treats NaN != NaN; compare source numeric values explicitly.
            for name in original.column_names:
                if name == "extra":
                    self.assertEqual(original[name].to_pylist(), result[name].to_pylist())
                else:
                    np.testing.assert_equal(original[name].to_pylist(), result[name].to_pylist())
            updated = json.loads((source / "meta/info.json").read_text())
            for name, value in info["features"].items():
                self.assertEqual(updated["features"][name], value)
            for name in OUTPUT_COLUMNS:
                self.assertEqual(updated["features"][name]["shape"], [55])
                self.assertEqual(result[name].type.list_size, 55)
            log_path = source / "qc/conversion_errors.jsonl"
            log = [json.loads(s) for s in log_path.read_text().splitlines()]
            self.assertEqual(len(log), 1)
            self.assertEqual(log[0]["source_row"], 1)
            self.assertEqual(log[0]["result"], "row_zeroed")
            self.assertEqual((source / "videos/keep.mp4").read_bytes(), b"video unchanged")
            self.assertEqual((source / "meta/keep.txt").read_text(), "original metadata")
            # 重跑不重复追加列，结果一致；错误报告只保留本轮，避免重复累计。
            rerun = convert_dataset(source, batch_size=2)
            self.assertEqual(rerun["status"], "complete_with_errors")
            again = pq.read_table(data)
            self.assertEqual(again.column_names, original.column_names + list(OUTPUT_COLUMNS))
            for name in OUTPUT_COLUMNS:
                self.assertTrue(result[name].equals(again[name]))
            self.assertEqual(json.loads((source / "meta/info.json").read_text()), updated)
            self.assertEqual([json.loads(s) for s in log_path.read_text().splitlines()], log)

    def test_write_failure_preserves_original_and_retry_finishes(self):
        info, row, _ = fixture()
        original = pa.Table.from_pylist([row] * 3)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "meta").mkdir()
            (source / "data").mkdir()
            info_path = source / "meta/info.json"
            info_path.write_text(json.dumps(info))
            info_before = info_path.read_bytes()
            first, second = source / "data/first.parquet", source / "data/second.parquet"
            for path in (first, second):
                pq.write_table(original, path)
            before = first.read_bytes()
            real_write = pq.ParquetWriter.write_table
            calls = 0

            def fail_second_batch(writer, table, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated write failure")
                return real_write(writer, table, *args, **kwargs)

            # 第一份文件写到一半失败，第二份继续成功；第一份原文件必须完整保留。
            with patch.object(pq.ParquetWriter, "write_table", new=fail_second_batch):
                summary = convert_dataset(source, batch_size=2)
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["files"], 1)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(info_path.read_bytes(), info_before)
            self.assertTrue(pq.read_table(second).select(original.column_names).equals(original))
            self.assertFalse(list(source.rglob("*.tmp")))
            errors = (source / "qc/conversion_errors.jsonl").read_text()
            self.assertIn('"result": "file_failed"', errors)

            # 模拟进程被强制中断留下临时文件，重跑仍应从完整原文件重新计算。
            first.with_name(first.name + ".tmp").write_bytes(b"interrupted temporary file")
            summary = convert_dataset(source, batch_size=2)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["files"], 2)
            self.assertFalse(list(source.rglob("*.tmp")))
            for path in (first, second):
                result = pq.read_table(path)
                self.assertEqual(result.column_names, original.column_names + list(OUTPUT_COLUMNS))
                self.assertTrue(result.select(original.column_names).equals(original))

    def test_metadata_failure_can_be_retried_after_parquet_replacement(self):
        info, row, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "meta").mkdir()
            (source / "data").mkdir()
            info_path = source / "meta/info.json"
            info_path.write_text(json.dumps(info))
            before = info_path.read_bytes()
            data = source / "data/example.parquet"
            pq.write_table(pa.Table.from_pylist([row]), data)

            def fail_info(path, value):
                if path == info_path:
                    raise OSError("simulated metadata write failure")
                return write_json(path, value)

            with patch("convert_robocoin_55d.write_json", side_effect=fail_info):
                summary = convert_dataset(source, batch_size=2)
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(info_path.read_bytes(), before)
            self.assertTrue(set(OUTPUT_COLUMNS) <= set(pq.read_schema(data).names))
            summary = convert_dataset(source, batch_size=2)
            self.assertEqual(summary["status"], "complete")
            self.assertTrue(set(OUTPUT_COLUMNS) <= set(json.loads(info_path.read_text())["features"]))


if __name__ == "__main__":
    unittest.main()
