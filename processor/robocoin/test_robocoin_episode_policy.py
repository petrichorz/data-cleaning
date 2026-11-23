"""覆盖整集判定、跨文件时间边界、原始列保留和重跑。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_robocoin_55d import OUTPUT_COLUMNS, convert_batch, convert_dataset, joint_mappings
from robocoin_episode_policy import apply_episode_policy
from test_convert_robocoin_55d import fixture


def episode_fixture(hand_count=6, count=4):
    info, template, _ = fixture(hand_count)
    sn = info["features"]["observation.state"]["names"]
    an = info["features"]["action"]["names"]
    rows = []
    for frame in range(count):
        row = copy.deepcopy(template)
        row.update(frame_index=frame, index=frame, timestamp=frame / 30)
        row["observation.state"][sn.index("left_arm_joint_1_rad")] += frame * .1
        values = dict(zip(sn, row["observation.state"]))
        row["action"] = [values[n] for n in an] + [93., 94.]  # 未命名尾部不同不阻止判定。
        row["eef_sim_pose_action"] = row["eef_sim_pose_state"].copy()
        row["gripper_open_scale_action"] = row["gripper_open_scale_state"].copy()
        rows.append(row)
    return info, rows


def converted_arrays(info, rows):
    mappings, _ = joint_mappings(info)
    table = pa.Table.from_pylist(rows)
    converted, _ = convert_batch(table, mappings,
                                 set(info["features"]) & {"gripper_open_scale_state", "gripper_open_scale_action"},
                                 lambda error: None, "example", 0)
    arrays = np.stack([np.asarray(converted[c].to_pylist(), dtype=np.float32) for c in OUTPUT_COLUMNS], axis=1)
    return table, mappings, arrays


class EpisodeTests(unittest.TestCase):
    def classify(self, info, rows):
        table, mappings, arrays = converted_arrays(info, rows)
        before = arrays.copy()
        result = apply_episode_policy(table, info, mappings, arrays)
        return before, arrays, result

    def test_dynamic_copy_shifts_all_slots_and_only_derived_action(self):
        info, rows = episode_fixture()
        rows = [rows[2], rows[0], rows[3], rows[1]]  # 存储顺序故意不同于时间顺序。
        before, after, report = self.classify(info, rows)
        self.assertEqual(report["mode"], "derived_next_state")
        for i, row in enumerate(rows):
            if row["frame_index"] == 3:
                self.assertTrue((after[i, [1, 3]] == 0).all())
            else:
                j = next(j for j, r in enumerate(rows) if r["frame_index"] == row["frame_index"] + 1)
                np.testing.assert_array_equal(after[i, 1], before[j, 0])
                np.testing.assert_array_equal(after[i, 3], before[j, 2])
        np.testing.assert_array_equal(after[:, [0, 2]], before[:, [0, 2]])

    def test_partial_equal_and_raw_precision_difference_do_not_shift(self):
        info, rows = episode_fixture()
        for column, index, change in (("action", 0, .1), ("eef_sim_pose_action", 3, .1),
                                      ("gripper_open_scale_action", 1, .1), ("action", 0, 1e-10)):
            with self.subTest(column=column, change=change):
                modified = copy.deepcopy(rows)
                modified[2][column][index] += change
                before, after, report = self.classify(info, modified)
                self.assertEqual(report["mode"], "source_native")
                np.testing.assert_array_equal(after, before)

    def test_over_capacity_hand_is_still_compared(self):
        info, rows = episode_fixture(hand_count=12)
        before, after, report = self.classify(info, rows)
        self.assertEqual(report["mode"], "derived_next_state")
        self.assertTrue((after[:, :, 30:42] == 0).all())
        idx = info["features"]["action"]["names"].index("left_hand_joint_12_rad")
        rows[1]["action"][idx] += .1
        before, after, report = self.classify(info, rows)
        self.assertEqual(report["reason"], "shared_raw_components_differ")
        np.testing.assert_array_equal(after, before)

    def test_static_short_and_invalid_frames_keep_source(self):
        for expected, frames in (("too_short", [0, 1]), ("duplicate_frame_index", [0, 1, 1, 2]),
                                 ("noncontiguous_frame_index", [0, 1, 3, 4]),
                                 ("missing_or_invalid_frame_index", [0, None, 2, 3])):
            info, rows = episode_fixture(count=len(frames))
            for row, frame in zip(rows, frames):
                row["frame_index"] = frame
            before, after, report = self.classify(info, rows)
            self.assertEqual(report["reason"], expected)
            np.testing.assert_array_equal(after, before)
        info, rows = episode_fixture()
        for row in rows:
            row["observation.state"] = rows[0]["observation.state"].copy()
            row["action"] = rows[0]["action"].copy()
        # 仅 EEF 运动不足以触发同事的核心运动门槛。
        for i, row in enumerate(rows):
            row["eef_sim_pose_state"][0] = row["eef_sim_pose_action"][0] = float(i)
        before, after, report = self.classify(info, rows)
        self.assertEqual(report["reason"], "static_ambiguous")
        np.testing.assert_array_equal(after, before)

    def test_nonfinite_masks_and_semantic_mismatch_keep_source(self):
        info, rows = episode_fixture()
        for row in rows:
            row["gripper_open_scale_state"][0] = row["gripper_open_scale_action"][0] = np.nan
        before, after, report = self.classify(info, rows)
        self.assertEqual(report["reason"], "nonfinite:gripper")
        np.testing.assert_array_equal(after, before)
        info, rows = episode_fixture()
        table, mappings, arrays = converted_arrays(info, rows)
        arrays[0, 3, 0] = 0
        report = apply_episode_policy(table, info, mappings, arrays)
        self.assertEqual(report["reason"], "invalid_or_different_masks")
        mappings["action"][0][0].reverse()
        report = apply_episode_policy(table, info, mappings, arrays)
        self.assertTrue(report["reason"].startswith("different_field_semantics:"))

    def test_no_gripper_is_not_a_missing_required_semantic(self):
        info, rows = episode_fixture(hand_count=12)
        for c in ("gripper_open_scale_state", "gripper_open_scale_action"):
            del info["features"][c]
            for row in rows:
                del row[c]
        _, arrays, report = self.classify(info, rows)
        self.assertEqual(report["mode"], "derived_next_state")
        self.assertTrue((arrays[:, :, 28:42] == 0).all())

    def test_invalid_episode_and_missing_frame_are_reported(self):
        info, rows = episode_fixture()
        table, mappings, arrays = converted_arrays(info, rows)
        before = arrays.copy()
        result = apply_episode_policy(table, info, mappings, arrays, invalid_episode=True)
        self.assertEqual(result["reason"], "missing_or_invalid_episode_index")
        result = apply_episode_policy(table.drop(["frame_index"]), info, mappings, arrays)
        self.assertEqual(result["reason"], "missing_or_invalid_frame_index")
        np.testing.assert_array_equal(arrays, before)

    def test_cross_file_episode_original_order_and_idempotent_rerun(self):
        info, rows = episode_fixture(hand_count=12, count=5)
        another = copy.deepcopy(rows[:3])
        for row in another:
            row["episode_index"] = 8
        # 差异出现在另一文件：不能只看第一份文件的相等片段就移位。
        another[2]["gripper_open_scale_action"][0] += .5
        # 同一集跨文件、row group、4096 类 I/O 批次，两个集交错存储。
        pieces = [[rows[3], another[0], rows[0]], [rows[4], another[2], rows[1]], [rows[2], another[1]]]
        expected = {}
        for ep_rows in (rows, another):
            _, arrays, result = self.classify(info, ep_rows)
            self.assertEqual(result["mode"], "derived_next_state" if ep_rows is rows else "source_native")
            for i, row in enumerate(ep_rows):
                expected[(row["episode_index"], row["frame_index"])] = arrays[i]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "meta").mkdir()
            (source / "data").mkdir()
            (source / "meta/info.json").write_text(json.dumps(info))
            for i, part in enumerate(pieces):
                pq.write_table(pa.Table.from_pylist(part), source / f"data/{i}.parquet", row_group_size=2)
            for batch_size in (2, 1):
                summary = convert_dataset(source, batch_size)
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["shifted_episodes"], 1)
                self.assertEqual(summary["retained_episodes"], 1)
                for i, part in enumerate(pieces):
                    result = pq.read_table(source / f"data/{i}.parquet")
                    original = pa.Table.from_pylist(part)
                    self.assertTrue(result.select(original.column_names).equals(original))
                    for j, row in enumerate(part):
                        actual = np.stack([result[c][j].as_py() for c in OUTPUT_COLUMNS])
                        np.testing.assert_array_equal(actual, expected[(row["episode_index"], row["frame_index"])])
                policies = [json.loads(line) for line in (source / "qc/episode_policy.jsonl").read_text().splitlines()]
                self.assertEqual(len(policies), 2)
                self.assertFalse(list(source.glob("qc/.conversion-*")))

    def test_cross_file_error_keeps_original_file_and_row_location(self):
        info, rows = episode_fixture()
        rows[2]["eef_sim_pose_action"] = [0.] * 11
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "meta").mkdir()
            (source / "data").mkdir()
            (source / "meta/info.json").write_text(json.dumps(info))
            pq.write_table(pa.Table.from_pylist([rows[3], rows[0]]), source / "data/a.parquet")
            pq.write_table(pa.Table.from_pylist([rows[1], rows[2]]), source / "data/b.parquet")
            summary = convert_dataset(source, 1)
            self.assertEqual(summary["status"], "complete_with_errors")
            self.assertEqual(summary["zeroed_rows"], 1)
            self.assertEqual(summary["error_rows"], 1)
            self.assertEqual(summary["retained_episodes"], 1)
            errors = [json.loads(s) for s in (source / "qc/conversion_errors.jsonl").read_text().splitlines()]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["source_file"], "data/b.parquet")
            self.assertEqual(errors[0]["source_row"], 1)
            self.assertEqual(errors[0]["frame_index"], 2)
            result = pq.read_table(source / "data/b.parquet")
            for c in OUTPUT_COLUMNS:
                self.assertTrue(np.all(np.asarray(result[c][1].as_py()) == 0))
            self.assertEqual(result["eef_sim_pose_action"][1].as_py(), [0.] * 11)


if __name__ == "__main__":
    unittest.main()
