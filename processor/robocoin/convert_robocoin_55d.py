#!/usr/bin/env python3
"""Add 55D columns to RoboCOIN Parquet files in place, preserving original rows and columns."""

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from robocoin_episode_policy import apply_episode_policy


# 原始列保留，另加两列 55D 数值和两列 mask；mask=1 表示该维度有来源。
# 布局：臂关节 0:14，EEF 位置和四元数 14:28，夹爪 28:30，手指 30:42。
# 腰、头、底盘及预留槽位 42:55 不接入，数值和 mask 始终为 0。
OUTPUT_COLUMNS = ("state_unified", "action_unified", "mask_state", "mask_action")
# 每组来源列都按 state、action 排列，转换时分别读取，互不替换。
RAW_COLUMNS = ("observation.state", "action")
EEF_COLUMNS = ("eef_sim_pose_state", "eef_sim_pose_action")
GRIP_COLUMNS = ("gripper_open_scale_state", "gripper_open_scale_action")
SOURCE_COLUMNS = RAW_COLUMNS + EEF_COLUMNS + GRIP_COLUMNS
OUTPUT_FEATURE = {"dtype": "float32", "shape": [55], "names": None}


def write_json(path, value):
    # 先写临时文件，完整写完再替换目标，避免留下半份 JSON。
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def joint_mappings(info):
    """Return (source indexes, output start) per state/action; do not repair names."""
    mappings = {column: [] for column in RAW_COLUMNS}
    names = {}
    notes = []
    # 从元数据名称找到原始数组下标；不补造名称，也不修改原始数组。
    for column in RAW_COLUMNS:
        listed = info["features"][column].get("names")
        if isinstance(listed, dict):
            listed = listed.get("motors")
        if not isinstance(listed, list) or not all(isinstance(n, str) for n in listed):
            raise ValueError(f"{column}: expected a list of component names")
        names[column] = listed
        # 左右臂按名称识别、按数字编号排序，因此原数组右臂在前也不影响输出。
        # 每侧必须从 1 连续编号且为 6 或 7 维；6 维时第 7 个槽位保持空缺。
        # 名称中的 _rad 只用于匹配，不据此换算或检查数值单位。
        for side, start in (("left", 0), ("right", 7)):
            pairs = sorted(
                (int(match[1]), index)
                for index, name in enumerate(listed)
                if (match := re.fullmatch(side + r"_arm_joint_(\d+)_rad", name))
            )
            numbers = [number for number, _ in pairs]
            if len(pairs) not in (6, 7) or numbers != list(range(1, len(pairs) + 1)):
                raise ValueError(f"{column}: {side} arm must have contiguous joints 1..6 or 1..7; got {numbers}")
            mappings[column].append(([index for _, index in pairs], start))

    # 左右手独立判断：每侧最多放 6 个完整通道，超过就整侧不接入，不截取。
    for side, start in (("left", 30), ("right", 36)):
        hands = {}
        reason = None
        for column, listed in names.items():
            candidates = [(index, name) for index, name in enumerate(listed) if name.startswith(side + "_hand_joint")]
            pairs = []
            for index, name in candidates:
                match = re.fullmatch(side + r"_hand_joint_(\d+)_rad", name)
                if match is None:
                    reason = "unrecognized_hand_names"
                else:
                    pairs.append((int(match[1]), index, name))
            # 按数字编号排序，而不是字符串排序，避免把 joint_10 排到 joint_2 前。
            hands[column] = sorted(pairs)
            if len({number for number, _, _ in pairs}) != len(pairs):
                reason = "duplicate_hand_numbers"
        counts = {column: len(pairs) for column, pairs in hands.items()}
        if not reason:
            if any(count > 6 for count in counts.values()):
                reason = "more_than_6_channels"
            elif all(count == 0 for count in counts.values()):
                reason = "no_hand_fields"
            # state/action 排序后必须对应同一组名称，仅数量相同还不够。
            elif [p[2] for p in hands[RAW_COLUMNS[0]]] != [p[2] for p in hands[RAW_COLUMNS[1]]]:
                reason = "state_action_hand_names_differ"
        # 无法映射时记录原因，该侧保持全零，不影响另一侧、EEF 或夹爪。
        if reason:
            notes.append({"side": side, "reason": reason, "counts": counts})
        else:
            for column, pairs in hands.items():
                mappings[column].append(([index for _, index, _ in pairs], start))
    return mappings, notes


def convert_batch(table, mappings, declared_grips, report, source_file, row_offset):
    """Mask invalid components independently; structural row errors still zero the row."""
    size = len(table)
    # 未接入的槽位保持零；有效数值恰好为零时，mask 仍为 1。
    outputs = [np.zeros((size, 55), dtype=np.float32) for _ in OUTPUT_COLUMNS]
    state, action, state_mask, action_mask = outputs
    columns = [c for c in (*SOURCE_COLUMNS, "episode_index", "frame_index", "index") if c in table.column_names]
    rows = table.select(columns).to_pylist()
    # 本来没有夹爪允许留空；声明过或实际存在的来源列必须能正常读取。
    expected = set(RAW_COLUMNS + EEF_COLUMNS) | declared_grips | (set(table.column_names) & set(GRIP_COLUMNS))
    bad_rows = 0
    for i, row in enumerate(rows):
        vectors = {}
        errors = []
        structural_error = False
        masked = [set(), set()]
        for column in SOURCE_COLUMNS:
            if column not in expected:
                continue
            raw = row.get(column)
            if raw is None:
                structural_error = True
                errors.append({"field": column, "error_type": "missing_value", "message": "Missing source vector"})
                continue
            try:
                values = np.asarray(raw, dtype=np.float64)
                if values.ndim != 1:
                    raise ValueError("Expected a one-dimensional vector")
                # 原始向量的 NaN/Inf 仍全部记录；未映射分量不会影响有效的 55D 槽位。
                bad = np.flatnonzero(~np.isfinite(values))
                if len(bad):
                    errors.append({"field": column, "error_type": "nonfinite", "indices": bad.tolist(),
                                   "values": [str(values[j]) for j in bad], "message": "Source contains NaN/Inf or null"})
                width = 12 if column in EEF_COLUMNS else 2 if column in GRIP_COLUMNS else None
                if width is not None and len(values) != width:
                    raise ValueError(f"Expected {width} components, got {len(values)}")
                if column in RAW_COLUMNS:
                    for indexes, _ in mappings[column]:
                        if max(indexes) >= len(values):
                            raise ValueError(f"Mapped index {max(indexes)} exceeds vector width {len(values)}")
                vectors[column] = values
            except (ValueError, TypeError, OverflowError) as exc:
                structural_error = True
                errors.append({"field": column, "error_type": "invalid_vector", "message": str(exc)})

        if not structural_error:
            # state/action 独立，左右侧独立；不做下一帧替换。
            for k, (out, mask) in enumerate(((state, state_mask), (action, action_mask))):
                eef = vectors[EEF_COLUMNS[k]]
                # 臂关节、手指、位置、夹爪原值复制，每个数值单独判断是否可用。
                parts = [(RAW_COLUMNS[k], start, vectors[RAW_COLUMNS[k]][indexes])
                         for indexes, start in mappings[RAW_COLUMNS[k]]]
                parts += [(EEF_COLUMNS[k], 14, eef[0:3]), (EEF_COLUMNS[k], 21, eef[6:9])]
                if GRIP_COLUMNS[k] in vectors:
                    parts.append((GRIP_COLUMNS[k], 28, vectors[GRIP_COLUMNS[k]]))
                for field, start, values in parts:
                    with np.errstate(over="ignore", invalid="ignore"):
                        converted = values.astype(np.float32)
                    good = np.isfinite(converted)
                    out[i, start:start + len(values)] = np.where(good, converted, 0)
                    mask[i, start:start + len(values)] = good
                    masked[k].update((start + np.flatnonzero(~good)).tolist())
                    overflow = np.flatnonzero(np.isfinite(values) & ~good)
                    if len(overflow):
                        errors.append({"field": field, "error_type": "float32_overflow",
                                       "output_field": OUTPUT_COLUMNS[k],
                                       "output_indices": (start + overflow).tolist(),
                                       "message": "Finite source values overflowed float32"})

                # 固定左位置/欧拉角、右位置/欧拉角；外禀 xyz 转 xyzw 四元数。
                # 一个欧拉角无效就屏蔽该侧四元数四维，位置及另一侧照常保留。
                for source_start, target_start in ((3, 17), (9, 24)):
                    angles = eef[source_start:source_start + 3]
                    if not np.isfinite(angles).all():
                        masked[k].update(range(target_start, target_start + 4))
                        continue  # 上面的原始向量检查已记录具体 NaN/Inf 来源。
                    try:
                        quat = Rotation.from_euler("xyz", angles, degrees=False).as_quat()
                        if not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1, atol=1e-6):
                            raise ValueError("Euler conversion did not produce a finite unit quaternion")
                        out[i, target_start:target_start + 4] = quat
                        mask[i, target_start:target_start + 4] = 1
                    except (ValueError, TypeError, OverflowError) as exc:
                        masked[k].update(range(target_start, target_start + 4))
                        errors.append({"field": EEF_COLUMNS[k], "error_type": "conversion_failed",
                                       "indices": list(range(source_start, source_start + 3)),
                                       "output_field": OUTPUT_COLUMNS[k],
                                       "output_indices": list(range(target_start, target_start + 4)),
                                       "message": str(exc)})

        if structural_error:
            # 向量缺失、维数错误或非数值等结构问题仍沿用原有整行清零规则。
            bad_rows += 1
            for out in outputs:
                out[i] = 0
        if errors:
            # result 是本行处理结果；索引列表只列因异常屏蔽的槽位，不含正常补零。
            result = "row_zeroed" if structural_error else "components_masked" if any(masked) else "source_only"
            for error in errors:
                report({"source_file": source_file, "source_row": row_offset + i,
                        "episode_index": row.get("episode_index"), "frame_index": row.get("frame_index"),
                        "index": row.get("index"), "result": result,
                        "masked_state_indices": sorted(masked[0]), "masked_action_indices": sorted(masked[1]),
                        **error})

    # 只追加派生列；原始列、行数和顺序全部保留。
    for name, array in zip(OUTPUT_COLUMNS, outputs):
        table = table.append_column(name, pa.FixedSizeListArray.from_arrays(pa.array(array.ravel()), 55))
    return table, bad_rows


def prepare_episodes(source, files, info, mappings, declared_grips, report, summary, temporary_dir):
    """先索引完整 episode，再转换；四个派生列暂存磁盘，避免整个子数据集常驻内存。"""
    groups, episodes, file_offsets = [], {}, []
    total = 0
    for file_number, path in enumerate(files):
        file_offsets.append(total)
        with pq.ParquetFile(path) as parquet:
            offset = 0
            for group in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(group).num_rows
                if "episode_index" in parquet.schema_arrow.names:
                    ids = parquet.read_row_group(group, columns=["episode_index"])["episode_index"].to_pylist()
                else:
                    ids = [None] * count
                by_episode = {}
                for row, episode in enumerate(ids):
                    # 无效 ID 不与其他文件合并，照常转 55D，但禁止移位并记录原因。
                    key = episode if type(episode) is int and episode >= 0 else ("invalid", file_number)
                    by_episode.setdefault(key, []).append(row)
                group_id = len(groups)
                groups.append((file_number, group, offset, count))
                for episode, indexes in by_episode.items():
                    episodes.setdefault(episode, []).append((group_id, np.asarray(indexes, dtype=np.int64)))
                offset += count
            total += offset
    # 这里只暂存新增的四列；不复制视频或整份原始数据集。空数据集也可正常写回。
    outputs = np.memmap(Path(temporary_dir) / "outputs.bin", mode="w+", dtype=np.float32,
                        shape=(max(total, 1), 4, 55))
    file_bad_rows = [0] * len(files)

    @lru_cache(maxsize=1)
    def read_group(group_id):
        # 相邻 episode 常在同一个 row group，缓存一个，避免重复解压读取。
        file_number, group, _, _ = groups[group_id]
        with pq.ParquetFile(files[file_number]) as parquet:
            columns = [c for c in (*SOURCE_COLUMNS, "episode_index", "frame_index", "index")
                       if c in parquet.schema_arrow.names]
            return parquet.read_row_group(group, columns=columns)

    policy_path = source / "qc/episode_policy.jsonl"
    with policy_path.open("w", encoding="utf-8") as policy_log:
        for number, (episode, pieces) in enumerate(episodes.items(), 1):
            tables, locations, file_numbers = [], [], []
            for group_id, indexes in pieces:
                file_number, _, offset, _ = groups[group_id]
                tables.append(read_group(group_id).take(pa.array(indexes)))
                locations.append(file_offsets[file_number] + offset + indexes)
                file_numbers.append(np.full(len(indexes), file_number, dtype=np.int64))
            table = pa.concat_tables(tables, promote_options="default")
            positions = np.concatenate(locations)
            owners = np.concatenate(file_numbers)
            structural_rows = set()

            def episode_error(error):
                # convert_batch 使用集内行号；这里恢复为原文件、原行号用于定位。
                row = error["source_row"]
                owner = int(owners[row])
                error["source_file"] = str(files[owner].relative_to(source))
                error["source_row"] = int(positions[row] - file_offsets[owner])
                if error["result"] == "row_zeroed" and row not in structural_rows:
                    file_bad_rows[owner] += 1
                    structural_rows.add(row)
                report(error)

            converted, _ = convert_batch(table, mappings, declared_grips, episode_error, "", 0)
            arrays = np.stack([np.asarray(converted[c].to_pylist(), dtype=np.float32)
                               for c in OUTPUT_COLUMNS], axis=1)
            result = apply_episode_policy(table, info, mappings, arrays, invalid_episode=isinstance(episode, tuple))
            outputs[positions] = arrays
            result.update(episode_index=None if isinstance(episode, tuple) else episode,
                          source_files=[str(files[i].relative_to(source)) for i in sorted(set(owners.tolist()))])
            policy_log.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            policy_log.flush()
            summary["episodes"] += 1
            shifted = result["mode"] == "derived_next_state"
            summary["shifted_episodes" if shifted else "retained_episodes"] += 1
            reasons = summary["episode_policy_reasons"]
            reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
            if number % 100 == 0 or number == len(episodes):
                print(f"{source.name}: {number}/{len(episodes)} episodes, "
                      f"{summary['shifted_episodes']} next-state targets prepared", flush=True)
    outputs.flush()
    return outputs, file_offsets, file_bad_rows


def convert_dataset(source, batch_size):
    # 直接在原子数据集里追加字段，qc 也放在该子数据集下。
    qc = source / "qc"
    qc.mkdir(exist_ok=True)
    # 每次运行重新生成报告；中断后重跑时，从原始列重新计算所有文件。
    summary = {"source": str(source), "status": "incomplete", "files": 0, "rows": 0,
               "zeroed_rows": 0, "error_rows": 0, "masked_rows": 0, "errors": 0, "columns_added": list(OUTPUT_COLUMNS),
               "episodes": 0, "shifted_episodes": 0, "retained_episodes": 0, "episode_policy_reasons": {}}
    with (qc / "conversion_errors.jsonl").open("w", encoding="utf-8") as log:
        last_error_row = None

        def report(error):
            nonlocal last_error_row
            if "source_row" in error:
                key = (error["source_file"], error["source_row"])
                if key != last_error_row:
                    summary["error_rows"] += 1
                    summary["masked_rows"] += error["result"] == "components_masked"
                    last_error_row = key
            log.write(json.dumps(error, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            summary["errors"] += 1

        try:
            # 先标记本轮未完成，避免中断后仍看到上轮的完成状态。
            write_json(qc / "conversion_summary.json", summary)
            (qc / "episode_policy.jsonl").write_text("")
            info = json.loads((source / "meta/info.json").read_text())
            mappings, notes = joint_mappings(info)
            summary["unmapped_hands"] = notes
            summary["joint_mappings"] = mappings
            # 无手指或超过 6 维是正常的未接入情况；名称歧义另写错误日志。
            for note in notes:
                if note["reason"] in ("unrecognized_hand_names", "duplicate_hand_numbers", "state_action_hand_names_differ"):
                    report({"error_type": "hand_mapping", "result": "hand_masked", **note})
            declared_grips = set(info["features"]) & set(GRIP_COLUMNS)
            files = sorted((source / "data").rglob("*.parquet"))
            if not files:
                raise ValueError("No data Parquet files found")
            # 先准备所有完整 episode，再写回；文件边界和 I/O 批次都不是 episode 边界。
            # 临时派生列由磁盘承载，内存主要保存一个 row group 和一个 episode。
            with TemporaryDirectory(prefix=".conversion-", dir=qc) as temporary_dir:
                outputs, file_offsets, bad_counts = prepare_episodes(
                    source, files, info, mappings, declared_grips, report, summary, temporary_dir)
                try:
                    for number, path in enumerate(files, 1):
                        relative = path.relative_to(source)
                        temporary = path.with_name(path.name + ".tmp")
                        row_offset = 0
                        file_bad_rows = bad_counts[number - 1]
                        try:
                            with pq.ParquetFile(path) as parquet:
                                # 重跑时只移除旧的四个派生字段，再从原始列计算，避免重复追加。
                                schema = parquet.schema_arrow
                                for index in range(len(schema) - 1, -1, -1):
                                    if schema[index].name in OUTPUT_COLUMNS:
                                        schema = schema.remove(index)
                                original_columns = schema.names
                                for name in OUTPUT_COLUMNS:
                                    schema = schema.append(pa.field(name, pa.list_(pa.float32(), 55)))
                                # 分批写到同目录临时文件；读写句柄关闭后才替换原文件。
                                # 写入失败时原文件仍在，不会留下半个 Parquet 覆盖原文件。
                                with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                                    for batch in parquet.iter_batches(batch_size=batch_size, columns=original_columns):
                                        original = pa.Table.from_batches([batch]).select(original_columns)
                                        start = file_offsets[number - 1] + row_offset
                                        converted = original
                                        for k, name in enumerate(OUTPUT_COLUMNS):
                                            values = np.ascontiguousarray(outputs[start:start + len(batch), k])
                                            converted = converted.append_column(name, pa.FixedSizeListArray.from_arrays(
                                                pa.array(values.ravel()), 55))
                                        writer.write_table(converted)
                                        row_offset += len(batch)
                            temporary.replace(path)
                            summary["files"] += 1
                            summary["rows"] += row_offset
                            summary["zeroed_rows"] += file_bad_rows
                            print(f"{source.name}: {number}/{len(files)} files, {row_offset} rows, {file_bad_rows} zeroed", flush=True)
                        except Exception as exc:
                            # 文件读写失败时保留原文件，移除临时结果，记录错误并继续下一个文件。
                            temporary.unlink(missing_ok=True)
                            report({"source_file": str(relative), "error_type": type(exc).__name__,
                                    "message": str(exc), "result": "file_failed"})
                finally:
                    outputs._mmap.close()

            if summary["files"] == len(files):
                # 所有文件完成后更新原 info.json，仅登记四个 55D 字段。
                # 原字段定义、其他元数据和视频都保留，不计算新的训练统计。
                for name in OUTPUT_COLUMNS:
                    info["features"][name] = dict(OUTPUT_FEATURE)
                write_json(source / "meta/info.json", info)
                summary["status"] = "complete_with_errors" if summary["errors"] else "complete"
        except Exception as exc:
            # 元数据或关节定义等子数据集级错误也写入 qc，完成状态保持 incomplete。
            report({"error_type": type(exc).__name__, "message": str(exc), "result": "dataset_failed"})
        finally:
            write_json(qc / "conversion_summary.json", summary)
    print(f"{source.name}: {summary['status']}; errors={summary['errors']}; qc={qc}", flush=True)
    return summary


def main():
    # 支持输入整个 RoboCOIN 根目录或单个子数据集；--datasets 可按完整名称筛选。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="RoboCOIN root, or one subdataset directory")
    parser.add_argument("--datasets", nargs="+", help="Process only these exact subdataset names")
    parser.add_argument("--batch-size", type=int, default=4096, help="Parquet write batch rows; episode conversion is never split (default: 4096)")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    datasets = [source] if (source / "meta/info.json").is_file() else sorted(p.parent.parent for p in source.glob("*/meta/info.json"))
    if args.datasets:
        missing = set(args.datasets) - {p.name for p in datasets}
        if missing:
            parser.error(f"Unknown datasets: {sorted(missing)}")
        datasets = [p for p in datasets if p.name in args.datasets]
    if not datasets:
        parser.error("No subdatasets with meta/info.json found")
    failed = False
    # 顺序处理子数据集；有行级错误但文件完整时仍算完成，incomplete 才返回失败码。
    for dataset in datasets:
        summary = convert_dataset(dataset, args.batch_size)
        failed |= summary["status"] == "incomplete"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
