#!/usr/bin/env python3
"""Add 55D columns to RoboCOIN Parquet files in place, preserving original rows and columns."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


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
    """Convert rows independently; any nonfinite source value invalidates both outputs."""
    size = len(table)
    # 先全部补零，只有成功填入的槽位才把 mask 置 1。
    # 有效数值恰好为 0 时，mask 仍然为 1，不能把它当作缺失。
    outputs = [np.zeros((size, 55), dtype=np.float32) for _ in OUTPUT_COLUMNS]
    state, action, state_mask, action_mask = outputs
    columns = [c for c in (*SOURCE_COLUMNS, "episode_index", "frame_index", "index") if c in table.column_names]
    rows = table.select(columns).to_pylist()
    # 原始 state/action 和 EEF 必须存在。夹爪若根本未提供，允许留空；
    # 若元数据声明了夹爪或文件已有夹爪列，则该列每行都必须可读。
    expected = set(RAW_COLUMNS + EEF_COLUMNS) | declared_grips | (set(table.column_names) & set(GRIP_COLUMNS))
    bad_rows = 0
    for i, row in enumerate(rows):
        vectors = {}
        errors = []
        # 先检查本行所有来源向量，再转换。NaN/Inf 检查覆盖整个向量，
        # 包括最终不接入 55D 的手指、腰等分量，以及没有名称的尾部数值。
        for column in SOURCE_COLUMNS:
            if column not in expected:
                continue
            raw = row.get(column)
            if raw is None:
                errors.append({"field": column, "error_type": "missing_value", "message": "Missing source vector"})
                continue
            try:
                values = np.asarray(raw, dtype=np.float64)
                if values.ndim != 1:
                    raise ValueError("Expected a one-dimensional vector")
                bad = np.flatnonzero(~np.isfinite(values))
                if len(bad):
                    errors.append({"field": column, "error_type": "nonfinite", "indices": bad.tolist(),
                                   "values": [str(values[j]) for j in bad], "message": "Source contains NaN/Inf or null"})
                width = 12 if column in EEF_COLUMNS else 2 if column in GRIP_COLUMNS else None
                if width is not None and len(values) != width:
                    raise ValueError(f"Expected {width} components, got {len(values)}")
                if column in RAW_COLUMNS:
                    # 只要求实际要读取的下标存在；原数组有额外分量不影响映射。
                    for indexes, _ in mappings[column]:
                        if max(indexes) >= len(values):
                            raise ValueError(f"Mapped index {max(indexes)} exceeds vector width {len(values)}")
                vectors[column] = values
            except (ValueError, TypeError, OverflowError) as exc:
                errors.append({"field": column, "error_type": "invalid_vector", "message": str(exc)})

        if not errors:
            try:
                # state/action 各转各的，不判断两者是否相等，也不做下一帧替换。
                for k, (out, mask) in enumerate(((state, state_mask), (action, action_mask))):
                    field = RAW_COLUMNS[k]
                    with np.errstate(over="ignore", invalid="ignore"):
                        # 臂关节和可接入的手指直接复制原值，不做单位换算或归一化。
                        for indexes, start in mappings[RAW_COLUMNS[k]]:
                            out[i, start:start + len(indexes)] = vectors[RAW_COLUMNS[k]][indexes]
                            mask[i, start:start + len(indexes)] = 1
                        field = EEF_COLUMNS[k]
                        eef = vectors[field]
                        # 沿用 OpenWAM 的固定顺序：左位置、左欧拉角、右位置、右欧拉角。
                        out[i, 14:17], out[i, 21:24] = eef[0:3], eef[6:9]
                        angles = eef[[3, 4, 5, 9, 10, 11]].reshape(2, 3)
                        # 沿用其弧度制、外禀 xyz 约定，转成 [x, y, z, w] 四元数。
                        # 每帧独立转换，不参考上一帧调整四元数正负号。
                        quat = Rotation.from_euler("xyz", angles, degrees=False).as_quat()
                        if not np.isfinite(quat).all() or not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-6):
                            raise ValueError("Euler conversion did not produce finite unit quaternions")
                        out[i, 17:21], out[i, 24:28] = quat
                        mask[i, 14:28] = 1
                        # 夹爪沿用 gripper_open_scale 的左右顺序和原值，不翻转或裁剪。
                        if GRIP_COLUMNS[k] in vectors:
                            field = GRIP_COLUMNS[k]
                            out[i, 28:30] = vectors[GRIP_COLUMNS[k]]
                            mask[i, 28:30] = 1
                    field = OUTPUT_COLUMNS[k]
                    # 原值即使有限，写成 float32 后也可能溢出；此时同样按整行错误处理。
                    if not np.isfinite(out[i]).all():
                        raise ValueError("55D result contains NaN/Inf after float32 conversion")
            except (ValueError, TypeError, OverflowError) as exc:
                errors.append({"field": field, "error_type": "conversion_failed", "message": str(exc)})

        if errors:
            # 任意来源或转换出错，本行新增的 state/action 和两份 mask 一起清零。
            # 原始行仍保留；逐项记录文件、行号、episode/frame，方便后续质检定位。
            bad_rows += 1
            for out in outputs:
                out[i] = 0
            for error in errors:
                report({"source_file": source_file, "source_row": row_offset + i,
                        "episode_index": row.get("episode_index"), "frame_index": row.get("frame_index"),
                        "index": row.get("index"), "result": "row_zeroed", **error})

    # 在原 Arrow 表末尾追加四列，保留原始列、行数和顺序。
    for name, array in zip(OUTPUT_COLUMNS, outputs):
        table = table.append_column(name, pa.FixedSizeListArray.from_arrays(pa.array(array.ravel()), 55))
    return table, bad_rows


def convert_dataset(source, batch_size):
    # 直接在原子数据集里追加字段，qc 也放在该子数据集下。
    qc = source / "qc"
    qc.mkdir(exist_ok=True)
    # 每次运行重新生成报告；中断后重跑时，从原始列重新计算所有文件。
    summary = {"source": str(source), "status": "incomplete", "files": 0, "rows": 0,
               "zeroed_rows": 0, "errors": 0, "columns_added": list(OUTPUT_COLUMNS)}
    with (qc / "conversion_errors.jsonl").open("w", encoding="utf-8") as log:
        def report(error):
            log.write(json.dumps(error, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            summary["errors"] += 1

        try:
            # 先标记本轮未完成，避免中断后仍看到上轮的完成状态。
            write_json(qc / "conversion_summary.json", summary)
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
            for number, path in enumerate(files, 1):
                relative = path.relative_to(source)
                temporary = path.with_name(path.name + ".tmp")
                row_offset = 0
                file_bad_rows = 0
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
                            for batch in parquet.iter_batches(batch_size=batch_size):
                                original = pa.Table.from_batches([batch]).select(original_columns)
                                converted, bad = convert_batch(original, mappings,
                                                               declared_grips, report, str(relative), row_offset)
                                writer.write_table(converted)
                                row_offset += len(batch)
                                file_bad_rows += bad
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
    parser.add_argument("--batch-size", type=int, default=4096, help="Parquet rows processed at a time (default: 4096)")
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
