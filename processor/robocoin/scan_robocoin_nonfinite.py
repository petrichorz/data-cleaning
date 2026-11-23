#!/usr/bin/env python3
"""只读并行扫描 RoboCOIN，打印含 NaN/Inf 的文件、episode 和帧。"""

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# 检查原始来源的全部分量，包括未接入 55D 的手指和未命名尾部。
SOURCE_COLUMNS = ("observation.state", "action", "eef_sim_pose_state", "eef_sim_pose_action",
                  "gripper_open_scale_state", "gripper_open_scale_action")
PRINT_LOCK = None


def init_worker(lock):
    global PRINT_LOCK
    PRINT_LOCK = lock


def emit(record):
    # 多进程共用输出锁，一条 JSON 不会被另一个进程的内容打断。
    with PRINT_LOCK:
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def scan_file(path, columns, batch_size):
    rows = bad_rows = bad_values = 0
    try:
        with pq.ParquetFile(path) as parquet:
            present = [c for c in columns if c in parquet.schema_arrow.names]
            if not present:
                raise ValueError("文件中没有任何指定的待检查列")
            ids = [c for c in ("episode_index", "frame_index") if c in parquet.schema_arrow.names]
            selected = list(dict.fromkeys(present + ids))
            # 每次只读一批、只读所需列；进程之间并行，不额外开启 Arrow 读线程。
            for batch in parquet.iter_batches(batch_size=batch_size, columns=selected, use_threads=False):
                problems = {}
                for name in present:
                    column = batch.column(batch.schema.get_field_index(name))
                    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type) or pa.types.is_fixed_size_list(column.type):
                        flat = pc.list_flatten(column)
                        parents = pc.list_parent_indices(column).to_numpy()
                        lengths = pc.fill_null(pc.list_value_length(column), 0).to_numpy()
                        starts = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
                    else:
                        flat = column
                        parents = np.arange(len(column))
                        starts = parents
                    # 整数没有 NaN/Inf；null 是缺失值，单独于本次 NaN/Inf 扫描。
                    if pa.types.is_integer(flat.type) or pa.types.is_null(flat.type):
                        continue
                    if not pa.types.is_floating(flat.type):
                        raise ValueError(f"{name}: 不支持的数值类型 {flat.type}")
                    bad = pc.indices_nonzero(pc.fill_null(pc.or_(pc.is_nan(flat), pc.is_inf(flat)), False)).to_numpy()
                    for index in bad:
                        row = int(parents[index])
                        value = flat[int(index)].as_py()
                        field = problems.setdefault(row, {}).setdefault(name, {"indices": [], "values": []})
                        field["indices"].append(int(index - starts[row]))
                        field["values"].append("NaN" if np.isnan(value) else "+Inf" if value > 0 else "-Inf")
                # 每个有问题的帧打印一行，可同时列出多个来源字段，避免重复打印整帧。
                for row, fields in sorted(problems.items()):
                    emit({"type": "nonfinite", "path": str(path), "parquet_row": rows + row,
                          **{key: batch.column(batch.schema.get_field_index(key))[row].as_py() if key in ids else None
                             for key in ("episode_index", "frame_index")},
                          "fields": fields})
                    bad_values += sum(len(f["indices"]) for f in fields.values())
                bad_rows += len(problems)
                rows += len(batch)
        return {"rows": rows, "bad_rows": bad_rows, "bad_values": bad_values, "failed": 0}
    except Exception as exc:
        # 无法读取也明确输出，不能把没扫描完的文件当成没有 NaN/Inf。
        emit({"type": "scan_error", "path": str(path), "message": f"{type(exc).__name__}: {exc}"})
        return {"rows": rows, "bad_rows": bad_rows, "bad_values": bad_values, "failed": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/RoboCOIN/"))
    parser.add_argument("--prefixes", nargs="+", help="仅扫描目录名匹配这些前缀的子数据集，区分大小写")
    parser.add_argument("--workers", type=int, default=4, help="并行扫描文件的进程数，默认 4")
    parser.add_argument("--batch-size", type=int, default=4096, help="每批读取的行数，默认 4096")
    parser.add_argument("--columns", nargs="+", default=SOURCE_COLUMNS, help="覆盖默认的六个原始来源列")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers 和 --batch-size 必须为正整数")
    if root.is_file() and root.suffix == ".parquet":
        if args.prefixes:
            parser.error("直接指定 Parquet 时不使用 --prefixes")
        files = [root]
    elif root.is_dir():
        datasets = [root] if (root / "data").is_dir() else sorted(p for p in root.iterdir() if (p / "data").is_dir())
        if args.prefixes:
            datasets = [p for p in datasets if any(p.name.startswith(prefix) for prefix in args.prefixes)]
        files = [p for dataset in datasets for p in sorted((dataset / "data").rglob("*.parquet"))]
    else:
        parser.error(f"目录或 Parquet 文件不存在：{root}")
    if not files:
        parser.error("没有匹配的 data/*.parquet 文件")

    totals = {"rows": 0, "bad_rows": 0, "bad_values": 0, "failed": 0}
    context = mp.get_context("spawn")
    lock = context.Lock()
    completed = 0
    # 最多保留 workers*2 个待完成任务，避免全量数据文件过多时堆积任务对象。
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=init_worker, initargs=(lock,)) as pool:
        remaining = iter(files)
        pending = {}
        while pending or completed < len(files):
            while len(pending) < args.workers * 2:
                path = next(remaining, None)
                if path is None:
                    break
                pending[pool.submit(scan_file, path, args.columns, args.batch_size)] = path
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                path = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    with lock:
                        print(json.dumps({"type": "scan_error", "path": str(path), "message": str(exc)}, ensure_ascii=False), flush=True)
                    result = {"failed": 1}
                for key, count in result.items():
                    totals[key] += count
                completed += 1
            with lock:
                print(f"扫描 {completed}/{len(files)} 文件；异常帧 {totals['bad_rows']}；读取失败 {totals['failed']}",
                      file=sys.stderr, flush=True)
    print(json.dumps({"type": "summary", "files": len(files), **totals}, ensure_ascii=False), flush=True)
    # 0=未发现异常，1=发现 NaN/Inf，2=扫描不完整（文件读取失败等）。
    return 2 if totals["failed"] else 1 if totals["bad_rows"] else 0


if __name__ == "__main__":
    sys.exit(main())
