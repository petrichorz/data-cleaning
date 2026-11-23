#!/usr/bin/env python3
"""按完整 episode 原地追加 InternData-A1 的四个 55D 字段；任务目录间并行。"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import fcntl
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from interndata_a1_mapping import OUTPUT_COLUMNS, OUTPUT_FEATURE, ROBOT_KEYS, build_mapping, convert_episode, source_columns


def sync_path(path, directory=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if directory else 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        sync_path(temporary)
        temporary.replace(path)
        sync_path(path.parent, directory=True)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"已有转换进程使用此目录：{path.parent}") from None
        yield


def fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def load_mapping(source):
    info = json.loads((source / "meta/info.json").read_text())
    stats_path = source / "meta/stats.json"
    try:
        stats = json.loads(stats_path.read_text())
    except (OSError, ValueError):
        stats = {}
    mapping = build_mapping(info, stats)
    mapping.update(gripper_stats_source=str(stats_path), quaternion_order="xyzw",
                   pose_frame="source_robot_frame", training_normalization="none",
                   single_arm_placement="left", action_source="actions.*", target_offset=0)
    return info, mapping


def discover_datasets(root):
    found, visited = [], set()
    for directory, children, _ in os.walk(root, followlinks=True):
        path = Path(directory)
        physical = path.resolve()
        if physical in visited:
            children[:] = []
            continue
        visited.add(physical)
        if (path / "meta/info.json").is_file():
            found.append(path)
            children[:] = []
        else:
            children[:] = sorted(name for name in children
                                 if name not in ("data", "videos", "qc", "meta") and not name.startswith("."))
    return sorted(found)


def index_episodes(source, files, mapping, info):
    """逐行组建立位置索引，用 episode manifest 验证完整轨迹。"""
    expected = {}
    manifests = sorted((source / "meta/episodes").rglob("*.parquet"))
    if not manifests:
        raise ValueError("Missing meta/episodes Parquet manifest")
    for manifest in manifests:
        table = pq.read_table(manifest, columns=["episode_index", "length"], use_threads=False)
        for row in table.to_pylist():
            episode, length = row["episode_index"], row["length"]
            if type(episode) is not int or episode < 0 or type(length) is not int or length < 1:
                raise ValueError(f"Invalid episode manifest entry: {row}")
            if episode in expected:
                raise ValueError(f"Duplicate episode_index in manifest: {episode}")
            expected[episode] = length
    groups, episodes, offsets = [], defaultdict(list), []
    required = set(source_columns(mapping)) | {"episode_index", "frame_index"}
    total = 0
    for file_number, path in enumerate(files):
        offsets.append(total)
        with pq.ParquetFile(path) as parquet:
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            offset = 0
            for group in range(parquet.num_row_groups):
                ids = parquet.read_row_group(group, columns=["episode_index"], use_threads=False)["episode_index"].to_pylist()
                rows = defaultdict(list)
                for row, episode in enumerate(ids):
                    if type(episode) is not int or episode < 0:
                        raise ValueError(f"{path}: invalid episode_index at row {offset + row}")
                    rows[episode].append(row)
                group_id = len(groups)
                groups.append((file_number, group, offset))
                for episode, indexes in rows.items():
                    episodes[episode].append((group_id, np.asarray(indexes, dtype=np.int64)))
                offset += len(ids)
            total += offset
    if set(episodes) != set(expected):
        raise ValueError("Episode IDs in data differ from the complete episode manifest")
    for episode, pieces in episodes.items():
        count = sum(len(indexes) for _, indexes in pieces)
        if count != expected[episode]:
            raise ValueError(f"Episode {episode}: {count} data rows != manifest length {expected[episode]}")
    if info.get("total_frames", total) != total or info.get("total_episodes", len(expected)) != len(expected):
        raise ValueError("Data counts differ from meta/info.json")
    return groups, episodes, offsets, total


def prepare_episodes(source, files, mapping, info, temporary_dir, report, summary):
    groups, episodes, offsets, total = index_episodes(source, files, mapping, info)
    outputs = np.memmap(Path(temporary_dir) / "outputs.bin", mode="w+", dtype=np.float32,
                        shape=(max(total, 1), 4, 55))
    columns = [*source_columns(mapping), "episode_index", "frame_index"]

    @lru_cache(maxsize=1)
    def read_group(group_id):
        file_number, group, _ = groups[group_id]
        with pq.ParquetFile(files[file_number]) as parquet:
            return parquet.read_row_group(group, columns=columns, use_threads=False)

    try:
        with (source / "qc/episode_policy.jsonl").open("w") as policy:
            for number, (episode, pieces) in enumerate(sorted(episodes.items()), 1):
                tables, positions, owners = [], [], []
                for group_id, indexes in pieces:
                    file_number, _, offset = groups[group_id]
                    tables.append(read_group(group_id).take(pa.array(indexes)))
                    positions.append(offsets[file_number] + offset + indexes)
                    owners.append(np.full(len(indexes), file_number, dtype=np.int64))
                table = pa.concat_tables(tables, promote_options="default")
                positions, owners = np.concatenate(positions), np.concatenate(owners)
                frames = table["frame_index"].to_pylist()
                if any(type(frame) is not int or frame < 0 for frame in frames):
                    raise ValueError(f"Episode {episode}: missing or invalid frame_index")
                order = np.argsort(frames, kind="stable")
                if not np.array_equal(np.asarray(frames)[order], np.arange(len(frames))):
                    raise ValueError(f"Episode {episode}: frame_index must be unique, contiguous and start at 0")
                table = table.take(pa.array(order))
                positions, owners = positions[order], owners[order]
                converted, errors, raw = convert_episode(table, mapping)

                def episode_error(error):
                    row = error.pop("row")
                    owner = int(owners[row])
                    report(dict(error, source_file=str(files[owner].relative_to(source)),
                                source_row=int(positions[row] - offsets[owner]),
                                episode_index=episode, frame_index=row))

                for error in errors:
                    episode_error(error)
                # 核对原始数值，不使用 float32 派生列掩盖精度差异，不改写源动作。
                action, successor = raw["action"][:-1], raw["state"][1:]
                finite = np.isfinite(action) & np.isfinite(successor)
                mismatched = np.any(finite & (action != successor), axis=1)
                verifiable = finite.all(axis=1)
                for row in np.flatnonzero(mismatched):
                    episode_error({"row": int(row), "field": "actions.*",
                                   "error_type": "next_state_mismatch", "result": "source_action_retained",
                                   "message": "Source action differs from next observed state; no temporal shift applied"})
                converted[-1, [1, 3]] = 0
                outputs[positions] = converted
                record = dict(episode_index=episode, rows=len(table), mode="source_next_state",
                              action_source="actions.*", target_offset=0, temporal_shift_applied=False,
                              terminal_frame_index=len(table) - 1, terminal="derived action and mask zero",
                              next_state_exact_pairs=int((verifiable & ~mismatched).sum()),
                              mismatched_pairs=int(mismatched.sum()), unverifiable_pairs=int((~verifiable).sum()),
                              source_files=[str(files[i].relative_to(source)) for i in sorted(set(owners.tolist()))])
                policy.write(json.dumps(record) + "\n")
                summary["episodes"] += 1
                summary["terminal_rows_masked"] += 1
                summary["next_state_exact_pairs"] += record["next_state_exact_pairs"]
                summary["next_state_mismatched_pairs"] += record["mismatched_pairs"]
                summary["next_state_unverifiable_pairs"] += record["unverifiable_pairs"]
                if number % 100 == 0 or number == len(episodes):
                    print(f"{source.name}: {number}/{len(episodes)} complete episodes prepared", flush=True)
        outputs.flush()
        return outputs, offsets
    except BaseException:
        outputs._mmap.close()
        raise


def rewrite_file(path, outputs, offset, batch_size):
    """原字段 schema/metadata、行顺序保持；临时文件刷盘关闭后才替换。"""
    temporary = path.with_name(path.name + ".55d.tmp")
    rows = 0
    try:
        with pq.ParquetFile(path) as parquet:
            original_schema = pa.schema([field for field in parquet.schema_arrow if field.name not in OUTPUT_COLUMNS],
                                        metadata=parquet.schema_arrow.metadata)
            schema = original_schema
            for name in OUTPUT_COLUMNS:
                schema = schema.append(pa.field(name, pa.list_(pa.float32(), 55)))
            with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                for batch in parquet.iter_batches(batch_size=batch_size, columns=original_schema.names, use_threads=False):
                    table = pa.Table.from_batches([batch])
                    for k, name in enumerate(OUTPUT_COLUMNS):
                        values = np.ascontiguousarray(outputs[offset + rows:offset + rows + len(batch), k])
                        table = table.append_column(name, pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 55))
                    writer.write_table(table)
                    rows += len(batch)
        sync_path(temporary)
        temporary.replace(path)
        sync_path(path.parent, directory=True)
        return rows
    finally:
        temporary.unlink(missing_ok=True)


def _convert_locked(source, batch_size):
    qc = source / "qc"
    summary = dict(source=str(source), status="incomplete", files=0, rows=0, errors=0,
                   error_rows=0, masked_rows=0, zeroed_rows=0, error_types={},
                   episodes=0, terminal_rows_masked=0, next_state_exact_pairs=0,
                   next_state_mismatched_pairs=0, next_state_unverifiable_pairs=0,
                   columns_added=list(OUTPUT_COLUMNS), action_source="actions.*", target_offset=0)
    error_rows, masked_rows, zeroed_rows = set(), set(), set()
    with (qc / "conversion_errors.jsonl").open("w") as log:
        def report(error):
            if "source_row" in error:
                key = (error["source_file"], error["source_row"])
                error_rows.add(key)
                if error["result"] == "row_zeroed":
                    zeroed_rows.add(key)
                elif error["result"] == "components_masked":
                    masked_rows.add(key)
            kind = error["error_type"]
            summary["error_types"][kind] = summary["error_types"].get(kind, 0) + 1
            summary["errors"] += 1
            log.write(json.dumps(error, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()

        try:
            write_json(qc / "conversion_summary.json", summary)
            (qc / "episode_policy.jsonl").write_text("")
            info_path = source / "meta/info.json"
            info_original = fingerprint(info_path)
            info, mapping = load_mapping(source)
            summary["mapping"] = mapping
            files = sorted((source / "data").rglob("*.parquet"))
            if not files:
                raise ValueError("No data Parquet files found")
            if any(path.is_symlink() for path in files):
                raise ValueError("Data Parquet files must not be symlinks")
            originals = {path: fingerprint(path) for path in files}
            with TemporaryDirectory(prefix=".conversion-", dir=qc) as temporary:
                outputs, offsets = prepare_episodes(source, files, mapping, info, temporary, report, summary)
                try:
                    for path, original in originals.items():
                        if fingerprint(path) != original:
                            raise RuntimeError(f"Source changed while preparing episodes: {path}")
                    for number, path in enumerate(files):
                        if fingerprint(path) != originals[path]:
                            raise RuntimeError(f"Source changed before write: {path}")
                        count = rewrite_file(path, outputs, offsets[number], batch_size)
                        summary["files"] += 1
                        summary["rows"] += count
                        print(f"{source.name}: {number + 1}/{len(files)} files written, {count} rows", flush=True)
                finally:
                    outputs._mmap.close()
            if fingerprint(info_path) != info_original:
                raise RuntimeError("meta/info.json changed during conversion")
            for name in OUTPUT_COLUMNS:
                info["features"][name] = dict(OUTPUT_FEATURE)
            write_json(info_path, info)
            summary["status"] = "complete_with_errors" if summary["errors"] else "complete"
        except Exception as error:
            report(dict(error_type=type(error).__name__, message=str(error), result="dataset_failed"))
        finally:
            summary.update(error_rows=len(error_rows), masked_rows=len(masked_rows - zeroed_rows), zeroed_rows=len(zeroed_rows))
            write_json(qc / "conversion_summary.json", summary)
    print(f"{source.name}: {summary['status']}, {summary['errors']} errors", flush=True)
    return summary


def convert_dataset(source, batch_size=4096, log_output=False):
    source = Path(source).resolve()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    qc = source / "qc"
    qc.mkdir(exist_ok=True)
    with exclusive_lock(qc / ".conversion.lock"):
        if log_output:
            with (qc / "conversion.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
                return _convert_locked(source, batch_size)
        return _convert_locked(source, batch_size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="required dataset family root or one task bucket")
    parser.add_argument("--workers", "-j", type=int, default=4, help="parallel task buckets (default: 4)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Parquet I/O rows; never splits episode semantics")
    parser.add_argument("--datasets", nargs="+", help="exact task paths relative to the family root")
    parser.add_argument("--robots", nargs="+", choices=list(ROBOT_KEYS.values()), help="select embodiment keys")
    parser.add_argument("--dry-run", action="store_true", help="inspect mapping without writing any files")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be positive")
    root = args.source.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")
    datasets = discover_datasets(root)
    names = {path: path.relative_to(root).as_posix() for path in datasets}
    if args.datasets:
        missing = set(args.datasets) - set(names.values())
        if missing:
            parser.error(f"Unknown task paths: {sorted(missing)}")
        datasets = [path for path in datasets if names[path] in args.datasets]
    if args.robots:
        datasets = [path for path in datasets
                    if ROBOT_KEYS.get(json.loads((path / "meta/info.json").read_text()).get("robot_type")) in args.robots]
    if not datasets:
        parser.error("No matching task buckets with meta/info.json")
    if args.dry_run:
        for path in datasets:
            info, mapping = load_mapping(path)
            scales = {arm["side"]: arm["gripper_scale"]["value"] for arm in mapping["arms"]}
            print(f"{names[path]}: {mapping['embodiment']}, {mapping['arm_layout']}, "
                  f"{info.get('total_frames', '?')} rows, gripper scales={scales}")
        print(f"Preview only: {len(datasets)} task buckets; no files changed.")
        return 0
    qc = root / "qc"
    qc.mkdir(exist_ok=True)
    with exclusive_lock(qc / ".batch_conversion.lock"):
        result = dict(root=str(root), status="incomplete", total=len(datasets), completed=0, rows=0, errors=0, results=[])
        summary_path = qc / "conversion_batch_summary.json"
        write_json(summary_path, result)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(datasets)), mp_context=context) as pool:
            futures = {pool.submit(convert_dataset, path, args.batch_size, True): path for path in datasets}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    summary = future.result()
                except Exception as error:
                    summary = dict(source=str(path), status="incomplete", rows=0, errors=1, message=str(error))
                result["results"].append(summary)
                result["completed"] += 1
                result["rows"] += summary["rows"]
                result["errors"] += summary["errors"]
                print(f"[{result['completed']}/{len(datasets)}] {names[path]}: "
                      f"{summary['status']}, {summary['rows']} rows, {summary['errors']} errors", flush=True)
                write_json(summary_path, result)
        statuses = Counter(record["status"] for record in result["results"])
        result["status"] = "incomplete" if statuses["incomplete"] else "complete_with_errors" if result["errors"] else "complete"
        result["status_counts"] = dict(statuses)
        write_json(summary_path, result)
        print(f"{result['status']}; {result['rows']} rows; summary: {summary_path}", flush=True)
        return 1 if result["status"] == "incomplete" else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
