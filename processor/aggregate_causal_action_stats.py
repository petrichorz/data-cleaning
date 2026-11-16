# SPDX-License-Identifier: OpenMDW-1.1
"""Aggregate per-dataset statistics without reading trajectories.

python processor/aggregate_causal_action_stats.py --dataset-root /data/parent --output /data/stats.json

独立脚本，仅依赖 NumPy；规则来自 cosmos-framework-ascend 的同名工具。
一个调用合并一个来源组，由输入目录及可选前缀决定组成员，不自动猜机型。
Global quantiles cannot be recovered from local quantiles. q01/q99 store the minimum source q01 and maximum source q99, respectively.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

STATS_RELATIVE_PATH = Path("meta/causal_action_stats.json")


def aggregate_statistics(paths, *, bounds="q01_q99"):
    """合并全量矩；分位数只保存子集外包范围，不冒充总体分位数。"""
    if bounds not in ("q01_q99", "min_max"):
        raise ValueError("Unknown bounds")
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Expected nonempty, distinct statistics paths")
    records = [json.loads(p.read_text()) for p in paths]
    reference = records[0]["provenance"]
    # 只允许统计口径相同的子集；mask 和样本数可以不同。
    contract_keys = (
        "template_id",
        "geometry",
        "population",
        "start_stride",
        "std_ddof",
        "rotation_groups",
    )
    # 当前 VeOmni 文件使用 columns；原 Cosmos 文件使用 read_options 及 split 配置。
    optional_contract = (
        "columns",
        "read_options",
        "split",
        "split_seed",
        "split_val_ratio",
    )
    roots = []
    for path, record in zip(paths, records):
        provenance = record["provenance"]
        if provenance.get("method") not in ("exact", "reservoir") or provenance.get("std_ddof") != 0:
            raise ValueError(f"{path}: expected directly fitted population statistics")
        if provenance.get("kind") != "fitted":
            raise ValueError(f"{path}: expected fitted statistics")
        for key in contract_keys:
            if key not in provenance or provenance[key] != reference.get(key):
                raise ValueError(f"{path}: incompatible statistics: {key}")
        for key in optional_contract:
            if (key in provenance) != (key in reference) or provenance.get(key) != reference.get(key):
                raise ValueError(f"{path}: incompatible statistics: {key}")
        if "columns" not in provenance and "read_options" not in provenance:
            raise ValueError(f"{path}: missing columns/read_options contract")
        source_roots = provenance.get("dataset_roots")
        if source_roots is None:
            source_roots = [provenance.get("dataset_root")]
        if (
            not isinstance(source_roots, list)
            or not source_roots
            or any(not isinstance(root, str) or not root for root in source_roots)
        ):
            raise ValueError(f"{path}: missing source dataset roots")
        roots.extend(str(Path(root).resolve()) for root in source_roots)
        if type(provenance.get("partial")) is not bool:
            raise ValueError(f"{path}: missing partial flag")
        for key in ("blocks", "available_blocks"):
            if type(provenance.get(key)) is not int or provenance[key] < 0:
                raise ValueError(f"{path}: invalid {key}")
        if provenance["blocks"] > provenance["available_blocks"]:
            raise ValueError(f"{path}: blocks exceeds available_blocks")
    if len(set(roots)) != len(roots):
        raise ValueError("Overlapping source datasets")
    width = len(records[0]["state"]["valid_counts"])
    if not width:
        raise ValueError("Empty statistics dimensions")
    rotation_indexes = []
    for group in reference["rotation_groups"]:
        if len(group) != 4 or len(set(group)) != 4 or any(type(i) is not int or not 0 <= i < width for i in group):
            raise ValueError("Invalid quaternion indexes")
        rotation_indexes.extend(group)
    if len(set(rotation_indexes)) != len(rotation_indexes):
        raise ValueError("Overlapping quaternion groups")

    def merge(kind):
        counts = np.zeros(width, dtype=np.int64)
        mean, m2 = np.zeros(width), np.zeros(width)
        minimum, maximum = np.full(width, np.inf), np.full(width, -np.inf)
        qs = ("q01", "q99")
        qmin = {q: np.full(width, np.inf) for q in qs}
        qmax = {q: np.full(width, -np.inf) for q in qs}
        rows = 0
        for path, record in zip(paths, records):
            data = record[kind]
            n = np.asarray(data["valid_counts"])
            row_counts = data["count"]
            if (
                not isinstance(row_counts, list)
                or len(row_counts) != 1
                or type(row_counts[0]) is not int
                or row_counts[0] < 0
            ):
                raise ValueError(f"{path}: {kind}: Invalid total row count")
            rows_i = row_counts[0]
            if n.shape != (width,) or not np.issubdtype(n.dtype, np.integer) or (n < 0).any() or (n > rows_i).any():
                raise ValueError(f"{path}: {kind}: Invalid per-channel counts")
            values = {key: np.asarray(data[key], dtype=np.float64) for key in ("mean", "std", "min", "max", *qs)}
            if any(v.shape != (width,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError(f"{path}: {kind}: Invalid statistics shape or values")
            active = n > 0
            if not np.array_equal(active, data["valid"]) or (values["std"] < 0).any():
                raise ValueError(f"{path}: {kind}: Invalid validity mask or standard deviation")
            if any(
                (values[left][active] > values[right][active]).any()
                for left, right in (("min", "q01"), ("q01", "q99"), ("q99", "max"))
            ):
                raise ValueError(f"{path}: {kind}: Invalid quantile or extrema ordering")
            # 每个子集都必须具有完整四元数组，不能让互补的错误 mask 在聚合后被掩盖。
            for group in reference["rotation_groups"]:
                if active[group].any() and not active[group].all():
                    raise ValueError(f"{path}: {kind}: Invalid quaternion coverage")
            total = counts + n
            delta = values["mean"] - mean
            # std 是总体标准差；恢复 M2 后加入组间均值差，不能直接平均 std。
            m2 += values["std"] ** 2 * n + delta**2 * counts * n / np.maximum(total, 1)
            mean += delta * n / np.maximum(total, 1)
            counts = total
            rows += rows_i
            minimum = np.minimum(minimum, np.where(active, values["min"], np.inf))
            maximum = np.maximum(maximum, np.where(active, values["max"], -np.inf))
            for q in qs:
                qmin[q] = np.minimum(qmin[q], np.where(active, values[q], np.inf))
                qmax[q] = np.maximum(qmax[q], np.where(active, values[q], -np.inf))
        valid = counts > 0

        def clean(x):
            return np.where(valid, x, 0)

        metrics = {
            "min": clean(minimum).tolist(),
            "max": clean(maximum).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(m2, 0) / np.maximum(counts, 1)).tolist(),
            "count": [rows],
            "valid_counts": counts.tolist(),
            "q01": clean(qmin["q01"]).tolist(),
            "q99": clean(qmax["q99"]).tolist(),
        }
        low, high = (minimum, maximum) if bounds == "min_max" else (qmin["q01"], qmax["q99"])
        low, high = clean(low).astype(np.float32), clean(high).astype(np.float32)
        for group in reference["rotation_groups"]:
            indexes = np.asarray(group)
            if valid[indexes].all():
                low[indexes], high[indexes] = -1, 1
        return dict(low=low.tolist(), high=high.tolist(), valid=valid.tolist(), **metrics)

    provenance = {key: reference[key] for key in (*contract_keys, *optional_contract) if key in reference}
    provenance.update(
        kind="fitted",
        method="source_quantile_envelope",
        bounds=bounds,
        quantiles=[0.01, 0.99],
        quantile_semantics="q01=min(source_q01); q99=max(source_q99); not global quantiles",
        quaternion_bounds=[-1, 1],
        dataset_roots=roots,
        input_statistics=[str(p) for p in paths],
        partial=any(r["provenance"]["partial"] for r in records),
        sources=[r["provenance"] for r in records],
    )
    for key in (
        "blocks",
        "available_blocks",
        "skipped_ranges",
        "discarded_action_steps",
    ):
        if all(key in r["provenance"] for r in records):
            provenance[key] = sum(r["provenance"][key] for r in records)
    return dict(state=merge("state"), action=merge("action"), provenance=provenance)


def discover_dataset_roots(paths, prefixes=()):
    """遇到 meta/info.json 就停止向下扫描，不遍历大体积 data/videos。"""
    roots = set()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        found = False
        for directory, children, _ in os.walk(path):
            root = Path(directory)
            if (root / "meta/info.json").is_file():
                found = True
                children[:] = []
                if not prefixes or root.name.startswith(tuple(prefixes)):
                    roots.add(root)
            else:
                children[:] = sorted(name for name in children if name not in ("data", "videos", "qc", ".git"))
        if not found:
            raise ValueError(f"No LeRobot datasets under {path}")
    if not roots:
        raise ValueError("No datasets match the selected prefixes")
    return sorted(roots)


def save_statistics(result, output):
    """验证和合并成功后才写入，临时文件与目标同目录，失败保留旧结果。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix=".aggregate_stats_", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", nargs="+", type=Path, required=True, help="同一个来源组的目录")
    parser.add_argument("--prefix", nargs="+", default=[], help="仅选择目录名以这些前缀开头的子数据集")
    parser.add_argument("--bounds", choices=("q01_q99", "min_max"), default="q01_q99")
    parser.add_argument("--output", type=Path, required=True, help="来源组统计的输出位置")
    args = parser.parse_args(argv)
    roots = discover_dataset_roots(args.dataset_root, args.prefix)
    paths = [root / STATS_RELATIVE_PATH for root in roots]
    # 不允许覆盖任何已发现子集的原始统计，包含被 prefix 排除的子集。
    all_roots = discover_dataset_roots(args.dataset_root) if args.prefix else roots
    if args.output.resolve() in {(root / STATS_RELATIVE_PATH).resolve() for root in all_roots}:
        parser.error("Output must not overwrite per-dataset statistics")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Missing statistics; no output written:\n" + "\n".join(missing))
    result = aggregate_statistics(paths, bounds=args.bounds)
    save_statistics(result, args.output)
    provenance = result["provenance"]
    print(
        f"Aggregated {len(paths)} datasets into {args.output}; "
        f"blocks={provenance['blocks']}; partial={provenance['partial']}"
    )
    print("q01/q99 are source envelopes, not global quantiles.")


if __name__ == "__main__":
    main()
