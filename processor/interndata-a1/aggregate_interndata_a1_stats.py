# SPDX-License-Identifier: OpenMDW-1.1
"""按机器人型号汇总 InternData-A1 子集统计，只读取 JSON，不读取轨迹或视频。"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


# 复用已有的输入校验、矩合并、分位数包络和原子写入，避免维护两份规则。
_shared_path = Path(__file__).resolve().parents[1] / "robocoin/aggregate_causal_action_stats.py"
_spec = importlib.util.spec_from_file_location("_interndata_a1_stats_common", _shared_path)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

ROBOT_TYPES = {
    "Franka": "franka",
    "ARX Lift-2": "lift2",
    "Genie-1": "genie1",
    "AgileX Split Aloha": "split_aloha",
}
TEMPLATE_ID = "unified55-xyzw-absolute-gripper-hand-v4"
GRIPPER_INDICES = (28, 29)
ROTATION_GROUPS = [[17, 18, 19, 20], [24, 25, 26, 27]]


def discover_robot_groups(dataset_roots, robots=None):
    """按 info 中的 robot_type 分组；兼容任务层级和包内重复目录。"""
    groups = {robot: [] for robot in (ROBOT_TYPES.values() if robots is None else robots)}
    for root in _common.discover_dataset_roots(dataset_roots):
        info = json.loads((root / "meta/info.json").read_text())
        robot_type = info.get("robot_type")
        if robot_type not in ROBOT_TYPES:
            raise ValueError(f"{root}: unknown robot_type {robot_type!r}")
        robot = ROBOT_TYPES[robot_type]
        if robot in groups:
            groups[robot].append(root)
    return groups


def aggregate_statistics(paths, *, bounds="q01_q99"):
    """保留实测统计，仅将有效夹爪的训练边界设为 [0, 1]。"""
    result = _common.aggregate_statistics(paths, bounds=bounds)
    provenance = result["provenance"]
    if provenance["template_id"] != TEMPLATE_ID:
        raise ValueError(f"Expected template_id={TEMPLATE_ID}, got {provenance['template_id']!r}")
    if provenance["rotation_groups"] != ROTATION_GROUPS:
        raise ValueError("Expected Unified55 xyzw quaternion groups")
    for kind in ("state", "action"):
        stats = result[kind]
        if len(stats["valid"]) != 55:
            raise ValueError(f"{kind}: expected 55 dimensions")
        for index in GRIPPER_INDICES:
            if stats["valid"][index]:
                stats["low"][index], stats["high"][index] = 0.0, 1.0
    provenance.update(
        data_source="interndata-a1",
        gripper_indices=list(GRIPPER_INDICES),
        gripper_bounds=[0.0, 1.0],
        gripper_bounds_policy="fixed_for_valid_channels",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", nargs="+", type=Path, required=True,
                        help="InternData-A1 根目录或任务目录，可填写多个")
    parser.add_argument("--output-dir", type=Path, required=True, help="按型号输出汇总统计的目录")
    parser.add_argument("--robots", nargs="+", choices=list(ROBOT_TYPES.values()),
                        help="仅处理指定型号，默认处理全部四种")
    parser.add_argument("--bounds", choices=("q01_q99", "min_max"), default="q01_q99",
                        help="关节和位置的边界；夹爪始终为 [0,1]，有效四元数为 [-1,1]")
    parser.add_argument("--dry-run", action="store_true", help="显示输入并验证汇总，不写文件")
    args = parser.parse_args(argv)
    try:
        groups = discover_robot_groups(args.dataset_root, args.robots)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    success, skipped, failed = 0, 0, []
    for robot, roots in groups.items():
        output = args.output_dir / f"stats_interndata_a1_{robot}.json"
        print(f"\n{robot}: {len(roots)} 个子数据集 -> {output}", flush=True)
        if not roots:
            print("跳过：没有匹配的子数据集。", flush=True)
            skipped += 1
            continue
        paths = [root / _common.STATS_RELATIVE_PATH for root in roots]
        if args.dry_run:
            for path in paths:
                print(f"  {'存在' if path.is_file() else '缺少'}: {path}", flush=True)
        try:
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing per-dataset statistics:\n" + "\n".join(missing))
            result = aggregate_statistics(paths, bounds=args.bounds)
            result["provenance"].update(robot_type=next(
                name for name, key in ROBOT_TYPES.items() if key == robot), embodiment=robot)
            if not args.dry_run:
                _common.save_statistics(result, output)
            provenance = result["provenance"]
            print(f"{'验证通过' if args.dry_run else '汇总成功'}: blocks={provenance['blocks']}; "
                  f"partial={provenance['partial']}", flush=True)
            success += 1
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            print(f"汇总失败 {robot}: {error}", file=sys.stderr, flush=True)
            failed.append(robot)

    print(f"\n{'预览结束' if args.dry_run else '结束'}：成功 {success}，未匹配 {skipped}，失败 {len(failed)}。",
          flush=True)
    print("q01/q99 是子集分位数的包络，不是全量总体分位数；有效夹爪 low/high 固定为 [0,1]。",
          flush=True)
    if failed:
        print("失败型号：" + ", ".join(failed), file=sys.stderr, flush=True)
    return 1 if failed or success == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
