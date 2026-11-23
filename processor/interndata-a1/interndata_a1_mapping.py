"""InternData-A1 原始字段到 Unified55；不依赖 OpenWAM 或训练框架。"""

import re

import numpy as np
import pyarrow as pa


OUTPUT_COLUMNS = ("state_unified", "action_unified", "mask_state", "mask_action")
OUTPUT_FEATURE = {"dtype": "float32", "shape": [55], "names": None}
ROBOT_KEYS = {"Franka": "franka", "ARX Lift-2": "lift2", "Genie-1": "genie1",
              "AgileX Split Aloha": "split_aloha"}
FULL_OPEN = {"franka": 0.08, "lift2": 0.088, "genie1": 5.74, "split_aloha": 0.1}
POSE_NAMES = ["position.x", "position.y", "position.z", "quaternion.w",
              "quaternion.x", "quaternion.y", "quaternion.z"]


def choose_scale(embodiment, stats, column):
    """与 OpenWAM 一致：Franka 用桶内 state.max 的对数距离选择 0.08/1。"""
    primary = FULL_OPEN[embodiment]
    selection = {"value": primary, "state_column": column, "method": "declared_full_open",
                 "observed_max": None, "observed_mean": None}
    try:
        maximum = float(np.ravel(stats[column]["max"])[0])
        if not np.isfinite(maximum):
            raise ValueError("nonfinite max")
        selection["observed_max"] = maximum
        try:
            mean = float(np.ravel(stats[column]["mean"])[0])
            if np.isfinite(mean):
                selection["observed_mean"] = mean
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    except (KeyError, IndexError, TypeError, ValueError):
        selection["method"] = "fallback_missing_or_invalid_bucket_stats"
        return selection
    if embodiment == "franka" and maximum > 0:
        selection["method"] = "nearest_declared_stroke_in_log_space"
        if abs(np.log(maximum)) < abs(np.log(maximum / primary)):
            selection["value"] = 1.0
            mean = selection["observed_mean"]
            if mean is None or mean <= primary:
                selection["warning"] = "alt_scale_not_corroborated_by_mean"
    if maximum / selection["value"] > 1.25:
        selection["range_warning"] = "bucket_max_exceeds_1.25_full_open"
    return selection


def build_mapping(info, stats=None):
    features = info["features"]
    robot = info.get("robot_type")
    if robot not in ROBOT_KEYS:
        raise ValueError(f"Unknown robot_type {robot!r}; define its gripper scale explicitly")
    single = "states.ee_to_robot_pose" in features
    dual = any(f"states.{side}_ee_to_robot_pose" in features for side in ("left", "right"))
    if single == dual:
        raise ValueError("Expected exactly one single-arm or bimanual pose schema")
    mapping = {"robot_type": robot, "embodiment": ROBOT_KEYS[robot],
               "arm_layout": "single_arm" if single else "bimanual", "arms": []}
    for side, joint_start, pose_start, grip_slot in (("left", 0, 14, 28), ("right", 7, 21, 29)):
        if single and side == "right":
            continue
        prefix = "" if single else side + "_"
        arm = {"side": side, "joint_start": joint_start, "pose_start": pose_start,
               "grip_slot": grip_slot}
        for kind, stream in (("state", "states"), ("action", "actions")):
            joint = f"{stream}.{prefix}joint.position"
            pose = f"{stream}.{prefix}ee_to_robot_pose"
            grip = f"{stream}.{prefix}gripper.position"
            for column in (joint, pose, grip):
                if column not in features:
                    raise ValueError(f"Missing feature: {column}")
            names = features[joint].get("names")
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ValueError(f"{joint}: expected joint component names")
            pattern = re.compile(re.escape(prefix) + r"joint_(\d+)")
            matches = [pattern.fullmatch(name) for name in names]
            if not all(matches):
                raise ValueError(f"{joint}: unrecognized joint names {names}")
            numbers = [int(match[1]) for match in matches]
            if len(numbers) not in (6, 7) or sorted(numbers) != list(range(len(numbers))):
                raise ValueError(f"{joint}: expected contiguous joints 0..5 or 0..6")
            if features[joint].get("shape") != [len(numbers)]:
                raise ValueError(f"{joint}: shape does not match joint names")
            pose_names = features[pose].get("names")
            if (features[pose].get("shape") != [7] or not isinstance(pose_names, list)
                    or len(pose_names) != 7 or set(pose_names) != set(POSE_NAMES)):
                raise ValueError(f"{pose}: expected xyz + named wxyz quaternion")
            if features[grip].get("shape") != [1]:
                raise ValueError(f"{grip}: expected one gripper component")
            arm[kind] = {"joint_column": joint, "joint_width": len(numbers),
                         "joint_indexes": np.argsort(numbers).tolist(),
                         "joint_names": [names[i] for i in np.argsort(numbers)],
                         "pose_column": pose,
                         "pose_indexes": [pose_names.index(name) for name in POSE_NAMES],
                         "grip_column": grip}
        if arm["state"]["joint_names"] != arm["action"]["joint_names"]:
            raise ValueError(f"{side}: state/action joint semantics differ")
        arm["gripper_scale"] = choose_scale(mapping["embodiment"], stats or {}, arm["state"]["grip_column"])
        mapping["arms"].append(arm)
    return mapping


def source_columns(mapping):
    return [arm[kind][key] for arm in mapping["arms"] for kind in ("state", "action")
            for key in ("joint_column", "pose_column", "grip_column")]


def _values(table, column, width):
    """兼容源数据中的 scalar gripper、list 和 fixed_size_list，返回结构错误行。"""
    array = table[column].combine_chunks()
    if width == 1 and (pa.types.is_floating(array.type) or pa.types.is_integer(array.type)):
        return array.to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 1), []
    is_list = pa.types.is_list(array.type) or pa.types.is_large_list(array.type)
    fixed = pa.types.is_fixed_size_list(array.type)
    if (fixed or is_list) and array.null_count == 0:
        lengths_ok = array.type.list_size == width if fixed else bool(np.all(np.diff(array.offsets) == width))
        if lengths_ok:
            try:
                return array.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, width), []
            except (TypeError, ValueError, OverflowError):
                pass  # 逐行定位非数值等结构错误。
    values = np.full((len(table), width), np.nan)
    invalid = []
    for row, raw in enumerate(array.to_pylist()):
        try:
            if raw is None:
                raise ValueError("missing vector")
            vector = np.asarray(raw, dtype=np.float64)
            if width == 1 and vector.ndim == 0:
                vector = vector.reshape(1)
            if vector.shape != (width,):
                raise ValueError(f"expected {width} components, got {vector.shape}")
            values[row] = vector
        except (ValueError, TypeError, OverflowError) as error:
            invalid.append((row, str(error)))
    return values, invalid


def convert_episode(table, mapping):
    """分别映射同一原始行；时间边界由调用方处理。raw 用于精确核对下一帧。"""
    outputs = np.zeros((len(table), 4, 55), dtype=np.float32)
    errors, structural = [], set()
    raw = {"state": [], "action": []}

    def assign(values, slots, kind_index, field, source_indexes):
        with np.errstate(over="ignore", invalid="ignore"):
            converted = values.astype(np.float32)
        good = np.isfinite(converted)
        outputs[:, kind_index, slots] = np.where(good, converted, 0)
        outputs[:, kind_index + 2, slots] = good
        for row, component in zip(*np.nonzero(~good)):
            if int(row) in structural:
                continue
            errors.append({"row": int(row), "field": field,
                           "source_index": int(source_indexes[component]),
                           "output_field": OUTPUT_COLUMNS[kind_index],
                           "output_indices": [int(slots[component])],
                           "error_type": "nonfinite" if not np.isfinite(values[row, component]) else "float32_overflow",
                           "result": "components_masked"})

    for arm in mapping["arms"]:
        for kind_index, kind in enumerate(("state", "action")):
            spec = arm[kind]
            arrays = []
            for key, width in (("joint_column", spec["joint_width"]), ("pose_column", 7), ("grip_column", 1)):
                values, invalid = _values(table, spec[key], width)
                arrays.append(values)
                for row, message in invalid:
                    structural.add(row)
                    errors.append({"row": row, "field": spec[key], "error_type": "invalid_vector",
                                   "message": message, "result": "row_zeroed"})
            joints, pose, grip = arrays
            joints = joints[:, spec["joint_indexes"]]
            pose = pose[:, spec["pose_indexes"]]  # xyz + wxyz
            raw[kind].extend((joints, pose, grip))
            jslots = np.arange(arm["joint_start"], arm["joint_start"] + spec["joint_width"])
            pslots = np.arange(arm["pose_start"], arm["pose_start"] + 3)
            assign(joints, jslots, kind_index, spec["joint_column"], spec["joint_indexes"])
            assign(pose[:, :3], pslots, kind_index, spec["pose_column"], spec["pose_indexes"][:3])
            scaled_grip = grip / arm["gripper_scale"]["value"]
            assign(scaled_grip, np.array([arm["grip_slot"]]),
                   kind_index, spec["grip_column"], [0])
            # 忽略 float32 表示 0.1 等常量带来的微小越界；真实有限越界保留并记录。
            outside = np.isfinite(scaled_grip[:, 0]) & ((scaled_grip[:, 0] < -1e-6) | (scaled_grip[:, 0] > 1 + 1e-6))
            for row in np.flatnonzero(outside):
                errors.append({"row": int(row), "field": spec["grip_column"],
                               "output_field": OUTPUT_COLUMNS[kind_index], "output_indices": [arm["grip_slot"]],
                               "error_type": "gripper_out_of_range", "result": "finite_value_retained",
                               "scaled_value": float(scaled_grip[row, 0])})
            quat = pose[:, [4, 5, 6, 3]]  # xyzw，只重排，不归一化、不调整符号。
            with np.errstate(over="ignore", invalid="ignore"):
                good_quat = np.isfinite(quat).all(axis=1) & (np.abs(np.linalg.norm(quat, axis=1) - 1) <= 1e-4)
            qslots = np.arange(arm["pose_start"] + 3, arm["pose_start"] + 7)
            outputs[:, kind_index, qslots] = np.where(good_quat[:, None], quat, 0).astype(np.float32)
            outputs[:, kind_index + 2, qslots] = good_quat[:, None]
            for row in np.flatnonzero(~good_quat):
                if int(row) not in structural:
                    errors.append({"row": int(row), "field": spec["pose_column"],
                                   "output_field": OUTPUT_COLUMNS[kind_index], "output_indices": qslots.tolist(),
                                   "error_type": "invalid_quaternion", "result": "components_masked"})
    if structural:
        outputs[sorted(structural)] = 0
    for error in errors:
        if error["row"] in structural:
            error["result"] = "row_zeroed"
    return outputs, errors, {kind: np.concatenate(parts, axis=1) for kind, parts in raw.items()}
