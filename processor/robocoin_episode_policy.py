"""整集动作判定：沿用同事的原值比较、mask 和运动条件，只修改派生 action。"""

import numpy as np

RAW = ("observation.state", "action")


def component_names(info, column):
    names = info["features"][column]["names"]
    return names["motors"] if isinstance(names, dict) else names


def apply_episode_policy(table, info, mappings, outputs, invalid_episode=False):
    """outputs 为 (帧数, 4, 55)，列序是 state/action/mask_state/mask_action。"""
    result = {"mode": "source_native", "rows": len(table),
              "comparison": "exact mapped source values and shared named raw components; full episode"}

    def keep(reason):
        return dict(result, reason=reason)

    # 文件中的行顺序不代表时间顺序；帧号不完整时保留初步转换，不猜下一帧。
    if invalid_episode:
        return keep("missing_or_invalid_episode_index")
    if "frame_index" not in table.column_names:
        return keep("missing_or_invalid_frame_index")
    frames = table["frame_index"].to_pylist()
    if any(type(v) is not int or v < 0 for v in frames):
        return keep("missing_or_invalid_frame_index")
    order = np.argsort(frames, kind="stable")
    ordered_frames = np.asarray(frames)[order]
    if np.any(np.diff(ordered_frames) == 0):
        return keep("duplicate_frame_index")
    if np.any(np.diff(ordered_frames) != 1):
        return keep("noncontiguous_frame_index")
    if len(table) < 3:
        return keep("too_short")

    # 同一输出槽位必须来自同一组关节名称；原始数组顺序不同允许按名称重排。
    names = {column: component_names(info, column) for column in RAW}
    fields = [{start: indexes for indexes, start in mappings[column]} for column in RAW]
    if fields[0].keys() != fields[1].keys():
        return keep("different_mapped_fields")
    pairs = []
    for start, si in fields[0].items():
        ai = fields[1][start]
        if [names[RAW[0]][i] for i in si] != [names[RAW[1]][i] for i in ai]:
            return keep(f"different_field_semantics:{start}")
        pairs.append((RAW[0], si, RAW[1], ai, f"joint_slot_{start}"))
    # EEF 沿用已确认的固定左右布局，比较原欧拉角，不用转换后的四元数代替。
    pairs.append(("eef_sim_pose_state", list(range(12)), "eef_sim_pose_action", list(range(12)), "eef"))
    grips = [f"gripper_open_scale_{side}" for side in ("state", "action")]
    present = [c in info["features"] or c in table.column_names for c in grips]
    if present[0] != present[1]:
        return keep("different_mapped_fields:gripper")
    if all(present):
        pairs.append((grips[0], [0, 1], grips[1], [0, 1], "gripper"))

    # 必须全部有限且严格相等；任何字段不同就保留整集，不单独移动某个语义。
    try:
        arrays = {c: np.asarray(table[c].to_pylist(), dtype=np.float64)
                  for c in {p[0] for p in pairs} | {p[2] for p in pairs}}
        for sc, si, ac, ai, label in pairs:
            x, y = arrays[sc][:, si], arrays[ac][:, ai]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                return keep("nonfinite:" + label)
            if not np.array_equal(x, y):
                return keep("source_values_differ:" + label)
        if any(len(set(ns)) != len(ns) for ns in names.values()):
            return keep("ambiguous_shared_component_names")
        shared = [n for n in names[RAW[0]] if n in names[RAW[1]]]
        # 包含超出 55D 容量的手指；未命名尾部不补名，也不猜测语义。
        si, ai = [[names[c].index(n) for n in shared] for c in RAW]
        x, y = arrays[RAW[0]][:, si], arrays[RAW[1]][:, ai]
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.array_equal(x, y):
            return keep("shared_raw_components_differ")
    except (KeyError, ValueError, TypeError, IndexError, OverflowError):
        return keep("invalid_source_vectors")

    state, action, ms, ma = (outputs[:, i] for i in range(4))
    core = np.r_[0:14, 28:30]
    used = (ms[:, core] > 0).any(0)
    if not used.any() or not (ms[:, core][:, used] > 0).all() or not np.array_equal(ms, ma):
        return keep("invalid_or_different_masks")
    if not np.any(np.abs(np.diff(state[order][:, core][:, used].astype(np.float64), axis=0)) > 1e-6):
        return keep("static_ambiguous")

    # 整个 55D action 和 mask 一起移动；按原行位置存回，原始列完全不动。
    action[order[:-1]] = state[order[1:]]
    ma[order[:-1]] = ms[order[1:]]
    action[order[-1]] = 0
    ma[order[-1]] = 0
    return dict(result, mode="derived_next_state", reason="dynamic_full_episode_exact_copy",
                source_step=1, terminal_frame_index=int(ordered_frames[-1]),
                terminal="all derived action values and masks zero")
