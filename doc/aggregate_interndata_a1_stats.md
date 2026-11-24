# InternData-A1 按机器人型号汇总统计量

先用训练仓库的统计工具为每个任务计算 55D 的 `meta/causal_action_stats.json`，
再运行本脚本。数据下载自带的 `meta/stats.json` 和根目录的 `meta/stats_*.json`
不属于这套 55D 训练统计，不能代替子集的 `causal_action_stats.json`。

在 `data-cleaning` 目录执行，Python 环境只需 NumPy：

```bash
bash processor/interndata-a1/aggregate_interndata_a1_stats.sh \
  /mnt/sfs_turbo/public/datasets/openwam-interndata-a1 \
  /path/to/training_statistics
```

也可在 Bash 顶部手动填写 `dataset_root`、`output_dir`，然后不带参数运行。
用 `PYTHON_BIN=/path/to/python` 指定解释器。

只合并部分型号或预览：

```bash
bash processor/interndata-a1/aggregate_interndata_a1_stats.sh \
  /path/to/InternData-A1 /path/to/training_statistics \
  --robots franka lift2 --dry-run
```

Python 入口支持多个根目录，也可以直接传入若干任务目录：

```bash
python -B processor/interndata-a1/aggregate_interndata_a1_stats.py \
  --dataset-root /path/to/InternData-A1 \
  --output-dir /path/to/training_statistics
```

脚本递归发现 `meta/info.json`，找到任务后停止向下扫描，避免遍历轨迹和视频。
根据元数据的 `robot_type` 分组，跨 `articulation_tasks`、`basic_tasks`、
`long_horizon_tasks` 合并同一种机器人，不依赖任务目录的固定深度或名字。

| robot_type | 输出文件 |
| --- | --- |
| Franka | `stats_interndata_a1_franka.json` |
| ARX Lift-2 | `stats_interndata_a1_lift2.json` |
| Genie-1 | `stats_interndata_a1_genie1.json` |
| AgileX Split Aloha | `stats_interndata_a1_split_aloha.json` |

复用 [RoboCOIN 汇总规则](aggregate_causal_action_stats.md)：按有效样本数合并矩，
`q01=min(子集 q01)`、`q99=max(子集 q99)`，不是全量总体分位数。
默认 `low/high` 取该分位数包络；`--bounds min_max` 改用实测极值。
在此基础上，对 `state` 和 `action` 均应用：

- 有效四元数的 `low/high` 固定为 `[-1,1]`。
- 有效夹爪维度 `28/29` 的 `low/high` 固定为 `[0,1]`，与 `--bounds` 无关。
- 夹爪的 `min/max/mean/std/q01/q99/valid_counts` 保留实测汇总值，便于追查越界。
- 缺失或全部无效的夹爪保持 `valid=false`、`low=high=0`；Franka 单臂右侧不会被启用。

只接受 `unified55-xyzw-absolute-gripper-hand-v4` 模板的 55D 直接计算结果。
模板、窗口、读取列及偏移等统计口径必须兼容，不能把已经汇总的统计再次作为输入。
输出 provenance 记录来源、型号、夹爪维度和固定边界策略。
固定边界只改变训练归一化参数，不裁剪原数据，也不消除负值或大于 1 的源数据问题。

缺少子集统计或输入不兼容时，该型号失败，不更新它已有的输出；继续处理其他型号，
最终返回非零退出码。没有下载的型号会跳过；全部未匹配也返回非零退出码。
未知 `robot_type` 会在分组阶段报错，避免静默漏掉子集。
`--dry-run` 同样检查缺失文件和统计兼容性，但不创建输出目录或写文件。
任意子集 `partial=true` 会传递到汇总结果并打印；不能将其当成完整来源统计。
每份成功输出均原子替换，不改写子集统计、Parquet、视频或训练配置。

验证：

```bash
python -B -m unittest discover -s processor/interndata-a1 -p 'test_aggregate_interndata_a1_stats.py' -v
```
