# 按数据来源合并统计量

`processor/aggregate_causal_action_stats.py` 只读取各子数据集的
`meta/causal_action_stats.json`，不重新读 Parquet 或视频，不修改子数据集统计。
依赖只有 NumPy，不依赖训练仓库、Torch 或 LeRobot 库。

## 使用

在 data-cleaning 目录执行，把 RoboCOIN 下所有子数据集合成一份来源统计：

```bash
python processor/aggregate_causal_action_stats.py \
  --dataset-root /efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/RoboCOIN \
  --output /efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/robocoin_causal_action_stats.json
```

一个调用对应一个来源组。脚本不根据机器人名称自动分组；不同来源分别运行、指定不同输出。
`--dataset-root` 可填写多个目录，它们的子集会合到同一份结果中。
目录重复或包含关系会在发现阶段去重。

如果希望按机器人前缀单独合并，例如只合并 AIRBOT_MMK2：

```bash
python processor/aggregate_causal_action_stats.py \
  --dataset-root /efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/RoboCOIN \
  --prefix AIRBOT_MMK2 \
  --output /efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/mmk2_causal_action_stats.json
```

前缀区分大小写，可填写多个，匹配的是子数据集目录名。不要把机器人型号与数据来源自动等同。

## 合并规则

沿用 `cosmos-framework-ascend/tools/aggregate_causal_action_stats.py` 的规则，适配当前
VeOmni 输出的 `dataset_root`、`columns` 元数据，也支持原 Cosmos 的输入格式。

| 字段 | 怎么合并 |
| --- | --- |
| count / valid_counts | 分别累加总行数、每个维度的有效样本数 |
| mean / std | 按各维有效样本数合并；标准差包含子集之间均值不同的影响，不直接平均标准差 |
| min / max | 在该维有效的子集中取最小值、最大值 |
| q01 / q99 | 取最小的子集 q01、最大的子集 q99，形成覆盖范围；不是总体分位数 |
| low / high | 默认使用上述 q01/q99 范围；传 `--bounds min_max` 改用 min/max |
| 四元数 low / high | 有效四元数四维固定为 [-1, 1]；实测统计字段保持原结果 |
| 全部无效的维度 | 数值填 0，valid=false，不让无效补零参与统计 |

均值和标准差可以从现有统计量合并；只有各子集的分位数，无法恢复全量数据的真实 q01/q99。
输出用 `method=source_quantile_envelope` 和 `quantile_semantics` 明确标注这个区别。

只允许合并直接计算得到的 exact/reservoir 统计。模板、窗口大小、动作读取列和偏移等口径
必须一致；mask 可以不同。缺失统计文件、重复数据来源或不兼容的口径会报错，不静默漏掉子集。
任意子集为调试结果 `partial=true`，合并结果也为 true，并在结束时打印。
输出保留输入路径和各子集的 provenance，便于追溯。

脚本只扫描目录和小 JSON，不需要轨迹读取的多进程并行。所有检查及合并成功后才原子替换输出，
禁止输出覆盖本次输入目录中的子数据集统计。合并结果仍包含 state/action 的 low/high/valid，
可用于现有训练的统计配置；脚本不自动修改训练配置。
