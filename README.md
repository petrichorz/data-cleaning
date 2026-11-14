# data-cleaning
数据清洗

## 转换规则文档

[OpenWAM RoboCOIN → 55D 语义映射规则](doc/robocoin_55d_mapping.md)：记录双臂关节、EEF、夹爪、灵巧手映射，腰/头/底盘不接入的规则，整集动作判定与下一帧目标规则，以及后续质检项。

[后处理质检讨论记录](doc/robocoin_postprocess_qc.md)：记录已确认的 action/state 整集检查、运动门槛与下一帧目标规则，现已接入转换流程。

## RoboCOIN 转 55D

脚本：[processor/convert_robocoin_55d.py](processor/convert_robocoin_55d.py)。
按已确定的字段映射，新增 `state_unified`、`action_unified`、`mask_state`、`mask_action`，
均为长度 55 的 float32 数组。先按原始行映射，再按完整 episode 沿用同事的动作策略：
映射来源和原始同名分量（含未接入的手指）全程严格相等、语义和 mask 一致、至少 3 帧，
且臂关节或夹爪有超过 `1e-6` 的相邻帧变化，才令派生 `action[t]=state[t+1]`，mask 同步移动，
末帧派生 action/mask 清零。任一条件不满足就保留初步转换结果，不局部移动夹爪或 EEF。
不做单位换算、训练归一化或四元数符号调整，原始列保持原样。

在仓库根目录安装依赖并运行：

```bash
python -m pip install -r processor/requirements.txt

python processor/convert_robocoin_55d.py \
  /path/to/RoboCOIN
```

可以只处理指定子数据集，名称需要完整匹配：

```bash
python processor/convert_robocoin_55d.py \
  /path/to/RoboCOIN \
  --datasets AIRBOT_MMK2_mobile_car leju_robot_hotel_services_ah
```

输入也可以直接指向一个子数据集。脚本直接给原来的 `data/*.parquet` 增加四列，
不再需要 `--output`，也不另外保存一份数据集。
按 `episode_index` 跨文件汇集完整 episode，按 `frame_index` 判断先后，再按原文件行位置写回。
帧号缺失、重复或不连续时不移位，原因记录在 QC 中。`--batch-size` 默认 4096，只控制写回时
读取原文件和写 Parquet 的批次行数，不控制 episode 判定范围。默认顺序处理子数据集。
临时新增列使用磁盘暂存（每帧约 880 字节），不把所有 55D 数组常驻内存；
内存主要用于行位置索引、一个原始 row group 和一个完整 episode。

输出内容：

- `data/`：保留原始列、行数和行顺序，追加四个 55D 字段。先写同目录临时文件，完整写完并关闭后替换原文件，最终路径和文件数量不变。
- `meta/info.json`：所有文件处理成功后，在原文件的 `features` 中登记新增四列；原有字段定义和其他元数据保留，不生成新训练统计。
- `videos/`：保持原样。
- `qc/conversion_errors.jsonl`：所有转换错误，含文件、行号、episode、帧号、字段/索引和处理结果。
- `qc/conversion_summary.json`：转换行数、错误行数、局部屏蔽行数、整行置零行数、错误数、关节映射和未接入手部的原因；另有 `episodes`、`shifted_episodes`、`retained_episodes` 及判定原因计数。
- `qc/episode_policy.jsonl`：每个 episode 的来源文件、处理模式、原因；移位时记录末帧帧号。报告计数表示已完成判断，是否全部写回以转换 `status` 为准。

NaN/Inf 或 null 数值元素只将对应输出维度的值和 mask 置零；state/action、左右侧独立。
某侧任一欧拉角无效或姿态转换失败时，仅屏蔽该侧四元数四维，位置仍照常处理。
float32 溢出也只屏蔽对应维度；未映射或未命名分量的 NaN/Inf 只记录，不影响有效槽位。
整个必需向量缺失、长度不符等结构错误仍将该行四个新增向量全部置零，记录后继续。
原始列中的错误数值仍保留，不删除该行。正常没有夹爪、手指超出容量等情况按映射规则补零。

双臂定义不是每侧连续 6/7 个关节时，记录子数据集错误，继续下一个子数据集。
完整 episode 准备阶段读取失败时，记录错误，该子数据集停止写回，避免使用不完整轨迹做判断。
写回某个文件失败时保留该原文件，继续其他文件；子数据集标为 `incomplete`，
已成功的文件保留新增列，本轮不更新 `meta/info.json`。
整个子数据集不是一次性替换：中断时可能只有部分文件已增加四列，应以 `qc/` 汇总的完成状态为准。

中断或失败后可直接重跑同一命令。每次都从原始列重新计算所有文件，
已有的四个派生字段会重新生成，不重复追加；上次中断留下的临时文件会重写。
三个 QC 报告每次重新生成，反映本轮结果。重复执行不会再次移动上轮的派生 action。
正常退出会清理 `qc/.conversion-*` 临时新增列；强制终止可能留下该临时目录，确认没有任务使用后可删除，不影响重跑。
同一个子数据集不要同时启动多个转换进程。

有转换错误记录的子数据集标为 `complete_with_errors`，不会因此中断命令；
任一子数据集为 `incomplete` 时命令退出码为 1，否则为 0。即使退出码为 0，也应查看 `qc/` 汇总。

运行测试：

```bash
python -B -m unittest discover -s processor -p 'test_*.py' -v
```

### 按子数据集前缀并行转换

启动脚本：[processor/convert_robocoin_55d.sh](processor/convert_robocoin_55d.sh)。
默认根目录为 `/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/RoboCOIN/`，默认 4 并发。
编辑脚本顶部的 `prefixes` 数组，保留本批次需要的前缀，将其他行注释掉；目前列出的 15 个前缀全部启用。
按直接子数据集的目录名前缀匹配，区分大小写；多个前缀命中同一目录也只处理一次。

```bash
bash processor/convert_robocoin_55d.sh

# 可选：改变并发数、Python 解释器或根目录。
CONVERT_JOBS=8 PYTHON_BIN=/path/to/python bash processor/convert_robocoin_55d.sh /path/to/RoboCOIN
```

每个进程转换一个子数据集，内部按完整 episode 处理并写回原文件；完成一个就接着取下一个。
`BATCH_SIZE` 可调整写回的 I/O 批次行数，默认 4096，不拆分 episode。原地追加及重跑行为与 Python 脚本一致。
各子数据集的详细运行日志写到自己的 `qc/conversion.log`，每次重跑覆盖。

总体进度条按子数据集数量计数，例如 `[##########--------------------] 33% 完成 5/15 失败 0`。
完成数包含成功和失败；终端原地刷新，重定向或 `nohup` 日志中逐行输出。
失败数统计进程返回非零的子数据集，单个失败后继续其他任务，最终返回非零退出码。
结束后自动打印有问题的子数据集、错误数、局部屏蔽行数、整行置零行数及错误类型/字段计数，并写入
**根目录的 `qc/conversion_batch_summary.json`**，不用逐个打开子数据集报告。
汇总包含各问题数据集的完整摘要、前三条错误示例和详细日志路径；仅统计本批次选中的目录，每次重跑覆盖。
`complete_with_errors` 仍算已完成，但会进入问题清单；单个转换失败、摘要缺失或损坏也会列入，且先汇总再返回失败码。
没有手部字段或手指超过 6 维是正常映射规则，不单独列为错误。
不要同时对同一个子数据集启动多个转换任务。依赖 Bash、GNU find/sort/xargs，以及 Python 转换依赖。

### 只读并行检查 NaN/Inf

一键启动：[processor/scan_robocoin_nonfinite.sh](processor/scan_robocoin_nonfinite.sh)。
默认扫描此前的 RoboCOIN 根目录，4 个进程并行；顶部 `prefixes` 可选择本批次前缀，默认 15 个全部启用。

```bash
bash processor/scan_robocoin_nonfinite.sh

# 可选：指定并发数、解释器、根目录，并保存异常明细和最终总计。
SCAN_JOBS=8 PYTHON_BIN=/path/to/python bash processor/scan_robocoin_nonfinite.sh /path/to/RoboCOIN > nonfinite.jsonl
```

`BATCH_SIZE` 默认 4096；终端显示扫描进度。Bash 保留 Python 扫描器的退出码，不修改数据。
也可以直接运行 Python：

```bash
python processor/scan_robocoin_nonfinite.py --workers 4

# 可指定根目录和子数据集前缀，将结果保存为 JSONL。
python processor/scan_robocoin_nonfinite.py /path/to/RoboCOIN \
  --prefixes AIRBOT_MMK2_ G1edu-u3_ --workers 8 > nonfinite.jsonl
```

默认目录同并行转换 Bash。也支持传入单个子数据集目录或单个 Parquet 文件。
默认检查原始 `observation.state`、`action`、`eef_sim_pose_state/action`、
`gripper_open_scale_state/action` 的全部分量，包括未映射的手指和未命名分量。
仅扫描文件中实际存在的指定列；可用 `--columns` 指定其他数值列。
这是 NaN/Inf 扫描，不将 null 缺失值当作 NaN，也不检查有限值是否合理。

每个异常帧打印一行 JSON：文件绝对路径、Parquet 内行号、`episode_index`、`frame_index`、
出错字段、字段内分量下标和 NaN/+Inf/-Inf；下标均从 0 开始，缺少帧编号时打印 null，不推测编号。
多个文件并行扫描，输出顺序不固定；进度写到 stderr，最后一行 stdout 为总计。
退出码：0 无 NaN/Inf，1 发现 NaN/Inf，2 扫描不完整或参数错误。脚本不修改任何数据或 qc 文件。

## RoboCOIN 下载

在已挂载目标目录的机器上，使用现有 obsutil 配置运行：

```bash
bash download/download_robocoin.sh
```

递归下载 `obs://egodata/openwam/robocoin/` 到
`/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/`，保留子目录层级。
`-u` 跳过未变化的文件，补下载缺失、大小不同或比远端旧的文件；不清理目标目录。
`-j=32` 同时下载最多 32 个文件，`-p=4` 为每个大文件使用最多 4 个并行分段。
中断后重新运行，obsutil 会尝试使用已有的有效分段断点续传；沿用之前的用户、配置和断点目录。

参数说明：[obsutil 下载对象](https://support.huaweicloud.com/intl/zh-cn/utiltg-obs/obs_11_0018.html)。

## 并行解压

下载命令成功后，自动调用 `download/extract_robocoin.sh`，递归查找已有的
`*.tar.gz`，仅处理指定文件名前缀，默认同时解压 4 个包，各自解压到压缩包所在目录，保留包内路径。
默认仅解压，保留所有压缩包；可通过 `DELETE_ARCHIVES=1` 开启自动删除。

运行前，编辑解压脚本顶部的 `prefixes` 数组，取消注释本批次需要的前缀，例如：

```bash
prefixes=(
    "AI2_Alphabot_2_"
    "AIRBOT_MMK2_"
)
```

脚本列出了清单中的 15 个前缀，默认全部注释，未选择时退出。
按文件名开头匹配，区分大小写（`AIRBOT_MMK2_` 和 `Airbot_MMK2_` 是不同批次）；
递归匹配子目录中的压缩包，开启删除时也仅删除本批次匹配的包。
下载脚本仍然下载全部文件，此处筛选只作用于后续解压和可选删除。

不预扫描包内文件或检查剩余空间。

显示按压缩包数量统计的总体进度，每个包成功或失败后更新一次，例如：

```text
[##########--------------------]  33% 完成 10/30 失败 0
```

“完成”包含成功和失败的任务，百分比不是字节进度；达到 100% 后仍需看失败数和退出状态。
终端中原地刷新，重定向到日志时逐行记录。此进度只统计本批次匹配的包。

使用 `--skip-old-files` 跳过已有同名文件，公共目录中的重复文件不再导致报错，
各包独有的文件继续解压。已有文件不会被覆盖，也不会校验其内容或完整性。
普通解压失败不影响其他包继续处理；本轮结束后，只要有任务失败，脚本就返回非零状态。
失败可能留下部分解压文件，重试前需先处理不完整文件，避免将其跳过。

解压脚本可以独立运行，不会调用下载。处理好中断残留后，只运行解压步骤，
可避免重新下载已解压并删除的压缩包。在仓库根目录执行：

```bash
bash download/extract_robocoin.sh
```

使用 `EXTRACT_JOBS` 调整并发数，例如同时解压 8 个包：

```bash
EXTRACT_JOBS=8 bash download/extract_robocoin.sh
```

也可指定其他目录：

```bash
EXTRACT_JOBS=8 bash download/extract_robocoin.sh /path/to/RoboCOIN/
```

开启自动删除，例如以 8 并发解压：

```bash
DELETE_ARCHIVES=1 EXTRACT_JOBS=8 bash download/extract_robocoin.sh
```

`DELETE_ARCHIVES` 仅接受 `0` 或 `1`，默认 `0`。开启后，等待本轮所有包解压成功，
执行 `sync -f` 刷盘，再删除本轮开始时列出的压缩包；任意解压任务或刷盘失败，
均不会进入删除阶段。删除阶段若遇到错误会停止，已经删除的包不会恢复。
被跳过的已有文件不会做完整性校验，开启删除前应确认它们完整。
此开关也可传给下载脚本，下载完成后的解压步骤会继承它。

脚本依赖 GNU tar 和 GNU xargs；开启自动删除时还需要 GNU sync。
