# data-cleaning
数据清洗

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
