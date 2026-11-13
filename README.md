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

## 解压并释放压缩包空间

下载命令成功后，自动调用 `download/extract_robocoin.sh`，递归查找已有的
`*.tar.gz`，每次只解压一个包到它所在目录，保留包内路径。
每个包成功解压并刷盘后，立即删除该压缩包，再处理下一个。

直接逐包解压，不预扫描压缩包或检查剩余空间。

坏包、同名文件冲突或写入失败时停止并保留压缩包；不会覆盖已有文件。
失败可能留下部分解压文件，需要先检查并处理冲突后重试。
解压中断后应只运行解压步骤，避免重新下载已解压并删除的压缩包：

```bash
bash download/extract_robocoin.sh
```

脚本依赖 GNU tar 和 GNU sync。
