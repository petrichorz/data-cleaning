#!/usr/bin/env python3
"""并行原地解压 InternData-A1；全部成功并刷盘后删除本次处理的 tar.gz。

用法：
  python3 extract_interndata_a1.py --jobs 8
  python3 extract_interndata_a1.py /path/to/dataset --jobs 4 --dry-run
  python3 extract_interndata_a1.py --keep-archives
  python3 extract_interndata_a1.py --verify-existing --jobs 8

去掉包内的 InternData-A1/ 前缀，保留任务类别/机器人/任务名的目录结构。
临时目录位于数据集根目录，成功解压后通过同文件系统 rename 合并文件。
已有文件会被完整解压的文件替换，因此中断或失败后可以重新运行。
若之前仅在刷盘或删除阶段失败，可用 --verify-existing 校验已有文件后继续清理。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time


DEFAULT_ROOT = "/mnt/sfs_turbo/public/datasets/openwam-interndata-a1"
TEMP_PREFIX = ".interndata-a1-extract-"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("并行数必须大于 0")
    return number


def find_archives(root):
    def onerror(error):
        raise error

    archives = []
    for directory, dirs, files in os.walk(root, onerror=onerror):
        dirs[:] = [
            name for name in dirs
            if name not in (".cache", ".git") and not name.startswith(TEMP_PREFIX)
        ]
        for name in files:
            path = Path(directory) / name
            if name.endswith(".tar.gz") and path.is_file() and not path.is_symlink():
                archives.append(path)
    return sorted(archives)


def fingerprint(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def merge_tree(source, destination):
    """调用方持有合并锁；rename 避免复制视频和并发写坏公共 meta 文件。"""
    if destination.is_symlink():
        raise RuntimeError(f"目标路径是符号链接：{destination}")
    if source.is_dir() and destination.exists():
        if not destination.is_dir():
            raise RuntimeError(f"目标目录被文件占用：{destination}")
        for child in sorted(source.iterdir()):
            merge_tree(child, destination / child.name)
    else:
        os.replace(source, destination)


def extract_one(archive, root, decompressor, merge_lock, original):
    # 先解压到独立目录，tar 失败时不会发布半截文件。
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX, dir=root) as directory:
        staging = Path(directory)
        result = subprocess.run(
            [
                "tar", "--extract", "--file", str(archive),
                "--directory", str(staging), "--use-compress-program", decompressor,
                "--no-same-owner", "--no-same-permissions",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"tar 退出码 {result.returncode}")

        payload = staging / "InternData-A1"
        if not payload.is_dir() or set(staging.iterdir()) != {payload}:
            raise RuntimeError("压缩包顶层应只有 InternData-A1/ 目录")
        expected_task = archive.relative_to(root).with_suffix("").with_suffix("")
        if not (payload / expected_task).is_dir():
            raise RuntimeError(f"压缩包缺少任务目录：InternData-A1/{expected_task}")
        for path in staging.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise RuntimeError(f"压缩包包含非普通文件/目录：{path.relative_to(staging)}")
        files = [root / path.relative_to(payload) for path in payload.rglob("*") if path.is_file()]
        if fingerprint(archive) != original:
            raise RuntimeError("解压期间压缩包发生变化，请等下载完成后重试")

        with merge_lock:
            merge_tree(payload, root)
    return files


def verify_existing(archive, root, original):
    """逐字节比对已有文件，同时读到 gzip 流末尾以校验 CRC 和完整性。"""
    files = []
    with gzip.open(archive, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as contents:
            for member in contents:
                relative = PurePosixPath(member.name)
                parts = relative.parts
                if relative.is_absolute() or ".." in parts or not parts or parts[0] != "InternData-A1":
                    raise RuntimeError(f"压缩包路径不符合预期：{member.name}")
                target = root.joinpath(*parts[1:])
                if any(path.is_symlink() for path in (target, *target.parents) if path.is_relative_to(root)):
                    raise RuntimeError(f"目标路径含符号链接：{target}")
                if member.isdir():
                    if not target.is_dir():
                        raise RuntimeError(f"缺少目录：{target}")
                    continue
                if not member.isfile():
                    raise RuntimeError(f"压缩包包含非普通文件：{member.name}")
                if not target.is_file() or target.stat().st_size != member.size:
                    raise RuntimeError(f"文件缺失或大小不一致：{target}")
                with contents.extractfile(member) as expected, target.open("rb") as actual:
                    while True:
                        block = expected.read(1024 * 1024)
                        if not block:
                            if actual.read(1):
                                raise RuntimeError(f"文件存在多余内容：{target}")
                            break
                        if actual.read(len(block)) != block:
                            raise RuntimeError(f"文件内容不一致：{target}")
                files.append(target)
        while compressed.read(1024 * 1024):
            pass
    expected_task = archive.relative_to(root).with_suffix("").with_suffix("")
    if not files or not (root / expected_task).is_dir():
        raise RuntimeError(f"没有找到已解压任务文件：{expected_task}")
    if fingerprint(archive) != original:
        raise RuntimeError("校验期间压缩包发生变化，请等下载完成后重试")
    return files


def flush_extracted_files(root, files, jobs):
    """对本次文件及目录 fsync，避免 sync -f 涉及整个共享文件系统。"""
    def flush(path, directory=False):
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if directory:
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    files = sorted(set(files))
    print(f"正在逐文件刷盘：{len(files)} 个文件……", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # 消费 map 结果，让任一 fsync 失败都在删除归档之前抛出。
        list(pool.map(flush, files))
    directories = {root}
    for path in files:
        directories.update(parent for parent in path.parents if parent.is_relative_to(root))
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        flush(directory, directory=True)
    print("本次文件及目录刷盘成功。", flush=True)


def run(args, root, archives):
    decompressor = "pigz -d -p 1" if shutil.which("pigz") else "gzip -d"
    required = () if args.verify_existing else ("tar", "pigz" if shutil.which("pigz") else "gzip")
    for program in required:
        if not shutil.which(program):
            raise RuntimeError(f"缺少命令：{program}")

    originals = {path: fingerprint(path) for path in archives}
    failed = []
    files = set()
    merge_lock = threading.Lock()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(archives))) as pool:
        if args.verify_existing:
            futures = {pool.submit(verify_existing, path, root, originals[path]): path for path in archives}
        else:
            futures = {
                pool.submit(extract_one, path, root, decompressor, merge_lock, originals[path]): path
                for path in archives
            }
        for completed, future in enumerate(as_completed(futures), 1):
            archive = futures[future]
            try:
                files.update(future.result())
                operation = "内容校验成功" if args.verify_existing else "解压成功"
                print(f"[{completed}/{len(archives)}] {operation}：{archive.relative_to(root)}", flush=True)
            except Exception as error:
                failed.append(archive)
                print(f"[{completed}/{len(archives)}] 失败：{archive.relative_to(root)}\n{error}",
                      file=sys.stderr, flush=True)

    if failed:
        print(f"有 {len(failed)} 个包失败，保留全部压缩包；修复后可直接重跑。", file=sys.stderr)
        return 1
    if args.keep_archives:
        print("全部处理成功，按 --keep-archives 保留压缩包。", flush=True)
    else:
        flush_extracted_files(root, files, min(args.jobs, len(archives)))
        # 删除前检查完整输入快照；下载仍在写入时不删除任何包。
        for archive in archives:
            if fingerprint(archive) != originals[archive]:
                raise RuntimeError(f"压缩包发生变化，保留全部压缩包：{archive}")
        for archive in archives:
            archive.unlink()
            print(f"已删除：{archive.relative_to(root)}", flush=True)
    print(f"完成，用时 {time.monotonic() - started:.1f} 秒。", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, type=Path, help="数据集根目录")
    parser.add_argument("-j", "--jobs", type=positive_int, default=8, help="并行解压数，默认 8")
    parser.add_argument("--dry-run", action="store_true", help="仅列出计划，不解压、不删除")
    parser.add_argument("--keep-archives", action="store_true", help="解压成功后保留 tar.gz")
    parser.add_argument("--verify-existing", action="store_true", help="逐文件校验已有解压结果，成功后刷盘并删除 tar.gz；不重复解压")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"数据集目录不存在：{root}")

    if args.dry_run:
        archives = find_archives(root)
        print(f"找到 {len(archives)} 个包，计划并行数 {args.jobs}，目标：{root}")
        for archive in archives:
            print(f"  {archive.relative_to(root)} -> {archive.relative_to(root).with_suffix('').with_suffix('')}/")
        operation = "校验已有文件" if args.verify_existing else "解压"
        print("仅预览。实际执行时全部" + operation + "成功后" + ("保留" if args.keep_archives else "删除") + " tar.gz。")
        return 0

    # 避免两个脚本实例同时合并或删除同一批压缩包。锁文件需要保留。
    lock_fd = os.open(root / ".extract_interndata_a1.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("该目录已有解压脚本在运行") from None
        archives = find_archives(root)
        if not archives:
            print("没有找到 tar.gz 压缩包。")
            return 0
        size = sum(path.stat().st_size for path in archives)
        print(f"找到 {len(archives)} 个包，共 {size / 1e9:.2f} GB；"
              f"并行数 {min(args.jobs, len(archives))}；目标：{root}", flush=True)
        print("模式：" + ("校验已有解压文件" if args.verify_existing else "解压"), flush=True)
        return run(args, root, archives)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断，压缩包删除阶段可能未完成；可重新运行。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
