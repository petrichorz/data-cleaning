#!/usr/bin/env bash
set -euo pipefail

# 手动填写 InternData-A1 根目录；也可通过第一个命令行参数覆盖。
dataset_root=""

# 在不同任务目录之间并行，目录内部按完整 episode 处理。
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}
jobs=${CONVERT_JOBS:-4}
batch_size=${BATCH_SIZE:-4096}

usage() {
    printf '用法：bash %s [数据集根目录] [--dry-run] [其他转换选项]\n' "$0"
    printf '请填写脚本顶部的 dataset_root，或通过第一个参数传入路径。\n'
}
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 && $1 != -* ]]; then
    dataset_root=$1
    shift
fi
if [[ -z $dataset_root ]]; then
    usage >&2
    exit 2
fi
[[ -d $dataset_root ]] || { printf '数据集目录不存在：%s\n' "$dataset_root" >&2; exit 2; }

exec "$python_bin" -u "$script_dir/convert_interndata_a1_55d.py" \
    "$dataset_root" --workers "$jobs" --batch-size "$batch_size" "$@"
