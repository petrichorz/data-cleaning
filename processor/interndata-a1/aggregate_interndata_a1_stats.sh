#!/usr/bin/env bash
set -euo pipefail

# 手动填写，也可通过前两个命令行参数覆盖。
dataset_root=""
output_dir=""

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}

usage() {
    printf '用法：bash %s [InternData-A1根目录] [输出目录] [其他汇总选项]\n' "$0"
    printf '可填写脚本顶部的 dataset_root 和 output_dir 后直接运行。\n'
    printf '其他选项：--robots franka lift2 genie1 split_aloha、--bounds min_max、--dry-run。\n'
}
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 && $1 != -* ]]; then
    dataset_root=$1
    shift
fi
if [[ $# -gt 0 && $1 != -* ]]; then
    output_dir=$1
    shift
fi
if [[ -z $dataset_root || -z $output_dir ]]; then
    usage >&2
    exit 2
fi
[[ -d $dataset_root ]] || { printf '数据集目录不存在：%s\n' "$dataset_root" >&2; exit 2; }

# 只扫描目录和小 JSON，按型号依次合并即可，不启动轨迹读取进程。
exec "$python_bin" -B -u "$script_dir/aggregate_interndata_a1_stats.py" \
    --dataset-root "$dataset_root" --output-dir "$output_dir" "$@"
