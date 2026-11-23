#!/usr/bin/env bash
set -euo pipefail

# 保留本批次要扫描的子数据集名前缀，其他行注释掉；区分大小写。
prefixes=(
    "AI2_Alphabot_2_"
    "AIRBOT_MMK2_"
    "AgiBot-g1_"
    "Agilex_Cobot_Magic_"
    "Agilex_Split_Aloha_"
    "Airbot_MMK2_"
    "G1edu-u3_"
    "Galaxea_R1_Lite_"
    "Galbot_g1_"
    "R1_Lite_"
    "RMC-AIDA-L_"
    "Realman_RMC_AIDA_L_"
    "Tianqin_A2_"
    "alpha_bot_2_"
    "leju_robot_"
)
[[ ${#prefixes[@]} -gt 0 ]] || { printf '请先在脚本 prefixes 数组中填写本批次前缀。\n' >&2; exit 2; }

root=${1:-/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/RoboCOIN/}
jobs=${SCAN_JOBS:-4}
python_bin=${PYTHON_BIN:-python}
batch_size=${BATCH_SIZE:-4096}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# 并行由 Python 管理；只读扫描，异常明细打印到 stdout，进度打印到 stderr。
# 直接传递退出码：0 无 NaN/Inf，1 发现 NaN/Inf，2 扫描不完整或参数错误。
exec "$python_bin" -u "$script_dir/scan_robocoin_nonfinite.py" "$root" \
    --prefixes "${prefixes[@]}" --workers "$jobs" --batch-size "$batch_size"
