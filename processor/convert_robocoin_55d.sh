#!/usr/bin/env bash
set -euo pipefail

# 保留本批次要处理的子数据集名前缀，其他行注释掉；可选多个，区分大小写。
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
[[ ${#prefixes[@]} -gt 0 ]] || { printf '请先在脚本 prefixes 数组中填写本批次前缀。\n' >&2; exit 1; }

root=${1:-/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/RoboCOIN/}
jobs=${CONVERT_JOBS:-4}
python_bin=${PYTHON_BIN:-python}
batch_size=${BATCH_SIZE:-4096}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
converter=$script_dir/convert_robocoin_55d.py
[[ -d $root ]] || { printf '目录不存在：%s\n' "$root" >&2; exit 1; }
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { printf 'CONVERT_JOBS 必须为正整数\n' >&2; exit 1; }
[[ $batch_size =~ ^[1-9][0-9]*$ ]] || { printf 'BATCH_SIZE 必须为正整数\n' >&2; exit 1; }
[[ -f $converter ]] || { printf '转换脚本不存在：%s\n' "$converter" >&2; exit 1; }
[[ -f $script_dir/summarize_robocoin_55d.py ]] || { printf '缺少 summarize_robocoin_55d.py 汇总脚本\n' >&2; exit 1; }

# 先固定本轮清单：仅扫描根目录下带 meta/info.json 的直接子数据集。
# 多个前缀命中同一个子数据集时只加入一次；路径用 NUL 分隔，支持空格。
run_dir=$(mktemp -d)
dataset_list=$run_dir/datasets
trap 'rm -rf -- "$run_dir"' EXIT
find "$root" -mindepth 3 -maxdepth 3 -type f -path '*/meta/info.json' -print0 |
    sort -z |
    while IFS= read -r -d '' info; do
        dataset=${info%/meta/info.json}
        for prefix in "${prefixes[@]}"; do
            if [[ -n $prefix && ${dataset##*/} == "$prefix"* ]]; then
                printf '%s\0' "$dataset"
                break
            fi
        done
    done > "$dataset_list"

total=$(tr -cd '\000' < "$dataset_list" | wc -c)
if [[ $total -eq 0 ]]; then
    printf '没有匹配当前前缀且含 meta/info.json 的子数据集。\n'
    exit 0
fi
# 启动前检查解释器和依赖，避免每个子数据集都因环境问题失败。
"$python_bin" -c 'import numpy, pyarrow.parquet, scipy.spatial.transform'
printf '匹配 %d 个子数据集，并发 %d；原地追加 55D 字段。\n' "$total" "$jobs"

show_progress() {
    local completed=$1 failed=$2 filled bar empty
    filled=$((completed * 30 / total))
    printf -v bar '%*s' "$filled" ''
    printf -v empty '%*s' "$((30 - filled))" ''
    [[ ! -t 1 ]] || printf '\r'
    printf '[%s%s] %3d%% 完成 %d/%d 失败 %d' \
        "${bar// /#}" "${empty// /-}" "$((completed * 100 / total))" \
        "$completed" "$total" "$failed"
    if [[ ! -t 1 || $completed -eq $total ]]; then
        printf '\n'
    fi
}
show_progress 0 0

# 一个进程处理一个子数据集，内部 Parquet 仍逐个处理；空出名额就取下一个。
# 各进程的详细日志分开保存，仅向主进程发完成状态，避免进度条交错。
conversion_failed=0
if ! xargs -0 -r -n 1 -P "$jobs" bash -c '
    python_bin=$1
    converter=$2
    batch_size=$3
    run_dir=$4
    dataset=$5
    status=0
    if mkdir -p -- "$dataset/qc" &&
        "$python_bin" -u "$converter" "$dataset" --batch-size "$batch_size" \
            > "$dataset/qc/conversion.log" 2>&1; then
        :
    else
        printf "\n转换失败：%s；请查看 qc/conversion.log 和 qc/conversion_summary.json。\n" "$dataset" >&2
        status=1
    fi
    # 单独保存本轮退出状态，旧的成功摘要不能掩盖本轮进程失败。
    printf "%s\n" "$status" > "$run_dir/${dataset##*/}.status" || exit 1
    # 统一返回 0/1，使单个任务失败（包括退出码 255）也不会让 xargs 停止调度。
    printf "%s\n" "$status"
    exit "$status"
' _ "$python_bin" "$converter" "$batch_size" "$run_dir" < "$dataset_list" | (
    completed=0
    failed=0
    while IFS= read -r status; do
        completed=$((completed + 1))
        failed=$((failed + status))
        show_progress "$completed" "$failed"
    done
    [[ $completed -eq $total ]]
); then
    [[ ! -t 1 ]] || printf '\n'
    conversion_failed=1
fi

# 即使有转换失败，也先汇总本批次；缺失或损坏的摘要同样列为问题。
if ! "$python_bin" "$script_dir/summarize_robocoin_55d.py" "$root" "$dataset_list" "$run_dir"; then
    exit 1
fi
exit "$conversion_failed"
