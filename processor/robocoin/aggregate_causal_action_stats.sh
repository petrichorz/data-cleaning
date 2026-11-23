#!/usr/bin/env bash
set -euo pipefail

# 保留要合并的机器人前缀，其他行注释掉；每个前缀单独输出一份统计。
# 区分大小写，AIRBOT_MMK2_ 和 Airbot_MMK2_ 不会自动合并。
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

# 可以直接在这里填写输出目录；命令行第一个参数会覆盖它。
output_dir=""
output_dir=${1:-$output_dir}
root=${2:-/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/RoboCOIN/}
python_bin=${PYTHON_BIN:-python}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

[[ -n $output_dir && $# -le 2 ]] || {
    printf '用法：bash %s 输出目录 [RoboCOIN根目录]\n也可以在脚本中填写 output_dir 后直接运行。\n' "$0" >&2
    exit 2
}
[[ -d $root ]] || { printf '目录不存在：%s\n' "$root" >&2; exit 2; }
[[ ${#prefixes[@]} -gt 0 ]] || { printf '请先填写 prefixes。\n' >&2; exit 2; }
for prefix in "${prefixes[@]}"; do
    [[ $prefix =~ ^[a-zA-Z0-9_-]+$ && -n ${prefix%_} ]] || {
        printf '前缀只能包含字母、数字、下划线和连字符：%s\n' "$prefix" >&2
        exit 2
    }
done
"$python_bin" -c 'import numpy'
mkdir -p -- "$output_dir"

# 只处理根目录的直接子数据集，不扫描轨迹和视频。
# 每个前缀调用一次独立 Python 脚本；合并的是小 JSON，依次处理即可。
shopt -s nullglob
completed=0
success=0
skipped=0
failed_prefixes=()
for prefix in "${prefixes[@]}"; do
    datasets=()
    for dataset in "$root"/"$prefix"*/; do
        [[ -f ${dataset}meta/info.json ]] && datasets+=("${dataset%/}")
    done
    completed=$((completed + 1))
    printf '\n[%d/%d] %s：匹配 %d 个子数据集\n' \
        "$completed" "${#prefixes[@]}" "$prefix" "${#datasets[@]}"
    if [[ ${#datasets[@]} -eq 0 ]]; then
        printf '跳过：没有匹配的数据集，不生成文件。\n'
        skipped=$((skipped + 1))
        continue
    fi

    # 匹配前缀保留末尾的下划线；输出名称去掉该分隔符，避免出现 _.json。
    output=$output_dir/stats_robocoin_${prefix%_}.json
    if "$python_bin" -u "$script_dir/aggregate_causal_action_stats.py" \
        --dataset-root "${datasets[@]}" --output "$output"; then
        success=$((success + 1))
    else
        printf '合并失败：%s；继续处理其他前缀。\n' "$prefix" >&2
        failed_prefixes+=("$prefix")
    fi
done

printf '\n结束：成功 %d，未匹配 %d，失败 %d；输出目录：%s\n' \
    "$success" "$skipped" "${#failed_prefixes[@]}" "$output_dir"
if [[ ${#failed_prefixes[@]} -gt 0 ]]; then
    printf '失败前缀：%s\n' "${failed_prefixes[@]}" >&2
    exit 1
fi
[[ $success -gt 0 ]] || { printf '没有合并任何数据集，请检查根目录和前缀。\n' >&2; exit 1; }
