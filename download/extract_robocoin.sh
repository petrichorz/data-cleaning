#!/usr/bin/env bash
set -euo pipefail

# 取消注释本批次要处理的文件名前缀，可同时选择多个；区分大小写。
prefixes=(
    # "AI2_Alphabot_2_"
    # "AIRBOT_MMK2_"
    # "AgiBot-g1_"
    # "Agilex_Cobot_Magic_"
    # "Agilex_Split_Aloha_"
    # "Airbot_MMK2_"
    # "G1edu-u3_"
    # "Galaxea_R1_Lite_"
    # "Galbot_g1_"
    # "R1_Lite_"
    # "RMC-AIDA-L_"
    # "Realman_RMC_AIDA_L_"
    # "Tianqin_A2_"
    # "alpha_bot_2_"
    # "leju_robot_"
)
[[ ${#prefixes[@]} -gt 0 ]] || { printf '请先在脚本 prefixes 数组中填写本批次前缀。\n' >&2; exit 1; }

destination=${1:-/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/}
jobs=${EXTRACT_JOBS:-4}
delete_archives=${DELETE_ARCHIVES:-0}
[[ -d $destination ]] || { printf '目录不存在：%s\n' "$destination" >&2; exit 1; }
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { printf 'EXTRACT_JOBS 必须为正整数\n' >&2; exit 1; }
[[ $delete_archives == 0 || $delete_archives == 1 ]] || { printf 'DELETE_ARCHIVES 必须为 0 或 1\n' >&2; exit 1; }

# Snapshot the input list so newly extracted archives are not added to this run.
archive_list=$(mktemp)
trap 'rm -f -- "$archive_list"' EXIT
find "$destination" -type f -name '*.tar.gz' -print0 |
    while IFS= read -r -d '' archive; do
        for prefix in "${prefixes[@]}"; do
            if [[ -n $prefix && ${archive##*/} == "$prefix"* ]]; then
                printf '%s\0' "$archive"
                break
            fi
        done
    done > "$archive_list"

total=$(tr -cd '\000' < "$archive_list" | wc -c)
if [[ $total -eq 0 ]]; then
    printf '没有匹配当前前缀的压缩包。\n'
    exit 0
fi

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

# Workers only extract; optional deletion happens after all workers succeed.
# Workers emit one status line; a single reader updates the progress counter.
if ! xargs -0 -r -n 1 -P "$jobs" bash -c '
    set -euo pipefail
    archive=$1
    parent=$(dirname -- "$archive")

    # Skip existing files, including shared files repeated across archives.
    # Existing files must already be complete; tar does not compare their content.
    status=0
    if ! tar --extract --gzip --skip-old-files --no-same-owner \
        --file="$archive" --directory="$parent" >&2; then
        printf "\n解压失败，保留压缩包：%s\n" "$archive" >&2
        status=1
    fi

    printf "%s\n" "$status"
    exit "$status"
' _ < "$archive_list" | (
    completed=0
    failed=0
    while IFS= read -r status; do
        completed=$((completed + 1))
        failed=$((failed + status))
        show_progress "$completed" "$failed"
    done
); then
    [[ ! -t 1 ]] || printf '\n'
    printf '部分压缩包解压失败，请检查上方报错；所有压缩包均保留。\n' >&2
    exit 1
fi

if [[ $delete_archives == 1 && -s $archive_list ]]; then
    # Flush the destination filesystem before deleting the original input list.
    # A failed extraction or sync exits before any archive is deleted.
    sync -f "$destination"
    while IFS= read -r -d '' archive; do
        rm -- "$archive"
        printf '已删除压缩包：%s\n' "$archive"
    done < "$archive_list"
else
    printf '解压流程结束，压缩包已保留。\n'
fi
