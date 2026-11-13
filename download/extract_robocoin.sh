#!/usr/bin/env bash
set -euo pipefail

destination=${1:-/efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/}
[[ -d $destination ]] || { printf '目录不存在：%s\n' "$destination" >&2; exit 1; }

# Snapshot the archive list before extraction; never extract several at once.
archive_list=$(mktemp)
trap 'rm -f -- "$archive_list"' EXIT
find "$destination" -type f -name '*.tar.gz' -print0 > "$archive_list"

while IFS= read -r -d '' archive; do
    parent=$(dirname -- "$archive")

    # Keep existing files intact. Any collision or extraction error preserves
    # the archive and stops the script; partial output is retained for inspection.
    if ! tar --extract --gzip --keep-old-files --no-same-owner \
        --file="$archive" --directory="$parent"; then
        printf '解压失败，已停止并保留压缩包：%s\n' "$archive" >&2
        exit 1
    fi

    # Flush extracted data before deleting this archive. No batch deletion.
    sync -f "$parent"
    rm -- "$archive"
    printf '解压完成，已删除压缩包：%s\n' "$archive"
done < "$archive_list"
