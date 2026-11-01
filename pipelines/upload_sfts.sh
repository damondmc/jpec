#!/usr/bin/env bash
# Upload the narrow-band SFTs in data/ to Great Lakes, one frequency folder per directory.
#
#   pipelines/upload_sfts.sh              upload F_MIN to F_MAX
#   pipelines/upload_sfts.sh --dry-run    list what would be sent, transfer nothing
#
# Set F_MIN and F_MAX below to the band you want on the cluster; folders outside it are left
# behind. Extra arguments go to rsync. One ssh login is shared by every transfer, and an
# interrupted run resumes where it stopped rather than starting the file again: at tens of GB
# that matters. Files already on the far side with the same size and time are skipped, so
# re-running is cheap.
set -euo pipefail

HOST=greatlakes
REMOTE=/scratch/kriles_root/kriles0/damoncht/jpec/data
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/data"
F_MIN=250          # Hz, inclusive; the band pipelines/o4a.yaml searches
F_MAX=300          # Hz, inclusive

if [[ ! -d "$LOCAL" ]]; then
    echo "no $LOCAL to upload; run pipelines/fetch_sfts.sh first" >&2
    exit 1
fi

# folders whose leading number lies in [F_MIN, F_MAX], however the name is spelled
folders=$(find "$LOCAL" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; \
    | awk -v lo="$F_MIN" -v hi="$F_MAX" '
        match($0, /[0-9]+(\.[0-9]+)?/) {
            f = substr($0, RSTART, RLENGTH) + 0
            if (f >= lo && f <= hi) print $0
        }' | sort -n)
if [[ -z "$folders" ]]; then
    echo "no folders between $F_MIN and $F_MAX Hz under $LOCAL; it holds:" >&2
    find "$LOCAL" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort -n | head >&2
    exit 1
fi

# count and size of just those folders, so the total is what actually goes over the wire
count=0
bytes=0
empty=()
while IFS= read -r folder; do
    [[ -z "$folder" ]] && continue
    n=$(find "$LOCAL/$folder" -name '*.sft' | wc -l | tr -d ' ')
    [[ "$n" -eq 0 ]] && empty+=("$folder")
    count=$((count + n))
    bytes=$((bytes + $(find "$LOCAL/$folder" -name '*.sft' -exec stat -f %z {} + 2>/dev/null \
        | awk '{s += $1} END {print s + 0}')))
done <<< "$folders"
printf '%s folders, %s SFTs, %.1f GB, %s-%s Hz -> %s:%s\n' \
    "$(wc -l <<< "$folders" | tr -d ' ')" "$count" \
    "$(awk -v b="$bytes" 'BEGIN {print b / 1073741824}')" "$F_MIN" "$F_MAX" "$HOST" "$REMOTE"
if [[ ${#empty[@]} -gt 0 ]]; then
    echo "  warning: no SFTs in ${empty[*]}" >&2
fi

SOCKET="/tmp/upload_sfts.$$"
SSH=(ssh -o ControlPath="$SOCKET")
"${SSH[@]}" -o ControlMaster=yes -o ControlPersist=yes -fN "$HOST"
trap '"${SSH[@]}" -O exit "$HOST" 2>/dev/null' EXIT

"${SSH[@]}" "$HOST" "mkdir -p '$REMOTE'"

while IFS= read -r folder; do
    [[ -z "$folder" ]] && continue
    echo "  $folder"
    rsync -a --partial --append-verify --stats --include='*.sft' --exclude='*' \
        -e "${SSH[*]}" "$@" "$LOCAL/$folder/" "$HOST:$REMOTE/$folder/"
done <<< "$folders"

echo "done; on the cluster the search reads them as $REMOTE/*/*.sft"
