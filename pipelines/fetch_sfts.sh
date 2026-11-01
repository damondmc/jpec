#!/usr/bin/env bash
# Download narrow-band SFTs of H1 and L1 from CIT into data/, one frequency folder every STEP Hz.
#
#   pipelines/fetch_sfts.sh              download
#   pipelines/fetch_sfts.sh --dry-run    list what would be copied
#
# Extra arguments go to rsync. One ssh login is shared by every transfer, and an interrupted run
# resumes where it stopped.
set -euo pipefail

HOST=cit
REMOTE=/home/hoitim.cheung/SFTs/o4a_data/SFTs/narrowBand_age300yr/240days
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/data"
DETECTORS=(H1 L1)
F_MIN=250
F_MAX=300
STEP=2

SOCKET="/tmp/fetch_sfts.$$"
SSH=(ssh -o ControlPath="$SOCKET")
"${SSH[@]}" -o ControlMaster=yes -o ControlPersist=yes -fN "$HOST"
trap '"${SSH[@]}" -O exit "$HOST" 2>/dev/null' EXIT

mkdir -p "$LOCAL"
expected=$(( (F_MAX - F_MIN) / STEP + 1 ))
for det in "${DETECTORS[@]}"; do
    # folders whose leading number is F_MIN, F_MIN + STEP, ..., F_MAX, however it is spelled
    folders=$("${SSH[@]}" "$HOST" "ls '$REMOTE/$det'" | awk -v lo="$F_MIN" -v hi="$F_MAX" -v step="$STEP" '
        match($0, /[0-9]+(\.[0-9]+)?/) {
            f = substr($0, RSTART, RLENGTH) + 0
            if (f >= lo && f <= hi && f == int(f) && (f - lo) % step == 0) print $0
        }')
    if [[ -z "$folders" ]]; then
        echo "no folders between $F_MIN and $F_MAX Hz in $HOST:$REMOTE/$det; it holds:" >&2
        "${SSH[@]}" "$HOST" "ls '$REMOTE/$det' | head" >&2
        exit 1
    fi
    echo "$det: $(wc -l <<< "$folders" | tr -d ' ') of $expected frequencies found"
    for folder in $folders; do
        echo "  $det/$folder"
        mkdir -p "$LOCAL/$folder"
        rsync -a --partial --stats --include='*.sft' --exclude='*' -e "${SSH[*]}" "$@" \
            "$HOST:$REMOTE/$det/$folder/" "$LOCAL/$folder/"
    done
done
