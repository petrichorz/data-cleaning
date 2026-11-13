#!/usr/bin/env bash
set -euo pipefail

obsutil cp obs://egodata/openwam/robocoin/ \
  /efs-mi-east4-4/EmbodiedDatasets/openwam/RoboCOIN/ \
  -r -u -flat -f -j=32 -p=4

# Only extract after the entire download succeeds.
bash "$(dirname -- "${BASH_SOURCE[0]}")/extract_robocoin.sh"
