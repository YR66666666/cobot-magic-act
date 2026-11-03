#!/usr/bin/env bash
# 四臂使能/失能 包装脚本：自动挑选装有 piper_sdk 的 python
#   bash armctl.sh on|off|status
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/agilex/miniconda3/envs/aloha/bin/python
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" "$HERE/armctl.py" "$@"
