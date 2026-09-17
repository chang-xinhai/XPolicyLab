#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then echo 'Usage: process_data.sh BENCH CKPT ENV_CFG ACTION_TYPE --input DIR --depth-scale SCALE [flags]' >&2; exit 2; fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(dirname "$(dirname "$(dirname "$POLICY_DIR")")"):${PYTHONPATH:-}"
bench=$1; ckpt=$2; robot=$3; action=$4; shift 4
python -m XPolicyLab.policy.pointcloud_common.process_data --sampling idp3 --env-cfg-type "$robot" --action-type "$action" \
  --output "$POLICY_DIR/data/$bench-$ckpt-$robot-$action" "$@"
