#!/usr/bin/env bash
set -euo pipefail
if (( $# < 6 )); then echo 'Usage: train.sh BENCH CKPT ENV_CFG ACTION_TYPE SEED GPU [training flags]' >&2; exit 2; fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(dirname "$(dirname "$(dirname "$POLICY_DIR")")"):${PYTHONPATH:-}"
bench=$1; ckpt=$2; robot=$3; action=$4; seed=$5; gpu=$6; shift 6
action_dim=$(bash "$POLICY_DIR/../../utils/get_action_dim.sh" "$(dirname "$(dirname "$(dirname "$POLICY_DIR")")")" "$robot" "$action")
export CUDA_VISIBLE_DEVICES="$gpu"
python -m XPolicyLab.policy.pointcloud_common.training --policy R3D --seed "$seed" --expected-action-dim "$action_dim" \
  --data "$POLICY_DIR/data/$bench-$ckpt-$robot-$action" \
  --output "$POLICY_DIR/checkpoints/$bench-$ckpt-$robot-$action-$seed" "$@"
