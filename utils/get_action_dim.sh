#!/bin/bash
set -e

ROOT_DIR="$1"
env_cfg_type="$2"
action_type="${3:-joint}"

python3 -c '
import sys, os, json

root_dir = sys.argv[1]
env_cfg_type = sys.argv[2]

robot_action_dim_info = json.load(
    open(os.path.join(root_dir, "XPolicyLab", "utils", "robot", "_robot_info.json"), "r", encoding="utf-8")
)[env_cfg_type]

action_type = sys.argv[3]
if action_type not in ("joint", "ee"):
    raise ValueError("action_type must be joint or ee")
arm_dim = sum(robot_action_dim_info["arm_dim"]) if action_type == "joint" else 7 * len(robot_action_dim_info["arm_dim"])
print(arm_dim + sum(robot_action_dim_info["ee_dim"]))
' "${ROOT_DIR}" "${env_cfg_type}" "${action_type}"