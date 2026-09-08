"""Opt-in model update/save/reload check; never controls a robot.

Run from the parent workspace:
python XPolicyLab/scripts/smoke_pointcloud.py --policy iDP3 --output /tmp/idp3-check
For R3D also provide --pretrained-path /path/to/r3d-weights.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from XPolicyLab.policy.pointcloud_common.backend import build, config, loss, normalizer
from XPolicyLab.policy.pointcloud_common.training import load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["iDP3", "ManiFlow", "R3D"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained-path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.policy == "R3D" and not args.pretrained_path:
        parser.error("R3D smoke requires the official pretrained encoder")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    channels = 6 if args.policy == "R3D" else 3
    cfg = config(
        args.policy,
        action_dim=14,
        state_dim=14,
        num_points=64,
        channels=channels,
        pretrained_path=args.pretrained_path,
    )
    if args.policy == "ManiFlow":
        cfg.n_layer = 2
        cfg.n_emb = 128
        cfg.n_head = 4
    if args.policy in ("iDP3", "R3D"):
        cfg.down_dims = [64, 128, 256]
    if args.policy == "R3D":
        cfg.pointcloud_encoder_cfg.num_group = 16
    batch = {
        "obs": {
            "point_cloud": torch.rand(4, 2, 64, channels, device=args.device),
            "agent_pos": torch.rand(4, 2, 14, device=args.device),
        },
        "action": torch.rand(4, 16, 14, device=args.device),
    }
    model = build(args.policy, cfg).to(args.device)
    model.set_normalizer(
        normalizer(
            args.policy,
            {
                "point_cloud": batch["obs"]["point_cloud"].flatten(0, 1),
                "agent_pos": batch["obs"]["agent_pos"].flatten(0, 1),
                "action": batch["action"].flatten(0, 1),
            },
        )
    )
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    result = loss(model, batch, ema)
    value = result[0] if isinstance(result, tuple) else result
    assert torch.isfinite(value)
    value.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.obs_encoder.parameters()
    )
    optimizer.step()
    ema.load_state_dict(model.state_dict())
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "camera": "cam_head",
        "depth_scale": 1.0,
        "num_points": 64,
        "channels": channels,
        "sampling": "raster_uniform",
        "frame": "camera_optical",
    }
    save(args.output / "latest.ckpt", args.policy, cfg, model, ema, optimizer, 1, contract)
    restored, _ = load(args.output / "latest.ckpt", args.device)
    with torch.inference_mode():
        action = restored.predict_action(batch["obs"])["action"]
    assert action.shape == (4, 6, 14) and torch.isfinite(action).all()
    report = {
        "policy": args.policy,
        "loss": value.item(),
        "action_shape": list(action.shape),
        "encoder_gradient": True,
        "reload": True,
        "compact_smoke_model": True,
        "task_policy": False,
    }
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
