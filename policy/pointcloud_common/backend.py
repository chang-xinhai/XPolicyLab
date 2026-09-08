"""Load pinned algorithms without simulator imports or global package collisions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from omegaconf import OmegaConf
import hydra

POLICIES = {
    "iDP3": ("xpl_idp3", "idp3.yaml"),
    "ManiFlow": ("maniflow", "maniflow_pointcloud_policy.yaml"),
    "R3D": ("r3d", "r3d_robotwin2.yaml"),
}


def activate(name: str) -> str:
    package, _ = POLICIES[name]
    source = Path(__file__).resolve().parents[1] / name / "source"
    loaded = sys.modules.get(package)
    if loaded is not None:
        locations = list(getattr(loaded, "__path__", []))
        if getattr(loaded, "__file__", None):
            locations.append(loaded.__file__)
        if any(not Path(location).resolve().is_relative_to(source) for location in locations):
            raise RuntimeError(
                f"{package} already imported from another checkout; use a fresh process"
            )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return package


def config(
    name: str,
    *,
    action_dim: int,
    state_dim: int,
    num_points: int = 1024,
    channels: int = 3,
    horizon: int = 16,
    n_obs_steps: int = 2,
    n_action_steps: int = 6,
    pretrained_path: str | None = None,
):
    if pretrained_path is not None and not (Path(pretrained_path) / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing pretrained encoder: {pretrained_path}")
    package = activate(name)
    source = Path(__file__).resolve().parents[1] / name / "source"
    cfg = OmegaConf.load(source / package / "config" / POLICIES[name][1])
    cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps = horizon, n_obs_steps, n_action_steps
    cfg.shape_meta = {
        "action": {"shape": [action_dim]},
        "obs": {
            "point_cloud": {"shape": [num_points, channels], "type": "point_cloud"},
            "agent_pos": {"shape": [state_dim], "type": "low_dim"},
        },
    }
    cfg.policy.shape_meta = cfg.shape_meta
    cfg.policy.use_pc_color = channels == 6
    cfg.policy.num_inference_steps = 2 if name == "ManiFlow" else 16
    if name == "iDP3":
        if channels != 3:
            raise ValueError("Upstream iDP3 multi-stage encoder supports XYZ only")
        cfg.policy.point_downsample = False  # Input sampling belongs to the data contract.
        cfg.policy.pointcloud_encoder_cfg.num_points = num_points
    elif name == "ManiFlow":
        cfg.policy.visual_cond_len = num_points
        cfg.policy.downsample_points = False
        cfg.policy.pointcloud_encoder_cfg.num_points = num_points
    else:
        if channels != 6:
            raise ValueError("Pretrained R3D requires XYZRGB; do not fabricate missing color")
        cfg.policy.pointcloud_encoder_cfg.pretrained_weights_path = pretrained_path
        cfg.policy.pointcloud_encoder_cfg.use_pretrained_weights = pretrained_path is not None
    return OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))


def build(name: str, policy_config):
    activate(name)
    if name == "R3D" and policy_config.pointcloud_encoder_cfg.use_pretrained_weights:
        from safetensors import safe_open

        path = (
            Path(policy_config.pointcloud_encoder_cfg.pretrained_weights_path) / "model.safetensors"
        )
        with safe_open(path, framework="pt", device="cpu") as weights:
            if not any(key.startswith("pc_encoder.transformer.blocks.") for key in weights.keys()):
                raise ValueError("R3D checkpoint has no expected pretrained encoder tensors")
    return hydra.utils.instantiate(policy_config)


def normalizer(name: str, data: dict):
    package = activate(name)
    cls = importlib.import_module(f"{package}.model.common.normalizer").LinearNormalizer
    result = cls()
    result.fit(data=data, last_n_dims=1, mode="limits")
    return result


def loss(model, batch, ema_model):
    # ManiFlow requires the EMA model for its consistency targets.
    if model.__class__.__module__.startswith("maniflow."):
        return model.compute_loss(batch, ema_model=ema_model)
    return model.compute_loss(batch)


def source_receipt(name: str) -> dict:
    """Validate vendored bytes and identify the exact algorithm source."""
    import hashlib
    import json

    directory = Path(__file__).resolve().parents[1] / name
    manifest_path = directory / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Migrated source differs from manifest: {path}")
    return {
        "name": name,
        "repository": manifest["repository"],
        "commit": manifest["commit"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
