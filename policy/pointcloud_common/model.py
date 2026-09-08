"""Standard XPolicyLab observation/action adapter for migrated 3D policies."""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_batch_size
from .training import load
from .observation import action_layout, vector, points


class PointCloudModel(ModelTemplate):
    policy_name = None

    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = dict(model_cfg)
        self.device = self.cfg.get("device", "cuda:0")
        directory = Path(__file__).resolve().parents[1] / self.policy_name
        path = resolve_checkpoint_root(self.cfg, directory / "checkpoints")
        path = Path(path)
        self.model, checkpoint = load(path / "latest.ckpt" if path.is_dir() else path, self.device)
        if checkpoint["policy_name"] != self.policy_name:
            raise ValueError("Wrong policy checkpoint")
        self.layout = action_layout(
            get_robot_action_dim_info(self.cfg["env_cfg_type"]), self.cfg["action_type"]
        )
        if sum(dim for _, dim in self.layout) != self.model.action_dim:
            raise ValueError("Robot/checkpoint action dimension mismatch")
        shape = checkpoint["policy_config"]["shape_meta"]["obs"]["point_cloud"]["shape"]
        self.count, self.channels = shape
        contract = checkpoint.get("observation_contract")
        if (
            not contract
            or contract.get("sampling") != "raster_uniform"
            or contract.get("frame") != "camera_optical"
        ):
            raise ValueError("Checkpoint requires an explicit supported observation contract")
        if [contract["num_points"], contract["channels"]] != list(shape):
            raise ValueError("Checkpoint preprocessing/model shape mismatch")
        for key in ("camera", "depth_scale"):
            if key in self.cfg and self.cfg[key] != contract[key]:
                raise ValueError(f"Deploy {key} differs from training preprocessing")
            self.cfg[key] = contract[key]
        self.batch_size = get_batch_size(self.cfg["env_cfg_type"])
        self.histories = {}
        self.active = []

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self.active = []
        for i, obs in enumerate(obs_list):
            idx = int(obs.get("env_idx", i))
            self.active.append(idx)
            camera = obs["vision"][self.cfg.get("camera", "cam_head")]
            point = points(
                camera,
                count=self.count,
                channels=self.channels,
                depth_scale=float(self.cfg["depth_scale"]),
            )
            state = vector(obs["state"], self.layout)
            sample = {"point_cloud": point, "agent_pos": state}
            history = self.histories.setdefault(idx, deque(maxlen=self.model.n_obs_steps))
            while len(history) < self.model.n_obs_steps:
                history.append(sample)
            history.append(sample)

    def get_action_batch(self, env_idx_list=None):
        indices = self.active if env_idx_list is None else list(env_idx_list)
        if not indices:
            raise RuntimeError("update_obs must precede get_action")
        obs = {
            key: torch.as_tensor(
                np.stack([[row[key] for row in self.histories[i]] for i in indices]),
                device=self.device,
            )
            for key in ("point_cloud", "agent_pos")
        }
        with torch.inference_mode():
            actions = self.model.predict_action(obs)["action"].cpu().numpy()
        result = []
        for chunk in actions:
            episode = []
            for row in chunk:
                if not np.isfinite(row).all():
                    raise FloatingPointError("Nonfinite policy action")
                action = {}
                offset = 0
                for key, dim in self.layout:
                    value = row[offset : offset + dim].copy()
                    offset += dim
                    if key.endswith("ee_pose"):
                        norm = np.linalg.norm(value[3:])
                        if norm < 1e-6:
                            raise ValueError("Degenerate predicted quaternion")
                        value[3:] /= norm
                    action[key] = value
                episode.append(action)
            result.append(episode)
        return result

    def get_action(self):
        return self.get_action_batch()[0]

    def reset(self):
        self.histories.clear()
        self.active = []
