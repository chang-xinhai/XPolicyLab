"""Official iDP3 window/subset behavior over XPolicyLab feature episodes."""

import numpy as np
import torch

from XPolicyLab.policy.pointcloud_common.backend import activate
from XPolicyLab.policy.pointcloud_common.training import EpisodeDataset
from .recipe import CANDIDATE_POINTS, MODEL_POINTS, validate_contract


class IDP3Dataset(EpisodeDataset):
    def __init__(self, paths, horizon=16, n_obs_steps=2):
        super().__init__(paths, horizon=horizon, n_obs_steps=n_obs_steps)
        validate_contract(self.observation_contract)
        for episode in self.episodes:
            if episode["point_cloud"].shape[1:] != (CANDIDATE_POINTS, 3):
                raise ValueError("iDP3 training requires 10000 candidate XYZ points per frame")
        activate("iDP3")
        from xpl_idp3.model.vision_3d.point_process import uniform_sampling_numpy
        self._sample_points = uniform_sampling_numpy
        self.configure_windows(horizon, n_obs_steps, 15)

    def configure_windows(self, horizon: int, n_obs_steps: int, n_action_steps: int) -> None:
        from xpl_idp3.common.sampler import create_indices
        if not 1 <= n_obs_steps <= horizon or not 1 <= n_action_steps <= horizon - n_obs_steps + 1:
            raise ValueError("Invalid iDP3 observation/action horizons")
        self.horizon, self.n_obs_steps = int(horizon), int(n_obs_steps)
        self.indices = []
        for ep, rows in enumerate(self.episodes):
            native = create_indices(np.array([len(rows["state"])]), sequence_length=horizon,
                                    episode_mask=np.ones(1, dtype=bool),
                                    pad_before=n_obs_steps - 1, pad_after=n_action_steps - 1)
            self.indices.extend((ep, int(begin - sample_begin + n_obs_steps - 1))
                                for begin, _, sample_begin, _ in native)

    def __getitem__(self, index):
        ep, t = self.indices[index]
        rows = self.episodes[ep]
        indices = np.clip(np.arange(self.horizon) + t - self.n_obs_steps + 1,
                          0, len(rows["state"]) - 1)
        # Use the pinned helper: one fresh permutation shared by the entire
        # time window. Repeated boundary observations remain identical.
        cloud = self._sample_points(rows["point_cloud"][indices], MODEL_POINTS)
        return {
            "obs": {"point_cloud": torch.from_numpy(cloud),
                    "agent_pos": torch.from_numpy(rows["state"][indices])},
            "action": torch.from_numpy(rows["action"][indices]),
        }

    def normalization_data(self):
        # Official identity cloud/proprio normalizers require no fitted rows.
        return {"action": np.concatenate([ep["action"] for ep in self.episodes])}
