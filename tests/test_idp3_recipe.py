"""iDP3 data/model parity; no camera, robot or GPU required."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from XPolicyLab.policy.iDP3.dataset import IDP3Dataset
from XPolicyLab.policy.iDP3.recipe import camera_points, observation_contract, official_config
from XPolicyLab.policy.pointcloud_common.backend import activate, config, normalizer
from XPolicyLab.policy.pointcloud_common.training import EpisodeDataset, truncate_metrics

activate("iDP3")
from xpl_idp3.model.vision_3d.point_process import uniform_sampling_numpy
from xpl_idp3.common.replay_buffer import ReplayBuffer
from xpl_idp3.common.sampler import SequenceSampler


class IDP3RecipeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "episode.hdf5"
        self.contract = observation_contract("cam_head", 1.0)
        rng = np.random.default_rng(93)
        self.rows = {"point_cloud": rng.normal(size=(4, 10000, 3)).astype(np.float32),
                     "state": rng.normal(size=(4, 7)).astype(np.float32),
                     "action": rng.normal(size=(4, 7)).astype(np.float32)}
        with h5py.File(self.path, "w") as file:
            for key, value in self.rows.items():
                file[key] = value
            file.attrs["observation_contract"] = json.dumps(self.contract)

    def test_every_window_matches_official_sampler_and_subset(self):
        dataset = IDP3Dataset([self.path])
        replay = ReplayBuffer.create_empty_numpy()
        replay.add_episode(self.rows)
        reference = SequenceSampler(replay, sequence_length=16, pad_before=1, pad_after=14)
        self.assertEqual(len(dataset), len(reference))
        for index in range(len(dataset)):
            rows = reference.sample_sequence(index)
            np.random.seed(74)
            expected = uniform_sampling_numpy(rows["point_cloud"], 4096)
            np.random.seed(74)
            actual = dataset[index]
            np.testing.assert_array_equal(actual["obs"]["point_cloud"], expected)
            np.testing.assert_array_equal(actual["obs"]["agent_pos"], rows["state"])
            np.testing.assert_array_equal(actual["action"], rows["action"])

    def test_fresh_subsets_and_initial_padding(self):
        dataset = IDP3Dataset([self.path])
        np.random.seed(19)
        first, second = dataset[0], dataset[0]
        cloud = first["obs"]["point_cloud"]
        self.assertEqual(list(cloud.shape), [16, 4096, 3])
        self.assertFalse(torch.equal(cloud, second["obs"]["point_cloud"]))
        self.assertTrue(torch.equal(cloud[0], cloud[1]))
        np.random.seed(19)
        self.assertTrue(torch.equal(cloud, dataset[0]["obs"]["point_cloud"]))
        self.assertTrue(torch.equal(first["action"], second["action"]))

    def test_resume_removes_uncheckpointed_and_partial_metric_rows(self):
        log = Path(self.directory.name) / "metrics.jsonl"
        rows = [{"epoch": k, "global_step": 3 * k} for k in (1, 2, 3)]
        log.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"epoch":')
        truncate_metrics(log, epoch=2, global_step=6)
        self.assertEqual([json.loads(line) for line in log.read_text().splitlines()], rows[:2])

    def test_legacy_cloud_dataset_is_still_fixed(self):
        dataset = EpisodeDataset([self.path])
        self.assertTrue(torch.equal(dataset[0]["obs"]["point_cloud"],
                                    dataset[0]["obs"]["point_cloud"]))

    def test_rejects_fixed_cloud_cache_disguised_as_candidates(self):
        with h5py.File(self.path, "a") as file:
            del file["point_cloud"]
            file["point_cloud"] = self.rows["point_cloud"][:, :4096]
        with self.assertRaisesRegex(ValueError, "10000 candidate"):
            IDP3Dataset([self.path])

    def test_normalizer_leaves_geometry_and_proprioception_unchanged(self):
        norm = normalizer("iDP3", {"action": self.rows["action"]})
        cloud = torch.tensor(self.rows["point_cloud"])
        state = torch.tensor(self.rows["state"])
        torch.testing.assert_close(norm["point_cloud"].normalize(cloud), cloud, rtol=0, atol=0)
        torch.testing.assert_close(norm["agent_pos"].normalize(state), state, rtol=0, atol=0)
        action = norm["action"].normalize(torch.tensor(self.rows["action"]))
        torch.testing.assert_close(action.min(0).values, -torch.ones(7))
        torch.testing.assert_close(action.max(0).values, torch.ones(7))

    def test_native_model_defaults_preserve_reference(self):
        native = config("iDP3", action_dim=7, state_dim=7)
        official = official_config()
        self.assertEqual(native.num_inference_steps, official.policy.num_inference_steps)
        self.assertEqual(native.n_action_steps, official.n_action_steps)
        self.assertEqual(native.horizon, official.horizon)
        self.assertEqual(native.n_obs_steps, official.n_obs_steps)
        self.assertTrue(native.point_downsample)
        self.assertEqual(native.pointcloud_encoder_cfg.num_points, 4096)
        self.assertEqual(native.noise_scheduler.prediction_type, "sample")
        self.assertEqual(native.noise_scheduler.num_train_timesteps, 50)
        # Explicit downstream adapters still own their future/chunk semantics.
        adapted = config("iDP3", action_dim=10, state_dim=10, num_points=1024, n_action_steps=6)
        self.assertEqual(adapted.n_action_steps, 6)
        self.assertEqual(adapted.pointcloud_encoder_cfg.num_points, 1024)

    def test_camera_zero_padding_depth_bounds_and_replacement(self):
        camera = {"depth": np.array([[0.05, 0.2, 0.3, 1.0]], dtype=np.float32),
                  "intrinsic_matrix": np.eye(3)}
        result = camera_points(camera, count=4096, contract=self.contract, rng=np.random.RandomState(4))
        valid = result[result[:, 2] != 0]
        self.assertEqual(len(valid), 2)
        np.testing.assert_allclose(sorted(valid[:, 2]), [0.2, 0.3])
        self.assertEqual(result.dtype, np.float32)
        empty = camera_points({**camera, "depth": np.zeros((1, 4))}, count=4096,
                              contract=self.contract)
        self.assertFalse(np.any(empty))
        millimetres = observation_contract("cam_head", .001)
        boundary = camera_points({**camera, "depth": np.full((2, 2), 100, dtype=np.uint16)},
                                 count=4096, contract=millimetres)
        self.assertFalse(np.any(boundary))
        # A dense grid has >10000 distinct voxels: official camera sampling is
        # WITH replacement; dataset selection later is WITHOUT replacement.
        dense = {"depth": np.full((120, 120), 0.5, dtype=np.float32),
                 "intrinsic_matrix": np.diag([50., 50., 1.])}
        for count in (4096, 10000):
            result = camera_points(dense, count=count, contract=self.contract,
                                   rng=np.random.RandomState(42))
            self.assertLess(len(np.unique(result, axis=0)), count)

    def test_integer_depth_voxel_representatives_match_official_golden(self):
        # Generated from Humanoid-Teleoperation@8221c58e's unmodified
        # create_colored_point_cloud/grid_sample_pcd, XYZ cast to float32.
        # Multiplying by .001 instead of dividing by1000 changes these points.
        for millimetres, focal, expected in [
            (350, 100, "e88261297f8a8b6ff85512dcd29f8f3984b89dd0ad22a16baa60ef9094484e03"),
            (700, 200, "eba35c22e8baabcdff4218e49f6cb1e97f82387faff01663ffe2342a927abe0d"),
        ]:
            camera = {"depth": np.full((64, 64), millimetres, dtype=np.uint16),
                      "intrinsic_matrix": np.array([[focal, 0, 32], [0, focal, 32], [0, 0, 1]])}
            cloud = camera_points(camera, count=4096, contract=observation_contract("cam_head", .001),
                                  rng=np.random.RandomState(13))
            self.assertEqual(hashlib.sha256(cloud.tobytes()).hexdigest(), expected)

    def test_live_uses_model_count_and_repeats_sampled_initial_observation(self):
        from XPolicyLab.policy.pointcloud_common.model import PointCloudModel
        model = object.__new__(PointCloudModel)
        model.cfg = {"camera": "cam_head", "depth_scale": 1.0}
        model.count, model.channels, model.idp3_sampling = 4096, 3, True
        model.observation_contract = self.contract
        model.model = type("Network", (), {"n_obs_steps": 2})()
        model.layout = [("arm_joint_state", 6), ("ee_joint_state", 1)]
        model.histories = {}
        sample = {"vision": {"cam_head": {"depth": np.full((2, 2), .5),
                   "intrinsic_matrix": np.eye(3)}},
                  "state": {"arm_joint_state": np.zeros(6), "ee_joint_state": np.ones(1)}}
        with patch("XPolicyLab.policy.iDP3.recipe.camera_points", wraps=camera_points) as project:
            model.update_obs(sample)
            self.assertEqual(project.call_args.kwargs["count"], 4096)
        cloud = model.histories[0][0]["point_cloud"]
        np.testing.assert_array_equal(cloud, model.histories[0][1]["point_cloud"])
        self.assertEqual(cloud.shape, (4096, 3))


if __name__ == "__main__":
    unittest.main()
