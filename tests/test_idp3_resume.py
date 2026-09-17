"""Real optimizer/RNG continuation must equal uninterrupted training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from XPolicyLab.policy.iDP3.recipe import observation_contract, official_config
from XPolicyLab.policy.pointcloud_common import training


class IDP3ResumeTests(unittest.TestCase):
    def test_interrupted_epoch_resume_matches_uninterrupted_with_eight_workers(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            rng = np.random.default_rng(4)
            with h5py.File(data / "episode.hdf5", "w") as file:
                file["point_cloud"] = rng.random((4, 10000, 3), dtype=np.float32)
                file["state"] = rng.random((4, 7), dtype=np.float32)
                file["action"] = rng.random((4, 7), dtype=np.float32)
                file.attrs["observation_contract"] = json.dumps(observation_contract("cam_head", 1.))
            override = root / "compact.yaml"
            override.write_text("down_dims: [16, 32, 64]\ndiffusion_step_embed_dim: 16\n")
            reference = official_config()
            reference.training.checkpoint_every = 1

            def execute(output):
                argv = ["training", "--policy", "iDP3", "--data", str(data), "--output", str(output),
                        "--epochs", "2", "--batch-size", "2", "--device", "cpu", "--config", str(override)]
                with patch("sys.argv", argv), patch("XPolicyLab.policy.iDP3.recipe.official_config", return_value=reference):
                    training.main()

            baseline, resumed = root / "baseline", root / "resumed"
            execute(baseline)
            original_save = training.save

            def interrupt_after_save(*args, **kwargs):
                original_save(*args, **kwargs)
                raise InterruptedError("Injected interruption after epoch checkpoint")

            with patch.object(training, "save", side_effect=interrupt_after_save):
                with self.assertRaises(InterruptedError):
                    execute(resumed)
            # Simulate additional uncheckpointed metrics written before a crash.
            with (resumed / "metrics.jsonl").open("a") as log:
                log.write(json.dumps({"epoch": 2, "global_step": 4, "loss": 999.}) + "\n")
            execute(resumed)
            left = torch.load(baseline / "latest.ckpt", weights_only=False)
            right = torch.load(resumed / "latest.ckpt", weights_only=False)
            for group in ("model", "ema_model"):
                for key, tensor in left[group].items():
                    torch.testing.assert_close(tensor, right[group][key], rtol=0, atol=0, msg=key)
            self.assertEqual(left["training_state"]["lr_scheduler"], right["training_state"]["lr_scheduler"])
            self.assertEqual(left["training_state"]["ema_optimization_step"], right["training_state"]["ema_optimization_step"])
            self.assertEqual((baseline / "metrics.jsonl").read_text(), (resumed / "metrics.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
