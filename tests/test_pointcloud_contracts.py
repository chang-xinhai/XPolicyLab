"""Fast geometry/layout/provenance regressions; no GPU or robot required."""

import unittest
import numpy as np
from XPolicyLab.policy.pointcloud_common.observation import points, action_layout, vector
from XPolicyLab.policy.pointcloud_common.backend import source_receipt, config


class PointCloudContracts(unittest.TestCase):
    def test_projection_and_color_alignment(self):
        camera = {
            "depth": np.ones((2, 2), dtype=np.float32),
            "intrinsic_matrix": np.eye(3),
            "color": np.array(
                [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8
            ),
        }
        result = points(camera, count=4, channels=6, depth_scale=0.5)
        np.testing.assert_allclose(
            result[:, :3], [[0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]
        )
        np.testing.assert_array_equal(result[:, 3:], [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

    def test_invalid_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            points(
                {"depth": np.zeros((2, 2)), "intrinsic_matrix": np.eye(3)},
                count=4,
                channels=3,
                depth_scale=1,
            )

    def test_robot_layout_and_dimensions(self):
        layout = action_layout({"arm_dim": [6, 7], "ee_dim": [1, 2]}, "joint")
        self.assertEqual([d for _, d in layout], [6, 1, 7, 2])
        self.assertEqual(vector({k: np.zeros(n) for k, n in layout}, layout).shape, (16,))
        self.assertEqual(
            [d for _, d in action_layout({"arm_dim": [6], "ee_dim": [1]}, "ee")], [7, 1]
        )

    def test_migrated_source_digests(self):
        for name in ["iDP3", "ManiFlow", "R3D"]:
            self.assertEqual(len(source_receipt(name)["commit"]), 40)

    def test_missing_pretrained_weights_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            config(
                "R3D",
                action_dim=10,
                state_dim=10,
                channels=6,
                pretrained_path="/nonexistent/encoder",
            )


if __name__ == "__main__":
    unittest.main()
