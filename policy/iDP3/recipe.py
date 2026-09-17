"""Official iDP3 defaults and the two distinct point-sampling boundaries.

Model/training: Improved-3D-Diffusion-Policy@f5b27faa.
Camera/grid/candidate sampling: Humanoid-Teleoperation@8221c58e.
No DepthUMI pose, timing or experiment settings belong in this module.
"""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

SAMPLING = "idp3_grid_uniform.v1"
CANDIDATE_POINTS = 10000
MODEL_POINTS = 4096


def official_config():
    return OmegaConf.load(Path(__file__).parent / "source/xpl_idp3/config/idp3.yaml")


def observation_contract(camera: str, depth_scale: float) -> dict:
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive")
    return {
        "sampling": SAMPLING, "frame": "camera_optical", "camera": camera,
        "depth_scale": float(depth_scale), "num_points": CANDIDATE_POINTS,
        "model_num_points": MODEL_POINTS, "channels": 3,
        "near_m": 0.1, "far_m": 1.0, "grid_size_m": 0.005,
        "grid_representative": "first", "candidate_replace": True,
        "padding": "zeros", "shuffle": True, "crop": False,
        "train_sampling": "uniform_without_replacement_shared_window_indices",
        "live_sampling": "grid_then_uniform_direct_to_model_count",
        "collection_source": "8221c58e0005de7efcd071f2373a888751727811",
    }


def validate_contract(contract: dict) -> None:
    expected = observation_contract(contract.get("camera", ""), contract.get("depth_scale", 0))
    if not expected["camera"] or any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("iDP3 requires its declared official 10000/4096 observation contract")


def camera_points(camera: dict, *, count: int, contract: dict, rng=None) -> np.ndarray:
    """Official camera recipe: grid first, replacement sampling or zero padding.

Training conversion requests 10000; live acquisition requests 4096 directly.
The separate training dataset subsequently samples a fresh 4096-point subset.
"""
    validate_contract(contract)
    if count not in (CANDIDATE_POINTS, MODEL_POINTS):
        raise ValueError("Official iDP3 camera count must be 10000 or 4096")
    rng = np.random if rng is None else rng
    # Keep acquisition precision through clipping/grid. Converting integer-mm
    # depth to float32 first moves exactly 100mm above the strict 0.1m bound.
    # Keep the official division order too: multiplying by .001 changes
    # representatives on 5mm voxel boundaries even for valid 350mm depth.
    depth = np.asarray(camera["depth"]) / (1.0 / contract["depth_scale"])
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    intr = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    if (depth.ndim != 2 or intr.shape != (3, 3) or not np.isfinite(intr).all()
            or intr[0, 0] <= 0 or intr[1, 1] <= 0):
        raise ValueError("Expected registered depth [H,W] and finite camera intrinsics")
    # Build all points before sampling, matching the official raster/grid order.
    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    cloud = np.stack(((x - intr[0, 2]) * depth / intr[0, 0],
                      (y - intr[1, 2]) * depth / intr[1, 1], depth), axis=-1).reshape(-1, 3)
    cloud = cloud[(cloud[:, 2] > contract["near_m"]) & (cloud[:, 2] < contract["far_m"])]
    grid = np.floor(cloud / contract["grid_size_m"]).astype(np.int64)
    keys = grid[:, 0] + grid[:, 1] * 10000 + grid[:, 2] * 100000000
    _, indices = np.unique(keys, return_index=True)
    cloud = cloud[indices]
    if len(cloud) < count:
        cloud = np.concatenate((cloud, np.zeros((count - len(cloud), 3))), axis=0)
    else:
        cloud = cloud[rng.choice(len(cloud), count, replace=True)]
    rng.shuffle(cloud)
    return cloud.astype(np.float32)
