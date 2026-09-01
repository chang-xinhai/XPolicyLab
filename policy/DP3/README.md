# DP3

This directory contains XPolicyLab's complete DP3 implementation.  The Python
package remains named `diffusion_policy_3d` so checkpoints and Hydra targets
use the upstream public API:

```text
diffusion_policy_3d.policy.dp3.DP3
```

The implementation, replay buffer, sampler, normalizer, EMA helper, training
entrypoint, and reference configuration were migrated together.  Consumers
must add this directory to `PYTHONPATH` or install it with `python -m pip
install -e policy/DP3`.

DepthUMI keeps canonical demonstrations in HDF5 and materializes disposable
Zarr training caches with exactly these keys:

```text
data/point_cloud  float32 [T, N, 3|6]
data/state        float32 [T, D_state]
data/action       float32 [T, D_action]
meta/episode_ends int64   [episodes]
```

Dataset/task-specific HDF5 conversion remains outside this implementation.
See `PROVENANCE.md` for the immutable source receipt and `LICENSE` for the
preserved source license.
