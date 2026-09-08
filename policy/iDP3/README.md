# iDP3

**Contributor:** XPolicyLab contributors | **Paper:** Generalizable Humanoid Manipulation with 3D Diffusion Policies | **arXiv:** [2410.10803](https://arxiv.org/abs/2410.10803) | **Original code:** [YanjieZe/Improved-3D-Diffusion-Policy](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy)

Pinned upstream model code lives in `source/`; source revisions, licenses and file digests are in [PROVENANCE.md](PROVENANCE.md) and `SOURCE_MANIFEST.json`. This adapter supports standard RGB-D HDF5 processing, training and WebSocket inference, with single/dual-arm `joint` or `ee` layouts registered in the shared robot configuration. It does not contain robot control drivers.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Activate a CUDA PyTorch environment with torchvision and PyTorch3D already installed. The installer preserves the installed CUDA stack. The DepthUMI machine uses `depthumi` (Python 3.12, PyTorch 2.10/cu128); no separate per-model environment is necessary there.

```bash
conda activate depthumi
cd XPolicyLab/policy/iDP3
bash install.sh
```

## Data Processing

The source must use the standard trajectory layout (`vision/<camera>/colors`, `depths`, `intrinsic_matrix`; plural state/action keys). Depth and color must already be registered to the same optical frame/resolution, and rows must be time-aligned. Conversion never overwrites the input. Depth units are explicit: `--depth-scale 1` for metres, `0.001` for millimetres.

```bash
bash process_data.sh BENCH CKPT ENV_CFG ACTION_TYPE --input TRAJECTORY_DIR --depth-scale SCALE [--camera CAMERA --num-points N]
# Example: metre-depth trajectories with a registered RGB camera
bash process_data.sh RoboDojo stack_bowls arx_x5 joint --input /data/stack_bowls --depth-scale 1
```

The derived per-episode HDF5 feature files contain `point_cloud [T,N,C]`, `state [T,S]`, `action [T,A]` and an explicit `observation_contract` attribute. Standard-adapter sampling is deterministic raster-uniform in the camera frame, shared offline/live. This is an adapter choice, not the upstream sampling recipe. DepthUMI bypasses this converter and uses its existing calibrated CUDA-FPS pipeline and manifests.

## Training

```bash
bash train.sh BENCH CKPT ENV_CFG ACTION_TYPE SEED GPU [--epochs E --batch-size B --config POLICY_OVERRIDE_YAML]
bash train.sh RoboDojo stack_bowls arx_x5 joint 42 0 --epochs 300 --batch-size 8
```

The portable trainer uses AdamW, the upstream EMA implementation, train-only normalization, episode-boundary holds, and atomic checkpoint saves. It is a minimal integration recipe, not a claim to reproduce the paper's full training/augmentation schedule. `--config` merges policy overrides; do not change dimensions away from the converted data. Native action windows retain upstream indexing. DepthUMI supplies its separate current-TCP future-target adapter.

`checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/latest.ckpt` includes model/EMA/optimizer, policy config and observation preprocessing. This standalone checkpoint contract differs from DepthUMI checkpoints. The standalone trainer does not yet support resume; DepthUMI's trainer does.

## Evaluation

```bash
bash eval.sh BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV
# Debug transport check with synthetic valid metre-depth observations
DEBUG_RGBD=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 42 0 0 depthumi depthumi
# Repeat with DEBUG_OBS_ENCODED=1 to exercise encoded RGB transport.
```

`deploy.yml` adds `camera`, `depth_scale` and `device`. Camera and depth scale must match the training checkpoint. Images arrive decoded RGB; the adapter never decodes or swaps channels. EE quaternion outputs are normalized, and nonfinite actions are rejected. `reset()` clears all environment histories; batched calls support noncontiguous environment indices.

## Model Assets

No task-independent pretrained policy weights are required. Train a task checkpoint before evaluation; debug/smoke checkpoints are not task policies. The upstream multi-stage encoder accepts XYZ only.

## Notes

Default integration timing is observation history 2, horizon 16, action chunk 6; iDP3/R3D use DDIM 16 steps. Reference paper configs remain under `source/`. Tests use compact model overrides and synthetic data for transport checks; separate DepthUMI tests use real recorded features. No simulator success rate or physical deployment success is claimed by these integration checks.
