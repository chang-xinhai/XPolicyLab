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
bash process_data.sh BENCH CKPT ENV_CFG ACTION_TYPE --input TRAJECTORY_DIR --depth-scale SCALE [--camera CAMERA --seed SEED]
# Example: metre-depth trajectories with a registered RGB camera
bash process_data.sh RoboDojo stack_bowls arx_x5 joint --input /data/stack_bowls --depth-scale 1
```

The derived per-episode HDF5 feature files contain `point_cloud [T,10000,3]`, `state [T,S]`, `action [T,A]` and a versioned `observation_contract`. The camera recipe follows [the pinned official collector](https://github.com/YanjieZe/Humanoid-Teleoperation/blob/8221c58e0005de7efcd071f2373a888751727811/humanoid_teleoperation/scripts/multi_realsense.py#L328-L370): strict near/far bounds of 0.1/1.0 metres, 5mm grid retaining the first point in each voxel, no scene crop, random sampling **with replacement** when enough points exist, otherwise zero padding, then shuffle. This preserves the official rule even when the candidate array contains duplicates or zeros. The conversion seed is recorded; intrinsics and depths must already share an optical frame.

At **every dataset access**, the pinned official NumPy sampler draws a new 4096-point subset **without replacement**, using one index vector for the entire time window. Repeated initial observations therefore remain identical after sampling. Cache count10000 and model count4096 are distinct checkpoint fields. Fixed1024/4096 caches and raster-uniform caches are rejected for new native iDP3 training; old checkpoints retain their saved preprocessing and normalizers when loaded.

DepthUMI's separate loader, TCP/time semantics and calibrated camera pipeline are downstream adaptations. They do not automatically acquire this native recipe by changing only an algorithm name. Their experiment configurations are maintained separately.

## Training

```bash
bash train.sh BENCH CKPT ENV_CFG ACTION_TYPE SEED GPU [--epochs E --batch-size B --num-workers W --config POLICY_OVERRIDE_YAML]
bash train.sh RoboDojo stack_bowls arx_x5 joint 42 0
```

Model, optimizer, EMA and loader defaults are read directly from the pinned `source/xpl_idp3/config/idp3.yaml`:

| Component | Native iDP3 default |
| --- | --- |
| Point encoder | MultiStagePointNet, XYZ,4096 points,128 visual features |
| Observation / horizon / returned actions |2 /16 /15; return `action_pred[:,1:16]` |
| Diffusion |50 training timesteps, DDIM10 inference, `sample`/x0 prediction |
| U-Net |[256,512,1024], diffusion embedding128 |
| Normalization |Identity point cloud and state; limits-normalized actions |
| AdamW |lr1e-4, betas(.95,.999), eps1e-8, weight decay1e-6 |
| Schedule |Cosine;500 optimizer-step warmup |
| EMA |power.75, inv_gamma1, update_after_step0, min0/max.9999 |
| Loader |batch64, workers8, shuffle, pinned memory; no persistent workers |
| Budget |301 epochs |

The actual training set supplies action normalization. Episode windows and boundary padding use the pinned sampler; native action row t belongs to observation t. The native adapter preserves this indexing, while DepthUMI owns its separate future-only adapter. `--config` explicitly merges model overrides and saves their resolved values; the loader/model point count remains4096. CLI epoch/batch/worker overrides are also persisted. This adapter uses the explicitly supplied episode directory; it does not silently apply the GR1 benchmark's90-demo cap, create an evaluation split or import humanoid robot drivers. Those are dataset/platform choices, not universal model defaults.

The portable workspace uses local JSONL metrics instead of requiring WandB, saves every100 epochs and at the final epoch, and atomically commits checkpoints. These storage/logging choices do not change model, loss or optimizer semantics; they are not a claim of reproducing humanoid benchmark results.

`checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/latest.ckpt` includes model/EMA/optimizer, policy config, observation preprocessing, resolved training recipe, LR scheduler, EMA update count, Python/NumPy/Torch/CUDA RNG and loader RNG. `policy.yaml` and `training_recipe.yaml` expose the actual settings. This standalone checkpoint contract differs from DepthUMI checkpoints.

Like the official workspace, native iDP3 resumes when an existing compatible checkpoint is present; `--no-resume` explicitly starts a fresh run. Resume restores a completed epoch boundary, trims metric rows newer than that checkpoint, and checks the unchanged recipe, source revision and input file hashes. A changed epoch budget or dataset requires an explicitly new run rather than silently changing the schedule. Epoch-boundary replay with workers8 requires the saved nonpersistent-worker loader setting. Legacy checkpoints lacking training/RNG state remain available for inference but cannot resume with this training path.

## Evaluation

```bash
bash eval.sh BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV
# Debug transport check with synthetic valid metre-depth observations
DEBUG_RGBD=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 42 0 0 depthumi depthumi
# Repeat with DEBUG_OBS_ENCODED=1 to exercise encoded RGB transport.
```

`deploy.yml` adds `camera`, `depth_scale` and `device`. Camera and depth scale must match the training checkpoint. **Official live acquisition performs the same grid processing but samples directly to4096 points**; it does not build a10000-point pool followed by another draw. The model's enabled4096→4096 fallback is a no-op. History stores the actual sampled observations and repeats the first one for initial padding. Images arrive decoded RGB; the adapter never decodes or swaps channels. EE quaternion outputs are normalized, and nonfinite actions are rejected. `reset()` clears all environment histories; batched calls support noncontiguous environment indices.

## Model Assets

No task-independent pretrained policy weights are required. Train a task checkpoint before evaluation; debug/smoke checkpoints are not task policies. The upstream multi-stage encoder accepts XYZ only.

## Notes

Reference paper configs and all migrated model bytes remain unchanged under `source/`. The adapter's numerical behavior is checked against the pinned sampling helpers, including point subsets, padding, native action slicing and strict depth bounds. Synthetic training and transport checks establish code behavior; they establish neither simulator success rates nor physical deployment success. Platform-specific timing, action representations and DepthUMI experiments must be configured and validated separately.
