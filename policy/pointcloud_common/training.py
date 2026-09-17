"""Small, simulator-independent training entry over per-episode HDF5 features."""

from __future__ import annotations

import argparse
import importlib
import copy
import json
import random
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from .backend import activate, build, config, loss, normalizer, source_receipt


class EpisodeDataset(Dataset):
    """Derived features: point_cloud [T,N,C], state [T,S], action [T,A].

    Action row t is the target for observation t (upstream indexing). Episodes
    are never concatenated for window sampling; boundary padding is a hold.
    """

    def __init__(self, paths, horizon=16, n_obs_steps=2):
        self.episodes = []
        self.indices = []
        self.observation_contract = None
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        for path in paths:
            with h5py.File(path) as f:
                episode = {
                    k: np.asarray(f[k], dtype=np.float32)
                    for k in ("point_cloud", "state", "action")
                }
                contract = json.loads(f.attrs["observation_contract"])
                if self.observation_contract is not None and contract != self.observation_contract:
                    raise ValueError("Episodes use different observation preprocessing contracts")
                self.observation_contract = contract
            t = len(episode["state"])
            if not t or any(len(v) != t or not np.isfinite(v).all() for v in episode.values()):
                raise ValueError(f"Invalid or nonfinite episode: {path}")
            if (
                episode["point_cloud"].ndim != 3
                or episode["state"].ndim != 2
                or episode["action"].ndim != 2
            ):
                raise ValueError(f"Invalid feature ranks: {path}")
            self.indices.extend((len(self.episodes), i) for i in range(t))
            self.episodes.append(episode)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ep, t = self.indices[index]
        rows = self.episodes[ep]
        idx = np.clip(np.arange(self.horizon) + t - self.n_obs_steps + 1, 0, len(rows["state"]) - 1)
        return {
            "obs": {
                "point_cloud": torch.from_numpy(rows["point_cloud"][idx[: self.n_obs_steps]]),
                "agent_pos": torch.from_numpy(rows["state"][idx[: self.n_obs_steps]]),
            },
            "action": torch.from_numpy(rows["action"][idx]),
        }

    def normalization_data(self):
        return {
            target: np.concatenate([ep[source] for ep in self.episodes])
            for source, target in [
                ("point_cloud", "point_cloud"),
                ("state", "agent_pos"),
                ("action", "action"),
            ]
        }


def save(path, name, cfg, model, ema, optimizer, epoch, observation_contract=None,
         training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(
        {
            "contract": "xpolicylab.pointcloud.v1",
            "policy_name": name,
            "algorithm_source": source_receipt(name),
            "policy_config": OmegaConf.to_container(cfg, resolve=True),
            "observation_contract": observation_contract,
            "model": model.state_dict(),
            "ema_model": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "training_state": training_state,
        },
        temporary,
    )
    temporary.replace(path)


def load(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != "xpolicylab.pointcloud.v1":
        raise ValueError("Unsupported point-cloud checkpoint contract")
    if checkpoint.get("algorithm_source") != source_receipt(checkpoint["policy_name"]):
        raise ValueError("Checkpoint algorithm source differs from installed XPolicyLab")
    cfg = OmegaConf.create(checkpoint["policy_config"])
    # The policy checkpoint embeds encoder weights; no network/download on deployment.
    if checkpoint["policy_name"] == "R3D":
        cfg.pointcloud_encoder_cfg.use_pretrained_weights = False
    model = build(checkpoint["policy_name"], cfg)
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    return model.to(device).eval(), checkpoint


def truncate_metrics(path: Path, epoch: int, global_step: int) -> None:
    """Discard uncheckpointed rows after an interrupted epoch sequence."""
    if not path.exists():
        return
    retained = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break  # An interrupted final write is not a complete metric row.
        if row["epoch"] <= epoch and row["global_step"] <= global_step:
            retained.append(line)
    temporary = path.with_suffix(".partial")
    temporary.write_text("".join(line + "\n" for line in retained))
    temporary.replace(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, choices=["iDP3", "ManiFlow", "R3D"])
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None,
                   help="Resume the saved epoch boundary (official iDP3 default: enabled)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expected-action-dim", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained-path")
    p.add_argument("--config", type=Path, help="Explicit policy config override YAML")
    args = p.parse_args()
    official = None
    if args.policy == "iDP3":
        from XPolicyLab.policy.iDP3.recipe import official_config
        official = official_config()
    if args.epochs is None:
        args.epochs = int(official.training.num_epochs) if official else 300
    if args.batch_size is None:
        args.batch_size = int(official.dataloader.batch_size) if official else 8
    if args.num_workers is None:
        args.num_workers = int(official.dataloader.num_workers) if official else 0
    if args.resume is None:
        args.resume = bool(official.training.resume) if official else False
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    paths = sorted(args.data.glob("*.hdf5"))
    if not paths:
        raise ValueError("No derived HDF5 feature episodes found")
    if official:
        from XPolicyLab.policy.iDP3.dataset import IDP3Dataset
        dataset = IDP3Dataset(paths)
    else:
        dataset = EpisodeDataset(paths)
    ep = dataset.episodes[0]
    if args.expected_action_dim is not None and ep["action"].shape[-1] != args.expected_action_dim:
        raise ValueError("Dataset action dimensions do not match robot configuration")
    cfg = config(
        args.policy,
        action_dim=ep["action"].shape[-1],
        state_dim=ep["state"].shape[-1],
        num_points=dataset.observation_contract["model_num_points"] if official else ep["point_cloud"].shape[1],
        channels=ep["point_cloud"].shape[2],
        pretrained_path=args.pretrained_path,
    )
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    if official and (list(cfg.shape_meta.obs.point_cloud.shape) != [4096, 3]
                     or cfg.pointcloud_encoder_cfg.num_points != 4096):
        raise ValueError("iDP3 loader/model contract requires 4096 XYZ, separately from 10000 stored points")
    dataset.horizon, dataset.n_obs_steps = cfg.horizon, cfg.n_obs_steps
    if official:
        dataset.configure_windows(int(cfg.horizon), int(cfg.n_obs_steps), int(cfg.n_action_steps))
    if not len(dataset):
        raise ValueError("No training windows remain under the configured horizons and padding")
    model = build(args.policy, cfg)
    model.set_normalizer(normalizer(args.policy, dataset.normalization_data()))
    model.to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    package = activate(args.policy)
    ema_options = {k: v for k, v in OmegaConf.to_container(official.ema).items()
                   if k != "_target_"} if official else {}
    ema_updater = importlib.import_module(f"{package}.model.diffusion.ema_model").EMAModel(
        model=ema, **ema_options)
    optimizer_options = {k: v for k, v in OmegaConf.to_container(official.optimizer).items()
                         if k != "_target_"} if official else {"lr": 1e-4}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    if args.policy == "ManiFlow" and (args.batch_size < 2 or len(dataset) < args.batch_size):
        raise ValueError(
            "ManiFlow needs complete batches of at least two samples for flow/consistency loss"
        )
    generator = torch.Generator().manual_seed(args.seed) if official else None
    loader_options = dict(batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=args.policy == "ManiFlow")
    if official:
        loader_options.update(pin_memory=bool(official.dataloader.pin_memory),
                              persistent_workers=bool(official.dataloader.persistent_workers))
    loader = DataLoader(dataset, generator=generator, **loader_options)
    scheduler = None
    if official:
        from xpl_idp3.model.common.lr_scheduler import get_scheduler
        scheduler = get_scheduler(official.training.lr_scheduler, optimizer=optimizer,
                                  num_warmup_steps=int(official.training.lr_warmup_steps),
                                  num_training_steps=len(loader) * args.epochs)
    recipe = {"optimizer": optimizer_options, "ema": ema_options, "dataloader": loader_options,
              "epochs": args.epochs, "seed": args.seed,
              "lr_scheduler": str(official.training.lr_scheduler) if official else None,
              "lr_warmup_steps": int(official.training.lr_warmup_steps) if official else 0,
              "data_paths": [str(p.resolve()) for p in paths],
              "data_sha256": [file_digest(p) for p in paths] if official else None}
    start_epoch, global_step = 0, 0
    resumed = args.resume and (args.output / "latest.ckpt").exists()
    if resumed:
        previous = torch.load(args.output / "latest.ckpt", map_location=args.device, weights_only=False)
        state = previous.get("training_state")
        if (not state or state["recipe"] != recipe
                or previous["policy_name"] != args.policy
                or previous["policy_config"] != OmegaConf.to_container(cfg, resolve=True)
                or previous["observation_contract"] != dataset.observation_contract
                or previous["algorithm_source"] != source_receipt(args.policy)):
            raise ValueError("Resume requires unchanged recipe, policy, sources and preprocessing")
        model.load_state_dict(previous["model"], strict=True)
        ema.load_state_dict(previous["ema_model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["lr_scheduler"])
        ema_updater.optimization_step = state["ema_optimization_step"]
        start_epoch, global_step = previous["epoch"], state["global_step"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if generator is not None:
            generator.set_state(state["loader_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        truncate_metrics(args.output / "metrics.jsonl", start_epoch, global_step)
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / "policy.yaml")
    OmegaConf.save(OmegaConf.create(recipe), args.output / "training_recipe.yaml")
    with (args.output / "metrics.jsonl").open("a" if resumed else "w") as log:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            total = 0.0
            for batch in loader:
                batch = {
                    "obs": {k: v.to(args.device) for k, v in batch["obs"].items()},
                    "action": batch["action"].to(args.device),
                }
                result = loss(model, batch, ema)
                value = result[0] if isinstance(result, tuple) else result
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                ema_updater.step(model)
                global_step += 1
                total += value.item()
            row = {"epoch": epoch + 1, "loss": total / len(loader), "global_step": global_step,
                   "lr": optimizer.param_groups[0]["lr"]}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            cadence = int(official.training.checkpoint_every) if official else 50
            if (epoch + 1) % cadence == 0 or epoch + 1 == args.epochs:
                save(
                    args.output / "latest.ckpt",
                    args.policy,
                    cfg,
                    model,
                    ema,
                    optimizer,
                    epoch + 1,
                    dataset.observation_contract,
                    training_state={"recipe": recipe, "global_step": global_step,
                                    "lr_scheduler": scheduler.state_dict() if scheduler else None,
                                    "ema_optimization_step": ema_updater.optimization_step,
                                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                                    "torch_rng": torch.get_rng_state(),
                                    "loader_rng": generator.get_state() if generator is not None else None,
                                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                )


if __name__ == "__main__":
    main()
