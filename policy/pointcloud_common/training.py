"""Small, simulator-independent training entry over per-episode HDF5 features."""

from __future__ import annotations

import argparse
import importlib
import copy
import json
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


def save(path, name, cfg, model, ema, optimizer, epoch, observation_contract=None):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, choices=["iDP3", "ManiFlow", "R3D"])
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expected-action-dim", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained-path")
    p.add_argument("--config", type=Path, help="Explicit policy config override YAML")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    paths = sorted(args.data.glob("*.hdf5"))
    if not paths:
        raise ValueError("No derived HDF5 feature episodes found")
    dataset = EpisodeDataset(paths)
    ep = dataset.episodes[0]
    if args.expected_action_dim is not None and ep["action"].shape[-1] != args.expected_action_dim:
        raise ValueError("Dataset action dimensions do not match robot configuration")
    cfg = config(
        args.policy,
        action_dim=ep["action"].shape[-1],
        state_dim=ep["state"].shape[-1],
        num_points=ep["point_cloud"].shape[1],
        channels=ep["point_cloud"].shape[2],
        pretrained_path=args.pretrained_path,
    )
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    dataset.horizon, dataset.n_obs_steps = cfg.horizon, cfg.n_obs_steps
    model = build(args.policy, cfg)
    model.set_normalizer(normalizer(args.policy, dataset.normalization_data()))
    model.to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    package = activate(args.policy)
    ema_updater = importlib.import_module(f"{package}.model.diffusion.ema_model").EMAModel(
        model=ema
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if args.policy == "ManiFlow" and (args.batch_size < 2 or len(dataset) < args.batch_size):
        raise ValueError(
            "ManiFlow needs complete batches of at least two samples for flow/consistency loss"
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.policy == "ManiFlow"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / "policy.yaml")
    with (args.output / "metrics.jsonl").open("w") as log:
        for epoch in range(args.epochs):
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
                ema_updater.step(model)
                total += value.item()
            row = {"epoch": epoch + 1, "loss": total / len(loader)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
                save(
                    args.output / "latest.ckpt",
                    args.policy,
                    cfg,
                    model,
                    ema,
                    optimizer,
                    epoch + 1,
                    dataset.observation_contract,
                )


if __name__ == "__main__":
    main()
