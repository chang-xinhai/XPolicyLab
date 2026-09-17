"""Convert standard RGB-D trajectories into disposable per-episode features."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
from .observation import action_layout, points


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env-cfg-type", required=True)
    p.add_argument("--action-type", choices=["joint", "ee"], required=True)
    p.add_argument("--camera", default="cam_head")
    p.add_argument("--depth-scale", type=float, required=True)
    p.add_argument("--num-points", type=int)
    p.add_argument("--sampling", choices=["raster_uniform", "idp3"], default="raster_uniform")
    p.add_argument("--seed", type=int, default=42, help="Seed for stochastic candidate conversion")
    p.add_argument("--channels", type=int, choices=[3, 6], default=3)
    args = p.parse_args()
    if args.sampling == "idp3":
        from XPolicyLab.policy.iDP3.recipe import observation_contract, camera_points
        contract = observation_contract(args.camera, args.depth_scale)
        if args.num_points not in (None, contract["num_points"]) or args.channels != 3:
            raise ValueError("Official iDP3 conversion requires 10000 candidate XYZ points")
        args.num_points = contract["num_points"]
        project = lambda obs: camera_points(obs, count=args.num_points, contract=contract)
    else:
        args.num_points = args.num_points or 1024
        contract = {"camera": args.camera, "depth_scale": args.depth_scale,
                    "num_points": args.num_points, "channels": args.channels,
                    "sampling": "raster_uniform", "frame": "camera_optical"}
        project = lambda obs: points(obs, count=args.num_points, channels=args.channels,
                                     depth_scale=args.depth_scale)
    np.random.seed(args.seed)
    layout = action_layout(get_robot_action_dim_info(args.env_cfg_type), args.action_type)
    args.output.mkdir(parents=True, exist_ok=True)
    for source in sorted(args.input.glob("*.hdf5")):
        target = args.output / source.name
        if target.exists() or source.resolve() == target.resolve():
            raise FileExistsError(target)
        with h5py.File(source) as f:
            state = np.concatenate([f["state"][key + "s"][:] for key, _ in layout], axis=-1)
            action = np.concatenate([f["action"][key + "s"][:] for key, _ in layout], axis=-1)
            camera = f["vision"][args.camera]
            cloud = []
            for t in range(len(state)):
                intr = camera["intrinsic_matrix"]
                intr = intr[t] if intr.ndim == 3 else intr[:]
                obs = {"depth": camera["depths"][t], "intrinsic_matrix": intr}
                if args.channels == 6:
                    obs["color"] = decode_image_bit(camera["colors"][t])
                cloud.append(project(obs))
            with h5py.File(target.with_suffix(".partial"), "w") as out:
                for key, value in dict(
                    point_cloud=np.stack(cloud), state=state, action=action
                ).items():
                    out.create_dataset(key, data=value, compression="lzf")
                out.attrs["observation_contract"] = json.dumps(contract)
                out.attrs["source"] = str(source.resolve())
                out.attrs["sampling"] = contract["sampling"]
                out.attrs["sampling_seed"] = args.seed
                out.attrs["frame"] = "camera_optical"
                out.attrs["depth_scale"] = args.depth_scale
        target.with_suffix(".partial").replace(target)


if __name__ == "__main__":
    main()
