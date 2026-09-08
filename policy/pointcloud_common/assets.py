"""Download the upstream R3D encoder, recording its immutable revision/digest."""

import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision", help="Exact Hugging Face commit; defaults to resolving current main"
    )
    args = parser.parse_args()
    repo = "eddie-cui/r3d-weights"
    revision = args.revision or HfApi().model_info(repo).sha
    snapshot_download(
        repo,
        revision=revision,
        local_dir=args.output,
        allow_patterns=["model.safetensors", "README.md"],
    )
    weights = args.output / "model.safetensors"
    receipt = {
        "repository": repo,
        "revision": revision,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
