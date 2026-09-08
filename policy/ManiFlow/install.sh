#!/usr/bin/env bash
set -euo pipefail
# Activate the intended policy environment first. Preserve installed CUDA PyTorch.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"'
python -m pip install 'hydra-core>=1.3,<1.4' 'diffusers>=0.35,<0.36' 'timm>=1.0,<1.1' einops termcolor dill h5py safetensors huggingface_hub
python -m pip install 'websockets>=14,<16' msgpack-numpy 'pydantic>=2.5,<3'
python -c 'import torch, torchvision, pytorch3d.ops'
