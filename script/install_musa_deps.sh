#!/usr/bin/env bash
set -euo pipefail


python3 -m pip install -U pip setuptools wheel
python3 -m pip install \
  torchada \
  diffusers==0.36.0 \
  easydict \
  einops \
  opencv-python \
  matplotlib \
  ftfy \
  safetensors \
  Pillow \
  tqdm \
  "imageio[ffmpeg]" \
  websockets \
  msgpack

python3 - <<'PY'
import sys
import torch
import torch_musa
import torchada

print("device_type=musa")
print("torchada", getattr(torchada, "__version__", "ok"))
print("musa_available", torch.musa.is_available())
print("musa_count", torch.musa.device_count())
print("torchada_active", "torchada" in sys.modules)
PY
