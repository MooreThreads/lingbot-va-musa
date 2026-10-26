#!/usr/bin/env bash
set -euo pipefail


export LINGBOT_DEVICE=musa
LINGBOT_ROOT=${LINGBOT_ROOT:-/workspace/lingbot-va-s5000}
export PYTHONPATH="${LINGBOT_ROOT}/wan_va:${PYTHONPATH:-}"

cd "${LINGBOT_ROOT}"

python3 - <<'PY'
import sys
import os
sys.path.insert(0, os.path.join(os.environ.get("LINGBOT_ROOT", "/workspace/lingbot-va-s5000"), "wan_va"))
from utils.musa_bootstrap import bootstrap_musa
bootstrap_musa()

import torch
from torch.nn.attention.flex_attention import flex_attention
from utils.device import (
    default_attn_mode,
    get_device_type,
    get_dist_backend,
    get_torch_device,
    is_on_accelerator,
    is_torchada_active,
    supports_flex_attention,
    use_cuda_api,
)
from modules.model import WanAttention
from diffusers import AutoencoderKLWan

print("device_type", get_device_type())
print("torchada_active", is_torchada_active())
print("use_cuda_api", use_cuda_api())
print("dist_backend", get_dist_backend())
print("supports_flex", supports_flex_attention())
print("default_attn_mode(flex)", default_attn_mode("flex"))
print("diffusers AutoencoderKLWan", AutoencoderKLWan)

device = get_torch_device(0)
attn = WanAttention(dim=128, heads=8, dim_head=16, attn_mode=default_attn_mode("torch"))
attn = attn.to(device=device, dtype=torch.bfloat16)
x = torch.randn(1, 16, 128, device=device, dtype=torch.bfloat16)
out = attn(x, x, x, rotary_emb=None)
print("attn_out", out.shape, out.dtype, out.device)
print("on_accelerator", is_on_accelerator(out))

compiled_flex = torch.compile(flex_attention, dynamic=True)
q = torch.randn(1, 4, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
flex_out = compiled_flex(q, q, q)
flex_out.float().sum().backward()
print("flex_compile_out", flex_out.shape, flex_out.device)
print("flex_backward_grad", q.grad.shape, bool(torch.isfinite(q.grad).all()))
print("SMOKE_OK")
PY
