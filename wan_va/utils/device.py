# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Device helpers for CUDA and Moore Threads MUSA backends."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Optional

import torch

_DEVICE_TYPE: Optional[str] = None


def _ensure_torch_musa() -> None:
    if not hasattr(torch, "musa"):
        with contextlib.suppress(ImportError):
            import torch_musa  # noqa: F401


def is_torchada_active() -> bool:
    return "torchada" in sys.modules


def use_cuda_api() -> bool:
    """Use torch.cuda.* device strings/APIs (including torchada on MUSA)."""
    return is_cuda() or (is_musa() and is_torchada_active())


def get_device_type() -> str:
    """Return ``cuda``, ``musa``, or ``cpu``."""
    global _DEVICE_TYPE
    if _DEVICE_TYPE is not None:
        return _DEVICE_TYPE

    forced = os.environ.get("LINGBOT_DEVICE", "auto").lower()
    if forced in {"cuda", "musa", "cpu"}:
        _DEVICE_TYPE = forced
        return _DEVICE_TYPE

    _ensure_torch_musa()
    if hasattr(torch, "musa") and torch.musa.is_available():
        _DEVICE_TYPE = "musa"
    elif torch.cuda.is_available():
        _DEVICE_TYPE = "cuda"
    else:
        _DEVICE_TYPE = "cpu"
    return _DEVICE_TYPE


def is_musa() -> bool:
    return get_device_type() == "musa"


def is_cuda() -> bool:
    return get_device_type() == "cuda"


def is_accelerator_available() -> bool:
    return get_device_type() in {"cuda", "musa"}


def get_torch_device(rank: int = 0) -> torch.device:
    if use_cuda_api():
        return torch.device(f"cuda:{rank}")
    if is_musa():
        return torch.device(f"musa:{rank}")
    return torch.device("cpu")


def get_device_name(rank: int = 0) -> str:
    if is_musa():
        return torch.musa.get_device_name(rank)
    if use_cuda_api():
        return torch.cuda.get_device_name(rank)
    return "cpu"


def set_device(rank: int) -> None:
    if use_cuda_api():
        torch.cuda.set_device(rank)
    elif is_musa():
        torch.musa.set_device(rank)


def empty_cache() -> None:
    if use_cuda_api():
        torch.cuda.empty_cache()
    elif is_musa():
        torch.musa.empty_cache()


def synchronize() -> None:
    if use_cuda_api():
        torch.cuda.synchronize()
    elif is_musa():
        torch.musa.synchronize()


def is_on_accelerator(obj: torch.Tensor) -> bool:
    if not torch.is_tensor(obj):
        return False
    return obj.device.type in {"cuda", "musa"}


def get_dist_backend() -> str:
    override = os.environ.get("LINGBOT_DIST_BACKEND")
    if override:
        return override
    if use_cuda_api():
        return "nccl"
    if is_musa():
        return "mccl"
    return "gloo"


def default_attn_mode(preferred: str = "torch") -> str:
    if is_musa() and not is_torchada_active() and preferred in {"flex", "flashattn"}:
        return "torch"
    return preferred


def supports_flex_attention() -> bool:
    if os.environ.get("LINGBOT_DISABLE_FLEX", "0") == "1":
        return False
    return not (is_musa() and not is_torchada_active())
