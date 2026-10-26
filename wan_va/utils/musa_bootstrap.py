# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Bootstrap torchada and MUSA-specific patches before model imports."""
from __future__ import annotations

import os
import sys
import warnings

_TORCHADA_TRIED = False


def _should_use_torchada() -> bool:
    if os.environ.get("LINGBOT_DISABLE_TORCHADA", "0") == "1":
        return False
    forced = os.environ.get("LINGBOT_DEVICE", "auto").lower()
    if forced in {"cuda", "cpu"}:
        return False
    if forced == "musa":
        return True
    try:
        import torch

        return hasattr(torch, "musa") and torch.musa.is_available()
    except ImportError:
        return False


def bootstrap_torchada() -> bool:
    """Load torchada so CUDA APIs and flex_attention work on MUSA."""
    global _TORCHADA_TRIED
    if _TORCHADA_TRIED:
        return "torchada" in sys.modules
    _TORCHADA_TRIED = True

    if not _should_use_torchada():
        return False

    try:
        import torchada  # noqa: F401
        return True
    except ImportError:
        warnings.warn(
            "MUSA platform detected but torchada is not installed; "
            "flex attention will fall back to torch SDPA.",
            stacklevel=2,
        )
        return False


def patch_diffusers_for_musa() -> None:
    import diffusers.utils.import_utils as import_utils

    import_utils._flash_attn_available = False
    import_utils._flash_attn_3_available = False


def patch_flash_attn_for_musa() -> None:
    try:
        import flash_attn
        import flash_attn.flash_attn_interface as interface
    except ImportError:
        return

    if not hasattr(flash_attn, "flash_attn_func") and hasattr(interface, "flash_attn_func"):
        flash_attn.flash_attn_func = interface.flash_attn_func
    if not hasattr(flash_attn, "flash_attn_varlen_func") and hasattr(
        interface, "flash_attn_varlen_func"
    ):
        flash_attn.flash_attn_varlen_func = interface.flash_attn_varlen_func


def bootstrap_musa() -> None:
    bootstrap_torchada()
    from .device import is_musa

    if not is_musa():
        return
    patch_flash_attn_for_musa()
    patch_diffusers_for_musa()
