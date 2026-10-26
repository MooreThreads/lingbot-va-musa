#!/usr/bin/env python3

import argparse
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def flash_attention(query, key, value, mask):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
        ).transpose(1, 2)


def stream_major_permutation(batch_size, stream_lengths, device):
    sequence_length = sum(stream_lengths)
    indices = torch.arange(batch_size * sequence_length, device=device).reshape(
        batch_size, sequence_length
    )
    streams = indices.split(stream_lengths, dim=1)
    return torch.cat([stream.reshape(-1) for stream in streams])


def block_diagonal_mask(base_mask, batch_size):
    sequence_length = base_mask.shape[-2]
    key_length = base_mask.shape[-1]
    mask = torch.zeros(
        1,
        1,
        batch_size * sequence_length,
        batch_size * key_length,
        dtype=torch.bool,
        device=base_mask.device,
    )
    for batch_index in range(batch_size):
        query_slice = slice(batch_index * sequence_length, (batch_index + 1) * sequence_length)
        key_slice = slice(batch_index * key_length, (batch_index + 1) * key_length)
        mask[:, :, query_slice, key_slice] = base_mask
    return mask


def compare_attention(
    name,
    query,
    key,
    value,
    base_mask,
    query_permutation,
    key_permutation,
):
    batch_size, query_length, _, _ = query.shape
    key_length = key.shape[1]
    query_native = query.detach().clone().requires_grad_(True)
    key_native = key.detach().clone().requires_grad_(True)
    value_native = value.detach().clone().requires_grad_(True)
    query_legacy = query.detach().clone().requires_grad_(True)
    key_legacy = key.detach().clone().requires_grad_(True)
    value_legacy = value.detach().clone().requires_grad_(True)

    native_output = flash_attention(
        query_native,
        key_native,
        value_native,
        base_mask,
    )

    legacy_mask = block_diagonal_mask(base_mask, batch_size)
    legacy_mask = legacy_mask[:, :, query_permutation]
    legacy_mask = legacy_mask[:, :, :, key_permutation]
    legacy_query = query_legacy.reshape(batch_size * query_length, *query.shape[2:])[
        query_permutation
    ][None]
    legacy_key = key_legacy.reshape(batch_size * key_length, *key.shape[2:])[
        key_permutation
    ][None]
    legacy_value = value_legacy.reshape(batch_size * key_length, *value.shape[2:])[
        key_permutation
    ][None]
    legacy_output = flash_attention(
        legacy_query,
        legacy_key,
        legacy_value,
        legacy_mask,
    )[0]
    inverse_query_permutation = torch.argsort(query_permutation)
    legacy_output = legacy_output[inverse_query_permutation].reshape_as(native_output)

    output_gradient = torch.randn_like(native_output)
    (native_output.float() * output_gradient.float()).sum().backward()
    (legacy_output.float() * output_gradient.float()).sum().backward()

    def metrics(left, right):
        difference = (left.float() - right.float()).abs()
        denominator = torch.linalg.vector_norm(left.float()).clamp_min(1e-12)
        return {
            "max_abs": difference.max().item(),
            "mean_abs": difference.mean().item(),
            "relative_l2": (
                torch.linalg.vector_norm(difference) / denominator
            ).item(),
        }

    return {
        "name": name,
        "output": metrics(native_output, legacy_output),
        "query_grad": metrics(query_native.grad, query_legacy.grad),
        "key_grad": metrics(key_native.grad, key_legacy.grad),
        "value_grad": metrics(value_native.grad, value_legacy.grad),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("musa:0")
    dtype = torch.bfloat16
    stream_lengths = [960, 960, 128, 128]
    sequence_length = sum(stream_lengths)
    text_length = 512
    query_permutation = stream_major_permutation(
        args.batch_size, stream_lengths, device
    )
    text_permutation = torch.arange(
        args.batch_size * text_length, device=device
    )

    query = torch.randn(
        args.batch_size,
        sequence_length,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    token_ids = torch.arange(sequence_length, device=device)
    self_mask = (
        (token_ids[:, None] - token_ids[None, :]).abs() <= 256
    )[None, None]

    self_result = compare_attention(
        "self_attention",
        query,
        key,
        value,
        self_mask,
        query_permutation,
        query_permutation,
    )
    del key, value
    torch.musa.empty_cache()

    text_key = torch.randn(
        args.batch_size,
        text_length,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    text_value = torch.randn_like(text_key)
    cross_mask = torch.ones(
        1,
        1,
        sequence_length,
        text_length,
        device=device,
        dtype=torch.bool,
    )
    cross_result = compare_attention(
        "cross_attention",
        query,
        text_key,
        text_value,
        cross_mask,
        query_permutation,
        text_permutation,
    )
    result = {
        "backend": "SDPBackend.FLASH_ATTENTION",
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "stream_lengths": stream_lengths,
        "sequence_length": sequence_length,
        "text_length": text_length,
        "self_attention": self_result,
        "cross_attention": cross_result,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
