# LingBot-VA S5000 / MUSA Adaptation

This branch adapts LingBot-VA training and inference for Moore Threads MTT S5000 GPUs using `torch_musa`. It is an engineering compatibility branch; it does not include model weights, datasets, or a claim of upstream benchmark parity.

## Validated environment

| Item | Validated value |
|---|---|
| Accelerator | 8 x MTT S5000, 80 GiB per card |
| Driver | `5.2.0-server` |
| PyTorch / torch_musa | `2.11.0.post1` / `2.11.0.post1+b431668` |
| torchada | `0.1.86` |
| Distributed backend | MCCL |
| Training precision | BF16 |
| Production attention | PyTorch SDPA Flash -> MUSA fused attention |

## Validated data paths

| Dataset | Validation |
|---|---|
| LIBERO-LONG (`libero-long-lerobot`) | 8 cards, MBS=1, gradient accumulation 10, 5000 steps, Flash SDPA |
| RoboTwin (`robotwin-clean-and-aug-lerobot`) | 8 cards, MBS=40, 8 latent frames, 20-step clean run |

The two datasets use different shapes and batch settings; their step times and memory numbers are not directly comparable.

## LIBERO-LONG

The complete launcher, paths, and acceptance checks are documented in [`docs/libero_long_s5000.md`](docs/libero_long_s5000.md).

Core configuration:

```yaml
config: libero_train
batch_size: 1
gradient_accumulation_steps: 10
num_steps: 5000
save_interval: 200
attention: flashattn
precision: bf16
```

The validated unoptimized Flash run used approximately `24-25 s/step` and completed 5000 optimizer steps. LIBERO-LONG `libero_10` evaluation on the matching 500-episode set reached `471/500 = 94.2%` on the separate RTX 5090 evaluation machine.

## Compatibility scope

- MUSA device/runtime helpers replace scattered CUDA-only calls.
- Batch-native PyTorch SDPA Flash is the production attention path.
- MUSA MATH fallback and Flex dynamic/eager-mask controls are compatibility paths.
- `collate_fixed_latent_frames` is retained for fixed-frame datasets such as RoboTwin.
- The failed length-bucket experiment and profiling-only changes are not part of this adaptation patch.
- H1 fused RoPE, H5 block-granularity FSDP, and H6 `reshard_after_forward=0` are performance optimizations and are intentionally excluded from this adaptation branch.

## Limitations

- A successful training run is not a benchmark result.
- Flex dynamic/eager mask is slower than Flash and remains a diagnostic compatibility path.
- The final optimized training version must be delivered as a separate patch with its own authorship and matched evaluation evidence.
- The runtime image should be rebuilt and pinned before publishing release artifacts.
