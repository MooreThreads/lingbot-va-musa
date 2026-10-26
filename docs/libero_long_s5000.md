# LIBERO-LONG 在 MTT S5000 上的训练说明

本文说明如何使用本仓库的 S5000/MUSA **适配版**训练 LIBERO-LONG，并记录已经完成的训练验证环境，便于在同类摩尔线程设备上复现。

本文对应未优化的 Flash SDPA 路径，不包含 fused RoPE、FSDP block 粒度或关闭 `reshard_after_forward` 等性能优化。

本仓库还在 `robotwin-clean-and-aug-lerobot` 上完成过 8 卡 S5000 训练验证；其固定 8 latent frames、MBS=40 的配置与 LIBERO-LONG 不同，详见 [`s5000_musa_porting_report.md`](s5000_musa_porting_report.md)。

## 1. 已验证环境

| 项目 | 已验证配置 |
|---|---|
| 训练节点 | `worker36021` |
| 加速卡 | 8 × Moore Threads MTT S5000，单卡约 80 GiB |
| PyTorch | `2.11.0.post1` |
| torch_musa | `2.11.0.post1+b431668` |
| torchada | `0.1.86` |
| 训练精度 | BF16 |
| 分布式后端 | MCCL |
| Attention | PyTorch SDPA Flash → MUSA fused attention |
| 数据集 | LIBERO-LONG，LeRobot 格式 |

该环境完成过 8 卡、每卡 batch size 1、梯度累积 10、global batch 80 的 5000-step 后训练。日志到达 `step=4999` 并输出 `Training completed!`，稳态约为 24–25 s/step。

该 LIBERO-LONG 环境使用 PyTorch/torch_musa 2.11，与 `MUSA.md` 中较早的 RoboTwin 2.7.1 环境不是同一次验证，复现时不要混用两套版本号。

## 2. 仓库配置

LIBERO-LONG 训练配置已经存在：

[`wan_va/configs/va_libero_train_cfg.py`](../wan_va/configs/va_libero_train_cfg.py)

关键默认值如下：

| 参数 | 值 |
|---|---:|
| `batch_size` | `1` |
| `gradient_accumulation_steps` | `10` |
| `num_steps` | `5000` |
| `save_interval` | `200` |
| `learning_rate` | `1e-5` |
| `warmup_steps` | `10` |

实际路径和运行参数通过命令行覆盖，不需要把服务器路径写死在配置文件中。

## 3. 数据和权重

下面是已验证服务器的目录布局示例，可替换成实际挂载位置：

```text
/workspace/lingbot-va/data/libero-long-lerobot
/workspace/lingbot-va/weights/lingbot-va-base
/workspace/lingbot-va/train_out
```

数据集根目录应包含各 LeRobot 子数据集及对应的 `meta/info.json`、latent 和 action 数据。使用 classifier-free dropout 时，数据集根目录还需要 `empty_emb.pt`。

## 4. 启动训练

在包含 `torch_musa` 和 `torchada` 的容器内执行：

```bash
cd /workspace/lingbot-va-s5000

export LINGBOT_DEVICE=musa
export LINGBOT_ROOT=/workspace/lingbot-va-s5000
export CONFIG_NAME=libero_train
export NGPU=8
export MASTER_PORT=29501

bash script/run_va_posttrain_musa.sh \
  --dataset-path /workspace/lingbot-va/data/libero-long-lerobot \
  --pretrained-model-path /workspace/lingbot-va/weights/lingbot-va-base \
  --num-steps 5000 \
  --batch-size 1 \
  --gradient-accumulation-steps 10 \
  --save-interval 200 \
  --attn-mode flashattn \
  --disable-wandb \
  --save-root /workspace/lingbot-va/train_out/libero-long-flash-5000
```

`flashattn` 在本仓库中表示 PyTorch `scaled_dot_product_attention` 的 Flash backend，不是直接调用第三方 CUDA `flash-attn` 包。

## 5. 验收检查

训练不能只以进程启动或单步 loss 作为成功标准。至少检查：

1. 8 个 rank 完成分布式初始化，日志中没有 traceback、OOM 或非有限 loss/grad norm；
2. 日志连续到达 `step=4999`，并出现 `Training completed!`；
3. `checkpoint_step_5000/transformer` 包含配置文件、safetensors 权重或权重索引；
4. 最终 checkpoint 能被推理服务真实加载；
5. 如需声明任务效果，使用同一套 LIBERO `libero_10` 环境完成 10 个任务、共 500 个 episode 的评测。

已验证的未优化版本在统一 `libero_10` 评测中得到 `471/500 = 94.2%`。该评测在 NVIDIA RTX 5090 仿真机完成；S5000 用于训练，不能把评测机结果表述为 S5000 仿真执行结果。

## 6. 版本边界

- 正式 5000-step Flash 训练未启用 `collate_length_bucketed`；长度分桶属于失败探索，不应作为复现条件。
- 本适配版不包含 H1 fused RoPE、H5 FSDP block 粒度和 H6 `reshard_after_forward=0`。
- checkpoint 只保留最近一个属于交付时的存储策略，不是模型或 MUSA 适配的必要条件。
- Flex dynamic/eager BlockMask 是 LIBERO 变长序列的兼容性路径，不是本文 5000-step Flash 主路径。
