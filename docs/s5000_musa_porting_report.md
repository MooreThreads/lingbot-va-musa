# LingBot-VA 在 MTT S5000 / MUSA 上的适配与性能报告

## 1. 报告范围

本文记录 LingBot-VA 在 Moore Threads MTT S5000 / MUSA 环境中的工程适配、训练验证、推理服务验证和 RoboTwin 最小闭环结果。测试对象为 LingBot-VA 的 RoboTwin 三相机任务，目标是确认代码在 S5000 上能够完成真实数据读取、模型 forward/backward、多卡训练、推理服务启动和仿真 client 调用。

本文不把 RoboTwin smoke 当作官方成功率复现。当前服务器加载的是 `lingbot-va-base`，不是官方 RoboTwin 后训练权重 `lingbot-va-posttrain-robotwin`；RoboTwin 环境也采用 llvmpipe/software Vulkan 的最小可运行配置，不是正式 benchmark 环境。文中的 MFU 为基于静态 FLOPs 与稳态 step time 的 proxy 口径，适合内部 A/B 对比，不作为硬件绝对峰值利用率。

## 1.1 结论摘要

- LingBot-VA 训练主链路已在 8 张 MTT S5000 上完成真实 RoboTwin LeRobot 数据多步验证，当前稳定配置为 bf16、8 卡、每卡 MBS=40、`attn_mode=flashattn`、`train_latent_frames=8`。
- 关键问题不是 S5000/MUSA 不能用 Flash Attention；S5000/MUSA 可以通过 torchada/CUDA API 语义映射走底层 fused attention。真正的问题是官方默认 `flex` 路径虽然适合表达块级因果/window mask，但在当前 MUSA 大 batch 训练中存在编译、mask 构建和旧 batch flatten 布局带来的显存压力。
- 本 MR 将训练主路径切到 batch-native PyTorch SDPA Flash：保留 `[B, tokens, dim]`，按单样本构建共享 mask，通过 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` 使用 MUSA fused attention，避免旧 `[1, B × tokens]` 展平布局造成的 mask 膨胀。
- 8 卡 MBS40 的 20-step clean run 已通过，step1-19 mean 为 `28.211512s`，global batch size 为 `320`，吞吐约 `11.34 samples/s`，MFU proxy 约 `18.42%`，峰值显存约 `75.82GiB/卡`。
- 推理服务可在 S5000 单卡启动。默认 `25/50` sampling 的 server infer-only 为 `11.390s/chunk`，低步数 `5/10` 为 `2.446s/chunk`；按完整 32 个低层动作摊销分别约 `356ms/action` 和 `76ms/action`。
- 推理侧块级因果 KV-cache 闭环仍保留，RoboTwin/Libero client 会在每个 chunk 执行后通过 `compute_kv_cache=True` 更新历史 K/V；该机制对长 horizon、多 chunk rollout 有收益，但不能消除每个 chunk 内多步 sampling 的主要耗时。
- RoboTwin 已完成 10-step 最小 policy rollout smoke，验证环境初始化、观测获取、websocket infer、动作执行和视频输出链路；该结果只代表链路可运行，不代表任务成功率。

## 2. 系统与运行配置

| 项目 | 当前配置 |
|---|---|
| 服务器 / 容器 | `s5000` / `lingbot-va-211docker` |
| 容器来源 | 参考 `dreamzero-211`，`--privileged --network host --shm-size=64g -v /data:/workspace` |
| 加速卡 | MTT S5000，最多 8 卡可见 |
| 代码目录 | `/workspace/lingbot-va-s5000` |
| 模型目录 | `/workspace/lingbot-va/models/lingbot-va-base` |
| 数据集目录 | `/workspace/lingbot-va/datasets/robotwin-clean-and-aug-lerobot` |
| RoboTwin 工程 | `/workspace/lingbot-va-s5000/third_party/RoboTwin` |
| 渲染 smoke | SAPIEN + Mesa llvmpipe software Vulkan |
| GitLab 分支 | `s5000-musa-robotwin-support` |

当前运行约束：

- GPU7 需要保留给 dagger，因此 LingBot-VA 推理和 RoboTwin client 验证不占用 GPU7。
- 当前只同步了 `lingbot-va-base`，尚未同步 `lingbot-va-posttrain-robotwin`。
- RoboTwin 第三方工程只做最小可运行补齐，`curobo/open3d/pytorch3d` 等依赖不是完整官方评测状态。

## 2.1 配置归属与实验含义

| 配置 | 当前值 | 含义 |
|---|---|---|
| `LINGBOT_DEVICE` | `musa` | 显式选择 MUSA 设备路径。 |
| `attn_mode` | `flashattn` | 当前训练主路径，实际为 PyTorch SDPA Flash -> MUSA fused attention。 |
| `train_latent_frames` | `8` | 固定 latent/action 帧数，保证 batch 内 shape 可 collate。 |
| per-rank MBS | `40` | 当前 8 卡已验证稳定基线。 |
| `LINGBOT_FLASH_LEGACY_FLATTEN` | 默认关闭 | 关闭旧 `[1, B × tokens]` 展平路径；仅用于回退和一致性验证。 |
| `LINGBOT_INFER_ATTN_MODE` | `flashattn` | 推理服务 attention 后端覆盖项。 |
| `LINGBOT_VIDEO_STEPS` / `LINGBOT_ACTION_STEPS` | 可选 | 推理 sampling steps 覆盖项，用于延迟/质量 trade-off。 |
| `ROBOTWIN_ROOT` | 外部 RoboTwin 路径 | 避免 client 硬编码 `/path/to/your/robowin`。 |
| `LINGBOT_SAPIEN_SHADER` | `default` for smoke | llvmpipe/software Vulkan 下避免使用官方 `rt` shader。 |

## 3. 关键问题与代码适配

### 3.1 MUSA device 与 bootstrap

原代码多处默认使用 `torch.cuda.*`，在 S5000 上会遇到 device 字符串、cache 清理、分布式 backend 和第三方依赖探测不一致的问题。本 MR 新增：

- `wan_va/utils/device.py`：统一 `cuda/musa/cpu` device、`empty_cache()`、`synchronize()`、device count/name 和 distributed backend；
- `wan_va/utils/musa_bootstrap.py`：在 MUSA 环境下优先导入 torchada，并对 diffusers / flash-attn 可用性做最小兼容；
- `wan_va/distributed/util.py`、`wan_va/distributed/fsdp.py`、`wan_va/wan_va_server.py`：将散落的 CUDA-only 调用改为设备无关接口。

处理后，训练和推理服务可以在 torchada/MUSA 栈上使用 `cuda:` API 语义映射，也可以在必要时回退到 `musa:` 或 CPU 路径。

### 3.2 Attention 后端：问题不在“能不能 Flash”，而在 mask 与 layout

官方 LingBot-VA 训练默认使用 `flex` attention，这是合理的：模型中的 video/action token 需要块级因果、window 和条件 token 约束，FlexAttention 的 `BlockMask` 适合表达这类结构化 mask。

本次没有继续沿用 `flex` 作为大 batch 主路径，原因集中在工程实现层面：

1. 当前 MUSA/torchada 下 `flex` 有编译链路和稳定性成本；
2. 旧训练路径会把 batch 展平为 `[1, B × tokens, dim]`；
3. 大 batch 下 attention mask 和中间张量随 `B × tokens` 快速膨胀；
4. MBS32/MBS40 时，继续沿用旧布局会先受到 mask 显存、构建开销和编译稳定性的限制。

因此，本 MR 采用 batch-native SDPA Flash 路径，而不是简单替换成第三方 `flash_attn_func`：

```text
hidden states: [B, S, C] 保持 batch 维
attention mask: 按单样本构建并对 batch 共享
backend: PyTorch SDPA + SDPBackend.FLASH_ATTENTION -> MUSA fused attention
```

当前 `attn_mode=flashattn` 对应的实际调用为：

```python
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
```

这不是 Dao-AILab `flash-attn` 包的直接 kernel 调用，但在 S5000/MUSA 上能够落到 fused attention 路径，并解决当前大 batch 训练最突出的 mask/layout 问题。代码仍保留 `flash_attn_interface` / `flash_attn` 导入兼容和 `LINGBOT_FLASH_LEGACY_FLATTEN=1` 回退开关。

### 3.3 Batch-native 训练路径

新增 batch-native flash 训练路径后，`forward_train()` 不再默认把 batch 和 sequence 展成单条超长序列。相应变化包括：

- hidden states、text states、time embedding、RoPE/grid 信息保留 batch 维；
- dense attention mask 按单样本 sequence 构建并对 batch 共享；
- 与 legacy flatten 路径保持可回退和可对照；
- 新增 `script/validate_batch_native_flash.py` 验证 batch-native 与 legacy flatten 输出/梯度一致性。

该改动的目标不是改变模型语义，而是降低大 batch 下 mask 与中间张量规模，使 MBS32/MBS40 成为可运行配置。

### 3.4 数据加载与训练入口

RoboTwin LeRobot 数据中的 latent/action 帧数需要在 batch 内对齐。本 MR 增加 `collate_fixed_latent_frames`，将样本裁剪或 padding 到固定帧数，并在 RoboTwin 训练配置中设置 `train_latent_frames=8`。

训练入口增加常用覆盖项，避免改配置文件即可做 smoke 和性能实验：

- `--dataset-path`
- `--pretrained-model-path`
- `--batch-size`
- `--num-steps`
- `--load-worker`
- `--dataset-init-worker`
- `--gradient-accumulation-steps`
- `--attn-mode`
- `--disable-wandb`
- `--save-root`

### 3.5 推理服务参数化与 KV-cache 闭环

推理服务新增环境变量覆盖：

- `LINGBOT_MODEL_PATH`：切换 base 或后训练 checkpoint；
- `LINGBOT_INFER_ATTN_MODE`：切换推理 attention 后端；
- `LINGBOT_VIDEO_STEPS` / `LINGBOT_ACTION_STEPS`：调整 flow matching / diffusion sampling 迭代次数。

LingBot-VA 原有块级因果 KV-cache 闭环仍保留：reset 时预分配 cache，普通 chunk 生成动作，client 执行动作后通过 `compute_kv_cache=True` 回传 key frames 与已执行 action，server 再将真实观测/action 的 K/V 写入历史 cache。该机制能减少长 horizon 中历史上下文的重复计算，但不能消除每个 chunk 内多步 sampling 的主要耗时。

### 3.6 RoboTwin 最小闭环

RoboTwin 代码只做最小复现适配：

- `ROBOTWIN_ROOT` 替换硬编码路径；
- `LINGBOT_SAPIEN_SHADER` 允许在 llvmpipe/software Vulkan 环境中使用 `default` shader；
- client 能连接 LingBot-VA websocket server 并完成 10-step policy rollout smoke。

这里的目标是证明“推理服务能被仿真 client 调用”，不是提交正式 RoboTwin benchmark。

## 4. 验证结果

### 4.1 训练主链路

已验证以下链路可执行：

- 8 卡分布式初始化；
- 真实 RoboTwin LeRobot 数据读取；
- batch-native flash attention forward；
- backward；
- optimizer step 与 scheduler step；
- 多步 loss 正常输出；
- 性能日志和 MFU proxy 汇总。

当前稳定配置：

```bash
NGPU=8 bash script/run_va_posttrain_musa.sh \
  --dataset-path /workspace/lingbot-va/datasets/robotwin-clean-and-aug-lerobot \
  --pretrained-model-path /workspace/lingbot-va/models/lingbot-va-base \
  --batch-size 40 \
  --num-steps 20 \
  --attn-mode flashattn \
  --disable-wandb \
  --save-root /workspace/lingbot-va/train_out/robotwin-8gpu-mbs40-batch-native-flash-20step-clean-20260827
```

8 卡 MBS40 结果：

| 指标 | 数值 |
|---|---:|
| per-rank MBS | `40` |
| global batch size | `320` |
| 有效统计 step | `1-19` |
| mean step time | `28.211512s` |
| 吞吐 | `11.34 samples/s` |
| MFU proxy | `18.42%` |
| 峰值显存 | `75.82GiB/卡` |

MBS 探索：

| 实验 | 结果 | 结论 |
|---|---|---|
| 7 卡 MBS32 batch-native flash | 成功 | 稳态 step time 约 `27.12s`，MFU proxy 约 `15.33%`。 |
| 7 卡 MBS40 batch-native flash | 单步成功 | 显存边界探索。 |
| 8 卡 MBS40 batch-native flash | 20 step 成功 | 当前稳定基线。 |
| 8 卡 MBS42 / MBS44 | NaN | 主要风险转为数值稳定性，不建议作为默认配置。 |

### 4.2 Attention 一致性

`script/validate_batch_native_flash.py` 已用于对比 batch-native flash 与 legacy flatten 路径：

- self-attention 输出/梯度近似一致；
- cross-attention 输出/梯度近似一致；
- 误差在 bf16 数值范围内，可用于确认 layout 改造没有改变 attention 语义。

### 4.3 推理服务

已启动并验证两个单卡推理服务：

| 端口 | GPU | sampling steps | 用途 |
|---:|---:|---|---|
| 29056 | GPU0 | video 25 / action 50 | 默认质量配置 |
| 29057 | GPU1 | video 5 / action 10 | 低延迟验证配置 |

本次推理计时采用 timing-only 口径：每次正式测量前 reset streaming cache，reset 时间单独记录，不计入 infer；统计字段优先使用 websocket server 返回的 `server_timing.infer_ms`。

| 配置 | server infer-only 平均 | 中位数 | 完整 chunk ms/action | 完整 chunk 频率 | 首 chunk ms/action | 首 chunk 频率 |
|---|---:|---:|---:|---:|---:|---:|
| video 25 / action 50 | `11.390s/chunk` | `11.444s/chunk` | `356ms` | `2.81Hz` | `712ms` | `1.40Hz` |
| video 5 / action 10 | `2.446s/chunk` | `2.445s/chunk` | `76ms` | `13.08Hz` | `153ms` | `6.54Hz` |

单次 `infer(obs)` 返回 action shape 为 `[16, 2, 16]`，完整 chunk 包含 `32` 个低层动作，每个动作 16 维。RoboTwin 数据集 timestamp 间隔为 `0.02s`，目标控制频率为 `50Hz`，即约 `20ms/action`；当前同步推理仍低于实时目标，主要瓶颈是每个 chunk 内多步 flow matching / diffusion sampling。

原始 timing 文件：

- `docs/perf/base_v25_a50_flashattn_gpu0_reset_each.json`
- `docs/perf/base_v5_a10_flashattn_gpu1_reset_each.json`

### 4.4 RoboTwin smoke

已完成 `press_stapler` 10-step policy rollout smoke，覆盖：

- SAPIEN 环境初始化；
- 观测获取与格式转换；
- websocket policy infer；
- 动作执行；
- 视频输出。

该 smoke 只验证功能链路，不作为成功率指标。正式成功率复现需要同步 `lingbot-va-posttrain-robotwin`，使用完整 250-step、官方 seed/config 和完整硬件渲染环境。

## 5. 当前瓶颈与风险

| 问题 | 当前判断 | 后续方向 |
|---|---|---|
| 推理实时性 | 默认 `25/50` 约 `356ms/action`，低步数 `5/10` 约 `76ms/action`，均低于 `20ms/action` 目标。 | 优先验证异步执行、减少 sampling steps、优化 action chunk 消费策略。 |
| Batch 上限 | MBS40 稳定，MBS42/MBS44 出现 NaN。 | MBS40 作为当前基线；MBS42+ 单独做数值诊断。 |
| Attention mask | batch-native 已避免全局 flatten，但 `torch/flashattn` 路径仍使用 dense mask。 | 按样本/分段构建更紧凑 mask，或接入结构化 mask 后端。 |
| RoboTwin 环境 | 当前为 llvmpipe 最小 smoke 环境。 | 正式 rollout 切到完整硬件渲染环境。 |
| 模型权重 | 当前为 `lingbot-va-base`。 | 正式成功率复现需同步后训练权重。 |
| MFU | 当前为 proxy 估算。 | 继续用于内部 A/B，不作为硬件绝对峰值结论。 |

## 6. 复现入口

基础环境：

```bash
bash script/install_musa_deps.sh
export LINGBOT_DEVICE=musa
bash script/run_musa_smoke.sh
```

训练：

```bash
NGPU=8 bash script/run_va_posttrain_musa.sh \
  --dataset-path /workspace/lingbot-va/datasets/robotwin-clean-and-aug-lerobot \
  --pretrained-model-path /workspace/lingbot-va/models/lingbot-va-base \
  --batch-size 40 \
  --num-steps 20 \
  --attn-mode flashattn \
  --disable-wandb
```

推理服务：

```bash
export LINGBOT_MODEL_PATH=/workspace/lingbot-va/models/lingbot-va-base
export LINGBOT_INFER_ATTN_MODE=flashattn
bash script/run_launch_va_server_sync.sh
```

RoboTwin smoke：

```bash
export ROBOTWIN_ROOT=/workspace/lingbot-va-s5000/third_party/RoboTwin
export LINGBOT_SAPIEN_SHADER=default
```

## 7. 合并建议

当前 MR 适合作为 LingBot-VA 的 S5000/MUSA 工程适配版合并：训练、推理服务和最小 RoboTwin client 闭环均已验证，关键性能数据已落盘。合并时建议保持当前边界：不声明官方 RoboTwin 成功率复现，不把 llvmpipe smoke 作为正式 rollout benchmark，不把 MFU proxy 写成硬件绝对利用率。
