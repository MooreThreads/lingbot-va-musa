# LingBot-VA on Moore Threads MUSA / S5000

本仓库在 MTT S5000 上验证过两套 LeRobot 数据：

| 数据集 | 已验证训练配置 | 详细说明 |
|---|---|---|
| LIBERO-LONG (`libero-long-lerobot`) | 8 卡、MBS=1、梯度累积=10、5000 steps、Flash SDPA | [`docs/libero_long_s5000.md`](docs/libero_long_s5000.md) |
| RoboTwin (`robotwin-clean-and-aug-lerobot`) | 8 卡、MBS=40、固定 8 latent frames、Flash SDPA、20-step clean run | [`docs/s5000_musa_porting_report.md`](docs/s5000_musa_porting_report.md) |

两套验证的数据形状、batch size 和验收范围不同，不能直接比较 step time 或显存结果。

## 环境

`embodied-keep` 基于 **training-suite v2.1.4-rc1-4.3.5 (musa-train)**：

- Python 3.10
- PyTorch 2.7.1 + torch_musa
- **torchada**（CUDA API 兼容层，flex_attention 必需）
- 8× MTT S5000

## 安装依赖

```bash
docker exec -it embodied-keep bash
cd /workspace/lingbot-va
bash script/install_musa_deps.sh
```

## 冒烟测试

```bash
export LINGBOT_DEVICE=musa
bash script/run_musa_smoke.sh
```

看到 `SMOKE_OK` 且 `flex_compile_out` 在 `musa:0` 上成功，即表示 torchada + flex attention 通路正常。

## Flex Attention

MUSA 上 **不能** 直接用裸 `torch_musa` + `musa:` 设备跑 flex attention。

正确方式：

1. 安装并 `import torchada`（代码已在 `bootstrap_musa()` 里自动完成）
2. 使用 `cuda:0` 设备字符串（由 torchada 映射到 MUSA）
3. 训练时使用 `attn_mode="flex"` + `torch.compile(flex_attention)`（LingBot-VA 默认即是）

未安装 torchada 时，会自动降级为 `torch` attention（SDPA）。

## 推理 / 服务

```bash
export LINGBOT_DEVICE=musa
export PYTHONPATH=/workspace/lingbot-va/wan_va:$PYTHONPATH
python3 wan_va/wan_va_server.py --config-name va_demo_cfg
```

模型 `transformer/config.json` 里推理请用 **`attn_mode: "torch"`** 或 `"flashattn"`，不要用 `"flex"`。

## 训练

```bash
export LINGBOT_DEVICE=musa
bash script/run_va_posttrain_musa.sh
```

分布式后端默认 `nccl`（torchada 会自动映射到 MCCL）。

## 环境变量

| 变量 | 说明 |
|---|---|
| `LINGBOT_DEVICE=musa` | 强制使用 MUSA |
| `LINGBOT_DISABLE_TORCHADA=1` | 禁用 torchada（flex 将降级为 torch） |
| `LINGBOT_DIST_BACKEND=nccl` | 分布式后端（torchada 下推荐 nccl） |
| `LINGBOT_DISABLE_FLEX=1` | 禁用 flex attention 编译 |

## 注意

- 代码通过 `cuda:` 设备字符串 + torchada 运行在 MUSA 上
- 容器内通常无法访问 GitHub，代码需从本机打包同步
- 预训练权重需提前放到 `/workspace` 或 PVC 挂载路径
