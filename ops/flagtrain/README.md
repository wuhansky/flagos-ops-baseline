# FlagTrain 基线采集模块（ops/flagtrain）

每个模块对应 FlagTrain benchmark 的一个算子，采集的是 **benchmark 在 NVIDIA 上
使用的基线实现**，不是 FlagTrain 自己的 Triton 实现。每个模块像 `ops/vllm/` 一样
自包含：只依赖 torch / deepspeed / megatron，**不 import `flag_train` 或
`benchmark`**。shape、dtype、输入构造、参考实现和基线选择都复制在模块内（来源
注释在每个文件顶部），因此 benchmark 改动时需要同步对应的模块。

**DeepSpeed / Megatron 是硬依赖**：对应模块在 `native()` 中加载失败时直接抛错，
采集器不会收集该算子的基线，也不会回退到 torch 组合。

## 前置条件

- **deepspeed**：lamb / lion / mixed_moe_gemm / blocked_flash / evoformer_attn /
  wf6af16_linear；
- **megatron-core**：quick_gelu / quick_gelu_backward；
- packbits / unpackbits 只用 torch（DeepSpeed 的 pack_bits/unpack_bits 是 XPU-only
  且已在上游删除，没有可用的原生基线）；
- **NCU 权限**：否则 `num_kernels=0`、`bottle_neck_unit=None`，只能得到延迟。

## 模块与基线

| 文件 | OP_NAME | 基线实现 |
|---|---|---|
| `lamb.py` | `lamb` | DeepSpeed `fused_lamb`（必需） |
| `lion.py` | `multi_tensor_lion` | DeepSpeed FusedLion（必需） |
| `mixed_moe_gemm.py` | `mixed_moe_gemm` | DeepSpeed `cutlass_ops.mixed_moe_gemm`（必需） |
| `wf6af16_linear.py` | `wf6af16_linear` | DeepSpeed W6A16（必需；仅 compute capability 8.0） |
| `packbits.py` | `packbits` | `ref_packbits`（torch 原语组合） |
| `unpackbits.py` | `unpackbits` | `unpackbits_ref`（torch 原语组合） |
| `blocked_flash.py` | `blocked_flash` | DeepSpeed `BlockedFlashAttn`（必需） |
| `evoformer_attn.py` | `evoformer_attn` | DeepSpeed `DS4Sci_EvoformerAttention`（必需） |
| `quick_gelu.py` | `quick_gelu` | Megatron `quick_gelu`（必需） |
| `quick_gelu_backward.py` | `quick_gelu_backward` | Megatron `quick_gelu` 的 backward（必需）；forward 在计时区外 |

`mixed_moe_gemm` 用 DeepSpeed 的 pybind binding 而不是 `MixedMoEGEMM` 包装类
（0.19.8 的包装类把 8 个参数传给了 6 参数的 `moe_gemm`）。混合内核的权重字节序
是 CUTLASS 内部的 interleave，DeepSpeed 没有公开的预处理入口；采集器只测延迟和
NCU 指标、不比对数值，所以模块只保证逻辑 shape 与 packed footprint 一致，数值
正确性以 FlagTrain benchmark 的 torch 组合为准。

`quick_gelu` / `quick_gelu_backward` 在导入 Megatron 前调用
`megatron.core.jit.disable_jit_fuser()`，避免 `torch.compile` 生成的 Inductor
kernel 混入基线；FlagTrain 的测试和 benchmark 也做同样处理。

文件名与算子名一致（参照 `lamb.py`）。DeepSpeed/Megatron 在**首次**
`native()`/`build_inputs()` 时才 import：ops 发现阶段不会触发它们的 JIT，也不会
出现 `evoformer_attn.so` 之类的顶层扩展模块遮住同名 ops 文件。加载失败时直接抛错。

## 运行

```bash
cd /root/wuhan/flagos-ops-baseline

# 先检查 native 解析与输入构造
python tools/collect_baseline_nvidia.py --check --repo flagtrain \
  --ops multi_tensor_lion,mixed_moe_gemm,wf6af16_linear,packbits,unpackbits,blocked_flash,evoformer_attn,quick_gelu,quick_gelu_backward

# 完整采集
python tools/collect_baseline_nvidia.py --repo flagtrain \
  --ops multi_tensor_lion,mixed_moe_gemm,wf6af16_linear,packbits,unpackbits,blocked_flash,evoformer_attn,quick_gelu,quick_gelu_backward \
  --output op_flagtrain_perf_baseline.json \
  --report-dir ncu_reports \
  --reference-chip H800
```

`--reference-chip` 必须是采集所在的芯片（并存在于 `hardware_specs.json`）；
H20 目前不在表里，需要先补规格再作为 reference。

## 形状键与匹配

- **简单算子**（lamb / wf6af16_linear / packbits / unpackbits / blocked_flash /
  evoformer_attn / quick_gelu）：`key_shape` 是第一个张量的 shape（例如
  `[4096]`、`[128, 1024]`），与 FlagTrain 运行时的直接查找一致。
- **lion**：`(n_tensors, numel)` 中多个 case 共享第一个张量 `[numel]`，直接键会撞；
  使用语义键 `n8_numel1024`，并在 `config()` 中列出 3n 个张量形状，运行时用张量
  签名回退匹配。
- **mixed_moe_gemm**：同一 GEMM shape 下 `num_bits`（4/8）和 `activation`（6/2）
  不同，但第一个张量 shape 相同；使用语义键
  `1x128x2048x2048_bits4_act6`，`config()` 列出全部张量形状和两个标量，运行时
  先按签名、再用标量区分。
- **文件名**：与 OP_NAME 一致，参照 `lamb.py`。DeepSpeed/Megatron 在
  `native()`/`build_inputs()` 第一次调用时才 import，ops 发现阶段不会触发它们的
  JIT，因此不会出现 `evoformer_attn.so` 之类的顶层扩展模块遮住同名 ops 文件。

## 采集后的有效性检查

```text
__environment__.hardware.gpu.name 与 _scaling_factors.reference_chip 一致
每条记录 num_kernels > 0、bottle_neck_unit 非空
per_kernel 的名称来自预期的基线（不是 triton_poi_fused_* 之类的 Inductor kernel）
ncu_reports/ 下有对应 .ncu-rep
shape 键覆盖 benchmark 的 core shapes
```

`quick_gelu` 的 backward 记录只应包含 backward kernel；forward 在
`build_inputs` 中执行，位于 NCU range 和 do_bench 计时区之外。
