# Weight Reordering：Vela 权重重排机制详解

> 本文分析 NXP `ethos-u-vela` 图编译器中的 Weight Reordering（权重重排），重点说明它要解决的问题、Vela 的处理流程、典型模型 Case，以及对自研 NPU 图编译器的启发。
>
> **核心结论：** Weight Reordering 不改变卷积的数学语义，而是把模型中的通用权重布局转换成目标 NPU 的 Weight Stream 布局，使硬件能够按照自己的数据通路、Block 组织和 Weight Decoder 规则读取权重。

---

## 1. Weight Reordering 要解决什么问题？

TensorFlow Lite 模型中的权重布局服务于通用模型表示；NPU 的 MAC 阵列、Weight Decoder、SRAM Bank 和 DMA 则有自己的数据组织要求。两者通常不完全一致。

因此，Vela 需要在编译阶段完成权重布局转换：

- 权重数值保持不变（除量化 Zero Point 校正等编码准备外）。
- 权重在内存中的排列顺序可以改变。
- 可能需要为硬件 Block 补充 Padding。
- 最终生成 NPU 能够消费的 Weight Stream，必要时还会进行压缩。

### 1.1 数值与布局是两回事

假设原始权重为：

```text
W0, W1, W2, W3, W4, W5
```

重排后可能成为：

```text
W0, W3, W1, W4, W2, W5
```

这只是示意顺序。真实顺序由目标 NPU 的硬件格式、Block 参数和遍历策略决定。

重排必须满足：硬件按照新格式解释这些数据时，计算结果与原始模型一致。

### 1.2 为什么不能直接使用 TFLite 原始布局？

NPU 并不是简单地把权重数组逐个交给 MAC。典型数据路径包含：

```text
External Memory
      |
      v
     DMA
      |
      v
 Weight Stream
      |
      v
 Weight Decoder
      |
      v
  MAC Array <---- IFM
      |
      v
     OFM
```

Weight Decoder 可能要求：

- 按输入通道微块（IFM ublock）组织权重；
- 按输出通道微块（OFM ublock）组织权重；
- 按特定 Kernel 子块顺序组织；
- 对不完整 Block 补零；
- 满足硬件规定的编码、对齐和 Stream 格式。

因此，权重重排是模型格式与硬件数据通路之间的适配步骤。

---

## 2. 从 TFLite 原始权重布局开始

### 2.1 Conv2D 的 OHWI 布局

TFLite Conv2D Filter 通常采用 OHWI：

| 维度 | 含义 |
|---|---|
| O | Output Channels，输出通道 |
| H | Kernel Height，卷积核高度 |
| W | Kernel Width，卷积核宽度 |
| I | Input Channels，输入通道 |

举例：1×1 卷积，输入通道 2，输出通道 3。

```text
O0: a, b
O1: c, d
O2: e, f
```

按照 OHWI 展平后：

```text
a, b, c, d, e, f
```

这里每个输出通道对应一组输入通道权重。

卷积计算可写为：

\[
Y_o = \sum_{h,w,i} X_{h,w,i} \times W_{o,h,w,i}
\]

### 2.2 Vela 内部先做 OHWI → HWIO

在 Vela 的 TFLite Reader 中，普通 Conv2D、DepthwiseConv2D 等权重会被转换成 Vela 内部使用的维度顺序。

典型逻辑（以对应版本源码为准）：

```python
if op.type.is_depthwise_conv2d_op() or \
   op.type.is_conv2d_op() or \
   op.type == Op.FullyConnected:

    if inputs[1].values is not None:
        if op.type == Op.FullyConnected:
            inputs[1] = clone_and_reshape_tensor(
                inputs[1], (1, 0), False
            )
        else:
            inputs[1] = clone_and_reshape_tensor(
                inputs[1], (1, 2, 3, 0), False
            )
```

`clone_and_reshape_tensor()` 中的 transpose 将普通四维卷积权重从：

\[
[O,H,W,I] \rightarrow [H,W,I,O]
\]

也就是：

```text
OHWI -> HWIO
```

对于前面的例子，转置后按 HWIO 展平可以示意为：

```text
a, c, e, b, d, f
```

**重要：这只是 Vela 内部 Tensor 维度转换，不等于最终的 Ethos-U Weight Stream 重排。**

应区分两个层次：

1. TFLite Reader 的维度转换：让 Vela 内部采用统一的 Tensor 维度约定。
2. Weight Encoder 的硬件重排：根据 NPU Block、Traversal 和 Weight Stream 格式生成最终权重数据。

---

## 3. NPU 为什么需要 Block 化的权重布局？

### 3.1 输入通道和输出通道通常按硬件微块处理

卷积计算需要对输入通道进行累加。假设 IFM 有 64 个通道：

```text
C0 C1 C2 C3 ... C63
```

硬件可能以固定数量的通道为一个处理单元，例如示意性地每 8 个或 16 个通道组织一组数据。实际数值取决于具体 Ethos-U 型号、位宽和硬件配置。

```text
Block 0: C0  ... C7
Block 1: C8  ... C15
Block 2: C16 ... C23
...
```

权重布局必须与当前 IFM Block、OFM Block、Kernel 位置的计算关系相匹配。

### 3.2 Padding

假设某个硬件处理单元要求 8 个通道，而模型只有 5 个输入通道：

```text
原始权重：W0 W1 W2 W3 W4
补齐后：  W0 W1 W2 W3 W4  0  0  0
```

新增的零权重不改变结果：

\[
\sum_{i=0}^{4} X_i W_i +
X_5 \cdot 0 + X_6 \cdot 0 + X_7 \cdot 0
=
\sum_{i=0}^{4} X_i W_i
\]

Padding 让硬件能够使用规则的 Block 处理数据，但会增加编码前的权重体积。

### 3.3 重排与压缩不是同一件事

- **Reordering**：改变权重排列顺序。
- **Padding**：补齐硬件所需的 Block 或对齐空间。
- **Compression**：将重排后的权重编码成更紧凑的 Weight Stream。

三者可以在同一个编码流程中连续发生，但概念不同。

---

## 4. 两种重要的权重遍历策略

Vela 的 Weight Compressor 会为适用的卷积选择硬件遍历策略。常见选项包括：

1. `DEPTH_FIRST`
2. `PART_KERNEL_FIRST`

它们描述的是权重在硬件 Weight Stream 中的组织/遍历优先级，不是两种不同的卷积数学算法。

### 4.1 DEPTH_FIRST

可以把它理解为优先围绕 IFM Depth（输入通道深度）Block 组织权重，再组织 Kernel 和 OFM 相关数据。

示意：

```text
IFM Depth Block 0
  -> 对应 Kernel / OFM 权重

IFM Depth Block 1
  -> 对应 Kernel / OFM 权重

IFM Depth Block 2
  -> 对应 Kernel / OFM 权重
```

当输入通道较多且能较好填满硬件 IFM Block 时，Depth First 可能减少通道方向的填充浪费。

### 4.2 PART_KERNEL_FIRST

当输入通道较少时，若仍按较大的 IFM Depth Block 组织，可能产生较多无效填充。Part Kernel First 会改变 Kernel 子块与深度方向的组织优先级。

示意：

```text
Kernel 子块 0
  -> 对应 IFM / OFM 微块权重

Kernel 子块 1
  -> 对应 IFM / OFM 微块权重

Kernel 子块 2
  -> 对应 IFM / OFM 微块权重
```

它的目标是改善特定形状下的 Block 利用率。上图是概念示意，不是完整的底层字节排列。

### 4.3 Vela 如何选择？

`weight_compressor.py` 中的 `encode_weight_and_scale_tensor()` 会根据 IFM Depth、Kernel Size 和 IFM Bitdepth 估算两种遍历方式的利用率。

典型逻辑如下（具体代码应以分析所使用的 Vela commit 为准）：

```python
npu_tensor.hw_traversal = NpuBlockTraversal.DEPTH_FIRST

if npu_block_type == NpuBlockType.ConvolutionMxN:

    kernel_size = weights.shape[0] * weights.shape[1]

    depth_utilization = weights.shape[2] / round_up(
        weights.shape[2],
        32 if ifm_bitdepth == 8 else 16
    )

    part_kernel_utilization = (
        weights.shape[2] /
        round_up(weights.shape[2], 8)
    ) * (
        kernel_size /
        round_up(
            kernel_size,
            4 if ifm_bitdepth == 8 else 2
        )
    )

    if part_kernel_utilization >= depth_utilization or ifm_depth <= 8:
        npu_tensor.hw_traversal = \
            NpuBlockTraversal.PART_KERNEL_FIRST
```

这里的利用率是根据硬件 Block 粒度估算的布局利用率，不是实测 MAC 利用率，也不是整个网络的性能指标。

#### Case A：IFM Depth = 64，3×3，INT8

\[
U_d = \frac{64}{\operatorname{roundup}(64,32)} = 1
\]

\[
U_p =
\frac{64}{\operatorname{roundup}(64,8)}
\times
\frac{9}{\operatorname{roundup}(9,4)}
= 1 \times \frac{9}{12} = 0.75
\]

因此 Depth First 的估算利用率较高，选择 `DEPTH_FIRST`。

#### Case B：IFM Depth = 10，3×3，INT8

\[
U_d = \frac{10}{\operatorname{roundup}(10,32)}
= \frac{10}{32} = 0.3125
\]

\[
U_p =
\frac{10}{\operatorname{roundup}(10,8)}
\times
\frac{9}{\operatorname{roundup}(9,4)}
=
\frac{10}{16}\times\frac{9}{12}
= 0.46875
\]

因此 Part Kernel First 的估算利用率较高，选择 `PART_KERNEL_FIRST`。

#### Case C：IFM Depth ≤ 8

代码中还有明确的条件：

```python
if ifm_depth <= 8:
    npu_tensor.hw_traversal = NpuBlockTraversal.PART_KERNEL_FIRST
```

因此，对于该逻辑覆盖的 MxN 卷积，输入通道不超过 8 时会选择 Part Kernel First。

> 注意：上述公式和阈值是特定 Vela 代码版本与目标硬件配置下的实现细节，不应脱离对应源码版本当作所有 NPU 的通用规则。

---

## 5. Vela 的 Weight Encoding 源码流程

重点文件：

| 文件 | 主要职责 |
|---|---|
| `tflite_reader.py` | 读取 TFLite，转换内部权重维度 |
| `scheduler.py` | 选择 Block Config、切分和 Buffering |
| `weight_compressor.py` | 准备权重、选择 Traversal、组织权重编码 |
| `mlw_codec` | 执行底层 Weight Reordering / Encoding |
| `high_level_command_stream_generator.py` | 组织 NPU 命令流及权重使用信息 |

### 5.1 Scheduler 调用 Weight Compressor

在 Scheduler 创建操作信息时，会先确定 Block Config，再调用权重编码函数。概念上如下：

```python
block_config = self._get_block_config(
    ifm_shape,
    ifm2_shape,
    self.uses_scalar,
    ofm_shape
)

scheduler_op_info = SchedulerOpInfo(
    block_config, 0, ifm_shape, ifm2_shape, ofm_shape
)

if self.parent_op.weights:
    (
        scheduler_op_info.npu_weights_tensor,
        scheduler_op_info.npu_scales_tensor,
    ) = weight_compressor.encode_weight_and_scale_tensor(
        self.arch,
        self.parent_op,
        self.parent_op.weights,
        self.parent_op.bias,
        self.kernel,
        block_config,
        [0, self.ofm.shape.depth],
    )
```

这说明编码结果依赖：

- 目标架构 `arch`；
- 算子类型及参数；
- Kernel；
- Scheduler 选择的 `block_config`；
- 本次需要编码的 OFM Depth 范围。

所以，权重重排不是只根据原始 Tensor Shape 决定的，它与调度器为 NPU 选择的计算 Block 密切相关。

### 5.2 Zero Point 校正

典型处理：

```python
quant_buf = weight_tens.values.astype(np.int16)
zero_point = np.int16(weight_tens.quantization.zero_point)
weights = quant_buf - zero_point
```

例如：

```text
量化权重： [130, 128, 125, 140]
Zero Point: 128
校正结果： [  2,   0,  -3,  12]
```

减去 Zero Point 后，得到以零为中心的权重值，供后续硬件编码使用。

注意：这不是普通的布局重排，而是量化权重到硬件权重表示的准备步骤。

### 5.3 FullyConnected 的二维权重扩展

FullyConnected 的权重可能以二维矩阵表示。Vela 可以增加两个维度，使其进入统一的四维权重编码流程：

```python
if len(weights.shape) == 2:
    weights = np.expand_dims(
        np.expand_dims(weights, axis=0),
        axis=0
    )
```

概念上：

\[
[I,O] \rightarrow [1,1,I,O]
\]

可以把 FullyConnected 看成空间尺寸为 1×1 的矩阵乘/卷积形式，以复用相应编码路径。

### 5.4 Transpose Convolution 的 Kernel 翻转

对于特定的转置卷积内部算子，Vela 会先翻转 Kernel 的 H、W 方向：

```python
if op.type == Op.Conv2DBackpropInputSwitchedBias:
    weights = np.flip(weights, axis=(0, 1))
```

例如：

```text
原始 Kernel       翻转后 Kernel

1 2 3             9 8 7
4 5 6      ->     6 5 4
7 8 9             3 2 1
```

必须区分：

- Kernel 翻转：处理转置卷积的算子语义。
- Weight Reordering：处理硬件 Weight Stream 的数据布局。

两者不是同一件事，只是在编码准备阶段可能先后出现。

### 5.5 按 OFM Depth 切分 Weight Brick

Vela 可以根据 OFM Depth 范围提取当前需要编码的权重：

```python
brick_weights = weights[
    :, :,
    :,
    depth_offset:depth_offset + depth_length
]
```

例如：

```text
OFM Depth = 16
OFM Block Depth = 8

Weight Brick 0: Output Channel 0..7
Weight Brick 1: Output Channel 8..15
```

需要理解两个层次：

- **Brick 切分**：决定当前编码哪一段输出通道。
- **Traversal**：决定这段权重内部如何按照硬件顺序排列。

### 5.6 多核输出通道分配

Vela 的 `core_deinterleave()` 逻辑可概括为：

```python
def core_deinterleave(hwio, core, ncores):
    # Put weights back into OHWI
    ohwi = np.transpose(hwio, (3, 0, 1, 2))
    return ohwi[core:ohwi.shape[0]:ncores]
```

假设两个 Core，输出通道为 O0..O7：

```text
Core 0: O0, O2, O4, O6
Core 1: O1, O3, O5, O7
```

即按 Core 编号对输出通道做交错分配：

```python
core0 = weights[0::2]
core1 = weights[1::2]
```

这只是双 Core 的示意。实际 Core 数量取决于目标架构配置。

### 5.7 调用底层 `mlw_codec.reorder_encode()`

Python 层准备好权重、Block 和遍历策略后，会调用底层 Codec。典型参数包括：

| 参数 | 含义 |
|---|---|
| `ifm_ublock_depth` | 硬件 IFM 微块深度 |
| `ofm_ublock_depth` | 硬件 OFM 微块深度 |
| `weights_volume` | 当前权重数据 |
| `ofm_block_depth` | 当前 OFM Block 深度 |
| `is_depthwise` | 是否为 Depthwise Conv |
| `block_traversal` | Depth First / Part Kernel First |
| `ifm_bitdepth` | IFM 数据位宽 |
| `decomp_h` | Kernel 高度方向子块分解参数 |
| `decomp_w` | Kernel 宽度方向子块分解参数 |

概念调用：

```python
return mlw_codec.reorder_encode(
    ifm_ublock.depth,
    ofm_ublock.depth,
    weights_volume,
    ofm_block_depth,
    is_depthwise,
    block_traversal == NpuBlockTraversal.PART_KERNEL_FIRST,
    ifm_bitdepth,
    decomp_h,
    decomp_w,
)
```

底层 Codec 才是执行硬件权重重排和 Weight Stream 编码的关键位置。Python 层负责准备输入和选择参数。

---

## 6. Bias 和 Scale 必须与 Weight 通道对应

卷积的每个输出通道通常有对应的 Weight、Bias 和量化参数。如果 Weight 按 OFM 通道重新分配，Bias 和 Scale 也必须保持相同的通道映射。

原始对应关系：

| OFM Channel | Weight | Bias | Scale |
|---|---|---|---|
| O0 | W0 | B0 | S0 |
| O1 | W1 | B1 | S1 |
| O2 | W2 | B2 | S2 |
| O3 | W3 | B3 | S3 |

双 Core 交错分配后：

```text
Core 0: (W0, B0, S0), (W2, B2, S2)
Core 1: (W1, B1, S1), (W3, B3, S3)
```

如果只重排 Weight，却没有同步分配 Bias / Scale，输出就会错误。

Vela 的相关逻辑会对 Bias 和量化 Scale 使用与输出通道相同的切片/交错规则。

### 6.1 Bias、Scale、Shift 的打包

Vela 会把 Bias、Scale、Shift 等信息编码成硬件要求的格式。特定 Ethos-U 格式中，相关数据可打包为每通道 10 Bytes（80 bits）：

```text
Bias  : 40 bits
Scale : 32 bits
Shift :  6 bits
----------------
合计   : 78 bits
存储   : 80 bits = 10 Bytes
```

实际字段含义和编码限制应以目标 NPU TRM 及对应 Vela Codec 版本为准。

---

## 7. Vela 中常见的模型 Case

| Case | 是否进入 NPU Weight Encoding | 处理重点 |
|---|---|---|
| 普通 Conv2D | 是 | 内部维度转换、Block 切分、Traversal、重排和编码 |
| 3×3 Conv2D | 是 | 根据 IFM Depth、位宽和 Kernel Size 选择 Traversal |
| 1×1 Conv2D | 是 | 仍需按硬件格式编码，不代表可以直接使用原始布局 |
| FullyConnected | 是 | 二维权重扩展成统一四维形式 |
| DepthwiseConv2D | 是 | 使用 Depthwise 专用编码路径 |
| TransposeConv | 若映射到 NPU 支持的对应形式则是 | 可能先翻转 Kernel，再编码 |
| Dilated Conv2D | 若该配置由 NPU 支持则是 | 编码参数考虑 Dilation / 子 Kernel 分解 |
| AveragePool 转 Conv | 转换后的卷积若在 NPU 上执行则是 | 图优化生成的常数权重也要编码 |
| Resize 转 DepthwiseConv | 转换后的算子若在 NPU 上执行则是 | 生成的选择性 Kernel 进入 Depthwise 编码 |
| MaxPool | 通常否 | 没有学习型 Weight Tensor |
| CPU Fallback Conv | 否 | 不使用 NPU Weight Stream，由 CPU 路径执行 |

### 7.1 普通 Conv2D：IFM Channel 较大

例如：

```text
IFM    : 56 × 56 × 64
Kernel : 3 × 3
OFM    : 56 × 56 × 128
```

需要完成：

1. 读取模型权重；
2. OHWI 转为 Vela 内部布局；
3. Scheduler 选择 OFM / IFM Block；
4. Weight Compressor 选择 Traversal；
5. 按 OFM Depth 切分；
6. Codec 重排、Padding 和编码。

输入通道较大时，Depth First 可能具有更好的 IFM Block 利用率。

### 7.2 普通 Conv2D：IFM Channel 较小

例如：

```text
IFM    : 56 × 56 × 5
Kernel : 3 × 3
OFM    : 56 × 56 × 32
```

输入通道无法填满较大的硬件 IFM Block，可能产生较多填充浪费。Vela 的启发式规则可能选择 Part Kernel First。

### 7.3 DepthwiseConv2D

普通 Conv2D 的每个输出通道通常综合多个输入通道；DepthwiseConv2D 则按输入通道独立执行卷积，并受 depth multiplier 影响。

例如：

```text
Input Channels     = 4
Depth Multiplier   = 1
Output Channels    = 4
```

Vela 会将 Depthwise 标志传给底层 Codec，使用 Depthwise 专用编码逻辑。对应实现还会约束 Depthwise 不使用普通 MxN 卷积的 Part Kernel First 路径。

### 7.4 AveragePool 转换成卷积

某些图优化会把 Pooling 转换为卷积形式。以 2×2 平均池化为概念例子，其数学效果可以表示为：

```text
1/4  1/4
1/4  1/4
```

具体能否转换、如何表示量化 Scale、Padding 和 Stride，要看 Vela 的转换条件和目标硬件支持。

一旦图优化把算子变成 NPU 卷积形式，生成的常数权重也需要进入 NPU Weight Encoding。

### 7.5 Resize 转换成 DepthwiseConv

某些 Resize 配置可能被改写成硬件支持的卷积形式。若生成 DepthwiseConv 的选择性 Kernel，那么它虽然不是训练得到的权重，也仍然是硬件卷积所需的权重数据，必须走对应编码流程。

**重要原则：Weight Encoding 面向最终 NPU 执行形式，而不只面向原始模型中的算子名称。**

---

## 8. Weight Reordering 与 Weight Buffering 的区别

这两个概念经常被混淆。

| 比较项 | Weight Reordering | Weight Buffering |
|---|---|---|
| 解决的问题 | 权重如何排列 | 权重何时、分几次搬运 |
| 关注对象 | 数据布局和编码格式 | SRAM 容量、DMA 和调度 |
| 典型依据 | IFM / OFM ublock、Traversal | Weight Size、Buffer Size、Pass / Stripe |
| 结果 | 硬件可读取的 Weight Stream | 权重在存储层级中的搬运计划 |

假设权重总大小为 256 KB，而 NPU 可用于权重的快速 SRAM 只有 64 KB：

```text
Weight Slice 0
Weight Slice 1
Weight Slice 2
Weight Slice 3
```

调度器可能分批搬运，并采用 Double Buffering：

```text
时间片 1：
  Buffer A：NPU 使用 Slice 0
  Buffer B：DMA 搬运 Slice 1

时间片 2：
  Buffer A：DMA 搬运 Slice 2
  Buffer B：NPU 使用 Slice 1
```

重排决定每个 Slice 内部的硬件布局；Buffering 决定 Slice 的搬运和使用时序。

---

## 9. Weight Reordering 在 Vela 编译流程中的位置

需要区分图优化阶段对算子的改写，与后续真正生成硬件 Weight Stream 的阶段。

典型流程可以概括为：

```text
TFLite Reader
    |
    | 读取模型，转换内部 Tensor 维度
    v
Graph Optimization
    |
    | 算子改写、常量权重生成或形状调整
    v
Tensor Purpose / Pass Packing
    |
    v
NPU Subgraph Extraction
    |
    v
Scheduler
    |
    | 选择 Block Config、OFM 切分、Traversal
    v
Weight Compressor
    |
    | Zero Point correction
    | Weight Reordering
    | Padding
    | Weight Encoding / Compression
    v
Command Stream Generation
    |
    v
Serialization
    |
    v
Compiled Model / NPU Weight Stream
```

因此，DeepWiki 等资料在“Compilation Pipeline / Graph Optimization”中概括提到 Weight Reordering 时，不应据此认为底层硬件 Weight Stream 重排全部发生在最初的图优化 Pass 中。应沿 Scheduler 和 Weight Compressor 的调用链确认真正实现位置。

---

## 10. 对自研 NPU 图编译器的启发

如果自研 NPU 是固定功能卷积加速器，且包含 MAC Array、IFM SRAM、Weight SRAM、DMA 等模块，那么也需要定义自己的 Weight Layout。但不能直接照搬 Ethos-U 的编码格式。

### 10.1 硬件团队需要明确提供的参数

| 硬件参数 | 对 Weight Reordering 的影响 |
|---|---|
| MAC Array 结构 | 每次并行处理多少输入 / 输出通道 |
| IFM Channel Block | 输入通道切分粒度 |
| OFM Channel Block | 输出通道切分粒度 |
| Kernel Processing Order | Kernel 权重的组织顺序 |
| Weight SRAM Bank 数量 | 权重如何分布到不同 Bank |
| Bank Width | 数据打包和对齐 |
| Weight Data Width | 每个权重占用的位数 |
| Padding 规则 | 不完整 Block 如何处理 |
| DMA Burst Alignment | 外部存储传输对齐 |
| Multi-Core Mapping | 权重如何分配给不同 Core |
| Weight Decoder Format | 最终 Weight Stream 的语法和编码格式 |

还需要明确：权重是否压缩、是否支持稀疏、是否支持不同位宽、是否要求特定 Bank 交错方式。

### 10.2 建议的软件模块划分

```text
Graph Compiler
    |
    v
Operator Lowering
    |
    v
NPU Scheduler
    |
    v
Weight Layout Selection
    |
    +---- Depth-oriented Layout
    +---- Kernel-oriented Layout
    +---- Channel-oriented Layout
    |
    v
Weight Reordering
    |
    v
Weight Padding
    |
    v
Weight Compression (optional)
    |
    v
Weight SRAM / Flash Layout
    |
    v
Command Stream Generation
```

建议将 Layout Selection、Reordering、Padding、Compression 分成独立模块，不要把所有逻辑塞进 Conv2D 图优化代码中。这样 Depthwise、FullyConnected、TransposeConv 以及未来新增算子可以复用底层机制。

### 10.3 建议建立的验证

至少需要以下测试：

1. **数值一致性**：原始布局与重排后布局经硬件格式解释，计算结果一致。
2. **边界通道数**：IFM / OFM Depth 小于、等于和大于 Block 深度。
3. **非整除情况**：通道数不能整除 ublock / block 时，Padding 正确。
4. **不同 Kernel**：1×1、3×3、5×5，以及硬件支持的其他尺寸。
5. **不同位宽**：INT8、INT16 等硬件支持的格式。
6. **Depthwise**：不同 depth multiplier。
7. **Dilation**：不同膨胀系数。
8. **多核**：输出通道分配、Bias 和 Scale 映射一致。
9. **压缩往返**：编码 / 解码后权重值与预期一致。
10. **DMA / SRAM 对齐**：地址、长度和 Bank 访问符合硬件约束。

---

## 11. 调试时重点检查什么？

遇到 NPU 计算结果错误、权重读取越界或性能异常时，可以按以下顺序检查：

| 检查点 | 典型问题 |
|---|---|
| 原始 TFLite Filter Shape | 是否正确理解 OHWI 等维度语义 |
| Vela 内部 Tensor Shape | transpose 是否正确 |
| Quantization Zero Point | 是否正确校正权重 |
| Block Config | IFM / OFM Block 是否符合目标架构 |
| Traversal | Depth First / Part Kernel First 选择是否正确 |
| OFM Brick | 权重切片范围是否正确 |
| Core Deinterleave | 多核通道分配是否一致 |
| Depthwise 标志 | 是否走专用编码路径 |
| Padding | 补零位置和数量是否正确 |
| Weight Stream | Codec 输出是否符合 TRM |
| Bias / Scale | 是否与输出通道保持对应 |
| DMA / Buffering | 地址、长度、对齐和切换时序是否正确 |

建议在编译器中保留中间产物或调试输出：

```text
original_weights.npy
vela_internal_weights.npy
selected_block_config.txt
selected_traversal.txt
reordered_weights_debug.npy
encoded_weight_stream.bin
bias_scale_stream.bin
```

其中 `reordered_weights_debug.npy` 需要由调试版本的 Codec 或独立参考实现导出；正常生产流程未必会保留这个中间数组。

---

## 12. 关键结论

Weight Reordering 主要解决三个问题：

1. **硬件数据组织**：把通用模型权重转换成 NPU Weight Decoder 能够消费的布局。
2. **硬件 Block 利用率**：通过不同 Traversal 和 Padding 策略，减少不完整 Block 带来的浪费。
3. **存储与传输效率**：结合编码、压缩、切片和 Buffering，改善权重存储及读取效率。

研究 Vela 时建议按以下优先级阅读：

| 优先级 | 文件 / 模块 | 重点 |
|---|---|---|
| 1 | `ethosu/vela/weight_compressor.py` | 权重准备、Traversal 选择、OFM 切分和 Codec 调用 |
| 2 | `ethosu/vela/scheduler.py` | Block Config、权重切片和 Buffering |
| 3 | `mlw_codec` | 真正的底层 Weight Reordering / Encoding |
| 4 | `ethosu/vela/tflite_reader.py` | 原始 TFLite 权重到 Vela 内部布局的转换 |
| 5 | Ethos-U TRM | 硬件 Weight Stream 格式和约束 |

**下一步最值得深入的内容是 `mlw_codec.reorder_encode()`：** Python 层解释了策略如何选择、参数如何传递；底层 Codec 才能解释每个权重最终按什么字节顺序写入 Weight Stream。

---

## 13. 参考资料与源码入口

以下链接指向 Vela 仓库和相关文档。阅读时应固定具体 Git commit / release tag，因为实现细节可能随版本变化。

- [NXP Ethos-U Vela GitHub 仓库](https://github.com/nxp-imx/ethos-u-vela)
- [Vela `weight_compressor.py`](https://github.com/nxp-imx/ethos-u-vela/blob/main/ethosu/vela/weight_compressor.py)
- [Vela `scheduler.py`](https://github.com/nxp-imx/ethos-u-vela/blob/main/ethosu/vela/scheduler.py)
- [Vela `tflite_reader.py`](https://github.com/nxp-imx/ethos-u-vela/blob/main/ethosu/vela/tflite_reader.py)
- [Arm Ethos-U65 Technical Reference Manual](https://documentation-service.arm.com/static/60cb2a5b0320e92fa40b3787)

> **版本提示：** 本文中的函数名和代码片段用于解释 Vela 相关实现路径。不同分支、release 或后续版本可能存在差异。对源码做逐行审查时，应以实际使用的 Vela commit 为准；对硬件格式做最终判断时，应以对应 Ethos-U 型号的 TRM 为准。
