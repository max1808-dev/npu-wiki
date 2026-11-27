
> 本文分析 Ethos-U Vela 图编译器中 Bias Tensor 的补齐、约束检查、量化参数计算和硬件编码，并整理图优化过程中可能生成或拆分 Bias 的典型场景。
>
> 代码分析基于 NXP `ethos-u-vela` 项目。阅读具体实现时，应以所使用的 Vela 分支/commit 为准；不同版本的函数位置和细节可能变化。

## 1. 概述：Bias-and-Quantization 要解决什么问题？

Bias-and-Quantization 并不是单纯把 Bias 改成某种数据类型。它涉及三个层面：

1. **Tensor 结构**：Bias 是否存在、形状是否符合算子要求、Bias 数量是否与输出通道匹配。
2. **量化语义**：Bias 的整数值必须处于正确的累加器尺度；每个输出通道可能有不同的 Weight Scale，因此也可能有不同的 Bias Scale 和 Requantization 参数。
3. **硬件编码**：Vela 需要将 Bias 以及对应的量化缩放参数转换成 Ethos-U 硬件使用的编码格式。

核心目标是：在保持模型计算语义的前提下，将 Bias 表示转换成 NPU 可以正确执行的形式。

### 1.1 为什么 TFLite 中合法的 Bias 还需要 Vela 处理？

TensorFlow Lite 的模型格式和算子语义比较通用，而 NPU 对 Bias 的形状、类型、数值范围和存储格式有明确约束。

例如，TFLite Conv2D 可以没有 Bias；但某些硬件执行路径需要一个明确的 Bias 输入。Vela 可以在适当情况下补充全零 Bias。

另外，模型中的 Bias 通常是 INT32，而 NPU 的最终权重流可能采用专门的 Bias 编码格式。Vela 需要完成从模型表示到硬件表示的转换。

---

## 2. 先理解 Bias 在 Conv2D 中的作用

假设使用 NHWC 输入布局，卷积核采用 TFLite 常见的 OHWI 逻辑形状：

```text
Input:  [1, 8, 8, 16]
Weight: [3, 3, 16, 32]
Output: [1, 6, 6, 32]
Bias:   [32]
```

这里：

- `3 × 3`：卷积核高和宽。
- `16`：输入 Channel 数量。
- `32`：输出 Channel 数量。
- 每个 Output Channel 对应一个卷积核和一个 Bias。

对第 `c` 个输出通道：

$$
Y_c = \sum_{h,w,i} X_{h,w,i}\times W_{h,w,i,c}+B_c
$$

其中：

- `X`：输入数据。
- `W`：权重。
- `B_c`：第 `c` 个输出通道对应的 Bias。
- `Y_c`：输出数据。

因此，普通 Conv2D 的 Bias 数量应与 Output Channel 数量一致。本例中：

```python
bias.shape = [32]
```

### 2.1 模型逻辑形状与硬件物理布局

必须区分两个概念：

- **模型级 Tensor 的逻辑形状**：例如 Bias 是 `[32]`。
- **NPU 内部计算/存储使用的物理布局**：例如硬件一次处理若干个输出通道，可能需要分块、对齐或重新编码。

假设 NPU 一次处理 16 个 Output Channel，32 个通道可能分成两组处理。但这不意味着模型逻辑上的 Bias 应从 32 个元素改成 16 个。硬件分块属于后续实现细节，不能与模型 Tensor 形状混为一谈。

---

## 3. Bias 为什么需要专门处理？

### 3.1 Bias Tensor 可能不存在

TFLite 中 Conv2D 的 Bias 是可选输入。模型可能没有训练 Bias，或者转换后的模型没有提供 Bias Tensor。

例如：

```text
Conv2D
  Input 0: IFM
  Input 1: Weight
  Input 2: Bias (optional)
  Output:  OFM
```

没有 Bias 时，数学上相当于：

$$
Y=X*W
$$

如果某个 NPU 执行路径要求 Bias 输入，Vela 可以创建全零 Bias：

```python
bias = [0] * output_channels
```

全零 Bias 不改变计算结果，但可以满足后续硬件参数准备流程的输入要求。

### 3.2 Bias 的数据类型可能不符合硬件要求

典型整数卷积的数据类型组合：

| Tensor | 常见数据类型 |
|---|---|
| IFM | INT8 / UINT8 |
| Weight | INT8 |
| Bias | INT32 |
| Accumulator | 通常为更宽的整数累加格式 |
| OFM | INT8 / UINT8 |

IFM 和 Weight 的整数值相乘后，需要对大量乘积进行累加。Bias 要加到累加结果上，因此通常使用 INT32，而不是 INT8。

Vela 对某些 INT16 输入路径还会使用 INT64 Bias。需要注意，这并不代表所有 INT16 算子都必须使用 INT64 Bias；具体选择与 Vela 的算子路径和缩放策略有关。

### 3.3 Bias 的量化 Scale 必须与 IFM、Weight 匹配

量化整数表示的实数通常可写为：

$$
Real=(Quantized-ZeroPoint)\times Scale
$$

对于常见的整数 Conv2D，Bias 的量化尺度应与累加器尺度一致：

$$
S_{bias,c}=S_{ifm}\times S_{weight,c}
$$

其中 `c` 表示输出通道。若 Weight 使用 Per-Channel Quantization，每个通道的 Weight Scale 可以不同，因此 Bias Scale 也可能逐通道不同。

Bias 的 Zero Point 通常为 0。

如果 Bias 的整数值与 Scale 不匹配，即使整数值本身看起来合理，其代表的实数也会错误，最终输出随之偏离。

### 3.4 NPU 需要专门的 Bias 编码格式

Vela 的 Weight Compression 阶段会把 Bias 和对应的缩放参数编码为硬件使用的数据流。Ethos-U 的相关编码路径中，Bias 项使用 80 bit 的编码布局：

| 字段 | 位宽 | 含义 |
|---|---:|---|
| Bias | 40 bit | 有符号 Bias 值 |
| Scale | 32 bit | 量化乘数相关字段 |
| Shift | 6 bit | 缩放移位 |
| Reserved | 2 bit | 保留位 |
| **总计** | **80 bit** | **每项 10 Bytes** |

这里的 Scale/Shift 是硬件 Requantization 所需的编码参数，不应与 TFLite Bias Tensor 的 `scale` 元数据混为一谈。具体字段解释及位布局应以目标 Ethos-U 架构版本和 Vela 对应源码为准。

---

## 4. Bias 量化：完整数值例子

假设一个 Conv2D 有两个 Output Channel：

```text
IFM:
  dtype = INT8
  scale = 0.02
  zero_point = 0

Weight:
  dtype = INT8
  scales = [0.01, 0.02]

Bias:
  dtype = INT32
  values = [200, -50]

OFM:
  dtype = INT8
  scale = 0.05
  zero_point = 0
```

假设两个通道的卷积乘加结果（尚未加 Bias）分别是：

```text
Channel 0: 800
Channel 1: 550
```

### 4.1 累加 Bias

```text
Channel 0: 800 + 200 = 1000
Channel 1: 550 - 50 = 500
```

所以：

$$
Accumulator=[1000,500]
$$

### 4.2 计算累加器的真实尺度

Channel 0：

$$
S_0=0.02\times0.01=0.0002
$$

Channel 1：

$$
S_1=0.02\times0.02=0.0004
$$

对应真实数值：

$$
Y_0=1000\times0.0002=0.2
$$

$$
Y_1=500\times0.0004=0.2
$$

### 4.3 转换到 OFM 量化尺度

OFM Scale 为 0.05，Zero Point 为 0：

$$
Q_0=0.2/0.05=4
$$

$$
Q_1=0.2/0.05=4
$$

最终量化输出：

```text
OFM = [4, 4]
```

### 4.4 Vela 需要计算的 Requantization Scale

对于每个输出通道：

$$
Multiplier_c =
\frac{S_{ifm}\times S_{weight,c}}{S_{ofm}}
$$

因此：

```text
Channel 0: 0.02 × 0.01 / 0.05 = 0.004
Channel 1: 0.02 × 0.02 / 0.05 = 0.008
```

Vela 会进一步把这些实数缩放系数转换为硬件所需的整数 Multiplier 和 Shift。

**重要：**

- Bias Scale：`IFM Scale × Weight Scale`。
- Requantization Scale：`Bias Scale ÷ OFM Scale`。

二者不是同一个量。每个 Output Channel 可能有不同的参数。

---

## 5. Vela 源码：三个关键环节

主要涉及以下文件：

| 文件 | 主要职责 |
|---|---|
| `ethosu/vela/tflite_supported_operators.py` | 检查 Bias 的形状、类型、数值范围等约束 |
| `ethosu/vela/tflite_graph_optimiser.py` | 在图优化中补齐或构造 Bias，拆分/转换算子时同步处理 Bias |
| `ethosu/vela/weight_compressor.py` | 计算每通道缩放参数，并将 Bias 编码成硬件格式 |

### 5.1 约束检查：`tflite_supported_operators.py`

#### Bias 形状检查

相关函数：

```python
def constraint_bias_shape(op):
    "Optional Bias tensor must be of shape: 1D"
    bias = op.bias
    if bias:
        valid = len(bias.shape) == 1
        return valid, f"Tensor '{bias.name}' has shape: {bias.shape}"
    return True, "Op has no bias tensor"
```

Vela 检查 Bias 是否为一维 Tensor。

例如：

```python
bias.shape = [32]          # 1D
bias.shape = [1, 1, 1, 32]  # 4D，即使元素数量相同也不满足此约束
```

#### Bias 类型检查

相关检查要求 Bias 属于支持的类型集合，典型为：

```python
supported_bias_dtypes = {
    DataType.int32,
    DataType.int64,
}
```

因此不能把普通 INT8 Bias 或 FLOAT32 Bias 直接当作该 NPU 路径所需的 Bias。

#### INT64 Bias 的 40-bit 范围检查

相关函数：

```python
def constraint_bias_40bit(op):
    bias = op.bias
    if bias and bias.dtype == DataType.int64 and bias.values is not None:
        valid = all(len(bin(value)[2:]) <= 40 for value in bias.values)
```

这里要关注两个事实：

1. INT64 是 Tensor 的存储数据类型，不代表硬件最终可以接受任意 64-bit 数值。
2. 最终 Bias 编码字段有位宽限制，因此 Vela 需要检查数值是否能被编码。

注意：上述代码片段用于说明源码检查思路。判断负数的 signed bit-width 时，应以目标版本中的完整实现为准，不能仅凭简化片段推断边界行为。

### 5.2 缺失 Bias 的补齐：`fixup_bias_tensors()`

文件：

`ethosu/vela/tflite_graph_optimiser.py`

核心逻辑可概括为：

```python
def fixup_bias_tensors(op, arch, nng, dtype=None):
    if op.type.needs_bias() and op.bias is None:
        nr_biases = op.inputs[1].shape[-1]
        bias_values = [0] * nr_biases

        if dtype is None:
            dtype = (
                DataType.int64
                if op.ifm.dtype == DataType.int16
                else DataType.int32
            )

        bias_tensor = create_const_tensor(
            op.name + "_bias",
            [nr_biases],
            dtype,
            bias_values
        )

        bias_index = op.type.info.indices.biases[0]

        if bias_index < len(op.inputs):
            op.set_input_tensor(bias_tensor, bias_index)
        else:
            op.add_input_tensor(bias_tensor)

    return op
```

#### 第一步：判断是否需要 Bias

```python
if op.type.needs_bias() and op.bias is None:
```

只有算子需要 Bias 且当前没有 Bias 时才创建。

#### 第二步：确定 Bias 数量

```python
nr_biases = op.inputs[1].shape[-1]
```

例如 Weight 形状为：

```python
weight.shape = [3, 3, 16, 32]
```

则：

```python
nr_biases = 32
```

即按 Weight 最后一维取得输出通道数。该逻辑依赖当前算子的 Weight 形状约定，不能不加区分地套用到所有算子。

#### 第三步：创建全零 Bias

```python
bias_values = [0] * nr_biases
```

例如 32 个输出通道就创建 32 个零。

#### 第四步：选择默认数据类型

```python
dtype = (
    DataType.int64
    if op.ifm.dtype == DataType.int16
    else DataType.int32
)
```

典型默认选择：

| IFM 类型 | 默认补齐 Bias 类型 |
|---|---|
| INT8 | INT32 |
| UINT8 | INT32 |
| INT16 | INT64 |

这是该函数的默认行为，不代表所有图变换生成的 Bias 都使用同一规则。某些转换会显式传入 INT32，原因可能是它们希望走特定的缩放路径。

#### 第五步：插入正确的输入位置

```python
bias_index = op.type.info.indices.biases[0]
```

Vela 通过算子元信息找到 Bias 输入位置，而不是在所有算子上硬编码“Bias 一定是第三个输入”。

如果该位置已经存在，则替换；否则追加输入。

### 5.3 计算 Scale/Shift：`weight_compressor.py`

核心函数：

```python
_prepare_scale_and_bias()
```

该函数取得 IFM、OFM、Weight Scale 和 Bias 值，并计算逐通道 Requantization 参数。

核心数学关系：

```python
scales = [
    (np.double(ifm_scale) * np.double(weight_scale))
    / np.double(ofm_scale)
    for weight_scale in weight_scales
]
```

对应：

$$
Scale_c =
\frac{Scale_{IFM}\times Scale_{Weight,c}}
{Scale_{OFM}}
$$

随后将浮点缩放系数转换为硬件 Multiplier 和 Shift：

```python
quantised_scales = [
    quantise_scale(scale)
    for scale in scales
]
```

对于 Per-Channel Quantization，Weight Scale 不同，因此各输出通道会得到不同的缩放参数。

### 5.4 Bias 编码：`encode_bias()`

相关逻辑将每个通道的 Bias 与该通道的 Scale/Shift 一起编码：

```python
for j, core_bias in enumerate(core_biases):
    scale_stream.extend(
        encode_bias(
            np.int64(core_bias),
            *core_scales[j]
        )
    )
```

这里：

- `core_bias`：当前输出通道的 Bias。
- `core_scales[j]`：该通道对应的量化缩放参数。
- `encode_bias()`：生成硬件使用的编码数据。

因此最终并不是简单地把原始 Bias Tensor 原样搬到 NPU，而是把 Bias 和相应的 Requantization 参数组合成硬件格式。

---

## 6. Vela 中的典型 Case

以下场景展示了 Bias 不仅来自原始模型，也可能由图优化器主动创建、修改或拆分。

### Case 1：普通 Conv2D 没有 Bias

输入：

```text
IFM:    [1, 8, 8, 16]
Weight: [3, 3, 16, 32]
Bias:   None
OFM:    [1, 6, 6, 32]
```

Vela 对需要 Bias 的算子执行 `fixup_bias_tensors()`，生成：

```python
bias.shape = [32]
bias.dtype = INT32
bias.values = [0, 0, ..., 0]
```

结果：

```text
Conv2D
  IFM
  Weight
  Bias (32 个零)
  OFM
```

全零 Bias 不改变数学结果，但使后续硬件参数准备拥有明确的 Bias Tensor。

### Case 2：Weight 使用 Per-Channel Quantization

假设有 4 个输出通道：

```python
ifm_scale = 0.02
weight_scales = [0.01, 0.02, 0.03, 0.04]
ofm_scale = 0.05
```

则：

```python
requant_scales = [
    0.004,
    0.008,
    0.012,
    0.016,
]
```

| Output Channel | Weight Scale | Bias Scale | Requantization Scale |
|---|---:|---:|---:|
| 0 | 0.01 | 0.0002 | 0.004 |
| 1 | 0.02 | 0.0004 | 0.008 |
| 2 | 0.03 | 0.0006 | 0.012 |
| 3 | 0.04 | 0.0008 | 0.016 |

这里：

- Bias Scale = IFM Scale × Weight Scale。
- Requantization Scale = Bias Scale ÷ OFM Scale。

TFLite 常见量化规范中，Conv2D Bias 的 Scale 与对应输出通道的 IFM Scale × Weight Scale 一致，Bias Zero Point 为 0。Per-Axis Weight Quantization 会使不同通道的 Bias Scale 不同。

### Case 3：DepthwiseConv2D 的 Bias

假设：

```text
IFM:             [1, 8, 8, 16]
Depthwise Weight:[1, 3, 3, 16]
Depth multiplier: 1
OFM:             [1, 6, 6, 16]
```

则 Bias：

```python
bias.shape = [16]
```

如果 Depth Multiplier 为 2：

```text
IFM Channel = 16
OFM Channel = 32
```

那么 Bias 数量也应与 32 个输出通道对应。

DepthwiseConv2D 与普通 Conv2D 的 Weight 量化轴不同。TFLite 常见规范中，Conv2D Weight 的量化维度为 0，DepthwiseConv2D Weight 的量化维度为 3。Vela 在内部转换 Weight 布局时，必须维持 Scale 与输出通道的对应关系。

### Case 4：INT16 IFM，Bias 使用 INT64

`fixup_bias_tensors()` 的默认选择中：

```python
if op.ifm.dtype == DataType.int16:
    dtype = DataType.int64
else:
    dtype = DataType.int32
```

在 `weight_compressor.py` 中，还能看到 INT16 IFM + INT64 Bias 对应的特殊缩放路径：

```python
if ifm_dtype == DataType.int16 and bias_tens.dtype == DataType.int64:
    quantised_scales = [
        reduced_quantise_scale(scale)
        for scale in scales
    ]
else:
    quantised_scales = [
        quantise_scale(scale)
        for scale in scales
    ]
```

代码注释将该路径与参考实现使用 reduced scaling 联系起来。

因此要记住：

- INT16 IFM + INT64 Bias：可能走 Reduced Scaling。
- 其他组合：可能走 Full/常规 Scaling 路径。
- 不能简单认为 Bias 位宽越大越好；类型选择会影响后续缩放算法。

### Case 5：AveragePool 转换成 DepthwiseConv2D，生成特殊 Bias

Vela 可以在满足条件时，将 AveragePool 转换成 DepthwiseConv2D，以复用 NPU 的卷积计算能力。

示意：

```text
原始图：
  PAD
   |
AveragePool

转换后：
  DepthwiseConv2D（配合相应 padding 处理）
```

转换后需要为 DepthwiseConv2D 准备 Bias。

#### 情况 A：UINT8 IFM

代码路径会调用：

```python
fixup_bias_tensors(
    op, arch, nng, DataType.int32
)
```

创建全零 INT32 Bias。

#### 情况 B：其他输入类型

相关代码会根据 IFM Zero Point 和 Kernel 面积构造 Bias：

```python
bias_values = [
    op.ifm.quantization.zero_point * k_h * k_w
] * nr_biases
```

例如：

```text
IFM Zero Point = 5
Kernel = 3 × 3
Output Channel = 16
```

则：

$$
Bias=5\times3\times3=45
$$

生成：

```python
bias = [45] * 16
```

这个非零 Bias 是图变换为了保持转换前后计算语义而构造的修正项，不是原模型训练得到的 Bias。

该路径还可能配合调整 OFM Zero Point 等量化属性。应结合对应版本中完整的 `replace_pad_by_hw_pad()` 实现理解，不能只抽取 Bias 公式而忽略其余转换条件。

### Case 6：ResizeNearestNeighbor 转换为 DepthwiseConv2D

对于满足转换条件的 ResizeNearestNeighbor（例如特定 `align_corners` 设置），Vela 可以构造 DepthwiseConv2D 来实现相应的采样行为。

构造的 Depthwise Weight 通常通过 0/1 系数选择目标输入位置：

```text
选中位置：1
其他位置：0
```

转换后的 DepthwiseConv2D 需要 Bias，因此代码会调用类似：

```python
fixup_bias_tensors(
    op, None, None, DataType.int32
)
```

生成全零 INT32 Bias。

此时 Bias 并非来自原始 Resize 算子，而是图优化产生的新算子所需的辅助 Tensor。

### Case 7：Mean 转换成一个或多个 DepthwiseConv2D

Vela 可以把满足条件的 Mean 运算转换为一个或多个 DepthwiseConv2D，并配合 ADD、MUL 等后续运算完成结果组合。

示意：

```text
Mean
 |
 +-- DepthwiseConv2D 0 --+
 |                       |
 +-- DepthwiseConv2D 1 --+--> ADD / MUL --> OFM
 |                       |
 +-- DepthwiseConv2D 2 --+
```

对生成的 DepthwiseConv2D，Vela 会创建与 Channel 数量相匹配的全零 Bias。代码中的类型选择可能是：

```python
dtype = (
    DataType.int64
    if intermediate_op.ifm.dtype == DataType.int16
    else DataType.int32
)

bias_values = [0] * c
```

这再次体现：图优化器必须为自己生成的新算子准备硬件要求的参数，而不能只关注原始模型中的 Tensor。

### Case 8：ArgMax 转换为 DepthwiseConv2D，使用非零 Bias

ArgMax 不只是找最大值，还需要处理相等元素时选择哪个索引的问题。TensorFlow 的常见语义是多个最大值相同时返回最小索引。

Vela 的某些 ArgMax 转换路径会利用 DepthwiseConv2D、MaxPool 和 LUT 等操作组合实现目标语义。

其中一个构造 Bias 的思路是生成反向索引：

```python
reverse_idxs = list(reversed(range(c)))

bias_tensor = create_const_tensor(
    op.name + "_bias",
    [c],
    DataType.int64,
    reverse_idxs
)
```

例如：

```python
c = 3
bias = [2, 1, 0]
```

该 Bias 是图转换为实现 ArgMax 语义而构造的辅助数据，不是普通卷积训练得到的偏置。

具体实现要结合对应版本中 ArgMax 转换的完整 Weight、Bias、MaxPool 和 LUT 逻辑阅读。

### Case 9：Grouped Conv2D，拆分 Bias

Vela 的 `convert_conv_groups()` 可以将 Grouped Conv 拆分为多个独立 Conv，再通过 Split/Concat 等操作连接。

示意：

```text
              Split
                |
        +-------+-------+
        |       |       |
      Conv0   Conv1   Conv2
        |       |       |
        +-------+-------+
              Concat
```

假设：

```text
原始 Conv:
  Output Channel = 64
  Groups = 4
```

每组对应 16 个输出通道。Vela 需要同步拆分：

- Weight；
- Bias；
- 对应的量化参数；
- 输入输出连接关系。

源码中的逻辑会计算当前 Group 的输出通道范围：

```python
cg_oc_start = i * num_filters_cg
cg_oc_end = (i + 1) * num_filters_cg
```

并截取当前 Group 对应的 Bias：

```python
op.bias.values[..., cg_oc_start:cg_oc_end]
```

拆分后，每个 Conv 的 Bias 必须与其 Weight 和 Output Channel 一一对应。

---

## 7. Bias 不符合要求时，Vela 如何处理？

需要区分“可自动补齐”和“无法直接支持”。

| 问题 | 典型处理 |
|---|---|
| Bias 不存在 | 对需要 Bias 的算子，可以创建全零 Bias |
| Bias 维度不是 1D | 不满足对应算子约束，不能直接按该 NPU 路径执行 |
| Bias 类型为 INT8 | 不符合该 Bias 类型约束 |
| Bias 类型为 FLOAT32 | 不符合该 Bias 类型约束 |
| INT64 Bias 超出编码范围 | 无法按目标 Bias 编码格式正常编码 |
| Bias 不是常量 | 不满足常量 Bias 编码路径的要求 |
| Bias 数量与 Output Channel 不匹配 | 模型结构或算子输入不合法，需要修正 |
| Bias 量化语义错误 | 需要修正上游量化模型，不能假设 Vela 会自动修复 |

对于不满足 NPU 支持约束的算子，Vela 可以将其保留为 CPU 执行部分，而不是强行生成 NPU 命令。具体分配结果取决于算子支持检查、图划分和编译配置。

---

## 8. Bias-and-Quantization 在 Vela 编译流程中的位置

可以将相关流程概括为：

```text
                 TFLite Model
                       |
                       v
            Read IFM / Weight / Bias
                       |
                       v
          Supported Operator Constraints
          - Bias shape
          - Bias dtype
          - Bias value range
                       |
                       v
               Graph Optimisation
          - Add missing zero Bias
          - Generate special Bias
          - Split Bias with grouped Conv
          - Preserve quantization semantics
                       |
                       v
                Weight Compression
          - Read Bias values
          - Calculate per-channel scales
          - Calculate multiplier / shift
          - Encode Bias
                       |
                       v
           Ethos-U encoded weights / biases
                       |
                       v
                NPU command stream
```

需要注意，约束检查、图优化和权重压缩并不是一个单独、严格串行的 Bias pass；它们是 Vela 编译流程中不同模块承担的相关职责。上图是便于理解的逻辑视图。

---

## 9. Vela 是否会重新量化 Bias？

结论：**Vela 的 Bias 编码阶段不是一个通用的浮点模型量化器，也不会任意重新量化已有 Bias 的整数值。**

### 9.1 模型量化阶段

通常由 TensorFlow Lite Converter 或其他量化工具完成：

```text
Float Model
    |
    v
Quantization
    |
    +-- IFM    -> INT8 / UINT8
    |
    +-- Weight -> INT8
    |
    +-- Bias   -> 通常 INT32
    |
    v
Quantized TFLite Model
```

这个阶段确定 Bias 的整数值及其量化语义。

### 9.2 Vela 编译阶段

Vela 接收已经量化的模型。在 `_prepare_scale_and_bias()` 一类逻辑中，会读取：

```python
biases = tens.values

ifm_scale = ...
ofm_scale = ...
weight_scales = ...
```

然后根据 IFM、Weight、OFM 的 Scale 计算硬件需要的 Requantization 参数，并将 Bias 与这些参数编码。

因此：

- 上游量化工具必须正确生成 Bias。
- Bias 整数值必须处于正确的累加器尺度。
- Vela 负责准备硬件需要的缩放参数并完成编码。
- 如果上游 Bias 的整数值或量化语义本身错误，不能指望 Vela 自动纠正。

---

## 10. 对自研 NPU 图编译器的借鉴

这部分对开发自有 NPU 图编译器尤其有价值。

### 10.1 建立统一的 Hardware Tensor Preparation 层

不要让硬件接口直接依赖 TFLite 的原始 Tensor 表示。建议在图优化后、硬件代码生成前建立统一的参数准备阶段，负责：

- Bias 是否存在；
- Bias 形状和输出通道数量；
- Bias 数据类型；
- Bias 常量性；
- Bias 数值范围；
- Per-Channel Scale 对齐；
- 硬件 Bias/Multiplier/Shift 编码。

这样可以避免每个图变换 pass 各自实现一套不一致的处理逻辑。

### 10.2 Per-Channel Quantization 必须贯穿完整数据路径

不能只处理 Weight 的 Per-Channel Scale，而忽略 Bias 和输出通道之间的对应关系。

需要保证：

```text
Output Channel c
   |
   +-- Weight[c]
   |
   +-- Weight Scale[c]
   |
   +-- Bias[c]
   |
   +-- Bias Scale[c]
   |
   +-- Requantization Multiplier[c]
   |
   +-- Requantization Shift[c]
```

所有字段必须属于同一个输出通道。

### 10.3 图优化生成的新算子也必须准备 Bias

这是 Vela 非常值得借鉴的设计。

例如：

- AveragePool → DepthwiseConv2D；
- Resize → DepthwiseConv2D；
- Mean → 多个 DepthwiseConv2D；
- ArgMax → DepthwiseConv2D + MaxPool + LUT；
- Grouped Conv → 多个 Conv。

这些图变换可能产生新的算子、新的 Weight 和新的 Bias。图编译器必须保证变换前后的计算语义一致，并为新算子准备完整的硬件参数。

### 10.4 不要把“补零 Bias”和“语义修正 Bias”混为一谈

建议在自研编译器中明确区分：

| 类型 | 目的 | 示例 |
|---|---|---|
| Default Bias | 满足硬件接口，保持原计算不变 | 缺失 Bias 时补零 |
| Semantic Correction Bias | 补偿图变换带来的数值/语义差异 | Pool/Pad 转换中的 Zero Point 修正 |
| Algorithmic Bias | 帮助实现目标算子语义 | ArgMax 转换中的索引辅助 Bias |

这样便于调试、验证和追踪 Bias 的来源。

---

## 11. 总结

Bias-and-Quantization 可以归纳为三个层面：

| 层面 | 要解决的问题 | Vela 的处理 |
|---|---|---|
| Tensor 结构 | Bias 缺失、维度不符合要求 | 补零、检查维度、随算子拆分 |
| 数据类型和数值 | INT32/INT64 选择、数值范围 | 类型检查、编码范围检查 |
| 量化和硬件编码 | 每个 Channel 的 Scale 不同，硬件需要 Multiplier/Shift | 计算逐通道参数、编码 Bias |

最重要的三个认识：

1. **Bias 不只是 Conv2D 的一个可选输入。** 它的形状、类型、数值范围和量化尺度都必须满足硬件要求。
2. **Per-Channel Quantization 必须贯穿 Weight、Bias 和输出通道。** 每个输出通道可能需要不同的 Requantization 参数。
3. **图优化器必须为自己生成的新算子准备参数。** Bias 可能是原始模型提供的，也可能由 Vela 补齐、修正或拆分生成。

### 推荐源码阅读顺序

1. `tflite_supported_operators.py`
   - `constraint_bias_shape`
   - `constraint_bias_type`
   - `constraint_bias_40bit`

2. `tflite_graph_optimiser.py`
   - `fixup_bias_tensors`
   - `replace_pad_by_hw_pad`
   - `convert_mean_to_depthwise_conv`
   - `convert_conv_groups`
   - Resize / ArgMax 相关转换逻辑

3. `weight_compressor.py`
   - `_prepare_scale_and_bias`
   - `quantise_scale`
   - `reduced_quantise_scale`
   - `encode_bias`

读完这三个文件，就能把 Vela 对 Bias 的结构处理、量化参数计算和硬件编码串起来理解。
