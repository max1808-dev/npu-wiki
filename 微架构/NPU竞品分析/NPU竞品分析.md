我们想看下市场上针对图像处理目标市场的主流NPU产品。从这么几个维度来考虑：算力、算子覆盖率、实际模型运行效率、数据搬运效率、软件编译工具链，以及功耗和芯片成本。

![[Pasted image 20261126082351.png|749]]



- 有些 NPU 主要提供矩阵乘法和卷积计算能力，其他操作交给 CPU。    
- 有些 NPU 把 Resize、NMS、Softmax 等预处理和后处理也集成进来。    
- 有些 NPU 具有比较完整的图编译器，能够自动进行算子融合、分块、内存规划和调度。    
- 有些产品虽然标称算力很高，但模型中大量算子无法在 NPU 上执行，最终性能可能并不理想。
- 
# 1. 低功耗嵌入式 **NPU** / NPU IP
## **Arm Ethos-U 系列**

- U55、U65：较早期的低功耗微型 NPU。    
- U85：128～2048 MAC，1GHz 时最高 4 TOPS。    
- INT8 权重，INT8/INT16 激活。    
- 支持 CNN、RNN、Transformer，支持 2:4 稀疏计算。    
- 软件：TFLite Micro、CMSIS-Ethos-U、Vela Compiler。

## **NXP i.MX 93 / Arm Ethos-U65**

- INT8 权重，INT8/INT16 激活。    
- 主要面向量化 CNN。    
- 支持 TFLite、TFLite Micro。    
- NPU 不支持的算子可回退到 Cortex-A/M。

## **全志 V853**
- 最高 1 TOPS。    
- 内置 128KB 高速缓存。    
- 硬件神经网络引擎 + 可编程 PPU。    
- 支持 TensorFlow、TFLite、PyTorch、ONNX 等模型导入。    
- 面向 IPC、智能门锁、行车记录仪等视觉设备。

比较值得研究的是它的硬算子与软算子组合方式。

这里特别值得研究的是 **Arm Ethos-U85 和全志 V853**。前者有成熟的模型编译器和 TFLite 生态；后者采用硬件算子引擎与可编程引擎结合的设计思路。

# **2. 中端边缘视觉 NPU / SoC**

## **瑞芯微 RK3588**

- 支持 INT4、INT8、INT16、FP16、BF16、TF32。
- 6 TOPS。
- Conv、DepthwiseConv、Pooling、MatMul、Add、Mul、Concat、Resize、Sigmoid、Softmax 等。    
- RKNPU2 + RKNN-Toolkit2。    
- 支持模型转换、量化、算子融合及 CPU/GPU 自定义算子。 
它的优势是完整的 SoC 和模型部署生态。

## **TI TDA4VM**

- C7x DSP + MMA 深度学习矩阵加速器。  
- 8 TOPS
- 集成 VPAC、ISP、DMPAC。    
- 支持 Conv、DepthwiseConv、Pooling、Normalization、Activation、Elementwise 等。    
- 软件：TIDL、TVM、Edge AI 工具链。   

非常值得研究的是它如何让 NPU、DSP、视觉处理硬件协同工作。


## **Renesas RZ/V2H**

- DRP-AI3 加速器。
- 8 / 80 TOPS。
- 4 核 Cortex-A55。    
- 6MB 片上 SRAM。    
- 4 路 MIPI CSI-2。    
- 另外有 DRP 动态可重配置处理器，可以加速 OpenCV 等图像处理任务。    
- 软件：DRP-AI TVM。  

它的特点是 NPU 和可重构图像处理引擎结合。


## **爱芯元智 AX650A**

- 8 核 Cortex-A55。   
- 18 TOPS。
- 支持混合精度。    
- 8K@30fps ISP。    
- 支持 AI-HDR、AI-3DNR、去雾、鱼眼矫正等。    
- 集成 H.264/H.265 编解码。   

特别值得研究它的 AI 计算、ISP、视频处理如何集成。


## **算能 Sophgo CV186AH**

- 自研 TPU。    
- 7.2 TOPs。
- 面向视觉 AI 推理。    
- 集成 HDR、3D 降噪、3A、去雾等图像处理能力。    
- 支持多路视频编解码。    
- 目标检测、分割、行为分析、OCR 等应用。

## **Synaptics Astra SL1680**

- 四核 Cortex-A73。    
- 多 TOPS NPU。    
- ISP、4K 视频编解码。    
- SyNAP 工具链，针对 TFLite 推理优化。



## **3. 高性能独立 NPU 加速器**

## **Hailo-8**

- INT8/INT16，INT4 权重。    
- 26 TOPs。
- 数据流架构，计算与存储分布在多个计算 Cluster。    
- 支持 Conv、Pooling、Fully Connected、Activation、Split、Concat、Add。    
- 专门硬件支持 NMS、Bilinear Resize、Crop、Reshape、Softmax。    
- Hailo Dataflow Compiler + HailoRT。    

它的特点是把数据流、计算和内存组织一起考虑。

## **Hailo-10H**

- 支持 CNN、Transformer、LLM、VLM。 
- 40 TOPs。
- INT4/INT8/INT16 混合精度。    
- 具有外部 LPDDR4/4X 内存接口。    
- 支持 PCIe、USB 接入。


1. INT4、INT8、FP16 算力不能直接比较。例如 Hailo-10H 的 40 TOPS 是 INT4，INT8 是 20 TOPS。    
2. 稀疏算力不能直接与稠密算力比较。例如 RZ/V2H 的 80 TOPS 是稀疏模型指标，稠密是 8 TOPS。    
3. 有些厂商按 MAC=1 operation 计算，有些按 MAC=2 operations 计算，比较时必须统一口径。



## **1. 核心计算算子**

这是 NPU 最基础的能力。

|算子|重要程度|主要用途|
|---|---|---|
|Conv2D|极高|CNN 主计算|
|DepthwiseConv2D|极高|MobileNet 等轻量网络|
|FullyConnected|高|分类、特征映射|
|MatMul|高|Transformer、矩阵计算|
|ConvTranspose|中|上采样、分割网络|
|GroupConv|高|分组卷积网络|
## **2. 激活函数**

| 算子                 | 重要程度 |
| ------------------ | ---- |
| ReLU               | 极高   |
| ReLU6              | 高    |
| Sigmoid / Logistic | 极高   |
| Tanh               | 中    |
| LeakyReLU          | 高    |
| HardSwish          | 高    |
| SiLU / Swish       | 高    |
| GELU               | 中高   |

## **3. 数据处理算子**

|算子|重要程度|
|---|---|
|Add / Mul|极高|
|Concat|高|
|Split|高|
|Reshape|高|
|Transpose|高|
|Pad|高|
|Slice|中高|
|Resize|极高|
|Crop|高|
|Flatten|高|

## **4. Pooling 和归约算子**

|算子|重要程度|
|---|---|
|MaxPool|高|
|AveragePool|高|
|GlobalAveragePool|高|
|ReduceSum|中高|
|ReduceMax|中|
|ArgMax|中|

## **5. 视觉后处理算子**

| 算子                  | 用途     |
| ------------------- | ------ |
| Softmax             | 分类概率   |
| Sigmoid             | 目标置信度  |
| TopK                | 选择高分候选 |
| NMS                 | 非极大值抑制 |
| Bounding Box Decode | 边界框解码  |
| Resize / Crop       | 图像处理   |
| Normalize           | 输入归一化  |

其他考虑点:
- Kernel Size 支持哪些？    
- Stride 支持哪些？    
- Dilation 支持哪些？    
- Group Conv 是否支持？    
- Padding 是否支持 SAME / VALID？    
- 输入输出数据布局是否有限制？    
- INT8 是否支持 Per-channel Quantization？


## **NPU Hardware 硬件规格对比**

|厂商|产品|INT8 算力|其他算力|产品形态|主要应用|
|---|---|---|---|---|---|
|Arm|Ethos-U55|0.032～0.256 TOPS¹|INT16|NPU IP|MCU、低功耗视觉|
|Arm|Ethos-U65|0.256～0.512 TOPS¹|INT16|NPU IP|嵌入式视觉|
|Arm|Ethos-U85|0.256～4.096 TOPS²|INT16|NPU IP|CNN、Transformer|
|Rockchip 瑞芯微|RK3588|6 TOPS|支持 FP16、INT4|SoC|边缘视觉、工控|
|TI 德州仪器|TDA4VM|8 TOPS|C7x 浮点 DSP|SoC|ADAS、机器视觉|
|Renesas 瑞萨|RZ/V2H|8 TOPS Dense|80 TOPS Sparse|MPU|工业视觉、机器人|
|AXERA 爱芯元智|AX650A|18 TOPS³|72 TOPS INT4³|SoC|摄像头、视频分析|
|Hailo|Hailo-8|26 TOPS|混合精度|独立加速器|边缘视觉|
|Hailo|Hailo-8L|13 TOPS|混合精度|独立加速器|低功耗视觉|
|Hailo|Hailo-10H|20 TOPS|40 TOPS INT4|独立加速器|视觉、生成式 AI|

¹ 按 500 MHz 及相应 MAC 配置计算，非固定产品频率。
² 按 1 GHz 及相应 MAC 配置计算。
³ AX650A 官方网页标称 18 TOPS INT8，但另一个官方 PDF 标称 10.8 TOPS INT8，正式选型时需要确认产品版本。
爱芯元智半导体股份有限公司+1


### **1.1 硬件架构及存储对比**
|产品|计算架构|MAC/cycle|NPU 内部存储|外部存储|计算精度|
|---|---|---|---|---|---|
|Ethos-U55|专用神经网络计算引擎|32/64/128/256|IP 内部存储及系统 SRAM|系统总线访问|INT8、INT16|
|Ethos-U65|专用神经网络计算引擎|256/512|IP 内部存储及系统 SRAM|系统 SRAM、DDR|INT8、INT16|
|Ethos-U85|CNN、矩阵计算引擎|128～2048|系统配置决定|SRAM、DDR|INT8、INT16|
|RK3588|三核 NPU|未公开|内部 SRAM、系统内存|LPDDR4/4X/5|INT4、INT8、INT16、FP16|
|TDA4VM|C7x DSP + MMA|4000（按峰值算力折算）|片上共享存储|DDR|整数 MMA、浮点 DSP|
|RZ/V2H|AI-MAC + DRP|未公开|SoC 集成 6MB SRAM|LPDDR4/4X|量化整数计算|
|AX650A|专用 NPU|未公开|容量未公开|64-bit LPDDR4X|INT4、INT8、混合精度|
|Hailo-8|分布式 Dataflow Architecture，8 Clusters|未公开|分布式片上存储|模型中间计算主要使用片上资源|4/8/16-bit 权重、8/16-bit 激活|

参考资料：

Arm Insights+3

这里需要特别关注 Arm Ethos-U 和 Hailo 的差别：Arm Vela 重点解决有限 SRAM 下的计算调度和数据搬运，Hailo 则采用分布式计算与存储架构，将网络映射到不同计算资源。


## **表2：Operator Support 算子支持对比**

以下表格的“支持”指硬件或编译工具链具备对应计算能力，并不代表任意参数组合都能在 NPU 上执行。

|算子|Ethos-U55/U65|RK3588|TI TIDL|DRP-AI|Hailo-8|
|---|---|---|---|---|---|
|Conv2D|支持|支持|支持|支持|支持|
|DepthwiseConv2D|支持|支持|支持|支持|支持|
|GroupConv|参数受限|支持|参数受限|依赖转换|参数受限|
|ConvTranspose|参数受限|支持|参数受限|依赖转换|参数受限|
|FullyConnected|支持|支持|支持|支持|支持|
|MatMul|受限转换|支持|参数受限|依赖转换|参数受限|
|MaxPool|支持|支持|支持|支持|支持|
|AveragePool|支持|支持|支持|支持|支持|
|GlobalAveragePool|支持转换|支持|支持|支持转换|支持转换|
|GlobalMaxPool|支持转换|支持|需核对版本|依赖转换|参数受限|
|ReLU|支持融合|支持|支持|支持|支持|
|ReLU6|支持融合|支持|支持|支持|支持|
|LeakyReLU|支持|支持|支持|参数受限|支持|
|PReLU|参数受限|支持|参数受限|依赖转换|参数受限|
|Sigmoid|支持|支持|支持|参数受限|支持|
|Tanh|支持|支持|参数受限|参数受限|支持|
|HardSwish|支持|支持|依赖转换|依赖转换|参数受限|
|SiLU/Swish|可拆解|支持或转换|依赖转换|依赖转换|支持转换|
|GELU|需转换或回退|版本相关|版本相关|依赖转换|版本相关|
|Softmax|支持|支持|支持|参数受限|专用引擎|
|Add|支持|支持|支持|支持|支持|
|Sub|支持|支持|支持|依赖转换|参数受限|
|Mul|支持|支持|支持|参数受限|支持|
|Div|通常回退|参数受限|参数受限|依赖转换|参数受限|
|Maximum|支持|支持|参数受限|依赖转换|参数受限|
|Minimum|支持|支持|参数受限|依赖转换|参数受限|
|Concat|支持|支持|支持|支持|原生支持|
|Split|支持转换|支持|支持|依赖转换|原生支持|
|Reshape|编译期优化|支持|支持|编译期优化|专用引擎/转换|
|Transpose|参数受限|支持|参数受限|参数受限|参数受限|
|Pad|支持或融合|支持|支持|依赖转换|支持或融合|
|Slice|参数受限|支持|参数受限|依赖转换|参数受限|
|Resize Nearest|参数受限|支持|支持|参数受限|参数受限|
|Resize Bilinear|参数受限|支持|支持|参数受限|专用引擎|
|ReduceMean|支持|支持|参数受限|依赖转换|参数受限|
|ReduceSum|参数受限|支持|参数受限|依赖转换|参数受限|
|ArgMax|参数受限|支持|参数受限|依赖转换|版本相关|
|BatchNormalization|权重融合|权重融合|权重融合|权重融合|权重融合|
|LayerNormalization|通常回退|版本相关|版本相关|依赖转换|版本相关|
|Quantize|支持|支持|编译器处理|编译器处理|编译器处理|
|Dequantize|CPU/边界转换|边界转换|边界转换|边界转换|边界转换|
|NMS|CPU|通常 CPU|CPU/DSP|CPU/其他引擎|专用 NMS 引擎|

**核查状态说明：** Ethos-U 和 Hailo 已找到较明确的公开算子及硬件资料；RK3588 找到了算子支持文档，但版本较旧。TI TIDL 和 DRP-AI 两列目前属于初步技术归纳，尚未逐项对照对应 SDK 版本的算子支持清单，不能作为最终验收依据。

### **2.1 算子融合对比**

|Fusion Pattern|Ethos-U/Vela|RKNN|TI TIDL|DRP-AI|Hailo|
|---|---|---|---|---|---|
|Conv + Bias|支持|支持|支持|支持|支持|
|Conv + BatchNorm|权重折叠|权重折叠|权重折叠|权重折叠|权重折叠|
|Conv + ReLU|支持融合|支持融合|支持融合|支持融合|支持融合|
|Conv + ReLU6|支持融合|支持融合|支持融合|依赖编译器|支持融合或转换|
|Conv + Pad|条件融合|条件融合|条件融合|依赖编译器|条件融合|
|Conv + Sigmoid|条件融合|支持优化|依赖编译器|未确认|支持映射|
|Conv + SiLU|可拆解映射|支持优化|依赖编译器|未确认|支持映射|
|Conv + Add|条件优化|条件优化|条件优化|未确认|数据流映射|
|Conv + Pool|调度优化|条件优化|条件优化|未确认|数据流映射|
|多算子子图|支持|支持|支持|支持|支持|
|自定义 Fusion Pattern|可修改开源编译器|受厂商工具限制|受工具链限制|可扩展 TVM 前端|受 DFC 限制|

注意：**支持算子融合、支持连续执行、支持中间结果不写回 DDR，是三个不同的能力。** 不能仅通过模型中存在某个 Fusion Pattern 就判断硬件具备完整融合能力。


## **表3：Compiler & SDK 软件生态对比**

|功能|Arm Ethos-U|Rockchip|TI|Renesas|Hailo|
|---|---|---|---|---|---|
|图编译器|Vela|RKNN Toolkit2|TIDL|DRP-AI TVM|Dataflow Compiler|
|Runtime|Ethos-U Driver + TFLM|RKNN Runtime|TIDL Runtime|DRP-AI Runtime|HailoRT|
|TensorFlow Lite|支持|支持|支持|TVM 前端转换|支持转换路径|
|ONNX|间接转换|支持|支持|支持|支持|
|PyTorch|ExecuTorch/TOSA 路径|支持转换|ONNX 转换|ONNX/TVM 转换|ONNX 转换|
|TensorFlow|TFLite 转换|支持转换|TFLite 转换|TVM/ONNX 转换|支持|
|PTQ|支持，上游完成|支持|支持|支持|支持|
|QAT|支持量化模型|支持量化模型|支持量化模型|依赖前端|支持量化流程|
|INT8 量化|支持|支持|支持|支持|支持|
|INT16 量化|支持部分路径|支持部分路径|版本相关|版本相关|支持|
|INT4 量化|非主要路径|支持部分路径|非主要路径|未确认|支持低位宽权重|
|图优化|支持|支持|支持|支持|支持|
|算子融合|支持|支持|支持|支持|支持|
|常量折叠|支持|支持|支持|支持|支持|
|子图划分|支持|支持|支持|支持|支持编译网络划分|
|Tensor Layout 转换|支持|支持|支持|支持|支持|
|Tiling|支持|自动处理，算法未公开|自动处理|编译工具处理|资源映射|
|Memory Planning|支持|内部实现未公开|支持|TVM/工具链处理|分布式存储分配|
|DMA Scheduling|编译器生成调度|内部实现未公开|支持|工具链处理|Dataflow 调度|
|Double Buffering|编译调度支持|未公开|依赖实现|未公开|不同数据流架构|
|硬件命令生成|Command Stream|RKNN 模型|编译模型文件|DRP-AI Object|HEF|
|CPU Fallback|TFLM 等上层框架|依赖 Runtime 和算子|支持|TVM 混合执行|Host 侧另外处理|
|Bare-metal|支持|非主要部署方式|RTOS/SDK 路径|Linux 为主|非主要部署方式|
|RTOS|支持|Linux 为主|支持|依赖系统集成|Linux 为主|
|Linux|支持|支持|支持|支持|支持|
|Profiler|支持|支持|支持|支持性能评估|支持|
|Tensor Dump|支持调试|支持|支持|TVM 调试工具|支持分析工具|
|Model Zoo|有|有|有|有|有|
|图编译器源码|开源|闭源|核心闭源|部分开源|闭源|
|Runtime 源码|开源|部分公开|部分公开|部分开源|HailoRT 开源|
|自研 NPU 适配参考价值|高|中|中|高|架构参考价值高|

其中 Vela 明确实现了算子映射、融合、内存规划、Tiling、调度和硬件 Command Stream 生成，可针对不同 NPU MAC 配置及存储系统进行编译。

Arm Insights

### **3.1 软件源码及文档入口**

|项目|源码及文档|重点研究内容|
|---|---|---|
|Arm Vela|[ethos-u-vela](https://github.com/nxp-imx/ethos-u-vela?utm_source=chatgpt.com)|图编译、融合、Tiling、内存规划|
|Arm Supported Operators|[SUPPORTED_OPS.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/SUPPORTED_OPS.md?utm_source=chatgpt.com)|算子参数约束|
|Rockchip Toolkit2|[rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2?utm_source=chatgpt.com)|模型转换、量化、Runtime|
|Rockchip Model Zoo|[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo?utm_source=chatgpt.com)|模型适配、性能测试|
|TI|[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools?utm_source=chatgpt.com)|异构计算、子图划分|
|Renesas|[drp-ai_tvm](https://github.com/renesas-rz/rzv_drp-ai_tvm?utm_source=chatgpt.com)|TVM Backend、硬件适配|
|Hailo|[HailoRT](https://github.com/hailo-ai/hailort?utm_source=chatgpt.com)|Runtime、设备管理|
|Hailo Model Zoo|[hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo?utm_source=chatgpt.com)|模型优化、性能测试|
|Google|[TFLite Micro](https://github.com/tensorflow/tflite-micro?utm_source=chatgpt.com)|Bare-metal Runtime、算子注册|


## **表4：Model Benchmark 模型性能对比**

性能数据不能直接根据 TOPS 编造 FPS。这里将理论计算上限和实际公开测试分开。

### **4.1 统一模型测试基准**

|模型|应用|输入尺寸|典型计算量|主要考察能力|
|---|---|---|---|---|
|MobileNetV1 1.0|分类|224×224|1.14 GOP|DepthwiseConv|
|MobileNetV2 1.0|分类|224×224|0.60 GOP|Depthwise、Residual|
|ResNet18|分类|224×224|3.6 GOP|Conv、Add|
|ResNet50|分类|224×224|8.2 GOP|Conv、Residual|
|YOLOv5n|目标检测|640×640|4.5 GOP|Conv、Concat、SiLU|
|YOLOv5s|目标检测|640×640|16.5 GOP|算力、带宽|
|YOLOv8n|目标检测|640×640|8.7 GOP|算子融合、Split、Concat|
|YOLOv8s|目标检测|640×640|28.6 GOP|算力、内存调度|
|YOLOv8m|目标检测|640×640|78.9 GOP|多核调度、带宽|
|SSD-MobileNetV2|目标检测|300×300|随网络配置变化|Depthwise、后处理|
|UNet|图像分割|256×256|随网络配置变化|Upsample、Concat|
|DeepLabV3+|图像分割|512×512|随 Backbone 变化|DilatedConv、Resize|

统一将一次 MAC 计为两次运算，具体计算量仍需以实际模型文件为准。

### **4.2 理论最大 FPS 对比**

计算公式：

$$
FPS_{max}=\frac{TOPS\times1000}{GOP/model}
$$


|产品|INT8 TOPS|MobileNetV2|ResNet50|YOLOv8n|YOLOv8s|
|---|---|---|---|---|---|
|Ethos-U55（256 MAC，500MHz）|0.256|427|31|29|9|
|Ethos-U65（512 MAC，500MHz）|0.512|853|62|59|18|
|Ethos-U85（2048 MAC，1GHz）|4.096|6827|500|471|143|
|RK3588|6|10000|732|690|210|
|TDA4VM|8|13333|976|920|280|
|RZ/V2H Dense|8|13333|976|920|280|
|AX650A|18|30000|2195|2069|629|
|Hailo-8L|13|21667|1585|1494|455|
|Hailo-8|26|43333|3171|2989|909|

**这张表是理论计算上限，不是实际测试 FPS。** 未考虑 DDR 带宽、计算单元利用率、算子回退、数据搬运、前后处理及流水线开销。


## **一、完善 NPU 竞品分析框架**

|编号|分析维度|主要内容|优先级|状态|
|---|---|---|---|---|
|01|Hardware Specification|算力、频率、精度、产品定位|P0|已建立|
|02|Operator Support|支持的算子类型|P0|已建立|
|03|Compiler & SDK|编译器、Runtime、工具链|P0|已建立|
|04|Model Benchmark|FPS、Latency、模型性能|P0|已建立|
|05|Compute Architecture|计算阵列、并行度、计算流水线|P0|建议新增|
|06|Memory & Dataflow|SRAM、Bank、DMA、Tiling|P0|建议新增|
|07|Operator Constraints|算子参数、形状、精度限制|P0|建议新增|
|08|Quantization|量化策略、精度损失、混合精度|P0|建议新增|
|09|Power & Efficiency|功耗、面积、算力利用率|P1|建议新增|
|10|System Integration|CPU/NPU 协同、任务调度、部署方式|P1|建议新增|


## **二、表5：Compute Architecture（计算架构）**

6 TOPS 的两个 NPU，在执行同一个模型时，性能可能相差很大。原因往往是计算架构不同。

|分析项|具体分析内容|设计意义|
|---|---|---|
|Compute Architecture|Systolic Array、MAC Array、Dataflow|决定基本计算架构|
|MAC Array Size|MAC 阵列行列数|决定计算并行度|
|MACs per Cycle|每周期 MAC 数量|决定峰值算力|
|Input Channel Parallelism|输入通道并行度|影响 Conv2D 性能|
|Output Channel Parallelism|输出通道并行度|影响 Filter 并行计算|
|Spatial Parallelism|H/W 方向并行度|影响特征图处理效率|
|Multi-core|多核计算架构|影响任务并行能力|
|Depthwise Engine|是否有专用深度卷积计算单元|影响轻量级模型性能|
|Pooling Engine|是否有独立 Pooling 单元|减少 MAC 资源占用|
|Activation Engine|是否有独立激活单元|支持算子融合|
|Elementwise Engine|Add、Mul 等计算单元|影响残差网络性能|
|Vector Engine|向量计算单元|支持非卷积类算子|
|Accumulator|累加器位宽|影响精度和溢出|
|Sparsity|稀疏计算支持|提升有效算力|
|Pipeline|计算流水线组织|影响计算吞吐率|
|Operator Parallelism|不同算子能否同时执行|影响子图执行效率|
|Dynamic Shape|是否支持动态形状|影响模型适用范围|


## **三、表6：Memory & Dataflow（存储与数据流）**

这张表非常重要。对于嵌入式 NPU，很多时候性能瓶颈不是计算能力，而是数据搬运。

|分析项|具体分析内容|
|---|---|
|Internal SRAM Size|NPU 内部 SRAM 总容量|
|SRAM Bank Number|SRAM Bank 数量|
|SRAM Bank Width|每个 Bank 的访问位宽|
|SRAM Bank Depth|每个 Bank 的存储深度|
|SRAM Bandwidth|内部 SRAM 读写带宽|
|Bank Conflict|Bank 冲突及解决机制|
|Memory Partition|IFM、Weight、OFM、Bias 的存储划分|
|Shared SRAM|CPU 和 NPU 是否共享 SRAM|
|External Memory|DDR、Flash、系统 SRAM|
|DMA Channel|DMA 通道数量|
|DMA Bandwidth|DMA 搬运带宽|
|DMA Stride|是否支持跨行、跨通道搬运|
|DMA Scatter-Gather|是否支持非连续地址搬运|
|Double Buffering|是否支持 Ping-Pong Buffer|
|Compute-DMA Overlap|计算和 DMA 是否并行|
|IFM Tiling|输入特征图分块|
|OFM Tiling|输出特征图分块|
|Weight Tiling|权重分块|
|Partial Sum|部分累加结果保存机制|
|Weight Reuse|权重复用机制|
|IFM Reuse|输入特征图复用机制|
|OFM Reuse|输出特征图复用机制|
|Dataflow|Weight/Output/Input Stationary 等|
|Intermediate Tensor|中间 Tensor 是否需要写回 DDR|
|Cross-layer Scheduling|跨算子存储与计算调度|
|Compression|权重、特征图压缩|
|Cache Coherency|CPU 与 NPU 的缓存一致性|
结合我们之前研究的 NACC_DMAParamsInit、SRAM Bank、IFM Split、Kernel Split，这张表可以帮助我们理解其他 NPU 是如何解决相同问题的。

特别需要研究：

- 一个算子的数据超过内部 SRAM 容量时，如何切分？
    
- 切分策略是编译时决定，还是 Runtime 动态决定？
    
- 输入通道切分后，Partial Sum 如何保存？
    
- DMA 搬运下一块数据时，NPU 能否继续计算？
    
- 多个算子之间能否复用内部 SRAM？


## **四、表7：Operator Constraints（算子参数约束）**

之前的算子支持表还不够细致。

例如，两个 NPU 都支持 Conv2D，但是一个只支持有限的 Kernel Size、Stride 和 Dilation，另一个支持更灵活的参数配置，那么两者实际的模型覆盖率可能差别很大。

|算子|需要比较的参数|
|---|---|
|Conv2D|Kernel H/W|
|Conv2D|Stride H/W|
|Conv2D|Dilation H/W|
|Conv2D|Padding Top/Bottom/Left/Right|
|Conv2D|最大 Input Channel|
|Conv2D|最大 Output Channel|
|Conv2D|Channel Alignment|
|Conv2D|最大 Feature Map H/W|
|Conv2D|Group 数量|
|Conv2D|Batch Size|
|DepthwiseConv2D|Depth Multiplier|
|DepthwiseConv2D|Kernel、Stride、Dilation|
|ConvTranspose|Output Padding、Stride|
|FullyConnected|最大输入输出维度|
|MatMul|M/N/K 维度及对齐限制|
|MaxPool|Kernel、Stride、Padding|
|AveragePool|Kernel、Stride、Padding|
|Add/Mul/Sub|Broadcasting 支持|
|Concat|支持的 Axis|
|Split|支持的 Axis、Split Size|
|Transpose|支持的 Permutation|
|Resize|Nearest、Bilinear、坐标转换方式|
|Softmax|Axis、最大输入长度|
|Reduce|支持的 Axis|
|Pad|Constant、Reflect、Edge|
|Activation|激活函数及参数限制|
|通用约束|NHWC/NCHW Layout|
|通用约束|Tensor Rank|
|通用约束|Tensor Size|
|通用约束|数据地址对齐|
|通用约束|输入输出数据类型组合|

### **建议每个核心算子建立独立对比表**

例如 Conv2D：

|参数|Ethos-U55|Ethos-U65|RK3588|NACC|自研 NPU|
|---|---|---|---|---|---|
|Kernel Size||||||
|Stride||||||
|Dilation||||||
|Padding||||||
|Input Channel||||||
|Output Channel||||||
|Group||||||
|Layout||||||
|Input Precision||||||
|Weight Precision||||||
|Accumulator Precision||||||
|Fused Activation||||||
|Fused Pooling||||


[ethos-u-vela/SUPPORTED_OPS.md at lf-6.18.20_2.0.0 · nxp-imx/ethos-u-vela](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/SUPPORTED_OPS.md?utm_source=chatgpt.com)


## **五、表8：Quantization（量化能力）**

量化不仅是编译器的事情，也会直接影响 NPU 硬件设计。

|分析项|具体分析内容|
|---|---|
|Weight Precision|INT4、INT8、INT16|
|Activation Precision|INT8、INT16、FP16|
|Bias Precision|INT32、INT64 等|
|Accumulator Precision|累加器位宽|
|Symmetric Quantization|对称量化|
|Asymmetric Quantization|非对称量化|
|Per-Tensor Quantization|每个 Tensor 使用独立 Scale|
|Per-Channel Quantization|每个 Channel 使用独立 Scale|
|Per-Group Quantization|分组量化|
|Zero Point|Zero Point 硬件处理|
|Scale|Scale 的表示及处理方式|
|Requantization|Multiplier、Shift|
|Rounding Mode|舍入模式|
|Saturation|饱和处理|
|Mixed Precision|混合精度计算|
|PTQ|训练后量化|
|QAT|量化感知训练|
|Calibration|校准数据集及校准算法|
|Quantization Error|量化精度损失|
|Bit-exact|硬件和软件参考模型是否一致|

## **六、表9：Power & Efficiency（功耗与计算效率）**

| 分析项                       | 单位       | 评价目的        |
| ------------------------- | -------- | ----------- |
| Peak Compute              | TOPS     | 峰值算力        |
| Effective Compute         | TOPS     | 实际有效算力      |
| MAC Utilization           | %        | MAC 计算单元利用率 |
| NPU Active Power          | W        | NPU 工作功耗    |
| Idle Power                | mW       | 空闲功耗        |
| System Power              | W        | 整个系统功耗      |
| Energy Efficiency         | TOPS/W   | 计算能效        |
| Model Efficiency          | FPS/W    | 实际模型能效      |
| Energy per Inference      | mJ       | 单次推理能耗      |
| DDR Bandwidth             | GB/s     | 外部存储带宽      |
| DDR Traffic per Inference | MB       | 单次推理数据搬运量   |
| SRAM Bandwidth            | GB/s     | 内部存储带宽      |
| Compute Density           | TOPS/mm² | 单位面积算力      |
| Process Node              | nm       | 制造工艺        |
| DVFS                      | MHz/V    | 动态频率电压调整    |
| Clock Gating              | Yes/No   | 时钟门控        |
| Power Gating              | Yes/No   | 电源门控        |
| Thermal Throttling        | %        | 持续运行性能下降    |


## **七、表10：System Integration（系统集成能力）**

如果我们做的是面向 Bare-metal/RTOS 的嵌入式 NPU，这部分尤其重要。

|分析项|具体分析内容|
|---|---|
|Host CPU|ARM、RISC-V、DSP|
|Host Interface|MMIO、Custom Instruction、PCIe 等|
|Command Interface|Register、Command Stream、Descriptor|
|Command Queue|硬件命令队列|
|Interrupt|完成中断、异常中断|
|Execution Mode|同步、异步执行|
|Multi-task|多任务支持|
|Multi-model|多模型并发|
|Task Priority|任务优先级|
|Preemption|NPU 任务抢占|
|Context Switch|NPU 上下文切换|
|Real-time|确定性执行、最坏执行时间|
|Memory Protection|MPU、MMU、IOMMU|
|Cache Maintenance|Cache Flush、Invalidate|
|Zero-copy|CPU/NPU 零拷贝|
|ISP Integration|ISP 输出直接进入 NPU|
|Camera Integration|Camera Pipeline|
|CPU Fallback|不支持算子的 CPU 执行|
|Bare-metal Driver|裸机驱动|
|RTOS Driver|RTOS 驱动|
|Linux Driver|Linux 驱动|
|Error Handling|Timeout、Bus Error、Illegal Command|
|Recovery|NPU Reset、任务恢复|
|Performance Counter|Cycle、Stall、DMA、MAC Counter|
|Debug Interface|寄存器、Tensor Dump、Trace|
|Functional Safety|ECC、Parity、Watchdog 等|

这张表最终会影响 SDK Driver、HAL、Runtime、RTOS Porting 和调试工具的设计。



## **八、还应该补充一个关键内容：模型算子覆盖率**

不能只统计 NPU 支持多少种算子，还应该分析目标市场的实际模型使用了哪些算子，以及这些算子消耗了多少计算量。

|指标|计算方式|意义|
|---|---|---|
|Operator Type Coverage|支持的算子类型 / 模型算子类型|算子种类覆盖率|
|Operator Instance Coverage|NPU 执行节点数 / 总节点数|实际节点覆盖率|
|Compute Coverage|NPU 执行计算量 / 总计算量|计算量覆盖率|
|Latency Coverage|可加速部分的原始耗时 / 原始总耗时|潜在加速收益|
|Subgraph Count|NPU 子图数量|图划分碎片化程度|
|CPU/NPU Boundary|CPU/NPU 切换次数|异构调度开销|
|Fusion Ratio|融合减少的节点数 / 原始节点数|图优化程度|
|External Memory Traffic|每次推理外部存储访问量|数据搬运成本|

例如，一个 NPU 支持模型中 95% 的算子节点，但是剩下的 5% 占据了大量计算时间，那么实际加速效果仍然可能很差。

因此，建议将这些指标补充到原来的 Model Benchmark 表中。




