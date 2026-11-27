---
title: Vela 参考资料与阅读指南
tags:
  - 图编译器
  - Vela
  - Ethos-U
  - NPU
  - 参考资料
updated: 2026-11-26
---

# Vela 参考资料与阅读指南

## 1. 文档目的

本文汇总 Arm Ethos-U Vela 的源码分析、官方技术资料及工程文档，服务于 Vela 源码研究和自研 NPU 图编译器设计。每项资料均包含链接、内容总结、建议研究方向及使用限制。

## 2. 资料索引

| 编号  | 资料                              | 类型       | 阅读优先级 |
| --- | ------------------------------- | -------- | ----- |
| 01  | DeepWiki Overview               | 自动生成源码分析 | P0    |
| 02  | DeepWiki Compilation Pipeline   | 自动生成源码分析 | P0    |
| 03  | DeepWiki Configuration Options  | 自动生成源码分析 | P1    |
| 04  | DeepWiki Data Representation    | 自动生成源码分析 | P0    |
| 05  | DeepWiki TFLite Serialization   | 自动生成源码分析 | P1    |
| 06  | Arm CMSIS-Ethos-U Vela          | 官方文档     | P0    |
| 07  | Vela AGENTS.md                  | 仓库开发指南   | P0    |
| 08  | Arm Vela 技术博客                   | 官方入门文章   | P0    |
| 09  | Ethos-U Driver                  | 官方驱动文档   | P1    |
| 10  | PERFORMANCE.md                  | 仓库性能说明   | P0    |
| 11  | OPTIONS.md                      | 仓库配置说明   | P1    |
| 12  | API.md                          | 仓库低层接口说明 | P0    |
| 13  | NXP Vela 源码仓库                   | 一手源码     | P0    |
| 14  | CMSIS-Ethos-U Integration       | 官方系统集成指南 | P1    |
| 15  | BUILD.md / CHANGELOG.md         | 构建及版本资料  | P1    |
| 16  | DeepWiki Ethos-U85 Architecture | 自动生成架构分析 | P2    |

P0：首轮必读；P1：专题研究；P2：按目标硬件需求阅读。

## 3. DeepWiki 源码分析

### 3.1 Overview（整体架构）

- 链接：[DeepWiki — NXP Ethos-U Vela](https://deepwiki.com/nxp-imx/ethos-u-vela)

**内容总结：** 介绍 Vela 在模型部署中的角色、输入格式、目标 NPU 与模块组织。从高层看，Vela 将模型转换为包含硬件可执行内容的产物。对于 TFLite 部署路径，可加速部分可被替换为自定义算子，其他计算仍由 CPU 执行。文档还给出了模型读取、图优化、调度、内存管理、命令生成与序列化之间的联系。

**研究价值：** 建立模块责任表，画出离线编译器与运行时的边界。建议把文档中的源码引用作为阅读入口，而不是把自动生成的描述当成规范。

**建议输出：** Vela 软件架构图；模块与目录对应表；输入输出文件说明。

### 3.2 Compilation Pipeline（编译流水线）

- 链接：[Compilation Pipeline](https://deepwiki.com/nxp-imx/ethos-u-vela/2-compilation-pipeline)

**内容总结：** 按模型解析、图改写、Tensor 用途标记与 Pass Packing、NPU 子图提取、调度与存储规划、Command Stream 生成、输出写回等阶段组织内容，并另行介绍 Regor 编译路径及调试数据库。涉及 `ethosu/vela/vela.py`、`compiler_driver.py` 和 `ethosu/regor/compiler/compiler.cpp` 等源码入口。

**研究价值：** 适合作为源码调用链研究的主线。建议逐阶段记录输入对象、输出对象、变换前后不变量以及失败路径。特别关注算子支持判断与实际子图划分的关系，以及调度如何影响硬件命令。

**建议实验：** 构造 Conv2D → MaxPool → Add 小模型，记录原始图、优化图、CPU/NPU 分区、内存分配与最终命令。注意具体融合结果必须通过实验确认。

**限制：** 页面同时讨论 Python 与 Regor，不能把其中所有步骤视为同一版本、同一路径的固定顺序。

### 3.3 Configuration Options（配置选项）

- 链接：[Configuration Options](https://deepwiki.com/nxp-imx/ethos-u-vela/2.2-configuration-options)

**内容总结：** 介绍 Vela 如何接受不同 NPU 与系统配置，涉及目标加速器、系统参数、内存模式及编译策略。相同模型在不同配置下可能产生不同的编译结果与资源消耗。

**研究价值：** 自研编译器也应建立独立硬件描述层，将 SRAM 容量、计算能力、存储带宽、对齐、支持的数据类型等约束与通用图算法分离。建议追踪配置解析到调度器、分配器和命令生成器的传播路径。

**限制：** CLI 参数与默认值应以选定版本的 `OPTIONS.md` 和 `vela --help` 为准。

### 3.4 Data Representation（数据表示）

- 链接：[Data Representation](https://deepwiki.com/nxp-imx/ethos-u-vela/7-data-representation)

**内容总结：** 以 Tensor 为核心讨论编译器中的数据描述、存储及序列化。Tensor 不仅是数组，还需要携带 shape、dtype、量化信息等属性；编译器进一步需要表达数据布局、存储位置及中间结果的使用关系。

**研究价值：** 建议研究 Tensor 与 Operator、Graph、Subgraph 的关联方式，区分模型原始数据与编译期间派生的数据。形成自研 IR 的最小字段表，明确哪些字段属于逻辑语义，哪些属于物理执行计划。

### 3.5 TFLite Serialization（模型读写）

- 链接：[TFLite Serialization](https://deepwiki.com/nxp-imx/ethos-u-vela/7.2-tflite-serialization)

**内容总结：** 关注 TFLite FlatBuffer 与 Vela 内部表示之间的转换，以及编译完成后模型的序列化接口。

**研究价值：** 对照 TFLite Schema 研究 Tensor、Operator、Buffer、Subgraph 索引如何映射到 IR，跟踪自定义算子和常量如何写回模型。使用 Netron 辅助比较原始与编译后模型，但命令流内容仍需要专门的解析或调试工具。

### 3.6 其他专题入口

从 [DeepWiki 首页](https://deepwiki.com/nxp-imx/ethos-u-vela) 的目录进入 Graph Optimization、Scheduling、Memory Management、Command Stream Generation 等专题。建议优先读对应源码引用，再为每个专题建立独立笔记。未逐页核查的子页面不在此列出猜测的 URL。

## 4. Arm 官方技术资料

### 4.1 CMSIS-Ethos-U — Vela

- 链接：[Arm Vela 官方文档](https://arm-software.github.io/CMSIS-Ethos-U/main/vela/index.html)

**内容总结：** 覆盖安装、模型编译、目标硬件设置、`vela.ini` 配置、存储放置与诊断报告。文档说明如何比较不同 Ethos-U 配置，如何在 SRAM 容量与执行效率间选择优化策略，以及怎样检查 CPU fallback。也介绍固件中的模型数据放置与运行时集成要求。

**研究价值：** 这是理解 Vela 外部行为和搭建可复现 benchmark 的首选。应分别记录编译器内存估算和完整固件内存占用；两者不是同一个指标。

**建议实验：** 固定模型、编译器版本和硬件配置，只改变 Size/Performance 策略或 cache 预算，记录峰值 SRAM、输出大小和性能估算。

### 4.2 Arm 技术博客 — Vela Compiler

- 链接：[Vela Compiler: The first step to deploy your NN model on the Arm Ethos-U microNPU](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/vela-compiler-deploy-your-nn-model-on-the-arm-ethos-u-micronpu)

**内容总结：** 从开发者角度描述量化模型准备、Vela 安装、算子支持检查、配置准备和离线编译过程。文章强调：支持某个算子名称并不表示该算子的任意参数都能被 NPU 加速；不符合限制的计算可留在 CPU 路径。

**研究价值：** 很适合作为 Wiki 入门章节的参考。建议据此建立算子支持矩阵，将 dtype、shape、kernel、stride、padding、量化限制与 fallback 原因分开记录。

**限制：** 属于较早的文章，不应据此断定当前 Regor 后端、U85 或最新 CLI 的全部行为。

### 4.3 CMSIS-Ethos-U — Driver

- 链接：[Ethos-U Driver 官方文档](https://arm-software.github.io/CMSIS-Ethos-U/main/driver/index.html)

**内容总结：** 说明驱动如何接收编译产物与内存区域地址、启动 NPU、处理完成及故障中断，并支持同步或异步执行与 PMU 性能监控。驱动与平台的电源、缓存和 RTOS 接口也在其范围内；模型编译和框架 Tensor Arena 的分配则不属于驱动自身责任。

**研究价值：** 用于定义自研图编译器、TFLM custom operator、NPU 驱动三者的接口契约。需要明确命令格式、地址与偏移的解释、缓存一致性、错误返回和版本兼容性。

**建议输出：** Compiler–Runtime–Driver 接口图；NPU Invoke 时序图；硬件 bring-up 检查表。

### 4.4 CMSIS-Ethos-U — Integration

- 链接：[Ethos-U Integration](https://arm-software.github.io/CMSIS-Ethos-U/main/integration/index.html)

**内容总结：** 从整个 MCU 系统角度说明编译配置、链接布局、驱动内存访问配置与目标板验证之间的对应关系。尤其关注模型选择与内存规划是否符合物理硬件，以及如何通过实际运行检查编译估算。

**研究价值：** 对自研 NPU SDK 的整体设计很重要。建议建立统一内存区域定义，保证编译器输出、链接脚本、运行时和 DMA 地址空间一致。

## 5. 源码仓库中的工程资料

### 5.1 AGENTS.md — Regor 后端开发指南

- 链接：[AGENTS.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/AGENTS.md)

**内容总结：** 该文件面向 coding agent，描述 Python CLI 与 C++ Regor 后端的混合架构，以及开发安装、CMake 构建、Catch2 单元测试、Sanitizer、代码格式检查等工程操作。它明确指出自身不是面向普通读者的产品说明书。

**研究价值：** 快速识别代码责任边界、搭建可调试的开发环境和测试入口。自研项目也可以借鉴这种方式，将 Agent 指令与详细设计文档分离。

**限制：** 不足以代替完整设计规范，也不能说明所有版本的默认 backend。

### 5.2 PERFORMANCE.md — 性能估算

- 链接：[PERFORMANCE.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/PERFORMANCE.md)

**内容总结：** 说明编译完成后性能摘要与 CSV 报告中的各类指标，包括目标配置、时钟、带宽与周期等。原文特别提醒：估算用于辅助编译优化决策，并不代表真实硬件测量值；不同配置下的估算结果不能直接作为公平性能比较。

**研究价值：** 研究自研编译器的 cost model 时，应把计算成本和存储访问成本分开，同时使用硬件测量校准模型。建议记录预测 cycles、实际 cycles、误差及硬件配置，逐步找出带宽、等待和数据重读造成的偏差。

**建议输出：** 性能指标字典、成本模型说明、预测与 PMU 实测对照表。

### 5.3 OPTIONS.md — 编译配置规范

- 链接：[OPTIONS.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/OPTIONS.md)

**内容总结：** 描述 Vela 的编译参数与优化设置，例如目标加速器、内存和 cache 预算、优化策略等。用于确认命令行参数与编译行为的关系。

**研究价值：** 可作为自研图编译器 CLI 与硬件配置文件设计的参考。建议建立参数、默认值、约束、作用阶段和测试用例的对应表。

### 5.4 API.md — 低层代码生成接口

- 链接：[API.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/API.md)

**内容总结：** 介绍供外部工具使用的低层代码生成 API，即在已有计算描述的基础上产生 Ethos-U 可执行的指令数据。

**研究价值：** 对自研 NPU 尤其重要：可把图级优化与硬件代码生成拆分，通过稳定接口连接两者。建议进一步核实 API 中的 operation 描述、地址、量化、权重编码和命令生成相关字段，不应假设所有 API 都覆盖所有硬件版本。

**建议实验：** 研究能否绕开完整模型编译，通过低层接口生成单算子命令；与完整编译结果进行功能对照。

### 5.5 Vela 源码仓库

- 链接：[NXP GitHub 仓库](https://github.com/nxp-imx/ethos-u-vela)
- 链接：[Arm 上游 GitLab 仓库](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela)

**内容总结：** NXP 仓库提供 Vela 的实现及工程文档，面向包括 i.MX93 在内的 Ethos-U 部署场景。源码是确认实际编译行为的最终依据；Arm 上游可用于追踪版本变化与区分 NXP 下游改动。

**建议操作：**

```bash
git clone https://github.com/nxp-imx/ethos-u-vela.git
cd ethos-u-vela
git rev-parse HEAD
git branch -a
```

项目正式研究开始前，应选定 branch、commit SHA、目标 NPU 和 backend，并记录到 Wiki。上游 GitLab 链接来自仓库 BUILD.md，未单独验证其访问权限。

### 5.6 BUILD.md 与 CHANGELOG.md

- 链接：[BUILD.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/BUILD.md)
- 链接：[CHANGELOG.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/CHANGELOG.md)

**内容总结：** BUILD.md 用于了解源码构建；CHANGELOG.md 用于追踪功能、修复和版本演进。例如，后者记录了 U85 TOSA 算子融合改进及性能报告相关变化。

**研究价值：** 当资料与实际代码不符时，首先检查版本差异。建议用变更记录建立专题演进表，再通过 Git log、diff 和单元测试验证关键改动。

## 6. 硬件架构相关补充

### 6.1 DeepWiki — Ethos-U85 Architecture

- 链接：[Ethos-U85 Architecture](https://deepwiki.com/nxp-imx/ethos-u-vela/6.2-ethos-u85-architecture/)

**内容总结：** 从 Vela 实现角度介绍 U85 的硬件配置、存储系统、算子支持和影响编译决策的架构特性。

**研究价值：** 比较不同 NPU 架构如何影响编译后端设计。但此资料反映的是软件对硬件的建模，不是官方寄存器规范；实际硬件限制应使用对应硬件技术手册核查。

## 7. 建议阅读顺序与 Wiki 产出

| 阶段 | 阅读资料 | 建议产出 |
|---|---|---|
| 第一阶段：建立整体认知 | Overview、Arm 博客、官方 Vela 文档 | 架构图、输入输出说明 |
| 第二阶段：进入源码 | Pipeline、AGENTS、Data Representation | 调用链、IR 数据结构、版本基线 |
| 第三阶段：图优化 | DeepWiki 专题、实际优化源码 | 算子支持矩阵、融合规则、分区算法 |
| 第四阶段：执行计划 | Pipeline、OPTIONS、PERFORMANCE | Tiling、Scheduling、Memory Planning、Cost Model |
| 第五阶段：代码生成 | API、Serialization、Driver | 命令格式、模型写回、Runtime ABI |
| 第六阶段：系统验证 | Integration、PERFORMANCE、BUILD | Benchmark、正确性回归、真实硬件验证 |

## 8. 推荐的最小验证模型

| 模型 | 研究问题 |
|---|---|
| 单个 Conv2D | 权重布局、量化、kernel 参数、tile 和命令生成 |
| 单个 MaxPool | 通道独立处理、窗口参数、边界处理 |
| 单个 Add | 广播、输入量化差异、内存布局 |
| Conv2D + Activation | 融合是否合法、激活是否单独执行 |
| Conv2D + MaxPool | 中间 Tensor 是否物化、是否存在级联或缓冲复用 |
| NPU 支持算子 + 不支持算子 | CPU fallback、子图边界、运行时数据交接 |

建议使用固定随机种子生成输入，将 TFLite CPU 结果与目标 NPU 结果比较。记录模型 hash、Vela commit、硬件配置、编译命令、日志、输出模型和测试结果。

可从以下命令开始（选项可用性须按版本确认）：

```bash
vela --version
vela --help
vela --supported-ops-report
vela model.tflite --accelerator-config ethos-u55-128 --show-cpu-operations
```

## 9. 资料使用原则与当前限制

- DeepWiki 是自动生成资料，可用作源码索引，不能直接当作人工审查过的设计规范。
- Python 后端与 Regor 后端需分别建立调用链。当前资料确认存在 Regor 相关实现和文档，但尚未验证项目选定版本的实际后端选择规则。
- Arm 在线文档的 main 分支会更新，可能与 NXP 分支不一致；引用具体实现时应固定 commit。
- 本文是参考资料指南，不是完整源码审计；未执行模型编译、硬件测试或验证每个函数的实现。
