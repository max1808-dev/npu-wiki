---
title: Vela 许可证与商业化分析
tags: [图编译器, Vela, License, NPU, 开源合规]
updated: 2026-11-26
---

# Vela 许可证与商业化分析

相关页面：[[Vela参考资料与阅读指南]]

## 1. 结论与适用范围

Arm Vela 的项目许可证为 Apache-2.0。NXP fork 的 outgoing license 同样为 Apache-2.0，但包含 BSD-3-Clause 来源代码和其他第三方依赖。该许可证允许商业使用、修改、私有 fork、闭源衍生产品以及源码或二进制再发布，不要求向 Arm 回馈修改。前提是遵守适用许可证的义务。

本页核查对象为 NXP 仓库 `lf-6.18.20_2.0.0` 分支及 Arm PyPI 项目声明，并非对最终产品所有文件的逐项法律审计。正式选型时应锁定 commit SHA。

## 2. 官方证据

| 资料 | 核查结果 |
|---|---|
| [NXP LICENSE.txt](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/LICENSE.txt) | Apache License 2.0 正文 |
| [NXP SCR.txt](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/SCR.txt) | outgoing Apache-2.0；Arm v5.0.0 来源；NXP BSD-3-Clause 来源 |
| [NXP THIRDPARTY.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/THIRDPARTY.md) | C/C++ 第三方组件及版本 |
| [NXP SBOM](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/SBOM.spdx.json) | 软件物料清单入口，尚未逐项审核 |
| [Arm Vela upstream](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela) | 上游源码入口 |
| [Arm PyPI](https://pypi.org/project/ethos-u-vela/) | Arm 发布包许可证声明 Apache-2.0 |
| [Apache 官方许可证](https://www.apache.org/licenses/LICENSE-2.0) | 权利义务的权威英文文本 |
| [NXP setup.py](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/setup.py) | 文件级 Apache 与 BSD 声明的实例 |

## 3. 自研 NPU 使用场景

| 场景 | 结论 | 说明 |
|---|---|---|
| 私有 Git 仓库 | 允许 | 不要求公开 fork |
| 更换 Ethos-U backend | 允许 | 可增加自研 ISA、DMA、指令生成 |
| 修改图优化、融合、调度、内存规划 | 允许 | 对外分发时标注修改文件 |
| SDK 免费或收费交付 | 允许 | 随包提供适用许可证和声明 |
| 只提供 wheel、可执行文件或动态库 | 允许 | 不要求交付对应源码 |
| 自研代码使用商业许可证 | 允许 | 不得取消原有 Apache/BSD 代码的适用权利和义务 |
| 不向上游提交修改 | 允许 | 上游贡献属于自主选择 |
| 更换产品名称 | 允许 | 不得暗示 Arm/NXP 背书 |
| 删除原有适用版权声明 | 不允许 | 保留适用声明 |
| 使用 Arm 硬件 IP | 不由本许可授权 | 另行审查硬件 IP、规格及专利 |

## 4. Apache-2.0 条款解读

- 第 2 条：允许复制、修改及分发源码或目标形式。
- 第 3 条：专利许可仅涵盖贡献者有权授予、且其贡献所必然涉及的相关专利权利要求；针对作品或贡献提起规定的专利侵权诉讼时，相关专利许可终止。并非所有 NPU 专利的免责许可。
- 第 4 条：向接收方提供许可证副本；修改文件应带有醒目的修改声明；分发衍生源码时保留适用的原版权、专利、商标及署名声明；若原作品附带 NOTICE，则按条款传递其中适用内容。可以给自己的修改或衍生作品整体设置额外或不同许可条件，但仍须遵守原作品许可证。
- 第 5 条：主动提交上游的贡献通常适用 Apache-2.0，除非明确另作安排；不要求提交。
- 第 6 条：不授予一般商标使用权，合理描述来源及复现 NOTICE 的必要使用除外。
- 第 7—9 条：原作者通常不提供担保；商业产品的质量保证、支持及责任需要自行承担。

## 5. 第三方依赖核查

以下为指定分支公开清单中的记录，不代表最终安装包的完整清单。

| 组件 | 版本（声明值） | 许可证 | 交付注意事项 |
|---|---|---|---|
| FlatBuffers (C++) | v24.12.23 | Apache-2.0 | 许可证、适用声明 |
| fmt | 11.1.1 | MIT | 版权及许可文本 |
| Catch2 | v3.8.0 | BSL-1.0 | 核查是否只用于测试、实际分发形式 |
| Gemmlowp | 09d81e02ab15b41405caebeb5eb63fd12555aee3 | Apache-2.0 | 许可证、适用声明 |
| pybind11 | v2.13.6 | BSD-3-Clause | 版权、条件及免责声明 |
| FlatBuffers (SCR) | 24.3.25 | Apache-2.0 | 核实实际 Python 依赖版本 |
| NumPy | 1.26.4 | BSD-3-Clause | 同时审查 wheel 内捆绑库 |
| lxml | 5.3.0 | BSD（SCR 原文） | 核实精确许可及 libxml2/libxslt 等捆绑组件 |
| NXP 来源代码 | 按文件 | BSD-3-Clause | 源码和二进制分发均应按条款保留声明，不得暗示背书 |

上述依赖主要为宽松许可，但不能据此认定全部传递依赖均已合规。SCR 和 THIRDPARTY 的 FlatBuffers 版本不同，应区分依赖层级并以实际构建产物为准。BSL-1.0 在此指 Boost Software License，不是 Business Source License。

## 6. 推荐工程方案

采用私有 fork，保留原代码版权和许可证文件；在独立目录实现自研 Backend、Hardware Description、Command Generator，并尽量明确接口边界。新模块可采用公司许可。此架构有利于维护和审计，但不是 Apache-2.0 的强制要求。

修改原文件时可在保留原文件头的基础上增加如下声明（公司名、年份按实际情况填写）：

```python
# Modified by <Company> in <Year>.
# Changes: Adapted scheduling and code generation for the custom NPU.
```

Git commit 记录建议保留，但不能代替修改文件本身的醒目声明。

建议发布包结构（示例，非强制命名）：

```text
NPU-SDK/
  bin/ 或 wheels/
  LICENSES/
    Apache-2.0.txt
    BSD-3-Clause.txt
    MIT.txt
    ...
  THIRD_PARTY_NOTICES.txt
  NOTICE                 # 原作品存在适用 NOTICE 时按条款传递
  SBOM.spdx.json
  README.md
```

第三方声明文件应包含实际适用的原始版权和免责声明，而不是只有许可证名称。若提供源代码包，保留各文件原始适用声明和修改标记。若仅二进制交付，仍需要随包提供适用许可证和通知。

## 7. 发布前检查清单

- [ ] 固定上游仓库、分支及 commit SHA，保存原始许可证和来源。
- [ ] 扫描所有源码文件头、子模块、生成代码、模型、测试数据、文档及第三方组件的许可。
- [ ] 区分 Apache、BSD、MIT 等文件级许可证，处理特殊或未知许可。
- [ ] 检查修改文件是否具有修改声明。
- [ ] 生成实际发布物的 SBOM，覆盖 Python wheel、C/C++ 库及传递依赖。
- [ ] 检查最终安装包是否附带适用许可证、版权及 NOTICE。
- [ ] 检查品牌命名、Logo、营销材料及专利风险。
- [ ] 单独核查 TFLM、NPU Driver、RTOS、硬件规格、模型权重及数据集的许可。
- [ ] 将开源合规检查纳入 CI/release gate（如 ScanCode、ORT、REUSE 等工具）。
- [ ] 商业发布前完成公司法务及 IP 审核。

## 8. 硬件及模型边界

Vela 的软件许可证不等同于 Ethos-U RTL、硬件 IP、商标或单独授权的技术规格许可证；也不能证明自研 NPU 不侵犯其他专利。模型编译输出通常不会仅因使用 Vela 自动变成 Apache-2.0；但若输出包含受许可约束的代码、数据或模板，需要单独判断，原始模型及权重的许可仍然有效。

## 9. 选型判断

许可层面：可作为商业自研图编译器候选基线，无需仅因为使用 Vela 而公开自研源码。

本文为工程合规分析，不替代正式法律意见。当前未完成完整文件级扫描、最终二进制依赖审计或专利自由实施分析（FTO）。
