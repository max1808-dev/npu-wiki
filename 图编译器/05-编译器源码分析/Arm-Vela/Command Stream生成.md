# Command Stream生成

**状态：待固定版本后源码验证。**

候选源码入口：`high_level_command_stream_generator.py / register_command_stream_generator.py`（可能随版本变化）。

## 研究问题
任务如何变成寄存器命令？

## 实验与证据
追踪地址、DMA、计算和同步。

## 分析记录
| 项目 | 记录要求 |
|---|---|
| 输入输出 | 类型、shape、地址或命令 |
| 调用链 | commit、文件、函数及行号 |
| 算法 | 关键条件、循环、不变量 |
| 硬件依赖 | SRAM、DMA、MAC、ISA |
| 自研借鉴 | 通用部分与目标专用部分 |

关联：[[Vela研究基线]]、[[04-图编译器核心技术/00-Index|核心技术]]。
