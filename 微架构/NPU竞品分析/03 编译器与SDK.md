# 编译器与SDK

[[00 首页]] · [[11 官方资料索引]]

> 数据继承自前序调研，本次仅整理归档，未重新在线核验。未找到表示前序调研未取得充分证据。

| 能力 | Arm | Rockchip | TI | Renesas | Hailo |
| --- | --- | --- | --- | --- | --- |
| 编译器 | Vela | RKNN Toolkit2 | TIDL | DRP-AI TVM | Dataflow Compiler |
| Runtime | Ethos Driver/TFLM | RKNN Runtime | TIDL | DRP-AI Runtime | HailoRT |
| 源码 | 公开 | 核心未公开 | 核心未公开 | 部分公开 | Runtime公开 |
| 模型输入 | TFLite/TOSA | ONNX/TFLite | ONNX/TFLite | TVM/ONNX | ONNX |
| Tiling | Vela支持 | 未找到 | 未找到 | 未找到 | 未找到 |
| Memory Planning | Vela支持 | 未找到 | 未找到 | 未找到 | 未找到 |
| 命令输出 | Command Stream | RKNN | 编译模型 | DRP-AI Object | HEF |
| CPU回退 | 上层框架 | 依赖Runtime | 待版本核验 | TVM混合执行 | Host处理 |

## 资料入口

- [Vela](https://github.com/nxp-imx/ethos-u-vela)
- [Ethos Driver](https://github.com/ARM-software/CMSIS-Ethos-U)
- [RKNN](https://github.com/airockchip/rknn-toolkit2)
- [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools)
- [DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm)
- [HailoRT](https://github.com/hailo-ai/hailort)
- [TFLM](https://github.com/tensorflow/tflite-micro)
