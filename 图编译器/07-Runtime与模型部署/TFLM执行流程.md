# TFLM执行流程

TFLM 使用模型 FlatBuffer、算子 resolver 与 Tensor Arena。典型流程 GetModel → 构建 MicroInterpreter → AllocateTensors → 写输入 → Invoke → 读输出。Prepare 与 Eval 的细节随 kernel 和版本变化。

[官方源码](https://github.com/tensorflow/tflite-micro)
