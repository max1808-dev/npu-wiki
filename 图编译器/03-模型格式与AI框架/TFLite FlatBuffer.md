# TFLite FlatBuffer

TFLite 文件使用 FlatBuffer。Model 包含 operator_codes、subgraphs 和 buffers；SubGraph 包含 tensors、operators、inputs、outputs。Operator 的 opcode_index 指向 OperatorCode；输入输出索引指向所在子图的 Tensor；Tensor 的 buffer 索引关联常量数据。融合激活可在 builtin options 中表达，独立激活也可能作为节点存在。

[官方 schema](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/compiler/mlir/lite/schema/schema.fbs)
