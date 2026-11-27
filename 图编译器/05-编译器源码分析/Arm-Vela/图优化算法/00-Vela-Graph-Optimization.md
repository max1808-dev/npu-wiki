# Vela Graph Optimization — 专题索引

基线：nxp-imx/ethos-u-vela commit `395a5690`，传统 Python TFLite 路径。五页按功能而非实际 pass 执行顺序组织。

- [[01-Passthrough-Tensors]]
- [[02-Concat-and-Split]]
- [[03-Resize-Conversion]]
- [[04-Weight-Reordering]]
- [[05-Bias-and-Quantization]]

阅读方法：先看数学语义，再看源码条件、IR 改写、硬件接口与测试建议。示例为手算，未执行 Vela 或 NPU；没有穷举所有支持约束或 Regor 实现。
