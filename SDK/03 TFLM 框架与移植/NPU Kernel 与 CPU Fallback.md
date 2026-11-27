# NPU Kernel 与 CPU Fallback

[[NPU SDK 首页]] · [[03 TFLM 框架与移植 索引]]

## 基础内容与研究方向

以 Conv2D 为首个加速目标。检查 dtype、shape、stride、padding、量化和硬件限制；在设计允许时选择 CPU 路径。Fallback 不是自动保证的功能。

## 验证与产出

- [ ] 固定相关源码或规范版本
- [ ] 补充真实代码位置、接口或实验结果
- [ ] 记录结论与适用范围

## 证据与参考

[[总览]] · [[源码分析模板]]
