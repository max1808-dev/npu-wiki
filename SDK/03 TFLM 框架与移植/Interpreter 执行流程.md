# Interpreter 执行流程

[[NPU SDK 首页]] · [[03 TFLM 框架与移植 索引]]

## 基础内容与研究方向

典型流程：GetModel → 注册算子 → 创建 MicroInterpreter → AllocateTensors → 填充输入 → Invoke → 读取输出。具体函数签名随版本变化。

## 验证与产出

- [ ] 固定相关源码或规范版本
- [ ] 补充真实代码位置、接口或实验结果
- [ ] 记录结论与适用范围

## 证据与参考

[[总览]] · [[源码分析模板]]
