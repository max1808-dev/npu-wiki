# IR与编译阶段

IR（Intermediate Representation）用明确语义表示程序。Graph IR 表示算子连接，Tensor IR 可表示循环和索引，低层 IR 表示硬件任务、地址和同步。并非所有编译器都具有这些独立层。Pass 应明确输入不变量、输出不变量和失败行为。
