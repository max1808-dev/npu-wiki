# NPU Driver与命令流

Runtime 的 custom kernel 获取编译产物、输入输出缓冲和 scratch，driver 配置目标设备、提交命令并等待完成。命令中的偏移与运行时实际基址必须符合 ABI；缓存一致性、中断及超时是平台相关问题。

[Arm Driver 文档](https://arm-software.github.io/CMSIS-Ethos-U/main/driver/index.html)
