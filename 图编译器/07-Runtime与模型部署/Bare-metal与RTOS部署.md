# Bare-metal与RTOS部署

Bare-metal 关注 linker memory region、栈和 arena、IRQ；RTOS 增加互斥、任务同步、并发访问及超时恢复。CPU fallback 指编译分区中未卸载的算子由 CPU kernel 执行，不代表 NPU 执行失败后必然自动回退。
