# 案例-Conv2D与MaxPool

检查跨算子 tile 传递、pool window 对前一层输出的需求以及中间缓冲大小。

**状态：实验设计，尚未编译或真机验证。**

实验步骤：生成或取得模型 → 记录哈希与量化参数 → Netron 查看原图 → Vela 编译并保存日志 → 比较输出模型和 placement → 跟踪调度/内存/命令 → reference 与目标结果比较。

记录：目标、参数、峰值 SRAM、NPU/CPU 区域、周期估计、实测延迟和误差。
