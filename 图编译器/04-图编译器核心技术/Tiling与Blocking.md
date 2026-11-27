# Tiling与Blocking

根据 SRAM 容量、对齐和硬件 block 限制拆分输出空间或通道。输出 tile 的输入窗口包含 halo：跨度=(输出tile−1)×stride+有效kernel。输入通道切分往往需要跨块累加，不能逐块提前饱和。
