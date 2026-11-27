# Tensor与布局

Tensor 是多维数据。Shape `[1, 32, 32, 16]` 在 NHWC 表示 batch、高、宽、通道；Rank 为 4。INT8 连续存储占 16384 字节（不含对齐）。NCHW 是另一种轴顺序；stride 是相邻元素在存储中的间隔，与卷积 stride 不同。Blocked layout 可能引入通道填充。
