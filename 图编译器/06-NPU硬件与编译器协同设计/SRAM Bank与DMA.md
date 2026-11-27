# SRAM Bank与DMA

容量=bank数×depth×width/8。若 16 个 512×32 bit bank 加 16 个 256×32 bit bank，则容量为 48 KiB；这是按表达式计算的示例，并非已确认的 NACC 规格。实际编译需要 bank 并行限制、对齐、DMA 最大长度、stride、广播和同步。
