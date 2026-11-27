# Benchmark与回归

固定模型哈希、输入数据、量化方式、目标频率、内存配置、编译参数、runtime/driver版本。预热后多次测量，记录中位数和尾延迟。CI 依次执行 lint、unit、模型编译、差分验证和硬件回归；硬件不可用应标记 BLOCKED 而非 PASS。
