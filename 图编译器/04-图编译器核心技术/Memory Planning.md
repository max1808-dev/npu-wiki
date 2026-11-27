# Memory Planning

为张量计算 first/last use；生命周期不相交的张量可复用地址。考虑 alignment、bank、持久常量和 scratch；峰值占用不是所有 Tensor 大小之和。
