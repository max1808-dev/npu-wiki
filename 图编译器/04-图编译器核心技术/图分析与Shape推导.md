# 图分析与Shape推导

构建 def-use 关系，拓扑遍历推导输出 shape；检查广播、维度和数据类型。以卷积为例，有效 kernel = dilation × (K−1)+1，VALID 输出为 floor((I−有效kernel)/stride)+1。
