# Vela研究基线

已确认：Arm Vela 是面向 Ethos-U 的 AOT 编译器；TFLite 输出可以包含携带 NPU 命令流的自定义算子。官方指南列出 U55/U65/U85、内存配置和性能报告。

- [Arm 官方指南](https://arm-software.github.io/CMSIS-Ethos-U/main/vela/index.html)
- [NXP 仓库](https://github.com/nxp-imx/ethos-u-vela)
- [Arm 上游](https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela)
- [NXP BUILD.md](https://github.com/nxp-imx/ethos-u-vela/blob/lf-6.18.20_2.0.0/BUILD.md)

本版未能克隆仓库或执行 Vela，因此不锁定一个未经确认的 commit。传统 Python 路径与较新 C++ 后端分开研究；以下文件名为传统路径候选阅读入口，不是已验证调用链。

```bash
git clone https://github.com/nxp-imx/ethos-u-vela.git
cd ethos-u-vela
git rev-parse HEAD
git branch --show-current
find . -maxdepth 2 -type d
python -m pip install .
vela --help
```
记录 commit、Python 版本、目标 NPU、配置文件、memory mode 和模型哈希。
