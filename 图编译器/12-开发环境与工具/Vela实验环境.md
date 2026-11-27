# Vela实验环境

建议 Linux + Python virtual environment；具体 Python 版本按锁定源码 BUILD.md。
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
vela --help
vela model.tflite --accelerator-config ethos-u55-128
```
最后一条为示例，需按实际目标及版本 CLI 调整。保存完整命令、stdout/stderr、输出目录和配置。
