# NPU Wiki

[在线阅读](https://max1808-dev.github.io/npu-wiki/)

用 Obsidian 打开仓库根目录进行编辑。推送到 `main` 后，GitHub Actions 会转换笔记、构建站点并发布到 GitHub Pages。

## 本地预览

```sh
python -m pip install -r requirements.txt
python scripts/export_wiki.py
python -m mkdocs serve
```

编辑笔记后重新运行导出脚本。脚本从根目录的总览和 Software Stack 笔记以及主题目录读取 Markdown 和附件，将 Obsidian 双链转换为站内链接。构建日志会列出缺失或重名且无法确定目标的链接，这些链接在网页上显示为普通文字。
