# desed_task（上游代码，vendored）

本目录来自 DCASE 2022 Task 4 官方 baseline 仓库 [DCASE-REPO/DESED_task](https://github.com/DCASE-REPO/DESED_task)，提供 DESED 数据管线、评估指标（PSDS / Event-F1 / Intersection-F1）、ManyHotEncoder、学习率调度器与基线 CRNN。

为适配本项目的多分支训练，其中少量文件做过修改（例如 `utils/schedulers.py` 增加了 `decay_power` 参数，`dataio/` 与 `evaluation/` 有小幅调整），具体差异可用 `git log --oneline -- desed_task/` 查看。新增的模型、编码器、融合模块都在仓库根目录的 `sed_modeling/` 中，不在本目录内。

上游许可证与引用信息见仓库根目录的 `LICENSE` 与 `README.md`。
