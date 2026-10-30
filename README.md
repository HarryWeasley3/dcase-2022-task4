# 异构自监督模型互补融合的声音事件检测

**Heterogeneous Self-Supervised Representation Fusion for Sound Event Detection**  
DCASE 2022 Task 4 · DESED · PyTorch · MIT

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本仓库是本科毕业设计《异构自监督模型互补融合的声音事件检测研究》的实现代码。基于 DCASE 2022 Task 4 基线，将波形前端、编码器、时间对齐与共享解码器分离，接入 CRNN、BEATs、WavLM 三类表征，并通过统一入口配置分支组合与训练评估流程，考察通用音频表征与语音域表征在声音事件检测中的互补性。方法、实验结果与运行说明见 [docs/](docs/)。

主要结果：CRNN+BEATs 的 PSDS1 由 CRNN 基线的 0.356 提升至 0.522，Event-F1 由 43.42% 提升至 58.64%。三种三分支结构的 PSDS 与 Event-F1 均未超过双分支，其中残差门控结构的 Event-F1 为 57.00%。

## 主要结果

| 模型 | PSDS1 | PSDS2 | Event-F1 (%) | Inter-F1 (%) |
|---|---|---|---|---|
| CRNN（baseline） | 0.356 | 0.578 | 43.42 | 65.00 |
| BEATs（通用音频 SSL） | 0.465 | 0.780 | 52.65 | 65.80 |
| WavLM（语音域 SSL） | 0.375 | 0.678 | 45.75 | 59.80 |
| **CRNN + BEATs** | **0.522** | **0.816** | **58.64** | **74.36** |
| CRNN + WavLM | 0.362 | 0.666 | 46.75 | 62.20 |
| CRNN+BEATs+WavLM（特征拼接） | 0.486 | 0.750 | 56.25 | 77.53 |
| CRNN+BEATs+WavLM（固定式中层融合） | 0.492 | 0.734 | 55.82 | 77.41 |
| CRNN+BEATs+WavLM（残差门控） | 0.497 | 0.745 | 57.00 | 78.97 |

评估采用 DESED 合成训练集和合成验证集的强标签设定（`synth_only`）。实验使用单张 RTX 5090，随机种子固定为 42，每种结构训练一次。类别级结果和融合分析见 [docs/results.md](docs/results.md)。

## 方法

![特征拼接融合结构示意](docs/figures/architecture.png)

### 编码器

三个编码器遵循同一接口：输入波形，输出形状为 `[B, T, D]` 的序列特征，`output_dim` 在构造时确定。CRNN 编码器沿用上游的 log-mel 前端与七层卷积，保留频谱裁剪与实例归一化；BEATs 编码器基于 vendored 的最小官方实现，仅保留特征提取所需代码，支持冻结与全量微调两种模式；WavLM 编码器基于 torchaudio bundle 构建，并处理了官方 checkpoint 与 torchaudio 模型之间的参数前缀差异，以及 padding mask 到 `lengths` 的转换。更换前端只需修改 `model.branches` 配置，训练与评估流程保持不变。

### 时间对齐

三路编码器的时间分辨率来源不同：CRNN 由卷积下采样决定，BEATs 取决于 patch embedding 与 Transformer 序列长度，WavLM 由卷积特征提取器与 Transformer 下采样决定。损失计算、后处理与 PSDS 评估使用固定的标签帧栅格。框架通过两个模块集中处理时间缩放：`TimeAligner` 将编码器序列对齐至标签栅格，支持 interpolate、nearest、adaptive_avg、adaptive_max 四种策略；`FusionTimeAligner` 在融合前将 SSL 序列对齐至 CNN 的时间栅格。

### 融合与残差门控

特征拼接将对齐后的多路特征交给 Merge MLP；残差门控以 BEATs 为主路径，将 CRNN、WavLM 表征作为可学习的残差修正。双分支的基本形式为 `fused = beats_norm + α · gate · cnn_norm`，三分支使用两路独立门控。门控网络的末层偏置初始化为 -1.5，融合结果经过可选线性投影和时间对齐后进入共享 BiGRU 解码器。具体结构见 [体系结构说明](docs/architecture.md)。

框架主体位于 `sed_modeling/`，提供 3 个编码器、2 种对齐器、3 个融合模块与 7 条多分支装配路由。训练器沿用上游 Mean-Teacher 框架，并支持统一模型构造、按模块分组的学习率、波形域 mixup、教师共享冻结编码器及中间特征输出。`tests/` 包含层级融合装配与学习率调度的检查。

## 快速开始

BEATs 官方权重 `BEATs_iter3+_AS2M.pt` 与本项目微调得到的 `BEATs_full_finetune_best_0_78.pt` 均放置于 `pretrained/beats/` 目录。后者 345 MB，发布于 [Releases 的 beats-fullft-v1](https://github.com/HarryWeasley3/dcase-2022-task4/releases/tag/beats-fullft-v1)，可由 `BEATsEncoder` 直接加载。

```bash
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 训练：准备好数据与权重后，使用默认 rtx_5090 三分支残差门控配置
python train_sed.py --conf_file confs/experiment_unified.yaml --gpus 1

# 评估
python train_sed.py --conf_file confs/experiment_unified.yaml --gpus 1 \
    --test_from_checkpoint ./exp/<run>/version_0/<best>.ckpt
```

将 `model.branches.wavlm.enabled` 设为 `false` 可切换至 CRNN+BEATs 双分支。数据准备需按 `confs/experiment_unified.yaml` 中 `data.*` 的键组织 DESED 合成训练集与验证集。WavLM 权重、数据目录映射、显存不足时的冻结策略与常见报错见 [docs/reproduction.md](docs/reproduction.md)。

## 仓库结构

| 路径 | 说明 |
|---|---|
| `sed_modeling/` | 编码器、时间对齐、融合模块、共享解码器与模型装配（`build_sed_model`） |
| `confs/` | 统一配置 `experiment_unified.yaml`，含 `rtx_5090` 与 `hierarchical_ssl_stage_a` 两个 profile |
| `train_sed.py` | 训练、断点续训与评估入口 |
| `local/` | Mean-Teacher 训练器、数据工具与重采样 |
| `desed_task/` | DESED 数据管线、评估指标、学习率调度器与基线 CRNN |
| `scripts/` | 编码器权重提取与后处理阈值搜索 |
| `tests/` | 层级融合装配与学习率调度的单元测试 |
| `docs/` | 架构说明、实验结果、运行指南、模块说明与插图 |

`desed_task/` 与 `sed_modeling/third_party/beats/` 为上游代码，分别来自 DCASE baseline 与 Microsoft BEATs 的最小 vendored 子集，许可证保留在对应目录及根目录 `LICENSE` 中。

## 局限

实验采用单随机种子和单次训练，评估集为 DESED 合成验证集。WavLM 微调权重与 CRNN warm-start 权重未随仓库发布，目前提供 BEATs 微调权重。训练入口支持单卡运行，多卡训练未实现。运行条件见 [docs/reproduction.md](docs/reproduction.md)。

## License

本项目采用 MIT 许可证，见 [LICENSE](LICENSE)。`desed_task/` 与 `sed_modeling/third_party/beats/` 遵循其各自的上游许可证。
