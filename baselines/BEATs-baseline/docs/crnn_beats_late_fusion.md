# CRNN + BEATs Late Fusion

## 1. 目标

这一版不是 posterior fusion，也不是把两个独立模型的输出在推理时做加权平均。

当前实现的是模型内部的论文式特征级晚融合：

`audio -> CNN branch`
`audio -> BEATs branch -> align to CNN time`
`concat -> Merge MLP -> BiGRU -> strong / weak predictions`

并且尽量复用了当前工程已有的：

- mean-teacher trainer
- strong / weak BCE
- consistency loss
- 验证 / 测试 / PSDS / TSV 导出流程

## 2. 当前数据流

### CRNN 分支

`waveform -> log-mel -> scaler -> CNN -> cnn_sequence_features [B, T_cnn, D_cnn]`

这里复用的是当前 `CRNNEncoder` 的输出边界，也就是原始 CRNN 里 **CNN 之后、BiGRU 之前** 的时序特征。

### BEATs 分支

`waveform -> frozen BEATs -> beats_sequence_features [B, T_beats, D_beats]`

第一版默认：

- `freeze=True`
- 冻结前向走 `torch.no_grad()`

### 融合链路

1. 用 `FusionTimeAligner(method="adaptive_avg")` 把 `T_beats` 对齐到 `T_cnn`
2. 对齐后做 `concat([cnn_features, beats_features_aligned], dim=-1)`
3. concat 后送入 `MergeMLP`
4. `MergeMLP` 输出送入共享 `SEDDecoder`
5. `SEDDecoder` 内部继续走 `BiGRU -> strong/weak heads`

如果配置里 `target_frame_len` 与 `T_cnn` 不同，仍会通过原来的 `TimeAligner` 在 decoder 前对齐到标签栅格，尽量保持训练 / 评估流程兼容。

## 3. 哪些部分严格参考了论文式结构

这一版明确参考了你要求的关键链路：

- BEATs frame-level embedding
- 先把 BEATs 时间长度对齐到 CNN branch 输出长度
- 默认使用 `adaptive average pooling`
- 融合方式默认 `concat`
- concat 后有单独的 `Merge MLP`
- `Merge MLP` 后接 `BiGRU`
- 最终输出 frame-level strong prediction 和 clip-level weak prediction

## 4. 工程化折中点

这版不是重新写一套独立工程，而是在当前底座里做的增量式实现，因此有几个明确折中：

- 继续复用当前 mean-teacher trainer，而不是重写成论文原始训练工程
- 继续复用当前 strong / weak / consistency loss
- 当前示例配置仍是 `synth_only`
- 第一版默认冻结 BEATs，优先保证显存和吞吐
- 复用当前 `CRNNEncoder` 的 CNN 输出作为融合点，而不是重新拆分旧 `CRNN` 类

## 5. 运行方式

```bash
python train_sed.py --conf_file ./confs/crnn_beats_late_fusion_synth_only.yaml --synth_only --gpus 1
```

如果只做冒烟测试：

```bash
python train_sed.py --conf_file ./confs/crnn_beats_late_fusion_synth_only.yaml --synth_only --gpus 0 --fast_dev_run
```

## 6. 是否需要重新训练

需要重新训练。

原因很直接：这次改的是 **模型内部特征级融合结构**，包括：

- 新增 BEATs branch
- 新增对齐模块
- 新增 concat 融合
- 新增 Merge MLP
- 改变了进入 BiGRU 的特征分布

这不是推理时的 posterior fusion，所以不能直接复用旧 CRNN 或旧 BEATs 的测试结果。

## 7. 当前限制

- 真实 BEATs checkpoint 仍需要你本地提供
- 第一版只做了 frozen BEATs 版本
- 当前还没有把 WavLM 接到同样的 late fusion 位置
- 当前没有实现论文里更复杂的多阶段 SSL / BLOCK2 / BLOCK3 设计
