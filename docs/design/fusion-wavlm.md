# CRNN + WavLM Residual Gated Fusion

## 1. 目标

这一版不是三路融合，也不是 posterior fusion，而是在当前已经跑通的 `CRNN + BEATs residual gated fusion` 基础上，做一个更严格、变量更受控的对照实验：

`CRNN 主干 + WavLM 辅助分支`

核心目的不是追求一步到位的最高分，而是先回答：

- `WavLM` 作为辅助分支有没有独立价值
- 它和 `BEATs` 作为残差补充分支时，效果是否不同
- 后续是否值得再做三路融合或更复杂模块

## 2. 数据流

### CRNN 分支

`waveform -> log-mel -> scaler -> CNN -> cnn_feat [B, T_cnn, D_cnn]`

### WavLM 分支

`waveform -> frozen WavLM -> wavlm_feat [B, T_wavlm, D_wavlm]`

### 对齐

先用 `FusionTimeAligner(method="adaptive_avg")` 把 `wavlm_feat` 对齐到 `T_cnn`：

`wavlm_aligned [B, T_cnn, D_wavlm]`

### Residual Gated Fusion

1. `cnn_feat -> cnn_proj -> cnn_norm`
2. `wavlm_aligned -> wavlm_proj -> wavlm_norm`
3. `gate = sigmoid(gate_net([cnn_norm ; wavlm_norm]))`
4. `fused = cnn_norm + gate * wavlm_norm`
5. `fused -> optional post_fusion_proj -> shared SEDDecoder`

默认 gate 是按通道的，也就是输出 `[B, T, D_fuse]`。

## 3. 为什么现在先做 CRNN + WavLM，而不是三路融合

当前已有实验已经说明：

- `CRNN baseline` 是健康主干
- `WavLM-only` 更像语音导向补充分支，而不是当前环境声 SED 的主力 encoder
- 如果现在直接上 `CRNN + BEATs + WavLM`，变量会变得太多，难以判断 `WavLM` 的独立价值

所以这一版先做 `CRNN + WavLM residual gated fusion`，目的是得到一个严谨、可比较、变量受控的实验结果。

## 4. 为什么沿用 residual gate

当前诊断已经说明：

- `CRNN` 是主导分支
- 辅助分支更适合做“按需补充”，而不是无条件堆叠
- 裸 `concat` 会让后续 `MLP / BiGRU` 自己学“该信谁”，控制粒度太粗

Residual gated fusion 更适合当前这种“强主干 + 弱补充分支”的结构，因为：

- projection + LayerNorm 先缓解尺度不匹配
- gate 可以逐帧 / 逐通道决定 WavLM 是否值得补进来
- 残差式设计显式保住 CRNN 主路径，不让辅助分支反客为主

## 5. 和现有 BEATs residual gate 的关系

结构保持同类：

- 同样是 `CNN branch 后、BiGRU 前` 融合
- 同样先把辅助分支对齐到 `T_cnn`
- 同样使用 `projection + norm + gate + residual fusion`
- 同样复用共享 `SEDDecoder`

唯一关键变量变化是：

- 辅助分支从 `BEATs` 换成 `WavLM`

这正是后续论文中公平比较 `CRNN / CRNN+BEATs / CRNN+WavLM` 所需要的。

## 6. 运行方式

```bash
python train_sed.py --conf_file ./confs/crnn_wavlm_residual_gated_fusion.yaml --synth_only --gpus 1
```

冒烟测试：

```bash
python train_sed.py --conf_file ./confs/crnn_wavlm_residual_gated_fusion.yaml --synth_only --gpus 0 --fast_dev_run
```

## 7. 是否需要重新训练

需要重新训练。

原因是这次改的是模型内部融合模块与辅助分支类型，不是输出层后处理：

- 新增了 `CRNN + WavLM` 的残差门控路径
- 改变了进入 BiGRU 的融合特征分布
- 需要重新学习 gate 与后端参数

## 8. 当前限制

- 当前仍是 `freeze WavLM` 版本
- 还没有做 `WavLM` 微调
- 还没有做 class-aware / event-aware gate
- 还没有做 `CRNN + BEATs + WavLM` 三路融合
