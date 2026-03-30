# CRNN + BEATs Residual Gated Fusion

## 1. 目标

这一版不是 posterior fusion，也不是重新写一套训练框架，而是在当前已经跑通的 `CRNN + BEATs late fusion` 基础上，把：

`concat -> Merge MLP -> BiGRU`

升级成更贴合诊断结论的：

`projection + per-branch LayerNorm + gate + residual fusion -> BiGRU`

核心思想是：

- `CRNN` 仍然是主干
- `BEATs` 仍然是冻结的补充分支
- 模型显式学习“每一帧 / 每一通道到底该让 BEATs 补多少”

## 2. 数据流

### CRNN 分支

`waveform -> log-mel -> scaler -> CNN -> cnn_feat [B, T_cnn, D_cnn]`

### BEATs 分支

`waveform -> frozen BEATs -> beats_feat [B, T_beats, D_beats]`

### 对齐

先用 `FusionTimeAligner(method="adaptive_avg")` 把 `beats_feat` 对齐到 `T_cnn`：

`beats_aligned [B, T_cnn, D_beats]`

### Residual Gated Fusion

1. `cnn_feat -> cnn_proj -> cnn_norm`
2. `beats_aligned -> beats_proj -> beats_norm`
3. `gate = sigmoid(gate_net([cnn_norm ; beats_norm]))`
4. `fused = cnn_norm + gate * beats_norm`
5. `fused -> optional post_fusion_proj -> shared SEDDecoder`

默认 gate 是按通道的，也就是输出 `[B, T, D_fuse]`。

## 3. 为什么不用裸 concat

之前的诊断已经说明：

- `CRNN` 是主导分支
- `BEATs` 不是没用，但更像弱补充通道
- 裸 `concat` 会把两路特征无条件堆到一起
- 后续 `MLP / BiGRU` 被迫自己学“该信谁”，控制粒度太粗

Residual gated fusion 更适合当前这种“强主干 + 弱补充”的结构，因为：

- 先做独立 projection + LayerNorm，可以缓解尺度不匹配
- gate 可以逐帧 / 逐通道决定 BEATs 是否值得补进来
- 残差式设计显式保住 CRNN 主路径，不让 BEATs 反客为主

## 4. 和旧版 late fusion 的区别

旧版：

`CNN feat + BEATs aligned -> concat -> Merge MLP -> BiGRU -> heads`

新版：

`CNN feat / BEATs aligned -> independent proj + norm -> gate -> cnn + gate * beats -> optional post proj -> BiGRU -> heads`

保留的部分：

- CNN branch 后、BiGRU 前的融合位置
- BEATs 先对齐到 CNN 时间长度
- 共享 `SEDDecoder`
- mean-teacher trainer / loss / 评估流程

替换的部分：

- 用 `ResidualGatedFusion` 替代旧的 `concat + Merge MLP`

## 5. 运行方式

```bash
python train_sed.py --conf_file ./confs/crnn_beats_residual_gated_fusion_synth_only.yaml --synth_only --gpus 1
```

冒烟测试：

```bash
python train_sed.py --conf_file ./confs/crnn_beats_residual_gated_fusion_synth_only.yaml --synth_only --gpus 0 --fast_dev_run
```

## 6. 是否需要重新训练

需要重新训练。

原因是这次改的是模型内部融合模块，不是输出层后处理：

- 新增了 projection + LayerNorm
- 新增了 gate network
- 改变了进入 BiGRU 的特征分布

## 7. 当前限制

- 当前仍是 `freeze BEATs` 版本
- 还没有做 class-aware gate
- 还没有做 attention / cross-attention 版融合
- 还没有接 WavLM
