# Encoder-Decoder Refactor for Traditional SED

## 1. 目标

这次重构把原来的单体 CRNN baseline 改成了统一的传统 SED 底座：

`waveform -> encoder -> time alignment -> shared decoder -> strong / weak predictions`

设计目标不是“把 BEATs 粘到原 CRNN 里”，而是让后续比较尽量满足：

- 同一任务
- 同一数据
- 同一标签栅格
- 同一训练/验证/测试流程
- 同一套 decoder
- 主要只替换 encoder

当前已经支持：

- `encoder_type=crnn`
- `encoder_type=beats`

并且第一阶段的 BEATs 路径默认是冻结特征提取：

`waveform -> frozen BEATs -> aligned sequence features -> shared decoder`

## 2. 目录结构

新增的核心模块如下：

- `sed_modeling/encoders/crnn_encoder.py`
- `sed_modeling/encoders/beats_encoder.py`
- `sed_modeling/decoders/sed_decoder.py`
- `sed_modeling/modules/time_aligner.py`
- `sed_modeling/models/sed_model.py`

其中：

- `CRNNEncoder` 负责 waveform 到卷积时序特征
- `BEATsEncoder` 负责 waveform 到 BEATs 序列特征
- `TimeAligner` 负责把 encoder 时间维对齐到标签时间维
- `SEDDecoder` 负责统一的 temporal module + strong head + weak head
- `SEDModel` 负责把这些模块串起来，并提供统一 forward 接口

另外，BEATs 依赖的最小官方实现被 vendor 到：

- `sed_modeling/third_party/beats/`

这里只保留第一阶段真正需要的特征提取代码，避免引入整套 fairseq/hydra 依赖。

## 3. 现在的整体数据流

### CRNN 路径

`waveform`
-> `log-mel frontend`
-> `instance/dataset scaler`
-> `CNN frontend`
-> `sequence features [B, T_crnn, D]`
-> `TimeAligner`
-> `SEDDecoder`
-> `strong_preds [B, C, T_label]`, `weak_preds [B, C]`

### BEATs 路径

`waveform`
-> `frozen BEATs`
-> `sequence features [B, T_beats, D]`
-> `TimeAligner`
-> `SEDDecoder`
-> `strong_preds [B, C, T_label]`, `weak_preds [B, C]`

### Shared decoder

`SEDDecoder` 当前包含：

- 可选 `input_proj`
- 可选 `BiGRU temporal module`
- `strong head`
- `weak head`（attention pooling 或简单均值）

这样后续接 WavLM 时，原则上只需要新写一个 `WavLMEncoder`，decoder 与训练评估逻辑可以继续复用。

## 4. 时间对齐是怎么做的

这是这次重构里最关键的显式模块。

原因是不同 encoder 的时间长度天然不同：

- `T_crnn` 来自卷积下采样
- `T_beats` 来自 patch embedding + Transformer 序列长度
- `T_label` 来自数据集标签栅格

如果不显式对齐，就会把 resize 逻辑散落在 trainer 或 loss 里，后续很难保证比较公平，也不方便分析中间特征。

当前默认使用：

- `TimeAligner(method="interpolate", interpolate_mode="linear")`

接口固定为：

- 输入：`[B, T_enc, D]`
- 输出：`[B, T_label, D]`

已经预留了其他策略：

- `nearest`
- `adaptive_avg`
- `adaptive_max`

## 5. 为什么第一阶段优先冻结 BEATs

第一阶段的优先级不是端到端微调，而是：

1. 先跑通统一架构
2. 控制显存
3. 保持吞吐
4. 让 encoder 间比较更清楚

因此 BEATs 默认：

- `freeze: true`
- 前向包在 `torch.no_grad()` 中
- 优先只训练共享 decoder

这意味着第一阶段的结论更像：

“在同一个传统 SED decoder 上，冻结的预训练 encoder 提供了什么样的表征能力”

而不是：

“端到端大规模微调后哪个系统最强”

## 6. 配置切换方式

模型切换通过 `model.encoder_type` 完成。

### 保守兼容配置

- `confs/default.yaml`
- `confs/synth_only_d_drive.yaml`

这两份配置默认走 `encoder_type=crnn`，尽量保持原 baseline 行为。

### 面向公平比较的统一配置

- `confs/unified_crnn_synth_only_d_drive.yaml`
- `confs/unified_beats_synth_only_d_drive.yaml`

这两份配置都走共享 decoder，并显式打开：

- `model.align`
- `model.decoder`
- `model.teacher.share_frozen_encoder`

其中 `unified_beats_synth_only_d_drive.yaml` 还需要你提供真实的 BEATs checkpoint 路径。

## 7. 训练脚本层面的变化

`train_sed.py` 现在通过 `build_sed_model(config)` 创建模型，而不是直接实例化旧的 `CRNN`。

`local/sed_trainer.py` 也做了配套适配：

- 不再把 `MelSpectrogram + log + scaler + CRNN` 写死在 trainer 里
- 改为调用统一模型前向
- loss / EMA / 验证指标 / PSDS / 后处理基本保持原样

这保证了：

- 指标计算逻辑尽量不变
- decoder 和后处理逻辑不会因为换 encoder 被重写一遍

## 8. OOM / 吞吐方面做了什么

当前已经做的稳妥优化：

- BEATs 默认冻结
- 冻结 BEATs 前向使用 `torch.no_grad()`
- optimizer 只接收 `requires_grad=True` 的参数
- teacher 默认可共享冻结 encoder，避免 student/teacher 各拷一份大 encoder
- 中间特征通过 forward dict 暴露，不需要额外重复前向

## 9. 中间特征分析接口

统一模型 forward 已经预留了中间特征输出，便于后续比较：

- `encoder_sequence_features`
- `aligned_sequence_features`
- `encoder_frontend_features`
- `decoder_inputs`
- `decoder_frame_features`
- `attention_weights`

正常训练时不需要额外处理；如果后续想做可视化或表征分析，可以在调用时打开 `return_intermediates=True`。

## 10. 工程折中点

这次重构有两个明确的工程折中：

### 折中 1：CRNN encoder 采用“waveform frontend + CNN”边界

原始 baseline 里的卷积、RNN 和头部是强耦合写在一个类里的。为了尽量少破坏训练流程，同时把 decoder 真正统一出来，这里把：

- `log-mel frontend + CNN`

视为 `CRNNEncoder`，而把：

- `BiGRU + strong/weak heads`

收敛到共享 `SEDDecoder`。

这比“直接保留旧 CRNN 头，再给 BEATs 单独做一套头”更接近公平比较目标。

### 折中 2：当前 mixup 在统一框架下放到了 waveform 空间

这样不同 encoder 在训练入口看到的是同一份增强后音频，而不是在各自内部特征空间做不同版本的 mixup。  
这会让新架构下的 CRNN 与原论文 baseline 存在轻微训练细节差异，但换来的是 encoder 间更统一的比较入口。

## 11. 当前限制

- `encoder_type=beats` 需要你本地提供 BEATs checkpoint
- 当前只完成了“冻结 BEATs + 共享 decoder”的第一阶段
- 还没有实现 WavLM encoder
- 还没有实现离线特征缓存
- 还没有实现“只解冻最后若干层”的细粒度 BEATs 微调策略

## 12. 后续推荐扩展

下一步最自然的扩展顺序是：

1. 新增 `WavLMEncoder`
2. 给 BEATs/WavLM 增加“部分解冻最后 N 层”
3. 增加离线特征缓存，进一步降低训练显存和加快吞吐
4. 固定同一份 decoder 配置，系统比较 CRNN / BEATs / WavLM
5. 基于 forward 暴露的中间特征做可视化与互补性分析
