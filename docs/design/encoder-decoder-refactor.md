# 统一编码器与解码器

项目采用统一的声音事件检测数据流：

`waveform → encoder → time alignment → shared decoder → strong/weak predictions`

编码器负责提取时序特征，时间对齐模块负责匹配标签栅格，共享解码器负责时序建模和分类。单分支与多分支模型复用相同的训练、验证、后处理和评估流程。

## 1 模块组成

| 模块 | 位置 | 作用 |
|---|---|---|
| `CRNNEncoder` | `sed_modeling/encoders/crnn_encoder.py` | 从波形生成 log-mel 表征，并通过 CNN 提取时序特征 |
| `BEATsEncoder` | `sed_modeling/encoders/beats_encoder.py` | 提取通用音频自监督特征 |
| `WavLMEncoder` | `sed_modeling/encoders/wavlm_encoder.py` | 提取语音域自监督特征 |
| `TimeAligner` | `sed_modeling/modules/time_aligner.py` | 将时序特征对齐至标签长度 |
| `SEDDecoder` | `sed_modeling/decoders/sed_decoder.py` | 生成帧级和片段级预测 |
| `build_sed_model` | `sed_modeling/models/sed_model.py` | 根据配置装配分支、融合模块和解码器 |

三个编码器均输出形状为 `[B, T, D]` 的 `sequence_features`，并通过 `output_dim` 提供特征维度。启用多个分支时，模型在共享解码器之前完成时间对齐和特征融合。

## 2 CRNN 路径

CRNN 路径的数据流如下：

`waveform → MelSpectrogram → AmplitudeToDB → scaler → CNN → sequence features`

Mel 前端位于 `CRNNEncoder` 外部，由模型构建流程注入。CNN 输出位于 BiGRU 之前，因此能够与 BEATs、WavLM 的序列特征在相同层级融合。

## 3 自监督编码器路径

BEATs 和 WavLM 均支持冻结与全量微调。冻结模式使用 `torch.no_grad()`，优化器仅接收 `requires_grad=True` 的参数。教师模型可以共享冻结编码器，从而减少显存占用。

BEATs 的精简实现位于 `sed_modeling/third_party/beats/`。WavLM 通过 `torchaudio.pipelines` 构建，并将 padding mask 转换为模型所需的有效长度。

## 4 时间对齐

CRNN、BEATs、WavLM 的时间步数由各自的前端和下采样过程决定。融合前使用 `FusionTimeAligner` 将各分支对齐到路由指定的参考时间轴，融合后使用 `TimeAligner` 对齐到标签栅格。

`TimeAligner` 支持以下策略：

- `interpolate`
- `nearest`
- `adaptive_avg`
- `adaptive_max`

统一接口接收 `[B, T_enc, D]`，返回 `[B, T_label, D]`。

## 5 共享解码器

`SEDDecoder` 包含可选输入投影、BiGRU、帧级分类头和片段级分类头。帧级输出 `strong_preds` 的形状为 `[B, C, T_label]`，片段级输出 `weak_preds` 的形状为 `[B, C]`。片段级预测支持注意力池化和时间均值池化。

模型前向结果还包含编码器特征、对齐后特征、解码器输入、帧级特征和注意力权重。调用方可通过 `return_intermediates=True` 获取这些中间量。

## 6 训练入口

`train_sed.py` 通过 `build_sed_model(config)` 创建模型。`local/sed_trainer.py` 负责监督损失、一致性损失、EMA 教师、指标计算和结果导出。波形域 mixup 在编码器之前执行，因此各分支接收相同的增强音频。

分支通过 `confs/experiment_unified.yaml` 中的 `model.branches.*.enabled` 控制。对齐参数位于 `model.align`，解码器参数位于 `model.decoder`，融合参数位于 `model.fusion`。
