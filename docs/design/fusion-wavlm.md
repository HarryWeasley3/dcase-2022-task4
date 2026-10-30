# WavLM 分支与融合路径

WavLM 分支提供语音域自监督表征，可用于单分支模型、CRNN+WavLM 双分支模型和 CRNN+BEATs+WavLM 三分支模型。

## 1 编码器

`WavLMEncoder` 位于 `sed_modeling/encoders/wavlm_encoder.py`，通过 `torchaudio.pipelines` 构建模型。输入为 `[B, T]` 波形，输出 `[B, T_wavlm, D]` 序列特征。

编码器支持以下权重来源：

- `pretrained_checkpoint` 指向与 torchaudio 模型兼容的 state dict。
- `use_bundle_weights: true` 使用 torchaudio bundle 权重。

`freeze` 控制参数冻结，`output_layer` 控制输出层，`normalize_waveform` 控制波形归一化。padding mask 会转换为 torchaudio 接口使用的有效长度。

## 2 CRNN+WavLM

CRNN+WavLM 残差门控路由以 WavLM 为主分支。WavLM 特征和 CRNN 特征先对齐时间长度，再分别经过投影与 LayerNorm。融合形式为：

`fused = wavlm_norm + α · gate · cnn_norm`

融合结果进入共享 BiGRU 解码器。该结构用于比较语音域表征与局部时频表征的互补效果。

## 3 三分支融合

三分支结构同时使用 CRNN、BEATs 和 WavLM。特征级后期融合将 3 路特征拼接后交给 `MergeMLP`。残差门控路由先将 WavLM 作为 BEATs 的门控残差，再加入 CRNN 门控残差。

层级 SSL 路由从 BEATs 和 WavLM 的多个 Transformer 层提取特征，并在层内投影、门控和聚合后接入 CRNN 分支。各层选择参数位于 `model.fusion.hierarchical_ssl`。

## 4 配置

在 `confs/experiment_unified.yaml` 中启用 WavLM：

```yaml
model:
  branches:
    wavlm:
      enabled: true
      freeze: false
      use_bundle_weights: true
```

双分支模型启用 CRNN 和 WavLM，并关闭 BEATs。三分支模型同时启用 3 个分支。融合方式由 `model.fusion.strategy` 选择。
