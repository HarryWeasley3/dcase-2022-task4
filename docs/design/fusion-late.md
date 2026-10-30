# 特征级后期融合

特征级后期融合位于编码器与共享解码器之间。各分支先独立提取序列特征，再对齐时间长度并完成融合。

## 1 数据流

双分支 CRNN+BEATs 的数据流如下：

`waveform → CRNNEncoder → cnn_features`

`waveform → BEATsEncoder → beats_features → align to CNN time`

`cnn_features + beats_features → late fusion → SEDDecoder`

CRNN+WavLM 使用相同结构，并将 BEATs 分支替换为 WavLM。三分支结构同时对齐 BEATs 和 WavLM 特征。

## 2 时间对齐

`FusionTimeAligner` 将自监督分支的时间长度对齐到 CNN 特征长度。默认策略为 `adaptive_avg`。融合结果随后由 `TimeAligner` 对齐至标签栅格。

## 3 融合方式

双分支路由支持 `concat` 和 `add`：

- `concat`：沿特征维拼接两路特征，再送入 `MergeMLP`。
- `add`：先将两路特征映射到相同维度，再逐元素相加。

三分支路由使用 `concat`，将 CRNN、BEATs 和 WavLM 特征拼接后送入 `MergeMLP`。

`MergeMLP` 的计算顺序为线性映射、可选 LayerNorm、激活函数和 Dropout。输出特征进入共享 `SEDDecoder`，经过 BiGRU 和分类头生成 strong、weak 预测。

## 4 配置与运行

使用 `confs/experiment_unified.yaml` 配置模型：

- 通过 `model.branches.crnn.enabled`、`beats.enabled` 和 `wavlm.enabled` 选择分支。
- 将 `model.fusion.strategy` 设为 `late`。
- 通过 `model.fusion.late.combine` 选择双分支的 `concat` 或 `add`。

训练命令如下：

```bash
python train_sed.py --conf_file confs/experiment_unified.yaml --synth_only --gpus 1
```

特征级融合改变了解码器输入分布，因此各融合结构需要独立训练。
