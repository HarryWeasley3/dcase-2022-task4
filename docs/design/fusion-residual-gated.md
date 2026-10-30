# 残差门控融合

残差门控融合先将各分支映射到相同维度，再通过门控网络控制残差分支对主分支的贡献。该模块位于编码器与共享解码器之间。

## 1 双分支结构

CRNN+BEATs 使用 BEATs 作为主分支。两路特征完成投影和 LayerNorm 后，融合形式为：

`fused = beats_norm + α · gate · cnn_norm`

CRNN+WavLM 使用 WavLM 作为主分支：

`fused = wavlm_norm + α · gate · cnn_norm`

其中，`α` 为可学习标量，`gate` 由主分支和残差分支共同生成。帧级门控的形状为 `[B, T, 1]`，计算时沿特征维广播；通道级门控的形状为 `[B, T, D]`。

## 2 三分支结构

三分支路由以 BEATs 作为时间与特征主路径。计算分为两步：

1. 使用 WavLM 门控生成 `beats_norm + α_w · gate_w · wavlm_norm`。
2. 将上一步结果与 CRNN 特征输入第二个门控网络，生成最终融合特征。

第二个门控网络以已经融合 WavLM 的 BEATs 表征作为控制输入。CRNN 和 WavLM 使用独立门控与独立缩放系数。

## 3 模块组成

每个分支先经过线性投影和 LayerNorm。门控网络接收主分支与残差分支的拼接特征，并通过 sigmoid 输出权重。门控末层偏置初始化为 -1.5。融合结果可经过 `post_fusion_proj`，随后进入 `TimeAligner` 和共享 `SEDDecoder`。

`stop_gradient_on_gate_input` 仅阻断门控控制器输入的梯度。分支特征在残差计算中的梯度路径保持有效。

## 4 配置与运行

在 `confs/experiment_unified.yaml` 中启用所需分支，并设置：

- `model.fusion.strategy: residual_gated`
- `model.fusion.fusion_type: residual_gated`
- `model.fusion.main_branch: auto`

`model.fusion.residual_gated.gate_mode` 控制帧级或通道级门控，`learnable_alpha` 控制缩放系数是否参与训练。

```bash
python train_sed.py --conf_file confs/experiment_unified.yaml --synth_only --gpus 1
```
