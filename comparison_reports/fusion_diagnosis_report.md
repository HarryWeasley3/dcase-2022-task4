# Late Fusion 诊断报告

## 1. 结论总览

当前 `CRNN + BEATs late fusion` 没有明显超过 CRNN baseline，主因不像“BEATs 完全没被用上”，而更像是：BEATs 分支已经提供了局部有效信息，但收益主要集中在少数长持续或设备类；对于 `Dishes / Dog / Alarm_bell_ringing` 这类弱类和复杂多事件场景，当前 `concat + Merge MLP + shared BiGRU` 的 late fusion 仍然太粗。

直接证据是：去掉 BEATs 后，fusion 的 event macro 变化为 +6.36pp；打乱 BEATs 后变化为 +4.46pp；而去掉 CNN 后变化为 +38.53pp。如果 CNN 一去掉就接近崩掉，而去掉/打乱 BEATs 只带来中小幅退化，说明当前系统仍然是 CRNN 主导，BEATs 提供的是局部补充而不是主导信息源。

## 2. 四个推理对照实验结果

| variant       | intersection | event_macro | event_micro | segment_macro | segment_micro | pred_events | pred_files |
| ------------- | ------------ | ----------- | ----------- | ------------- | ------------- | ----------- | ---------- |
| fusion        | 0.583454     | 41.35       | 40.6        | 64.02         | 72.14         | 6019        | 2424       |
| zero_beats    | 0.559243     | 34.98       | 31.7        | 63.47         | 73.95         | 7510        | 2478       |
| zero_cnn      | 0.178606     | 2.82        | 6.0         | 19.97         | 13.08         | 2572        | 2351       |
| shuffle_beats | 0.564336     | 36.89       | 37.8        | 62.3          | 70.3          | 5716        | 2401       |
| crnn          | 0.650249     | 43.35       | 43.2        | 71.14         | 75.59         | 7203        | 2466       |

| class              | fusion_event | fusion_segment | fusion_pred | zero_beats_event | zero_beats_segment | zero_beats_pred | zero_cnn_event | zero_cnn_segment | zero_cnn_pred | shuffle_beats_event | shuffle_beats_segment | shuffle_beats_pred | crnn_event | crnn_segment | crnn_pred |
| ------------------ | ------------ | -------------- | ----------- | ---------------- | ------------------ | --------------- | -------------- | ---------------- | ------------- | ------------------- | --------------------- | ------------------ | ---------- | ------------ | --------- |
| Dishes             | 15.51        | 21.72          | 200         | 13.04            | 20.9               | 225             | 0.0            |                  | 0             | 10.58               | 15.41                 | 127                | 29.12      | 50.28        | 621       |
| Dog                | 15.88        | 33.24          | 130         | 12.57            | 39.83              | 405             | 0.0            |                  | 0             | 15.45               | 34.49                 | 136                | 32.4       | 59.67        | 376       |
| Alarm_bell_ringing | 21.18        | 59.6           | 249         | 24.76            | 66.58              | 409             | 0.0            |                  | 0             | 19.19               | 58.01                 | 236                | 20.5       | 63.85        | 330       |
| Cat                | 34.37        | 70.61          | 310         | 37.48            | 72.29              | 382             | 0.0            |                  | 0             | 31.41               | 69.9                  | 316                | 30.0       | 73.24        | 451       |
| Running_water      | 53.1         | 71.77          | 210         | 47.19            | 71.31              | 228             | 1.08           | 9.41             | 616           | 42.73               | 72.26                 | 237                | 49.47      | 71.41        | 349       |
| Frying             | 67.36        | 83.52          | 392         | 56.59            | 76.87              | 291             | 27.09          | 30.52            | 1956          | 61.35               | 80.47                 | 376                | 65.32      | 83.84        | 364       |
| Vacuum_cleaner     | 67.55        | 74.82          | 202         | 51.34            | 65.28              | 158             | 0.0            |                  | 0             | 61.09               | 70.01                 | 191                | 69.3       | 81.31        | 280       |

整体上，`zero_beats` 和 `shuffle_beats` 都没有像 `zero_cnn` 那样引起灾难性下降，说明当前模型主要还是由 CNN/CRNN 分支主导。

`zero_cnn` 的 event macro 只有 2.82%，说明单靠当前 frozen BEATs 支路并不足以支撑整体检测；`zero_beats` 的 event macro 仍有 34.98%，进一步说明 CRNN 是主力。

按类看，最依赖 BEATs 的类别主要是：`Vacuum_cleaner`(+16.20pp)，`Frying`(+10.77pp)，`Running_water`(+5.91pp)，`Dog`(+3.32pp)。

如果某些类别在 `zero_beats` 下降明显而 `shuffle_beats` 也下降，说明 BEATs 提供的不是纯噪声；如果 `shuffle_beats` 影响很小，则说明当前融合对 BEATs 的利用还不够深。

## 3. 特征统计结论

| feature       | mean      | std      | min        | max       | batch_mean_min | batch_mean_max | batch_std_min | batch_std_max |
| ------------- | --------- | -------- | ---------- | --------- | -------------- | -------------- | ------------- | ------------- |
| cnn_feat      | -0.037579 | 0.96922  | -16.287388 | 15.398331 | -0.083269      | 0.003813       | 0.763989      | 1.321312      |
| beats_feat    | -0.000732 | 0.220007 | -2.499089  | 1.960801  | -0.000767      | -0.000692      | 0.219734      | 0.220287      |
| beats_aligned | -0.000731 | 0.219707 | -2.459049  | 1.931884  | -0.000767      | -0.000692      | 0.219384      | 0.22003       |
| fused         | -0.005995 | 0.419213 | -16.287388 | 15.398331 | -0.012518      | -6.9e-05       | 0.353418      | 0.539365      |
| merge_mlp_out | 0.274675  | 0.904387 | -0.169971  | 27.594864 | 0.165391       | 0.463994       | 0.652949      | 1.323937      |

这些统计主要用来判断两个问题：一是 CNN 与 BEATs 是否存在明显尺度不匹配；二是 Merge MLP 后是否塌缩成近常数。

如果 `beats_feat / beats_aligned` 的 std 明显远小于 `cnn_feat`，而 `merge_mlp_out` 又没有把这两路拉回更均衡的范围，那就更支持“BEATs 被当成弱补充特征，而不是被充分利用”的判断。

## 4. 逐类 posterior / 可分性分析

| class              | event_f1 | segment_f1 | pos_mean | neg_mean | pos_ge_0.5 | neg_ge_0.5 | separation | diagnosis     |
| ------------------ | -------- | ---------- | -------- | -------- | ---------- | ---------- | ---------- | ------------- |
| Dishes             | 15.51    | 21.72      | 0.1756   | 0.0259   | 0.1005     | 0.0014     | 0.1498     | 表征与边界问题并存     |
| Dog                | 15.88    | 33.24      | 0.2353   | 0.0154   | 0.1974     | 0.0013     | 0.2199     | 表征与边界问题并存     |
| Alarm_bell_ringing | 21.18    | 59.6       | 0.4077   | 0.0097   | 0.4083     | 0.0008     | 0.398      | 更像边界/切段/后处理问题 |
| Cat                | 34.37    | 70.61      | 0.4928   | 0.0105   | 0.5317     | 0.0023     | 0.4823     | 表征与边界问题并存     |
| Running_water      | 53.1     | 71.77      | 0.5676   | 0.0276   | 0.5809     | 0.0052     | 0.54       | 表征与边界问题并存     |

这里重点看的是：正样本帧的 posterior 是否明显高于负样本帧，以及 `segment F1` 与 `event F1` 是否出现分离。

如果某类 `segment F1` 还有一定水平、但 `event F1` 很低，更像是边界/切段/后处理问题；如果 `segment` 和 `event` 都低，而且正负 posterior 几乎不可分，则更像表征不足。

## 5. 样本级 win/loss 分析

| group       | count | avg_delta | long_duration_ratio | short_event_ratio | animal_ratio | device_like_ratio | weak_focus_ratio | multi_event_ratio | avg_gt_events |
| ----------- | ----- | --------- | ------------------- | ----------------- | ------------ | ----------------- | ---------------- | ----------------- | ------------- |
| fusion_loss | 609   | -0.3761   | 0.514               | 0.7816            | 0.2693       | 0.5238            | 0.6601           | 0.9754            | 3.48          |
| fusion_win  | 375   | 0.3645    | 0.6427              | 0.6533            | 0.208        | 0.672             | 0.536            | 0.9653            | 3.34          |
| similar     | 1516  | 0.0058    | 0.6339              | 0.6168            | 0.2348       | 0.5726            | 0.6029           | 0.9446            | 3.14          |

### Fusion 明显优于 CRNN 的样本示例

| filename | delta              | gt_labels                 | gt_events |
| -------- | ------------------ | ------------------------- | --------- |
| 1633.wav | 1.0                | Blender                   | 1         |
| 1455.wav | 1.0                | Cat                       | 1         |
| 1231.wav | 1.0                | Blender                   | 1         |
| 1186.wav | 1.0                | Blender                   | 1         |
| 1416.wav | 1.0                | Vacuum_cleaner            | 1         |
| 1573.wav | 1.0                | Alarm_bell_ringing,Speech | 4         |
| 993.wav  | 0.8571428571428571 | Cat,Speech                | 4         |
| 1795.wav | 0.8                | Blender,Speech            | 3         |

### Fusion 与 CRNN 接近的样本示例

| filename | delta                | gt_labels                         | gt_events |
| -------- | -------------------- | --------------------------------- | --------- |
| 871.wav  | -0.1454545454545454  | Dishes,Frying,Speech              | 7         |
| 2287.wav | -0.1428571428571429  | Electric_shaver_toothbrush,Speech | 4         |
| 1926.wav | -0.1428571428571429  | Dishes,Speech                     | 4         |
| 2333.wav | -0.1428571428571429  | Speech,Vacuum_cleaner             | 4         |
| 408.wav  | -0.1428571428571429  | Dishes,Speech                     | 4         |
| 968.wav  | -0.1428571428571429  | Dishes,Speech                     | 4         |
| 2246.wav | -0.1428571428571429  | Dishes,Frying,Speech              | 4         |
| 1256.wav | -0.14141414141414138 | Cat,Speech                        | 6         |

### Fusion 明显差于 CRNN 的样本示例

| filename | delta | gt_labels                         | gt_events |
| -------- | ----- | --------------------------------- | --------- |
| 926.wav  | -1.0  | Dog                               | 2         |
| 2317.wav | -1.0  | Alarm_bell_ringing                | 1         |
| 234.wav  | -1.0  | Speech,Vacuum_cleaner             | 2         |
| 2445.wav | -1.0  | Dog                               | 2         |
| 803.wav  | -1.0  | Electric_shaver_toothbrush,Speech | 3         |
| 1564.wav | -1.0  | Blender,Speech                    | 2         |
| 1963.wav | -1.0  | Dog                               | 2         |
| 1195.wav | -1.0  | Dog                               | 1         |

这些分组采用 `file-level event F1 delta >= 0.15 / <= -0.15` 的规则。如果 win 组明显富集在长持续设备类，而 loss 组更偏 `Dog / Dishes / Alarm / Cat` 或多事件场景，就说明当前 late fusion 的收益具有很强的场景偏置。

## 6. 最可能的问题归因

最可能的主因是：BEATs 分支并非完全没被用上，但它在当前 late fusion 结构里只被粗粒度利用，主要帮助了少数长持续、设备纹理稳定的类别；对弱类、动物类、复杂多事件场景，它没有形成稳定互补。

次要原因是：两路特征在融合前后仍可能存在尺度和时间压缩上的不匹配。即便 adaptive average pooling 在工程上是正确的，它也可能把 BEATs 的局部时间细节过度平滑，让它更擅长提供“有这个声音的大致区间”，而不擅长改善细粒度边界。

因此当前问题更像“被用上了，但只对少数类有效”，而不是“完全没被用上”。从对照实验如果还能看到 `shuffle_beats` 比 `zero_beats` 更差，也会进一步支持这一点。

## 7. 最值得做的 1~3 个改动

1. 先保留当前底座不变，在 fusion 前后加入更明确的归一化或门控，让 BEATs 不只是弱补充通道。
2. 优先针对 `Dishes / Dog / Alarm_bell_ringing / Cat` 做类不平衡和阈值分析，确认问题是表征不足还是边界不稳。
3. 如果本轮证据显示 BEATs 只对少数长持续类有帮助，下一步更值得试更聪明的融合方式或 posterior fusion，而不是继续盲目长训当前 concat late fusion。

## 备注

本次排查没有重训模型，也没有改训练底座，只是在 `version_20` best checkpoint 上做了离线对照推理与统计。

CRNN baseline 采用旧 checkpoint 到当前统一 `SEDModel` 的最小权重映射，仅用于诊断推理。加载时缺失的非关键键为：`['encoder.mel_spec.spectrogram.window', 'encoder.mel_spec.mel_scale.fb']`。