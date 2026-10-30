# DCASE 2022 Task 4 运行说明

本文件说明环境安装、数据准备、权重放置、训练和评估方法。模型结构见 [体系结构说明](architecture.md)，实验结果见 [实验结果](results.md)。

## 1 环境安装

项目运行环境如下：

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 20.04 |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| GPU | RTX 5090 32GB |
| CPU | Intel(R) Xeon(R) Platinum 8470Q |
| 内存 | 90GB |

PyTorch 的 `+cu128` 表示 CUDA 12.8 构建。创建 Python 3.10 环境后，先安装配套的 PyTorch 和 torchaudio，再安装项目依赖：

```bash
conda create -n dcase2022 python=3.10 -y
conda activate dcase2022
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

检查 GPU 环境：

```bash
python -c "import torch, torchaudio; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

数据下载脚本依赖 ffmpeg：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 2 数据准备

### 2.1 下载 DESED

下载完整开发集：

```bash
python generate_dcase_task4_2022.py --basedir=../../data
```

仅下载 `synth_only` 训练所需的合成数据：

```bash
python generate_dcase_task4_2022.py --basedir=../../data --only_synth
```

默认输出目录为 `<basedir>/dcase/dataset`。训练脚本从 YAML 的 `data` 段读取路径，因此可以使用任意本地数据根目录。

### 2.2 数据配置

主配置 [confs/experiment_unified.yaml](../confs/experiment_unified.yaml) 使用以下路径：

| 配置键 | 内容 |
|---|---|
| `synth_folder`、`synth_tsv` | 合成训练音频与强标签 |
| `synth_val_folder`、`synth_val_tsv`、`synth_val_dur` | 合成验证音频、强标签与时长 |
| `strong_folder`、`strong_tsv` | 真实强标签训练集 |
| `weak_folder`、`weak_tsv` | 弱标签训练集 |
| `unlabeled_folder` | 无标签训练集 |
| `test_folder`、`test_tsv`、`test_dur` | 带标签测试集 |
| `eval_folder` | 无标签 evaluation set |
| `audio_max_len`、`fs`、`net_subsample` | 音频长度、采样率与网络下采样倍数 |

强标签 TSV 包含 `filename`、`onset`、`offset`、`event_label`。弱标签 TSV 包含 `filename`、`event_labels`。时长 TSV 包含 `filename`、`duration`。

下面给出一组路径示例：

```yaml
data:
  synth_folder: /data/dcase/audio/train/synthetic21_train/soundscapes_16k
  synth_folder_44k: /data/dcase/audio/train/synthetic21_train/soundscapes
  synth_tsv: /data/dcase/metadata/train/synthetic21_train/soundscapes.tsv
  synth_val_folder: /data/dcase/audio/validation/synthetic21_validation/soundscapes_16k
  synth_val_tsv: /data/dcase/metadata/validation/synthetic21_validation/soundscapes.tsv
  synth_val_dur: /data/dcase/metadata/validation/synthetic21_validation/durations.tsv
  test_folder: /data/dcase/audio/validation/validation_16k
  test_tsv: /data/dcase/metadata/validation/validation.tsv
  test_dur: /data/dcase/metadata/validation/validation_durations.tsv
```

相对路径以启动命令时的当前目录为基准。建议从仓库根目录启动训练和评估。

### 2.3 重采样

项目使用 16 kHz 音频。YAML 同时提供原始采样率目录和 16 kHz 输出目录时，训练入口会执行重采样并生成 duration TSV。也可以提前运行：

```bash
python local/resample_folder.py \
  --in_dir /data/dcase/audio/train/synthetic21_train/soundscapes \
  --out_dir /data/dcase/audio/train/synthetic21_train/soundscapes_16k \
  --target_fs 16000
```

输出目录需要具备写权限和足够的磁盘空间。

## 3 预训练权重

主配置涉及以下权重：

| 文件 | 用途 | 缺失时的行为 |
|---|---|---|
| `pretrained/beats/BEATs_iter3+_AS2M.pt` | BEATs 官方预训练权重 | 模型构建终止 |
| `pretrained/beats/BEATs_full_finetune_best_0_78.pt` | BEATs warm-start | 启用 warm-start 时模型构建终止 |
| `pretrained/wavlm/WavLM-Base+.pt` | WavLM 预训练权重 | `use_bundle_weights: true` 时使用 torchaudio bundle |
| `pretrained/wavlm/WavLM_full_finetune_best_0_65.pt` | WavLM warm-start | 跳过该项加载 |
| `pretrained/crnn/crnn_best.pt` | CRNN warm-start | 跳过该项加载 |
| `model.decoder_warmstart.checkpoint` 指向的文件 | 解码器 warm-start | 跳过该项加载 |

BEATs 官方权重文件需要包含 `cfg` 和 `model` 两个键。BEATs 全量微调权重发布于 [beats-fullft-v1](https://github.com/HarryWeasley3/dcase-2022-task4/releases/tag/beats-fullft-v1)，文件 sha256 为 `dff86d366a70079d7e94d2da787d9d216a4cf2a418d666a25fc79c2172432fec`。

WavLM 自定义文件使用 torchaudio WavLM 的 state dict 格式。使用 bundle 权重时，将 `model.branches.wavlm.use_bundle_weights` 设为 `true`。离线环境需要提前缓存对应权重。

编码器权重导出脚本如下：

```bash
python scripts/extract_wavlm_ckpt.py \
  --src /path/to/wavlm_only_run/epoch=xxx-step=xxxxx.ckpt \
  --out pretrained/wavlm/WavLM_full_finetune_best_0_65.pt

python scripts/extract_ckpt.py \
  --src /path/to/beats_only_run/epoch=xxx-step=xxxxx.ckpt \
  --base pretrained/beats/BEATs_iter3+_AS2M.pt \
  --out pretrained/beats/BEATs_full_finetune_best_0_78.pt
```

## 4 流程检查

`--fast_dev_run` 将每套数据限制为 2 个 batch，并将训练限制为最多 3 个 epoch。默认三分支配置仍会构建全部编码器，因此需要准备相应数据和 BEATs 权重。

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --log_dir ./exp/smoke_test \
  --gpus 0 \
  --fast_dev_run
```

`--gpus 0` 使用 CPU。CRNN 流程检查可以使用：

```bash
python train_sed.py --conf_file confs/default.yaml --synth_only --gpus 0 --fast_dev_run
```

构造测试和调度器测试通过以下命令运行：

```bash
python -m pytest tests/
```

## 5 训练

`train_sed.py` 通过 `--conf_file` 读取 YAML。主配置包含 `rtx_5090` 和 `hierarchical_ssl_stage_a` 两个 profile，顶层 `profile` 决定使用哪组覆盖参数。

默认配置使用三分支残差门控模型。训练命令如下：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --log_dir ./exp/unified_resgate_rtx5090 \
  --gpus 1 \
  --synth_only
```

主要训练参数为 `n_epochs: 120`、`batch_size: 8`、`precision: 32`、`seed: 42` 和 `ema_factor: 0.999`。优化器使用 Adam，基础学习率为 `1.5e-5`。调度器使用 warmup cosine，预热 10 个 epoch，最低学习率为 `1e-6`。

从 checkpoint 恢复完整训练状态：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --log_dir ./exp/unified_resgate_rtx5090 \
  --gpus 1 \
  --resume_from_checkpoint ./exp/unified_resgate_rtx5090/version_0/last.ckpt
```

使用 `--init_from_checkpoint` 可以仅加载匹配的模型参数。该选项与 `--resume_from_checkpoint` 互斥。

训练日志和 checkpoint 写入 `--log_dir`。TensorBoard 启动命令如下：

```bash
tensorboard --logdir ./exp
```

训练入口支持 CPU 和单卡 GPU。指定具体 GPU 时，可以通过 `CUDA_VISIBLE_DEVICES` 控制可见设备。

## 6 评估

### 6.1 带标签数据

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --test_from_checkpoint /path/to/epoch=xxx-step=xxxxx.ckpt \
  --log_dir ./exp/eval_devtest \
  --gpus 1
```

评估时，数据路径和运行参数以本次传入的 YAML 为准。主配置中的 `test_*` 默认指向合成验证集。结果保存在 `metrics_test/student/` 和 `metrics_test/teacher/`，内容包括 PSDS、Event-F1、Inter-F1 和预测 TSV。

### 6.2 无标签 evaluation set

先在 YAML 中设置 `data.eval_folder`，再运行：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --eval_from_checkpoint /path/to/epoch=xxx-step=xxxxx.ckpt \
  --log_dir ./exp/eval_official \
  --gpus 1
```

该模式将 batch size 设为 1，并在 `metrics_test/` 下输出多个阈值对应的预测 TSV。

## 7 常见问题

**缺少 `ffprobe`。** 安装 ffmpeg，并确认 `ffprobe` 位于 `PATH`。

**重采样或 duration TSV 生成失败。** 检查音频输入路径、输出目录写权限和剩余磁盘空间。

**显存不足。** 将 `model.branches.beats.freeze` 或 `model.branches.wavlm.freeze` 设为 `true`，再降低 `training.batch_size` 和 `batch_size_val`。冻结编码器时，可以启用 `model.teacher.share_frozen_beats` 或 `share_frozen_wavlm`。

**找不到 `asteroid`。** 安装 [requirements.txt](../requirements.txt)。调度器模块依赖 `asteroid.engine.schedulers`。

**出现 `checkpoint not found ... skipping`。** 该提示对应可选的 WavLM、CRNN 或解码器 warm-start 文件。训练会使用模块初始化参数继续运行。

## 8 许可证与第三方组件

根目录 [LICENSE](../LICENSE) 使用 MIT 许可证。DCASE 2022 Task 4 baseline 位于 `desed_task/`，BEATs 精简代码位于 `sed_modeling/third_party/beats/`。BEATs 的许可证保存在 [sed_modeling/third_party/beats/LICENSE](../sed_modeling/third_party/beats/LICENSE)。数据集和预训练权重同时遵循各自发布方的使用条款。
