# DCASE 2022 Task 4 复现说明

本文档说明如何在本仓库中准备环境、数据与预训练权重，并跑通训练和评估流程。它面向拿到仓库副本、希望重新训练或复现实验结果的读者，重点在"每一步实际要做什么、配置键对应什么、哪些东西仓库里没有"。模型的内部结构和三个编码器如何拼接请配合 [体系结构说明](architecture.md) 阅读，论文表格中的最终数字与逐类结果见 [实验结果](results.md)，本文只负责"怎么跑"。

需要先说明仓库的现状：本仓库不包含训练好的模型权重（唯一的例外是 BEATs 全量微调权重，发布在 Releases 的 `beats-fullft-v1`），也不包含 `exp/` 训练输出目录、tensorboard 日志和预测 TSV；`confs/` 中保留的是配置定义，`docs/results/data/` 中保留的是整理后的结果表，它们都不是可以直接加载的权重。因此下面的流程描述的是文件和命令应当如何组织，它还没有在一台干净的机器上端到端验证过：安装依赖、下载数据、放置权重、启动训练这几步都跑通之前，不应把本文当作已验证的复现记录。凡是本文没有把握的地方都直接写"需要确认"，请按你环境里的实际报错处理。

## 1. 实验环境与依赖安装

### 1.1 论文环境

论文表 2 给出的实验配置是：Ubuntu 20.04，Python 3.10.20，PyTorch 2.7.1+cu128，CUDA 13.0，Intel Xeon Platinum 8470Q CPU，90 GB 内存，单张 NVIDIA RTX 5090 32 GB 显卡。这里的 CUDA 13.0 指机器上 CUDA 运行时/驱动侧记录的版本，而 PyTorch wheel 的 `+cu128` 表示框架自带的是 CUDA 12.8 分支的运行库；两者可以共存，wheel 会携带自己的 CUDA 库，具体最低驱动版本请以 PyTorch 官方安装矩阵为准（需要确认）。RTX 5090 是 Blackwell 架构（sm_120），必须使用 cu128 及以上的 wheel，早期 CUDA 11.x 或 12.6 的 wheel 不能直接在这张卡上运行。论文只记录了主要框架版本，没有记录 pytorch-lightning、torchmetrics 等库的版本，因此 [requirements.txt](../requirements.txt) 中这些依赖给的是区间而不是钉死版本。

本仓库按 Linux + NVIDIA GPU 编写，路径和日志目录都按 Linux 习惯书写（例如 `TensorBoardLogger` 的目录拆分用 `/` 处理）。在 Windows 上做本地 clone 适合阅读和编辑代码，直接运行训练需要确认路径分隔符和依赖可用性，本文不作为支持目标。

### 1.2 创建 Python 环境

推荐用 conda 建立独立的 Python 3.10 环境，再按 cu128 索引安装框架，最后安装其余依赖：

```bash
conda create -n dcase2022 python=3.10 -y
conda activate dcase2022
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

论文记录的 Python 是 3.10.20，这个 patch 版本能否直接从 conda 默认频道装上需要确认；代码不依赖具体 patch 版本，装同一 3.10.x 系列即可。如果 cu128 索引中没有与 torch 2.7.1 配对的 torchaudio 2.7.1 wheel，上面的第二条命令需要换成索引里实际存在的成对版本（需要确认）。安装完成后可以先做一次简单自检：

```bash
python -c "import torch, torchaudio; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果环境装在离线集群上，需要提前把 wheel 和 torchaudio 的 WavLM 权重缓存准备好，这一步的缓存位置和行为需要确认。

### 1.3 系统依赖

数据下载脚本依赖 ffmpeg 提供的 ffprobe（上游 README 记录过 `FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'` 这个报错），在 Ubuntu 上安装系统包即可：

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 2. 数据准备

### 2.1 下载 DESED 数据

`generate_dcase_task4_2022.py` 是上游 DCASE 2022 Task 4 baseline 的数据下载脚本，它通过 `desed` 库下载 DESED 的真实数据部分（弱标签、无标签、validation 以及可选的 AudioSet 强标签）和合成的 `dcase_synth` 数据包，并把音频目录、metadata 目录以软链接方式整理到同一个数据集根目录。最常用的两种用法是：

```bash
# 下载完整开发集（真实数据 + 合成数据 + AudioSet 强标签）
python generate_dcase_task4_2022.py --basedir=../../data

# 只下载复现 synth_only 实验所需的合成数据
python generate_dcase_task4_2022.py --basedir=../../data --only_synth
```

当不指定 `--out_dir` 时，脚本会把数据放到 `<basedir>/dcase/dataset` 下（默认 `../../data/dcase/dataset`，相对启动命令时的当前目录）。脚本还会创建 `missing_files` 目录记录真实数据中下载失败的条目，上游流程要求把该清单发给任务组织方补齐，合成数据部分不受影响。需要注意 `<basedir>` 与 `--out_dir` 的组合存在一个由源码决定的小限制：合成数据压缩包的解压与软链接路径是按 `basedir` 推导的，当 `--out_dir` 不等于 `<basedir>/dcase/dataset` 时，`create_synth_dcase` 会去一个不存在的位置找 `dcase_synth`，实际行为需要确认；建议直接用默认 `--out_dir` 或下载后手工整理目录。

### 2.2 配置键与数据目录的映射

训练脚本本身不写死任何数据路径，所有路径都来自 YAML 的 `data` 段。主配置 [confs/experiment_unified.yaml](../confs/experiment_unified.yaml) 使用下面这些键：

| 配置键 | 含义 | 主配置中的默认值 |
| --- | --- | --- |
| `synth_folder` / `synth_folder_44k` | 合成训练集 16 kHz 目录 / 原始采样率目录 | `./runtime_data/dcase_synth/audio/train/synthetic21_train/soundscapes_16k` / 空 |
| `synth_tsv` | 合成训练集强标签 TSV | `./runtime_data/dcase_synth/metadata/train/synthetic21_train/soundscapes.tsv` |
| `strong_folder` / `strong_folder_44k` / `strong_tsv` | AudioSet 强标签训练集（仅在 `--strong_real` 且非 synth_only 时读取） | 空 |
| `weak_folder` / `weak_folder_44k` / `weak_tsv` | 弱标签训练集（非 synth_only 时读取） | 空 |
| `unlabeled_folder` / `unlabeled_folder_44k` | 无标签训练集（非 synth_only 时读取） | 空 |
| `synth_val_folder` / `synth_val_folder_44k` / `synth_val_tsv` / `synth_val_dur` | 合成验证集音频、标签与时长 TSV | `./runtime_data/dcase_synth/.../soundscapes_16k` 等 |
| `test_folder` / `test_folder_44k` / `test_tsv` / `test_dur` | 测试集音频、标签与时长 TSV | 与合成验证集相同 |
| `eval_folder` / `eval_folder_44k` | 官方 evaluation set 的无标签音频目录（`--eval_from_checkpoint` 用） | 空 |
| `audio_max_len` / `fs` / `net_subsample` | 截断长度（秒）、采样率、网络下采样倍数 | `10` / `16000` / `4` |

强标签 TSV 的列是 `filename`、`onset`、`offset`、`event_label`，弱标签 TSV 的列是 `filename`、`event_labels`，时长 TSV 的列是 `filename`、`duration`，这些格式由 `desed_task/dataio/datasets.py` 和 `local/utils.py` 的读取逻辑决定。主配置中 `test_folder`、`test_tsv`、`test_dur` 与合成验证集完全一致，这意味着按原文件跑 `--test_from_checkpoint` 得到的是"在合成验证集上自测"的指标，而不是 DESED 官方 dev-test 指标；若要评估真实 dev-test，需要把 `test_folder`、`test_tsv` 改为 validation 数据，并生成或指向对应的 `test_dur`。另外 `train_sed.py` 不会切换工作目录，YAML 里的相对路径、`--log_dir` 都相对启动命令时的当前目录解析，建议统一在仓库根目录启动，或者直接写绝对路径。

`./runtime_data/dcase_synth/...` 只是论文实验服务器上整理好的一种本地目录布局，不是数据下载脚本的默认输出。拿到下载好的数据后，需要按自己的数据集根目录修改 `data` 段的键。例如把数据放在 `/data/dcase`，一个可用的示例如下（把 `/data/dcase` 换成自己的路径即可）：

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

如果只做 synth_only 复现，`strong_*`、`weak_*`、`unlabeled_*` 保持空字符串即可，训练脚本在 `synth_only` 分支不会读取它们；但 `test_tsv` / `test_dur` 仍然必须有值，因为测试集对象在训练开始前就会构建。

### 2.3 重采样到 16 kHz 与 duration TSV

16 kHz 重采样由 `train_sed.py` 调用 `local/resample_folder.py` 完成。具体逻辑是：`resample_data_generate_durations` 只处理"`<键名>_44k` 和对应 16 kHz 键都非空"的数据集，用 `torchaudio.functional.resample` 把原始采样率音频写入 16 kHz 目录；随后若 `synth_val_dur`、`test_dur` 文件不存在（或本次发生了重采样），用 `generate_tsv_wav_durations` 扫描音频目录重新生成时长 TSV。主配置中所有 `*_44k` 键都是空字符串，所以按原文件运行不会自动重采样，`soundscapes_16k` 目录必须事先存在。两种准备方式都可以：一是同时在 YAML 里填好 `synth_folder_44k` 与 `synth_folder`（以及验证集、测试集对应键），第一次训练启动时自动完成转换，这要求 16 kHz 输出目录有写权限并有足够磁盘空间；二是手工调用重采样脚本，适合想先把数据准备好、再多次训练的场景：

```bash
python local/resample_folder.py \
  --in_dir /data/dcase/audio/train/synthetic21_train/soundscapes \
  --out_dir /data/dcase/audio/train/synthetic21_train/soundscapes_16k \
  --target_fs 16000
```

`local/resample_folder.py` 的跳过条件是"输出目录中 `*.wav` 数量等于输入目录"；如果上一次转换中途失败、只写了一部分文件，再次运行会从头重算并覆盖，因此不要把不完整的输出目录当成已完成。上游 README 也提醒过：首次运行会在数据目录旁边创建 16 kHz 副本，数据目录需要可写。

## 3. 预训练权重与 warm-start 检查点

### 3.1 权重清单与缺失时的行为

仓库不提供训练权重，`.gitignore` 也排除了 `pretrained/`、`*.pt`、`*.ckpt` 等路径（BEATs 全量微调权重例外，它发布在 Releases 里，不随仓库分发）。完整复现需要在本地按下面的路径准备权重。主配置 [confs/experiment_unified.yaml](../confs/experiment_unified.yaml) 引用了六份权重，缺失时的行为并不相同：

| 文件（相对仓库根目录） | 配置键 | 来源 | 文件缺失时的行为 |
| --- | --- | --- | --- |
| `pretrained/beats/BEATs_iter3+_AS2M.pt` | `model.branches.beats.pretrained_checkpoint` | Microsoft BEATs 官方权重 | 构建模型时直接抛 `FileNotFoundError` |
| `pretrained/beats/BEATs_full_finetune_best_0_78.pt` | `model.branches.beats.warmstart.checkpoint` | 本项目的 BEATs 全量微调产物，已发布在 Releases 的 [`beats-fullft-v1`](https://github.com/HarryWeasley3/dcase-2022-task4/releases/tag/beats-fullft-v1) | `warmstart.enable` 为 true 时抛 `FileNotFoundError` |
| `pretrained/wavlm/WavLM-Base+.pt` | `model.branches.wavlm.pretrained_checkpoint` | torchaudio WavLM bundle，或 torchaudio 兼容的 state dict | 文件不存在且 `use_bundle_weights` 为 true 时回退到 torchaudio bundle |
| `pretrained/wavlm/WavLM_full_finetune_best_0_65.pt` | `model.branches.wavlm.warmstart.checkpoint` | 本项目的 WavLM 全量微调产物，可用 `scripts/extract_wavlm_ckpt.py` 导出 | 打印 `checkpoint not found ... skipping` 后继续 |
| `pretrained/crnn/crnn_best.pt` | `model.branches.crnn.warmstart.checkpoint` | 纯 CRNN 实验的 best checkpoint（旧版 CRNN 权重格式，见 3.3） | 打印 `checkpoint not found ... skipping` 后继续 |
| `exp/unified_beats_synth_only_a800_finetune/version_5/epoch=55-step=8791.ckpt` | `model.decoder_warmstart.checkpoint` | 论文 BEATs 全量微调实验的 best checkpoint | 打印 `checkpoint not found ... skipping` 后继续 |

表中的"抛 `FileNotFoundError`"和"打印 skipping"都来自 `sed_modeling/encoders/beats_encoder.py` 与 `sed_modeling/models/sed_model.py` 的现有加载逻辑。对默认配置而言，前两项是硬依赖，后四项缺失只会让对应分支从随机初始化或 torchaudio 自带权重开始，不会阻止训练启动，但会改变与论文实验的初始化一致性。

### 3.2 BEATs 与 WavLM 预训练权重

`BEATs_iter3+_AS2M.pt` 需要从 Microsoft 官方 BEATs 发布页获取（仓库 vendored 代码来自 `microsoft/unilm` 的 `beats` 目录，具体下载链接以该仓库 README 为准，本文不提供第三方直链）。该文件必须同时包含 `cfg` 和 `model` 两个键，`sed_modeling/encoders/beats_encoder.py` 会用 `BEATsConfig(state["cfg"])` 构建结构、用 `load_state_dict(state["model"])` 加载权重。

WavLM 分支的代码路径支持两种来源：`pretrained_checkpoint` 指向的文件存在时，按 torchaudio WavLM 的 state dict 格式严格加载；文件不存在而 `use_bundle_weights: true` 时，改用 `torchaudio.pipelines.WAVLM_BASE_PLUS` bundle。论文实际使用的 `WavLM-Base+.pt` 是如何生成的，仓库中没有记录，这一步需要确认；如果只是想让流程跑起来，可以让该文件不存在、直接使用 bundle（要求能访问 torchaudio 的模型下载源，离线环境需要预先准备缓存）。如果希望本地有一份与 `WavLMEncoder` 加载逻辑一致的权重，可以用 `torch.save({"model": torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().state_dict()}, "pretrained/wavlm/WavLM-Base+.pt")` 从 torchaudio bundle 导出——这只是一种与代码约定一致的做法，并不保证等同于论文使用的文件。Microsoft 官方发布的 WavLM checkpoint 是 fairseq 格式，能否被当前的严格加载路径接受尚未验证，需要确认，不要假设改个文件名就能用。

### 3.3 项目内 warm-start 权重的准备

`BEATs_full_finetune_best_0_78.pt` 已经发布在 Releases 的 [`beats-fullft-v1`](https://github.com/HarryWeasley3/dcase-2022-task4/releases/tag/beats-fullft-v1)：下载后按原文件名放进 `pretrained/beats/` 即可，配置里的 `model.branches.beats.warmstart.checkpoint` 正好指向这个路径。文件约 345 MB，sha256 为 `dff86d366a70079d7e94d2da787d9d216a4cf2a418d666a25fc79c2172432fec`；它是 `scripts/extract_ckpt.py` 把全量微调的 BEATs 编码器权重合并回官方文件后得到的格式，`BEATsEncoder` 可以直接加载。

`WavLM_full_finetune_best_0_65.pt` 还没有发布，是论文多阶段训练的中间产物，当前仓库的 `confs/` 中没有保留生成它的单分支训练配置，`exp/` 目录也没有随仓库发布，因此要复现这个权重，需要先按单分支方式训练 WavLM-only 模型，再用脚本导出：

```bash
# 从 WavLM-only 训练产出的 checkpoint 中导出 WavLM 权重
python scripts/extract_wavlm_ckpt.py --src /path/to/wavlm_only_run/epoch=xxx-step=xxxxx.ckpt --out pretrained/wavlm/WavLM_full_finetune_best_0_65.pt
```

如果要自己重新生成 BEATs warm-start 权重，对应命令是：

```bash
# 从 BEATs-only 训练产出的 Lightning checkpoint 中合并出 BEATs warm-start 权重
python scripts/extract_ckpt.py --src /path/to/beats_only_run/epoch=xxx-step=xxxxx.ckpt --base pretrained/beats/BEATs_iter3+_AS2M.pt --out pretrained/beats/BEATs_full_finetune_best_0_78.pt
```

这两个脚本只负责从训练 checkpoint 里抽取编码器权重（BEATs 脚本还会把权重合并回官方文件的 `model` 键），它们不负责训练模型；训练单分支模型所需的配置、轮数和随机种子是否与论文完全一致，需要确认。`crnn_best.pt` 的情况更麻烦一些：当前 `build_sed_model` 统一走 encoder-decoder 结构，用 `confs/default.yaml` 训练出的 CRNN checkpoint 里编码器键是 `sed_student.encoder.cnn.*` 形式，而 `_load_crnn_encoder_warmstart` 只识别 `crnn_encoder.cnn.`、`sed_student.cnn.`、`cnn.` 这三种旧版 CRNN 权重前缀，两者并不匹配。因此用当前代码重新训练 CRNN 后不能直接把 checkpoint 改名成 `crnn_best.pt` 当 warm-start 用，需要自行做 key 重映射导出（仓库没有现成的转换脚本，这一步需要自行处理），或者使用旧版 CRNN 实验留下的 checkpoint。

## 4. 冒烟测试

`--fast_dev_run` 会在训练脚本层面把每套数据限制为 2 个 batch、最多 3 个 epoch，并把日志频率调成每步一次，用于检查数据管线、模型构建和日志记录是否正常。它仍然会完整构建模型、读取配置里的权重路径，因此必须是合成数据、BEATs 官方权重和 BEATs warm-start 权重都已就位才能按默认配置跑起来：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --log_dir ./exp/smoke_stage_a \
  --gpus 0 \
  --fast_dev_run
```

这里的 `--gpus 0` 在推荐安装的 Lightning 2.x 路径下会走 CPU（源码中显式设置 `accelerator="cpu"`），适合本地小规模检查；真正的实验必须用 GPU，否则 BEATs 与 WavLM 两个 Transformer 编码器在 CPU 上会非常慢。如果只是想先验证数据准备和 CRNN 通路，可以改用 `--conf_file confs/default.yaml --synth_only --fast_dev_run`，它不需要 BEATs 权重。

仓库还提供了两个不依赖真实权重的构造测试：`tests/test_hierarchical_ssl_fusion.py` 用假的编码器替代 BEATs/WavLM 来验证层级融合的装配和前向形状，`tests/test_scheduler_decay.py` 验证学习率调度器；安装 `pytest` 后可以运行 `python -m pytest tests/`，但这些测试在当前依赖环境下是否全部通过需要确认。

## 5. 训练

### 5.1 配置结构与 profile

`train_sed.py` 用 `--conf_file` 读取 YAML，并在加载时解析 `profile`：文件顶层的 `profile` 字段从同文件的 `profiles` 字典中选出一组覆盖项，与基础配置做深度合并。主配置 `confs/experiment_unified.yaml` 的默认 profile 是 `rtx_5090`，它只覆盖训练相关的数值，不覆盖融合策略，因此按原文件训练得到的是基础的 `residual_gated` 三路残差门控融合，也就是论文实验用的那条配置，单卡 5090 就能跑；同一个文件里还有一个 `hierarchical_ssl_stage_a` profile，它把 `model.fusion.strategy` 覆盖为 `hierarchical_ssl`，训练的是 CRNN + BEATs + WavLM 的层级 SSL 融合模型（论文之后的扩展）。仓库没有提供选择 profile 的命令行参数，切换需要直接编辑 YAML 顶部的 `profile` 值。

另外还有一份独立的 [confs/experiment_resgate_beats_anchor_topk.yaml](../confs/experiment_resgate_beats_anchor_topk.yaml)，它是残差门控 + BEATs 时间锚点 + 分组学习率 + top-k checkpoint 保存的配置变体，`training.synth_only` 同样为 true，可用于对照实验，与默认配置互不影响。

### 5.2 训练命令与关键超参

默认配置（`profile: rtx_5090`）的训练命令如下（配置里 `training.synth_only` 已经是 true，`--synth_only` 只是显式重申，二者等价）：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --log_dir ./exp/unified_resgate_rtx5090 \
  --gpus 1 \
  --synth_only
```

当前主配置的训练设定是：`n_epochs: 120`、`n_epochs_warmup: 10`、`batch_size: 8`、`batch_size_val: 8`、`num_workers: 4`、`accumulate_batches: 1`、`precision: 32`、`gradient_clip: 1.0`、`early_stop_patience: 20`、`median_window: 7`、`val_thresholds: [0.3, 0.4, 0.5, 0.6, 0.7]`、`n_test_thresholds: 50`、`ema_factor: 0.999`、`self_sup_loss: mse`、`mixup: soft`、`seed: 42`；优化器是 Adam，基础学习率 `1.5e-5`，调度器为 `warmup_cosine`（`min_lr: 1e-6`、`decay_power: 0.6`）。主配置的 `opt.param_groups` 没有开启，三路分支共享同一个学习率；需要分组学习率时使用 5.1 节提到的那份残差门控配置。模型侧，`model.branches` 同时启用 crnn、beats、wavlm 三个分支，BEATs 与 WavLM 的 `freeze` 都是 false（全量微调），融合策略是 `residual_gated`（`fusion_type: beats_main_residual_gated`，以 BEATs 为主路径），`fuse_dim: 256`、门控为 channel 模式、α 可学习；层级融合的 `beats_layers` / `wavlm_layers` 等参数只在 `hierarchical_ssl_stage_a` profile 下生效。具体结构参见 [体系结构说明](architecture.md)。

`--gpus` 的语义按代码实现：在 Lightning 2.x 路径下，字符串 `"0"` 表示 CPU，其余任何值都按单卡 GPU 处理并固定 `devices=1`，不解析卡号；如果要指定具体显卡，需要用 `CUDA_VISIBLE_DEVICES` 控制可见设备（这一点是从源码直接推断的，实际行为需要确认）。多卡不受支持：`--gpus "1,2"` 会直接抛 `NotImplementedError`。当 `synth_only` 生效时，同时传入 `--strong_real` 只会得到一条被忽略的警告。

### 5.3 恢复训练与权重初始化

训练中断后可以从 checkpoint 恢复完整训练状态，包括优化器、调度器和当前 epoch：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --log_dir ./exp/unified_hierarchical_ssl_stage_a_resume \
  --gpus 1 \
  --resume_from_checkpoint ./exp/unified_hierarchical_ssl_stage_a/version_0/last.ckpt
```

如果只想用某个模型权重初始化网络、但不恢复优化器和训练进度，用 `--init_from_checkpoint` 并传入一个包含 `state_dict` 的 Lightning checkpoint；它会按 key 名和形状匹配加载，并打印加载、缺失、形状不匹配的统计，匹配不到任何权重时会直接报错。两个参数互斥，同时传入会抛 `ValueError`；用法与上面的 resume 命令相同，只是把参数换成 `--init_from_checkpoint /path/to/crnn_beats.ckpt`。

### 5.4 输出与日志

训练日志通过 `TensorBoardLogger` 写到 `--log_dir` 指定的目录，使用 `tensorboard --logdir ./exp` 查看；checkpoint 由 `ModelCheckpoint` 以 `epoch=xxx-step=xxxxx.ckpt` 的文件名保存在同一目录，保存策略由 `training.checkpoint_top_k`（默认 1）和 `checkpoint_save_last`（默认 true）控制，监控指标是验证集上的 `val/obj_metric`。在 `synth_only` 设定下，验证集只有合成验证集，`val/obj_metric` 等于合成验证集上的 intersection-based F1，早停与选模都基于这个指标。训练结束时会打印 best checkpoint 路径，并在 `metrics_test/` 下写入测试集（默认即合成验证集）的 PSDS、event F1、intersection F1 结果；训练与 dev-test 的能耗 kWh 分别记录在 `training_codecarbon/`、`devtest_codecarbon/` 目录，evaluation 分支的能耗记录见 6.2 的说明。`scaler.ckpt` 只有在 `scaler.statistic` 为 `dataset` 时才会生成并缓存，主配置用的是 `instance` 统计，不会产生该文件。

## 6. 评估

### 6.1 在带标签的 dev-test 上评估

`--test_from_checkpoint` 会加载 checkpoint，用其中的 `hyper_parameters` 还原模型与训练配置，但 `data` 段以及 `batch_size_val`、`num_workers`、`median_window`、`n_test_thresholds`、`val_thresholds`、`precision`、`backend`、`synth_only` 这些运行期参数会以这次传入的 YAML 为准。因此评估时要传入数据路径正确的配置文件，例如：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --test_from_checkpoint /path/to/epoch=xxx-step=xxxxx.ckpt \
  --log_dir ./exp/eval_devtest \
  --gpus 1
```

按主配置原样运行时，`test_*` 指向合成验证集，得到的是自测指标；要评估 DESED dev-test，请先把 `test_folder`、`test_tsv`、`test_dur` 改成 validation 数据（见 2.2 节示例）。结果写入 `./exp/eval_devtest/metrics_test/student/` 与 `.../teacher/`，包含 PSDS scenario1、scenario2、event-based F1、intersection-based F1 以及保存下来的预测 TSV；能耗写入 `devtest_codecarbon/devtest_tot_kwh.txt`。

### 6.2 在无标签 evaluation set 上生成预测

`--eval_from_checkpoint` 走另一个分支：它把 `evaluation=True` 传给训练器，测试集改为 `data.eval_folder` 下的无标签音频，`batch_size_val` 被强制设为 1，不计算指标，只把不同阈值下的预测 TSV 保存到 `metrics_test/` 下，供官方评测使用：

```bash
python train_sed.py \
  --conf_file confs/experiment_unified.yaml \
  --eval_from_checkpoint /path/to/epoch=xxx-step=xxxxx.ckpt \
  --log_dir ./exp/eval_official \
  --gpus 1
```

使用前必须把 `eval_folder` 填成 evaluation set 的音频目录（主配置默认为空字符串，此时构建无标签数据集会失败）。另外从源码看，评估分支有两处需要留意：保存 teacher 目录下的多阈值预测文件时，写入的是 student 的全阈值缓冲区；写能耗文件的代码只存在于带标签测试分支中，`evaluation_codecarbon/eval_tot_kwh.txt` 按当前 `on_test_epoch_end` 的缩进结构实际不会被写入，tracker 也没有在该分支 stop（见 `local/sed_trainer.py`）。因此引用 evaluation 能耗或 teacher 预测前需要确认这些行为是否符合预期。

## 7. 常见问题

**下载数据时报 `FileNotFoundError: ffprobe`。** 说明系统里没有 ffmpeg，按 1.3 节安装即可；这个问题发生在下载真实数据阶段，与模型代码无关。

**重采样或生成 duration TSV 时报权限错误。** `resample_folder` 会往 YAML 中配置的 16 kHz 输出目录写文件，`generate_tsv_wav_durations` 会往 metadata 目录写 TSV，二者都需要写权限；把输出目录改到可写位置，或在首次运行前手工准备好 16 kHz 目录和 duration TSV。

**显存不足（OOM）。** 主配置默认对 BEATs 和 WavLM 都做全量微调，32 GB 显存下的 batch size 是 8；如果换成更小显存的卡或同时有其他任务占用显存，可以先把 `model.branches.beats.freeze` 和/或 `model.branches.wavlm.freeze` 改为 true（冻结的编码器走 `torch.no_grad()`，优化器也只接收 `requires_grad=True` 的参数），再降低 `training.batch_size` 和 `batch_size_val`。`model.teacher.share_frozen_beats` / `share_frozen_wavlm` 可以在编码器冻结时让教师模型共享同一份编码器权重，进一步减少显存占用。

**`--gpus 0` 到底是什么意思。** 在推荐安装的 Lightning 2.x 下它表示 CPU；其他值表示单卡 GPU，但不会选择具体卡号，需要选卡请设置 `CUDA_VISIBLE_DEVICES`。多卡训练目前会直接抛 `NotImplementedError`。

**训练日志或结果目录不正确。** `--log_dir` 会同时影响 TensorBoard 目录、checkpoint 目录和 `metrics_test/` 的输出位置；评估时也要显式传 `--log_dir`，否则默认写到 `./exp/2022_baseline` 这类共享路径，容易混淆不同实验的结果。

**Lightning 版本。** 仓库原 requirements 把 `pytorch-lightning` 钉在 1.5.1，这是上游 2021 年 baseline 的旧环境；当前代码对 Trainer 参数做了 1.x/2.x 两套兼容分支，建议按 2.x 安装。论文表 2 没有记录 Lightning 版本，实际使用哪个 2.x 小版本能完全复现训练行为需要确认。

**导入时报 `ModuleNotFoundError: No module named 'asteroid'`。** 即使你只想训练 CRNN，`train_sed.py` 导入 `desed_task.utils.encoder` 时也会触发 `desed_task/utils/__init__.py` 里的 `from .schedulers import ...`，而该模块依赖 asteroid 的 `BaseScheduler`；安装 [requirements.txt](../requirements.txt) 即可解决，但 asteroid 0.6.0 与 PyTorch 2.7 的兼容性还需要确认。

**启动或加载权重时看到旧键名和 skipping 提示。** BEATs 报错里出现的 `model.beats.checkpoint`、`model.beats.branch_checkpoint` 是加载器内部使用的旧键名，配置里对应 `model.branches.beats.pretrained_checkpoint` 与 `model.branches.beats.warmstart.checkpoint`，`sed_modeling/models/sed_model.py` 会自动映射，按新键名填路径即可。`checkpoint not found ... skipping` 则是 `decoder_warmstart`、CRNN warm-start、WavLM warm-start 这类可选权重缺失时的正常提示，训练会继续，但初始化与论文实验不一致；要严格复现请补齐 3.1 节表格中的文件。

## 8. 许可证、第三方代码与复现状态

仓库根目录的 [LICENSE](../LICENSE) 是标准 MIT 全文，Copyright (c) 2026 李想 (Li Xiang)。仓库中的 `desed_task/` 源自 DCASE 官方仓库 `DCASE-REPO/DESED_task` 的 DCASE 2022 Task 4 baseline（上游同样以 MIT 发布），保留该目录时请同时保留其归属说明；`sed_modeling/third_party/beats/` 是 Microsoft BEATs 官方代码的最小化 vendored 版本，其许可证单独放在 [sed_modeling/third_party/beats/LICENSE](../sed_modeling/third_party/beats/LICENSE)，Copyright (c) 2022 Microsoft，来源是 `microsoft/unilm` 的 `beats` 目录。DESED 数据集、AudioSet 数据、BEATs 与 WavLM 预训练权重的使用还受各自发布方条款约束，论文和商业使用前需要核对具体许可；这些内容不在本仓库 MIT 的覆盖范围内。

最后再强调一次：仓库不包含训练权重（BEATs 全量微调权重在 Releases 里单独发布），也不包含 `exp/` 结果目录、tensorboard 日志和预测文件，本文中的流程尚未在干净环境中端到端验证。当前可以确认的是代码和配置的静态结构（argparse 参数、配置键、加载逻辑、缺失时的提示行为都已按源码核对）；以下事项需要在真实环境中确认：论文所用各库的精确版本（尤其 pytorch-lightning、torchmetrics、scipy、codecarbon）、BEATs/WavLM 官版权重的具体下载方式与 WavLM-Base+.pt 的格式、单分支 warm-start 权重的训练配置与导出结果、decoder warm-start checkpoint 是否参与论文主实验，以及论文表格中的数字是否能由当前 `confs/experiment_unified.yaml` 精确复现。论文口径的结果数据请以 [实验结果](results.md) 和 [实验映射表](results/data/experiment_mapping_table.csv) 为准；架构细节请以 [体系结构说明](architecture.md) 为准。
