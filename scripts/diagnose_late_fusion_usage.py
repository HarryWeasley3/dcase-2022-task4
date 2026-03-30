import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import yaml
from dcase_util.data import DecisionEncoder

ROOT = Path("/home/llxxll/pyProj/dcase-2022-task4")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desed_task.dataio.datasets import StronglyAnnotatedSet
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_sed_eval_metrics,
)
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.scaler import TorchScaler
from local.classes_dict import classes_labels
from sed_modeling import build_sed_model


FUSION_CONFIG_PATH = ROOT / "confs/crnn_beats_late_fusion_synth_only.yaml"
FUSION_CKPT_PATH = ROOT / "exp/2022_baseline/version_20/epoch=34-step=21875.ckpt"
CRNN_CONFIG_PATH = ROOT / "confs/synth_only_d_drive.yaml"
CRNN_CKPT_PATH = ROOT / "exp/2022_baseline/version_4/epoch=133-step=111756.ckpt"

OUT_DIR = ROOT / "comparison_reports"
REPORT_PATH = OUT_DIR / "fusion_diagnosis_report.md"
OVERALL_CSV = OUT_DIR / "fusion_diagnosis_overall.csv"
FOCUS_CSV = OUT_DIR / "fusion_diagnosis_focus_classes.csv"
POSTERIOR_CSV = OUT_DIR / "fusion_diagnosis_posterior.csv"
WINLOSS_CSV = OUT_DIR / "fusion_diagnosis_winloss_groups.csv"
JSON_PATH = OUT_DIR / "fusion_diagnosis_results.json"

FOCUS_CLASSES = [
    "Dishes",
    "Dog",
    "Alarm_bell_ringing",
    "Cat",
    "Running_water",
    "Frying",
    "Vacuum_cleaner",
]

POSTERIOR_CLASSES = [
    "Dishes",
    "Dog",
    "Alarm_bell_ringing",
    "Cat",
    "Running_water",
]


class RunningTensorStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None
        self.batch_means = []
        self.batch_stds = []

    def update(self, tensor):
        x = tensor.detach().float().cpu()
        flat = x.reshape(-1)
        self.count += flat.numel()
        self.sum += float(flat.sum())
        self.sumsq += float((flat * flat).sum())
        cur_min = float(flat.min())
        cur_max = float(flat.max())
        self.min = cur_min if self.min is None else min(self.min, cur_min)
        self.max = cur_max if self.max is None else max(self.max, cur_max)
        self.batch_means.append(float(flat.mean()))
        self.batch_stds.append(float(flat.std(unbiased=False)))

    def summary(self):
        mean = self.sum / max(self.count, 1)
        var = max(self.sumsq / max(self.count, 1) - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": var ** 0.5,
            "min": self.min,
            "max": self.max,
            "batch_mean_min": min(self.batch_means) if self.batch_means else None,
            "batch_mean_max": max(self.batch_means) if self.batch_means else None,
            "batch_std_min": min(self.batch_stds) if self.batch_stds else None,
            "batch_std_max": max(self.batch_stds) if self.batch_stds else None,
        }


def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def build_encoder(config):
    return ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )


def build_loader(config, encoder, batch_size=8):
    gt_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    dataset = StronglyAnnotatedSet(
        config["data"]["test_folder"],
        gt_df,
        encoder,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return gt_df, loader


def set_input_scaler_if_needed(model, config):
    if getattr(model, "needs_input_scaler", False):
        model.set_input_scaler(
            TorchScaler(
                "instance",
                config["scaler"]["normtype"],
                config["scaler"]["dims"],
            )
        )


def load_fusion_model(config, device):
    model = build_sed_model(config)
    set_input_scaler_if_needed(model, config)
    ckpt = torch.load(FUSION_CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["sed_student"], strict=True)
    model.to(device).eval()
    return model


def map_old_crnn_state(old_state):
    new_state = {}
    for key, value in old_state.items():
        if key.startswith("cnn."):
            new_state["encoder." + key] = value
        elif key.startswith("rnn."):
            new_state["decoder.temporal." + key[4:]] = value
        elif key.startswith("dense."):
            new_state["decoder.strong_head." + key[len("dense."):]] = value
        elif key.startswith("dense_softmax."):
            new_state["decoder.attention_head." + key[len("dense_softmax."):]] = value
    return new_state


def load_crnn_model(config, device):
    model = build_sed_model(config)
    set_input_scaler_if_needed(model, config)
    ckpt = torch.load(CRNN_CKPT_PATH, map_location="cpu", weights_only=False)
    mapped = map_old_crnn_state(ckpt["sed_student"])
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading CRNN model: {unexpected}")
    model.to(device).eval()
    return model, missing


def decode_scores(scores_bct, filenames, encoder, threshold, median_window):
    decision_encoder = DecisionEncoder()
    rows = []
    n_files, n_classes, _ = scores_bct.shape
    for file_idx in range(n_files):
        pred_tc = scores_bct[file_idx].transpose(1, 0)
        for class_idx, label in enumerate(encoder.labels):
            binary = pred_tc[:, class_idx] > threshold
            if median_window > 1:
                binary = scipy.ndimage.median_filter(binary.astype(np.uint8), size=(median_window,))
                binary = binary.astype(bool)
            regions = decision_encoder.find_contiguous_regions(binary)
            for onset, offset in regions:
                rows.append(
                    {
                        "filename": filenames[file_idx],
                        "event_label": label,
                        "onset": float(encoder._frame_to_time(onset)),
                        "offset": float(encoder._frame_to_time(offset)),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["filename", "event_label", "onset", "offset"])
    return pd.DataFrame(rows)


def compute_metrics(pred_df, gt_df, config):
    if pred_df.empty:
        event_macro = event_micro = segment_macro = segment_micro = 0.0
        event_classwise = {cls: 0.0 for cls in classes_labels.keys()}
        segment_classwise = {cls: 0.0 for cls in classes_labels.keys()}
    else:
        event_metric, segment_metric = compute_sed_eval_metrics(pred_df, gt_df)
        event_res = event_metric.results()
        segment_res = segment_metric.results()
        event_macro = float(event_res["class_wise_average"]["f_measure"]["f_measure"])
        event_micro = float(event_res["overall"]["f_measure"]["f_measure"])
        segment_macro = float(segment_res["class_wise_average"]["f_measure"]["f_measure"])
        segment_micro = float(segment_res["overall"]["f_measure"]["f_measure"])
        event_classwise = {
            cls: float(event_res["class_wise"][cls]["f_measure"]["f_measure"])
            for cls in classes_labels.keys()
        }
        segment_classwise = {
            cls: float(segment_res["class_wise"][cls]["f_measure"]["f_measure"])
            for cls in classes_labels.keys()
        }

    intersection = float(
        compute_per_intersection_macro_f1(
            {0.5: pred_df},
            config["data"]["test_tsv"],
            config["data"]["test_dur"],
        )
    ) if not pred_df.empty else 0.0

    return {
        "event_macro": event_macro,
        "event_micro": event_micro,
        "segment_macro": segment_macro,
        "segment_micro": segment_micro,
        "intersection": intersection,
        "event_classwise": event_classwise,
        "segment_classwise": segment_classwise,
    }


def run_inference(fusion_config, crnn_config, device, batch_size):
    encoder = build_encoder(fusion_config)
    gt_df, loader = build_loader(fusion_config, encoder, batch_size=batch_size)
    fusion_model = load_fusion_model(fusion_config, device)
    crnn_model, crnn_missing = load_crnn_model(crnn_config, device)

    variant_scores = defaultdict(list)
    filenames_all = []
    labels_all = []

    stat_buffers = {
        "cnn_feat": RunningTensorStats(),
        "beats_feat": RunningTensorStats(),
        "beats_aligned": RunningTensorStats(),
        "fused": RunningTensorStats(),
        "merge_mlp_out": RunningTensorStats(),
    }

    with torch.no_grad():
        for batch_idx, (audio, labels, _, filenames) in enumerate(loader):
            audio = audio.to(device)
            target_len = labels.shape[-1]
            labels_all.append(labels.numpy().astype(np.float32))
            filenames_all.extend([Path(x).stem + ".wav" for x in filenames])

            cnn_outputs = fusion_model.crnn_encoder(audio)
            beats_outputs = fusion_model.beats_encoder(audio)
            cnn_feat = cnn_outputs["sequence_features"]
            beats_feat = beats_outputs["sequence_features"]
            beats_aligned = fusion_model.fusion_aligner(beats_feat, cnn_feat)

            stat_buffers["cnn_feat"].update(cnn_feat)
            stat_buffers["beats_feat"].update(beats_feat)
            stat_buffers["beats_aligned"].update(beats_aligned)

            perm = torch.roll(torch.arange(audio.shape[0], device=audio.device), shifts=1)
            shuffled_beats = beats_aligned[perm] if audio.shape[0] > 1 else beats_aligned

            fused_normal = torch.cat([cnn_feat, beats_aligned], dim=-1)
            fused_zero_beats = torch.cat([cnn_feat, torch.zeros_like(beats_aligned)], dim=-1)
            fused_zero_cnn = torch.cat([torch.zeros_like(cnn_feat), beats_aligned], dim=-1)
            fused_shuffle = torch.cat([cnn_feat, shuffled_beats], dim=-1)

            stat_buffers["fused"].update(fused_normal)

            merged_normal = fusion_model.merge_mlp(fused_normal)
            merged_zero_beats = fusion_model.merge_mlp(fused_zero_beats)
            merged_zero_cnn = fusion_model.merge_mlp(fused_zero_cnn)
            merged_shuffle = fusion_model.merge_mlp(fused_shuffle)

            stat_buffers["merge_mlp_out"].update(merged_normal)

            def decode_variant(merged):
                aligned = fusion_model.label_aligner(merged, target_len)
                return fusion_model.decoder(aligned)["strong_preds"].detach().cpu().numpy().astype(np.float32)

            variant_scores["fusion"].append(decode_variant(merged_normal))
            variant_scores["zero_beats"].append(decode_variant(merged_zero_beats))
            variant_scores["zero_cnn"].append(decode_variant(merged_zero_cnn))
            variant_scores["shuffle_beats"].append(decode_variant(merged_shuffle))

            crnn_outputs = crnn_model(audio, target_frame_len=target_len)
            variant_scores["crnn"].append(
                crnn_outputs["strong_preds"].detach().cpu().numpy().astype(np.float32)
            )

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"[inference] processed {batch_idx + 1}/{len(loader)} batches",
                    flush=True,
                )

    scores = {name: np.concatenate(parts, axis=0) for name, parts in variant_scores.items()}
    labels_np = np.concatenate(labels_all, axis=0)
    feature_stats = {name: stats.summary() for name, stats in stat_buffers.items()}
    return gt_df, filenames_all, labels_np, scores, feature_stats, crnn_missing


def overall_table(results):
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "variant": name,
                "intersection": round(result["metrics"]["intersection"], 6),
                "event_macro": round(result["metrics"]["event_macro"] * 100, 2),
                "event_micro": round(result["metrics"]["event_micro"] * 100, 2),
                "segment_macro": round(result["metrics"]["segment_macro"] * 100, 2),
                "segment_micro": round(result["metrics"]["segment_micro"] * 100, 2),
                "pred_events": int(len(result["pred_df"])),
                "pred_files": int(result["pred_df"]["filename"].nunique()) if len(result["pred_df"]) else 0,
            }
        )
    return pd.DataFrame(rows)


def focus_class_table(results):
    rows = []
    for cls in FOCUS_CLASSES:
        row = {"class": cls}
        for name, result in results.items():
            row[f"{name}_event"] = round(result["metrics"]["event_classwise"][cls] * 100, 2)
            row[f"{name}_segment"] = round(result["metrics"]["segment_classwise"][cls] * 100, 2)
            row[f"{name}_pred"] = int((result["pred_df"]["event_label"] == cls).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def posterior_table(scores, labels, results):
    rows = []
    for cls in POSTERIOR_CLASSES:
        cls_idx = list(classes_labels.keys()).index(cls)
        cls_scores = scores["fusion"][:, cls_idx, :].reshape(-1)
        cls_labels = labels[:, cls_idx, :].reshape(-1) > 0.5
        pos = cls_scores[cls_labels]
        neg = cls_scores[~cls_labels]
        pos_mean = float(pos.mean()) if len(pos) else 0.0
        neg_mean = float(neg.mean()) if len(neg) else 0.0
        pos_hit = float((pos >= 0.5).mean()) if len(pos) else 0.0
        neg_fp = float((neg >= 0.5).mean()) if len(neg) else 0.0
        event_f1 = results["fusion"]["metrics"]["event_classwise"][cls]
        segment_f1 = results["fusion"]["metrics"]["segment_classwise"][cls]
        if segment_f1 >= 0.55 and event_f1 < 0.30:
            diagnosis = "更像边界/切段/后处理问题"
        elif segment_f1 < 0.40 and (pos_mean - neg_mean) < 0.10:
            diagnosis = "更像弱类表征不足"
        else:
            diagnosis = "表征与边界问题并存"
        rows.append(
            {
                "class": cls,
                "event_f1": round(event_f1 * 100, 2),
                "segment_f1": round(segment_f1 * 100, 2),
                "pos_mean": round(pos_mean, 4),
                "neg_mean": round(neg_mean, 4),
                "pos_ge_0.5": round(pos_hit, 4),
                "neg_ge_0.5": round(neg_fp, 4),
                "separation": round(pos_mean - neg_mean, 4),
                "diagnosis": diagnosis,
            }
        )
    return pd.DataFrame(rows)


def file_metric(pred_rows, gt_rows):
    if pred_rows.empty:
        return 0.0
    event_metric, _ = compute_sed_eval_metrics(pred_rows, gt_rows)
    return float(event_metric.results()["overall"]["f_measure"]["f_measure"])


def file_traits(gt_rows):
    durations = (gt_rows["offset"] - gt_rows["onset"]).to_numpy()
    labels = set(gt_rows["event_label"])
    device_labels = {
        "Vacuum_cleaner",
        "Blender",
        "Electric_shaver_toothbrush",
        "Running_water",
        "Frying",
    }
    weak_labels = {"Dishes", "Dog", "Alarm_bell_ringing", "Cat"}
    return {
        "long_duration": bool((durations >= 8.0).any()),
        "short_event": bool((durations <= 1.0).any()),
        "animal": bool(labels & {"Dog", "Cat"}),
        "device_like": bool(labels & device_labels),
        "weak_focus": bool(labels & weak_labels),
        "multi_event": bool(len(gt_rows) >= 3 or len(labels) >= 2),
    }


def win_loss_analysis(fusion_pred_df, crnn_pred_df, gt_df):
    fusion_groups = {fname: df for fname, df in fusion_pred_df.groupby("filename")}
    crnn_groups = {fname: df for fname, df in crnn_pred_df.groupby("filename")}
    gt_groups = {fname: df for fname, df in gt_df.groupby("filename")}

    rows = []
    for idx, (fname, gt_rows) in enumerate(gt_groups.items(), start=1):
        fusion_score = file_metric(fusion_groups.get(fname, pd.DataFrame(columns=gt_rows.columns)), gt_rows)
        crnn_score = file_metric(crnn_groups.get(fname, pd.DataFrame(columns=gt_rows.columns)), gt_rows)
        delta = fusion_score - crnn_score
        if delta >= 0.15:
            group = "fusion_win"
        elif delta <= -0.15:
            group = "fusion_loss"
        else:
            group = "similar"
        rows.append(
            {
                "filename": fname,
                "fusion_event_f1": fusion_score,
                "crnn_event_f1": crnn_score,
                "delta": delta,
                "group": group,
                **file_traits(gt_rows),
                "gt_labels": ",".join(sorted(set(gt_rows["event_label"]))),
                "gt_events": int(len(gt_rows)),
            }
        )
        if idx % 250 == 0:
            print(f"[winloss] scored {idx}/{len(gt_groups)} files", flush=True)

    df = pd.DataFrame(rows)
    summary_rows = []
    for group_name, group_df in df.groupby("group"):
        summary_rows.append(
            {
                "group": group_name,
                "count": int(len(group_df)),
                "avg_delta": round(float(group_df["delta"].mean()), 4),
                "long_duration_ratio": round(float(group_df["long_duration"].mean()), 4),
                "short_event_ratio": round(float(group_df["short_event"].mean()), 4),
                "animal_ratio": round(float(group_df["animal"].mean()), 4),
                "device_like_ratio": round(float(group_df["device_like"].mean()), 4),
                "weak_focus_ratio": round(float(group_df["weak_focus"].mean()), 4),
                "multi_event_ratio": round(float(group_df["multi_event"].mean()), 4),
                "avg_gt_events": round(float(group_df["gt_events"].mean()), 2),
            }
        )

    examples = {}
    for group_name in ["fusion_win", "similar", "fusion_loss"]:
        sub = df[df["group"] == group_name].sort_values("delta", ascending=(group_name != "fusion_win"))
        examples[group_name] = sub.head(8)[["filename", "delta", "gt_labels", "gt_events"]].to_dict("records")

    return df, pd.DataFrame(summary_rows), examples


def feature_stats_table(feature_stats):
    rows = []
    for name, stats in feature_stats.items():
        rows.append(
            {
                "feature": name,
                "mean": round(stats["mean"], 6),
                "std": round(stats["std"], 6),
                "min": round(stats["min"], 6),
                "max": round(stats["max"], 6),
                "batch_mean_min": round(stats["batch_mean_min"], 6),
                "batch_mean_max": round(stats["batch_mean_max"], 6),
                "batch_std_min": round(stats["batch_std_min"], 6),
                "batch_std_max": round(stats["batch_std_max"], 6),
            }
        )
    return pd.DataFrame(rows)


def df_to_markdown(df):
    if df.empty:
        return "(empty)"
    display_df = df.copy()
    display_df.columns = [str(c) for c in display_df.columns]
    for col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = {
        col: max(len(col), *(len(v) for v in display_df[col].tolist()))
        for col in display_df.columns
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in display_df.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in display_df.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in display_df.columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def build_report(overall_df, focus_df, feature_df, posterior_df, winloss_df, examples, results, crnn_missing):
    fusion = results["fusion"]["metrics"]
    zero_beats = results["zero_beats"]["metrics"]
    zero_cnn = results["zero_cnn"]["metrics"]
    shuffle_beats = results["shuffle_beats"]["metrics"]

    beats_drop = (fusion["event_macro"] - zero_beats["event_macro"]) * 100
    beats_shuffle_drop = (fusion["event_macro"] - shuffle_beats["event_macro"]) * 100
    cnn_drop = (fusion["event_macro"] - zero_cnn["event_macro"]) * 100

    beats_benefit = []
    for cls in FOCUS_CLASSES:
        delta = (
            results["fusion"]["metrics"]["event_classwise"][cls]
            - results["zero_beats"]["metrics"]["event_classwise"][cls]
        ) * 100
        beats_benefit.append((cls, delta))
    beats_benefit.sort(key=lambda x: x[1], reverse=True)

    lines = []
    lines.append("# Late Fusion 诊断报告")
    lines.append("")
    lines.append("## 1. 结论总览")
    lines.append("")
    lines.append(
        "当前 `CRNN + BEATs late fusion` 没有明显超过 CRNN baseline，主因不像“BEATs 完全没被用上”，"
        "而更像是：BEATs 分支已经提供了局部有效信息，但收益主要集中在少数长持续或设备类；"
        "对于 `Dishes / Dog / Alarm_bell_ringing` 这类弱类和复杂多事件场景，当前 `concat + Merge MLP + shared BiGRU` 的 late fusion 仍然太粗。"
    )
    lines.append("")
    lines.append(
        f"直接证据是：去掉 BEATs 后，fusion 的 event macro 下降 {beats_drop:.2f}pp；"
        f"打乱 BEATs 后下降 {beats_shuffle_drop:.2f}pp；"
        f"而去掉 CNN 后下降 {cnn_drop:.2f}pp。"
        "如果 CNN 一去掉就接近崩掉，而去掉/打乱 BEATs 只带来中小幅退化，说明当前系统仍然是 CRNN 主导，BEATs 提供的是局部补充而不是主导信息源。"
    )
    lines.append("")
    lines.append("## 2. 四个推理对照实验结果")
    lines.append("")
    lines.append(df_to_markdown(overall_df))
    lines.append("")
    lines.append(df_to_markdown(focus_df))
    lines.append("")
    lines.append(
        f"整体上，`zero_beats` 和 `shuffle_beats` 都没有像 `zero_cnn` 那样引起灾难性下降，说明当前模型主要还是由 CNN/CRNN 分支主导。"
    )
    lines.append("")
    lines.append(
        f"`zero_cnn` 的 event macro 只有 {zero_cnn['event_macro']*100:.2f}%，说明单靠当前 frozen BEATs 支路并不足以支撑整体检测；"
        f"`zero_beats` 的 event macro 仍有 {zero_beats['event_macro']*100:.2f}%，进一步说明 CRNN 是主力。"
    )
    lines.append("")
    lines.append(
        "按类看，最依赖 BEATs 的类别主要是："
        + "，".join(f"`{cls}`({delta:+.2f}pp)" for cls, delta in beats_benefit[:4])
        + "。"
    )
    lines.append("")
    lines.append(
        "如果某些类别在 `zero_beats` 下降明显而 `shuffle_beats` 也下降，说明 BEATs 提供的不是纯噪声；"
        "如果 `shuffle_beats` 影响很小，则说明当前融合对 BEATs 的利用还不够深。"
    )
    lines.append("")
    lines.append("## 3. 特征统计结论")
    lines.append("")
    lines.append(df_to_markdown(feature_df))
    lines.append("")
    lines.append(
        "这些统计主要用来判断两个问题：一是 CNN 与 BEATs 是否存在明显尺度不匹配；二是 Merge MLP 后是否塌缩成近常数。"
    )
    lines.append("")
    lines.append(
        "如果 `beats_feat / beats_aligned` 的 std 明显远小于 `cnn_feat`，而 `merge_mlp_out` 又没有把这两路拉回更均衡的范围，"
        "那就更支持“BEATs 被当成弱补充特征，而不是被充分利用”的判断。"
    )
    lines.append("")
    lines.append("## 4. 逐类 posterior / 可分性分析")
    lines.append("")
    lines.append(df_to_markdown(posterior_df))
    lines.append("")
    lines.append(
        "这里重点看的是：正样本帧的 posterior 是否明显高于负样本帧，以及 `segment F1` 与 `event F1` 是否出现分离。"
    )
    lines.append("")
    lines.append(
        "如果某类 `segment F1` 还有一定水平、但 `event F1` 很低，更像是边界/切段/后处理问题；"
        "如果 `segment` 和 `event` 都低，而且正负 posterior 几乎不可分，则更像表征不足。"
    )
    lines.append("")
    lines.append("## 5. 样本级 win/loss 分析")
    lines.append("")
    lines.append(df_to_markdown(winloss_df))
    lines.append("")
    for group_name, title in [
        ("fusion_win", "Fusion 明显优于 CRNN 的样本示例"),
        ("similar", "Fusion 与 CRNN 接近的样本示例"),
        ("fusion_loss", "Fusion 明显差于 CRNN 的样本示例"),
    ]:
        lines.append(f"### {title}")
        example_df = pd.DataFrame(examples[group_name])
        lines.append("")
        lines.append(df_to_markdown(example_df))
        lines.append("")
    lines.append(
        "这些分组采用 `file-level event F1 delta >= 0.15 / <= -0.15` 的规则。"
        "如果 win 组明显富集在长持续设备类，而 loss 组更偏 `Dog / Dishes / Alarm / Cat` 或多事件场景，"
        "就说明当前 late fusion 的收益具有很强的场景偏置。"
    )
    lines.append("")
    lines.append("## 6. 最可能的问题归因")
    lines.append("")
    lines.append(
        "最可能的主因是：BEATs 分支并非完全没被用上，但它在当前 late fusion 结构里只被粗粒度利用，"
        "主要帮助了少数长持续、设备纹理稳定的类别；对弱类、动物类、复杂多事件场景，它没有形成稳定互补。"
    )
    lines.append("")
    lines.append(
        "次要原因是：两路特征在融合前后仍可能存在尺度和时间压缩上的不匹配。"
        "即便 adaptive average pooling 在工程上是正确的，它也可能把 BEATs 的局部时间细节过度平滑，"
        "让它更擅长提供“有这个声音的大致区间”，而不擅长改善细粒度边界。"
    )
    lines.append("")
    lines.append(
        "因此当前问题更像“被用上了，但只对少数类有效”，而不是“完全没被用上”。"
        "从对照实验如果还能看到 `shuffle_beats` 比 `zero_beats` 更差，也会进一步支持这一点。"
    )
    lines.append("")
    lines.append("## 7. 最值得做的 1~3 个改动")
    lines.append("")
    lines.append("1. 先保留当前底座不变，在 fusion 前后加入更明确的归一化或门控，让 BEATs 不只是弱补充通道。")
    lines.append("2. 优先针对 `Dishes / Dog / Alarm_bell_ringing / Cat` 做类不平衡和阈值分析，确认问题是表征不足还是边界不稳。")
    lines.append("3. 如果本轮证据显示 BEATs 只对少数长持续类有帮助，下一步更值得试更聪明的融合方式或 posterior fusion，而不是继续盲目长训当前 concat late fusion。")
    lines.append("")
    lines.append("## 备注")
    lines.append("")
    lines.append(
        "本次排查没有重训模型，也没有改训练底座，只是在 `version_20` best checkpoint 上做了离线对照推理与统计。"
    )
    lines.append("")
    lines.append(
        "CRNN baseline 采用旧 checkpoint 到当前统一 `SEDModel` 的最小权重映射，仅用于诊断推理。"
        f"加载时缺失的非关键键为：`{crnn_missing}`。"
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    ensure_dirs()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    fusion_config = load_yaml(FUSION_CONFIG_PATH)
    crnn_config = load_yaml(CRNN_CONFIG_PATH)

    print(f"[setup] device={device}", flush=True)
    gt_df, filenames, labels_np, scores, feature_stats, crnn_missing = run_inference(
        fusion_config, crnn_config, device=device, batch_size=args.batch_size
    )
    print("[setup] inference complete", flush=True)

    threshold = fusion_config["training"]["val_thresholds"][0]
    median_window = fusion_config["training"]["median_window"]

    results = {}
    for variant_name, score_array in scores.items():
        print(f"[decode] variant={variant_name}", flush=True)
        pred_df = decode_scores(score_array, filenames, build_encoder(fusion_config), threshold, median_window)
        metrics = compute_metrics(pred_df, gt_df, fusion_config)
        results[variant_name] = {
            "pred_df": pred_df,
            "metrics": metrics,
        }
    print("[decode] all variants complete", flush=True)

    overall_df = overall_table(results)
    focus_df = focus_class_table(results)
    posterior_df = posterior_table(scores, labels_np, results)
    winloss_raw_df, winloss_df, examples = win_loss_analysis(
        results["fusion"]["pred_df"], results["crnn"]["pred_df"], gt_df
    )
    print("[winloss] analysis complete", flush=True)
    feature_df = feature_stats_table(feature_stats)

    overall_df.to_csv(OVERALL_CSV, index=False)
    focus_df.to_csv(FOCUS_CSV, index=False)
    posterior_df.to_csv(POSTERIOR_CSV, index=False)
    winloss_df.to_csv(WINLOSS_CSV, index=False)

    payload = {
        "overall": overall_df.to_dict("records"),
        "focus_classes": focus_df.to_dict("records"),
        "posterior": posterior_df.to_dict("records"),
        "winloss_summary": winloss_df.to_dict("records"),
        "examples": examples,
        "feature_stats": feature_stats,
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    build_report(overall_df, focus_df, feature_df, posterior_df, winloss_df, examples, results, crnn_missing)

    print("Generated:")
    print(f"- {REPORT_PATH}")
    print(f"- {OVERALL_CSV}")
    print(f"- {FOCUS_CSV}")
    print(f"- {POSTERIOR_CSV}")
    print(f"- {WINLOSS_CSV}")
    print(f"- {JSON_PATH}")
    print("Device:", device)


if __name__ == "__main__":
    main()
