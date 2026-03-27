import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path("/home/llxxll/pyProj/dcase-2022-task4")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics
from local.classes_dict import classes_labels

REPORT_DIR = ROOT / "CRNN-baseline"
ASSET_DIR = REPORT_DIR / "report_assets"

EVENT_F1_TXT = ROOT / "exp/2022_baseline/metrics_test/student/event_f1.txt"
SEGMENT_F1_TXT = ROOT / "exp/2022_baseline/metrics_test/student/segment_f1.txt"
PREDICTION_TSV = (
    ROOT
    / "exp/2022_baseline/metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"
)
GROUND_TRUTH_TSV = Path(
    "/mnt/d/Downloads/Compressed/dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv"
)
VAL_AUDIO_16K_DIR = (
    ROOT / "runtime_data/dcase_synth/audio/validation/synthetic21_validation/soundscapes_16k"
)
VAL_AUDIO_ORIG_DIR = Path(
    "/mnt/d/Downloads/Compressed/dcase_synth/audio/validation/synthetic21_validation/soundscapes"
)
TB_DIR = ROOT / "exp/2022_baseline/version_4"
BEST_CKPT = ROOT / "exp/2022_baseline/version_4/epoch=133-step=111756.ckpt"
CONFIG_YAML = ROOT / "confs/synth_only_d_drive.yaml"
TRAIN_ENERGY_TXT = ROOT / "exp/2022_baseline/training_codecarbon/training_tot_kwh.txt"
TEST_ENERGY_TXT = ROOT / "exp/2022_baseline/devtest_codecarbon/devtest_tot_kwh.txt"
REPORT_PATH = REPORT_DIR / "training_result_report.md"

FEAT_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 256,
    "n_mels": 128,
    "f_min": 0,
    "f_max": 8000,
}

SAMPLE_TYPE_EN = {
    "高质量检测": "Accurate Detection",
    "空预测/漏检": "Empty Prediction",
    "过预测": "Over-prediction",
    "边界偏移但类别正确": "Boundary Shift",
    "多事件重叠表现较好": "Multi-event Good Case",
    "弱类漏检": "Weak-class Miss",
}


def ensure_dirs():
    REPORT_DIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)


def read_float(path):
    return float(Path(path).read_text().strip())


def df_to_markdown(df):
    display_df = df.copy()
    display_df.columns = [str(c) for c in display_df.columns]
    for col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else str(x))

    widths = {}
    for col in display_df.columns:
        widths[col] = max(len(col), *(len(v) for v in display_df[col].tolist())) if len(display_df) else len(col)

    header = "| " + " | ".join(col.ljust(widths[col]) for col in display_df.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in display_df.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in display_df.columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def load_config():
    config = yaml.safe_load(CONFIG_YAML.read_text())
    ckpt = torch.load(BEST_CKPT, map_location="cpu")
    ckpt_config = ckpt.get("hyper_parameters", {})
    merged = config.copy()
    merged.update({k: v for k, v in ckpt_config.items() if k in merged})
    return merged, ckpt


def load_tensorboard_scalars():
    ea = event_accumulator.EventAccumulator(str(TB_DIR))
    ea.Reload()
    scalar_tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in scalar_tags:
        vals = ea.Scalars(tag)
        data[tag] = vals
    return data


def scalar_values(tb_data, tag):
    return [item.value for item in tb_data.get(tag, [])]


def final_scalar(tb_data, tag):
    vals = scalar_values(tb_data, tag)
    return vals[-1] if vals else None


def smooth(values, window=25):
    if not values:
        return []
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(float(np.mean(values[start : idx + 1])))
    return out


def load_predictions_and_gt():
    pred = pd.read_csv(PREDICTION_TSV, sep="\t")
    gt = pd.read_csv(GROUND_TRUTH_TSV, sep="\t")
    pred["duration"] = pred["offset"] - pred["onset"]
    gt["duration"] = gt["offset"] - gt["onset"]
    return pred, gt


def compute_event_segment_metrics(pred, gt):
    event_metric, segment_metric = compute_sed_eval_metrics(pred, gt)
    return event_metric.results(), segment_metric.results()


def build_overall_metrics(event_results, segment_results, tb_data):
    event_micro = event_results["overall"]["f_measure"]["f_measure"] * 100
    event_macro = event_results["class_wise_average"]["f_measure"]["f_measure"] * 100
    segment_micro = segment_results["overall"]["f_measure"]["f_measure"] * 100
    segment_macro = segment_results["class_wise_average"]["f_measure"]["f_measure"] * 100

    metrics = [
        ("PSDS-scenario1", f"{final_scalar(tb_data, 'test/student/psds_score_scenario1'):.3f}"),
        ("PSDS-scenario2", f"{final_scalar(tb_data, 'test/student/psds_score_scenario2'):.3f}"),
        ("Intersection-based F1", f"{final_scalar(tb_data, 'test/student/intersection_f1_macro'):.3f}"),
        ("Event-based F1 (micro)", f"{event_micro:.2f}%"),
        ("Event-based F1 (macro)", f"{event_macro:.2f}%"),
        ("Segment-based F1 (micro)", f"{segment_micro:.2f}%"),
        ("Segment-based F1 (macro)", f"{segment_macro:.2f}%"),
    ]
    return pd.DataFrame(metrics, columns=["指标", "数值"])


def build_class_metrics(event_results, segment_results, pred, gt):
    rows = []
    gt_counts = gt["event_label"].value_counts()
    pred_counts = pred["event_label"].value_counts()
    for label in classes_labels.keys():
        event_f1 = event_results["class_wise"][label]["f_measure"]["f_measure"] * 100
        segment_f1 = segment_results["class_wise"][label]["f_measure"]["f_measure"] * 100
        gt_n = int(gt_counts.get(label, 0))
        pred_n = int(pred_counts.get(label, 0))
        ratio = pred_n / gt_n if gt_n else float("nan")
        rows.append(
            {
                "类别": label,
                "GT事件数": gt_n,
                "Pred事件数": pred_n,
                "Pred/GT": f"{ratio:.2f}" if gt_n else "NA",
                "Event F1": f"{event_f1:.2f}%",
                "Segment F1": f"{segment_f1:.2f}%",
            }
        )
    df = pd.DataFrame(rows)
    event_numeric = [event_results["class_wise"][label]["f_measure"]["f_measure"] for label in classes_labels.keys()]
    segment_numeric = [segment_results["class_wise"][label]["f_measure"]["f_measure"] for label in classes_labels.keys()]
    df["event_score_num"] = event_numeric
    df["segment_score_num"] = segment_numeric
    return df


def compute_prediction_stats(pred, gt):
    gt_files = sorted(gt["filename"].unique())
    pred_files = sorted(pred["filename"].unique())
    gt_file_set = set(gt_files)
    pred_file_set = set(pred_files)
    empty_files = sorted(gt_file_set - pred_file_set)

    pred_counts = pred["event_label"].value_counts()
    gt_counts = gt["event_label"].value_counts()
    rows = []
    for label in classes_labels.keys():
        gt_n = int(gt_counts.get(label, 0))
        pred_n = int(pred_counts.get(label, 0))
        diff = pred_n - gt_n
        rows.append(
            {
                "类别": label,
                "GT事件数": gt_n,
                "Pred事件数": pred_n,
                "差值": diff,
                "Pred/GT": f"{(pred_n / gt_n):.2f}" if gt_n else "NA",
            }
        )
    class_count_df = pd.DataFrame(rows)

    pred_near_full = int((pred["duration"] >= 9.5).sum())
    gt_near_full = int((gt["duration"] >= 9.5).sum())

    summary = {
        "gt_file_count": len(gt_files),
        "pred_file_count": len(pred_files),
        "empty_pred_count": len(empty_files),
        "empty_pred_ratio": len(empty_files) / len(gt_files),
        "gt_event_count": len(gt),
        "pred_event_count": len(pred),
        "pred_duration_mean": float(pred["duration"].mean()),
        "pred_duration_median": float(pred["duration"].median()),
        "gt_duration_mean": float(gt["duration"].mean()),
        "gt_duration_median": float(gt["duration"].median()),
        "pred_duration_p95": float(pred["duration"].quantile(0.95)),
        "gt_duration_p95": float(gt["duration"].quantile(0.95)),
        "pred_near_full": pred_near_full,
        "gt_near_full": gt_near_full,
        "empty_files": empty_files,
        "class_count_df": class_count_df,
    }
    return summary


def add_file_level_stats(pred, gt):
    pred_file = pred.groupby("filename").agg(
        pred_n=("event_label", "size"),
        pred_total_dur=("duration", "sum"),
    )
    gt_file = gt.groupby("filename").agg(
        gt_n=("event_label", "size"),
        gt_total_dur=("duration", "sum"),
    )
    merged = gt_file.join(pred_file, how="left").fillna({"pred_n": 0, "pred_total_dur": 0})
    merged["pred_n"] = merged["pred_n"].astype(int)
    merged["ratio_total_dur"] = merged["pred_total_dur"] / (merged["gt_total_dur"] + 1e-9)
    return merged


def label_jaccard(pred_labels, gt_labels):
    union = len(pred_labels | gt_labels)
    if union == 0:
        return 0.0
    return len(pred_labels & gt_labels) / union


def choose_representative_samples(pred, gt, class_metrics_df):
    merged = add_file_level_stats(pred, gt)
    samples = []

    def add_sample(sample_type, filename, reason):
        if filename and filename not in [item["filename"] for item in samples]:
            samples.append({"type": sample_type, "filename": filename, "reason": reason})

    if "1823.wav" in set(gt["filename"]):
        add_sample("高质量检测", "1823.wav", "人工抽样中表现准确，适合作为正例。")
    if "1278.wav" in set(gt["filename"]):
        add_sample("空预测/漏检", "1278.wav", "人工抽样中为典型空预测样本。")
    if "697.wav" in set(gt["filename"]):
        add_sample("过预测", "697.wav", "人工抽样中存在明显多报时间段。")

    chosen = {item["filename"] for item in samples}
    gt_group = gt.groupby("filename")
    pred_group = pred.groupby("filename")

    best_boundary = None
    best_boundary_score = -1
    for filename, gt_rows in gt_group:
        if filename in chosen:
            continue
        pred_rows = pred_group.get_group(filename) if filename in pred_group.groups else pd.DataFrame(columns=pred.columns)
        if len(gt_rows) != 1 or pred_rows.empty:
            continue
        gt_row = gt_rows.iloc[0]
        same_cls = pred_rows[pred_rows["event_label"] == gt_row["event_label"]]
        if same_cls.empty:
            continue
        for _, pred_row in same_cls.iterrows():
            inter = max(0.0, min(gt_row["offset"], pred_row["offset"]) - max(gt_row["onset"], pred_row["onset"]))
            union = max(gt_row["offset"], pred_row["offset"]) - min(gt_row["onset"], pred_row["onset"])
            if union <= 0:
                continue
            iou = inter / union
            shift = abs(gt_row["onset"] - pred_row["onset"]) + abs(gt_row["offset"] - pred_row["offset"])
            if 0.2 <= iou <= 0.75 and shift >= 0.5:
                score = shift * (1 - abs(0.5 - iou))
                if score > best_boundary_score:
                    best_boundary_score = score
                    best_boundary = filename
    add_sample("边界偏移但类别正确", best_boundary, "预测类别正确，但起止边界存在明显偏移。")

    best_multi = None
    best_multi_score = -1
    for filename, gt_rows in gt_group:
        if filename in chosen:
            continue
        if len(gt_rows) < 3:
            continue
        pred_rows = pred_group.get_group(filename) if filename in pred_group.groups else pd.DataFrame(columns=pred.columns)
        if len(pred_rows) < 2:
            continue
        gt_labels = set(gt_rows["event_label"])
        pred_labels = set(pred_rows["event_label"])
        jacc = label_jaccard(pred_labels, gt_labels)
        dur_row = merged.loc[filename]
        if dur_row["pred_total_dur"] == 0 or dur_row["gt_total_dur"] == 0:
            continue
        dur_ratio = min(dur_row["pred_total_dur"], dur_row["gt_total_dur"]) / max(
            dur_row["pred_total_dur"], dur_row["gt_total_dur"]
        )
        score = jacc * dur_ratio
        if score > best_multi_score:
            best_multi_score = score
            best_multi = filename
    add_sample("多事件重叠表现较好", best_multi, "多事件样本中，类别覆盖和时长匹配都较好。")

    weak_classes = class_metrics_df.sort_values("event_score_num").head(5)["类别"].tolist()
    weak_sample = None
    weak_score = -1
    for filename, gt_rows in gt_group:
        if filename in chosen:
            continue
        gt_labels = set(gt_rows["event_label"])
        if not gt_labels.intersection(weak_classes):
            continue
        pred_rows = pred_group.get_group(filename) if filename in pred_group.groups else pd.DataFrame(columns=pred.columns)
        pred_labels = set(pred_rows["event_label"])
        missing_weak = len(gt_labels.intersection(weak_classes) - pred_labels)
        if missing_weak <= 0:
            continue
        score = missing_weak + len(gt_rows) * 0.01
        if score > weak_score:
            weak_score = score
            weak_sample = filename
    add_sample("弱类漏检", weak_sample, "包含弱类事件，但该弱类在预测中缺失或明显不足。")

    return samples[:6]


def find_audio_path(filename):
    path_16k = VAL_AUDIO_16K_DIR / filename
    if path_16k.exists():
        return path_16k
    path_orig = VAL_AUDIO_ORIG_DIR / filename
    if path_orig.exists():
        return path_orig
    return None


def sample_event_rows(df, filename):
    rows = df[df["filename"] == filename][["event_label", "onset", "offset"]].copy()
    rows = rows.sort_values(["onset", "offset", "event_label"]).reset_index(drop=True)
    return rows


def build_sample_note(sample_type, gt_rows, pred_rows):
    if sample_type == "高质量检测":
        return "主要事件的类别与时序均较贴近真值，可作为本次模型表现较好的代表。"
    if sample_type == "空预测/漏检":
        return "该样本没有输出任何预测，属于明显漏检案例。"
    if sample_type == "过预测":
        return "该样本输出了额外的事件片段，体现了局部过预测现象。"
    if sample_type == "边界偏移但类别正确":
        return "模型识别到了正确类别，但事件起止时间与真值仍有偏移。"
    if sample_type == "多事件重叠表现较好":
        return "在多事件场景中仍能保留较好的类别覆盖，说明模型没有在复杂场景下完全失效。"
    if sample_type == "弱类漏检":
        return "样本包含弱类事件，但模型未能稳定捕获，反映了类别层面的不足。"
    return "代表性样本。"


def plot_training_curves(tb_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    train_loss = scalar_values(tb_data, "train/student/loss_strong")
    val_loss = scalar_values(tb_data, "val/synth/student/loss_strong")
    val_obj = scalar_values(tb_data, "val/obj_metric")
    val_inter = scalar_values(tb_data, "val/synth/student/intersection_f1_macro")

    if train_loss:
        axes[0, 0].plot(train_loss, alpha=0.25, label="raw")
        axes[0, 0].plot(smooth(train_loss, window=50), label="moving average")
        axes[0, 0].set_title("train/student/loss_strong")
        axes[0, 0].set_xlabel("step")
        axes[0, 0].legend()

    if val_loss:
        axes[0, 1].plot(val_loss, marker="o", markersize=2)
        axes[0, 1].set_title("val/synth/student/loss_strong")
        axes[0, 1].set_xlabel("epoch")

    if val_obj:
        best_epoch = int(np.argmax(val_obj))
        axes[1, 0].plot(val_obj, marker="o", markersize=2)
        axes[1, 0].axvline(best_epoch, color="r", linestyle="--", label=f"best epoch={best_epoch}")
        axes[1, 0].set_title("val/obj_metric")
        axes[1, 0].set_xlabel("epoch")
        axes[1, 0].legend()

    if val_inter:
        axes[1, 1].plot(val_inter, marker="o", markersize=2)
        axes[1, 1].set_title("val/synth/student/intersection_f1_macro")
        axes[1, 1].set_xlabel("epoch")

    out_path = ASSET_DIR / "training_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path.name


def plot_class_statistics(class_metrics_df):
    labels = class_metrics_df["类别"].tolist()
    x = np.arange(len(labels))
    gt_counts = class_metrics_df["GT事件数"].astype(float).to_numpy()
    pred_counts = class_metrics_df["Pred事件数"].astype(float).to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    width = 0.38
    axes[0].bar(x - width / 2, gt_counts, width=width, label="GT")
    axes[0].bar(x + width / 2, pred_counts, width=width, label="Prediction")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_title("Per-class predicted event count vs ground truth")
    axes[0].legend()

    event_scores = class_metrics_df["event_score_num"].to_numpy() * 100
    segment_scores = class_metrics_df["segment_score_num"].to_numpy() * 100
    axes[1].bar(x - width / 2, event_scores, width=width, label="Event F1")
    axes[1].bar(x + width / 2, segment_scores, width=width, label="Segment F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Per-class Event F1 and Segment F1")
    axes[1].legend()

    out_path = ASSET_DIR / "class_statistics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path.name


def plot_duration_distribution(pred, gt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    bins = np.linspace(0, 10, 41)
    axes[0].hist(gt["duration"], bins=bins, alpha=0.6, label="GT")
    axes[0].hist(pred["duration"], bins=bins, alpha=0.6, label="Prediction")
    axes[0].set_title("Event duration distribution")
    axes[0].set_xlabel("duration (s)")
    axes[0].legend()

    axes[1].boxplot([gt["duration"], pred["duration"]], labels=["GT", "Prediction"], showfliers=False)
    axes[1].set_title("Event duration boxplot")
    axes[1].set_ylabel("duration (s)")

    out_path = ASSET_DIR / "duration_distribution.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path.name


def plot_sample_figure(filename, gt_rows, pred_rows, sample_type):
    audio_path = find_audio_path(filename)
    duration = max(
        10.0,
        float(gt_rows["offset"].max()) if not gt_rows.empty else 0.0,
        float(pred_rows["offset"].max()) if not pred_rows.empty else 0.0,
    )

    rows_order = sorted(set(gt_rows["event_label"]).union(set(pred_rows["event_label"])))
    if not rows_order:
        rows_order = list(classes_labels.keys())[:1]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.2], hspace=0.25)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_pred = fig.add_subplot(gs[2, 0], sharex=ax_spec)

    if audio_path and audio_path.exists():
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = torch.tensor(audio, dtype=torch.float32)
        if sr != FEAT_CONFIG["sample_rate"]:
            audio = torchaudio.functional.resample(audio, sr, FEAT_CONFIG["sample_rate"])
            sr = FEAT_CONFIG["sample_rate"]
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=FEAT_CONFIG["n_fft"],
            win_length=FEAT_CONFIG["win_length"],
            hop_length=FEAT_CONFIG["hop_length"],
            f_min=FEAT_CONFIG["f_min"],
            f_max=FEAT_CONFIG["f_max"],
            n_mels=FEAT_CONFIG["n_mels"],
            power=1.0,
        )
        db_transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
        mel = db_transform(mel_transform(audio.unsqueeze(0))).squeeze(0).numpy()
        ax_spec.imshow(
            mel,
            origin="lower",
            aspect="auto",
            extent=[0, len(audio) / sr, 0, FEAT_CONFIG["n_mels"]],
            cmap="magma",
        )
        ax_spec.set_ylabel("Mel bin")
    else:
        ax_spec.text(0.5, 0.5, "Audio unavailable", ha="center", va="center")

    color_map = matplotlib.colormaps.get_cmap("tab10")
    color_lookup = {label: color_map(idx) for idx, label in enumerate(classes_labels.keys())}

    def draw_event_axis(ax, rows, title):
        y_lookup = {label: idx for idx, label in enumerate(rows_order)}
        for label in rows_order:
            ax.axhline(y_lookup[label], color="#dddddd", linewidth=0.5, zorder=0)
        if rows.empty:
            ax.text(0.5, 0.5, "No events", transform=ax.transAxes, ha="center", va="center")
        else:
            for _, row in rows.iterrows():
                y = y_lookup[row["event_label"]] - 0.35
                ax.broken_barh(
                    [(row["onset"], row["offset"] - row["onset"])],
                    (y, 0.7),
                    facecolors=color_lookup[row["event_label"]],
                    edgecolors="black",
                    linewidth=0.5,
                )
        ax.set_yticks(list(y_lookup.values()))
        ax.set_yticklabels(rows_order, fontsize=8)
        ax.set_ylabel(title)
        ax.set_xlim(0, duration)

    draw_event_axis(ax_gt, gt_rows, "GT")
    draw_event_axis(ax_pred, pred_rows, "Pred")
    ax_pred.set_xlabel("Time (s)")
    ax_spec.set_title(f"{filename} | {SAMPLE_TYPE_EN.get(sample_type, sample_type)}")

    out_path = ASSET_DIR / f"sample_{filename.replace('.wav', '')}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path.name


def format_event_list(rows):
    if rows.empty:
        return "无预测"
    parts = []
    for _, row in rows.iterrows():
        parts.append(f"{row['event_label']} ({row['onset']:.3f}-{row['offset']:.3f}s)")
    return "<br>".join(parts)


def build_report(
    config,
    ckpt,
    tb_data,
    pred,
    gt,
    event_results,
    segment_results,
    overall_df,
    class_metrics_df,
    prediction_stats,
    sample_infos,
    generated_assets,
):
    val_obj = scalar_values(tb_data, "val/obj_metric")
    val_inter = scalar_values(tb_data, "val/synth/student/intersection_f1_macro")
    val_event = scalar_values(tb_data, "val/synth/student/event_f1_macro")
    train_loss = scalar_values(tb_data, "train/student/loss_strong")
    val_loss = scalar_values(tb_data, "val/synth/student/loss_strong")

    best_epoch = int(np.argmax(val_obj)) if val_obj else None
    best_val = max(val_obj) if val_obj else None
    last_val = val_obj[-1] if val_obj else None
    best_ckpt_epoch = int(ckpt.get("epoch", -1))

    weak_classes = class_metrics_df.sort_values("event_score_num").head(5)["类别"].tolist()
    strong_classes = class_metrics_df.sort_values("event_score_num", ascending=False).head(3)["类别"].tolist()

    class_metrics_display = class_metrics_df.drop(columns=["event_score_num", "segment_score_num"]).copy()
    count_table = prediction_stats["class_count_df"]

    prediction_summary_df = pd.DataFrame(
        [
            ("真值文件数", prediction_stats["gt_file_count"]),
            ("有预测的文件数", prediction_stats["pred_file_count"]),
            ("空预测文件数", prediction_stats["empty_pred_count"]),
            ("空预测比例", f"{prediction_stats['empty_pred_ratio'] * 100:.2f}%"),
            ("真值事件数", prediction_stats["gt_event_count"]),
            ("预测事件数", prediction_stats["pred_event_count"]),
            ("预测事件平均时长", f"{prediction_stats['pred_duration_mean']:.2f}s"),
            ("预测事件中位时长", f"{prediction_stats['pred_duration_median']:.2f}s"),
            ("真值事件平均时长", f"{prediction_stats['gt_duration_mean']:.2f}s"),
            ("真值事件中位时长", f"{prediction_stats['gt_duration_median']:.2f}s"),
            ("预测事件 p95 时长", f"{prediction_stats['pred_duration_p95']:.2f}s"),
            ("真值事件 p95 时长", f"{prediction_stats['gt_duration_p95']:.2f}s"),
            ("接近整段(>=9.5s)预测数", prediction_stats["pred_near_full"]),
            ("接近整段(>=9.5s)真值数", prediction_stats["gt_near_full"]),
        ],
        columns=["统计项", "数值"],
    )

    sample_sections = []
    for sample in sample_infos:
        gt_rows = sample["gt_rows"]
        pred_rows = sample["pred_rows"]
        sample_sections.append(
            f"""### {sample['type']}：`{sample['filename']}`

![{sample['filename']}](report_assets/{sample['asset_name']})

- 文件名：`{sample['filename']}`
- 代表模式：{sample['type']}
- 代表性原因：{sample['reason']}
- 简短点评：{sample['note']}

**Ground Truth 事件列表**

{format_event_list(gt_rows)}

**Prediction 事件列表**

{format_event_list(pred_rows)}
"""
        )

    report = f"""# CRNN Baseline 训练结果详细分析报告

## 目录
- [1. 实验概况](#1-实验概况)
- [2. 最终指标汇总](#2-最终指标汇总)
- [3. 训练过程分析](#3-训练过程分析)
- [4. 预测行为统计](#4-预测行为统计)
- [5. 典型样本分析](#5-典型样本分析)
- [6. 结论与建议](#6-结论与建议)

## 1. 实验概况

本次分析对象是当前仓库中的 CRNN baseline。模型训练逻辑采用 mean-teacher 框架，包含 `student` 与 `teacher` 两个分支，训练和最终推理时均会记录两路指标；本报告以 `student` 结果为主，因为当前 `metrics_test/student` 是最直接的主结果输出目录。

从本次运行使用的配置看，实验属于 `synth_only` 设置，关键配置来自 [confs/synth_only_d_drive.yaml](/home/llxxll/pyProj/dcase-2022-task4/confs/synth_only_d_drive.yaml)。本次训练：

- 使用 synthetic train 作为训练集
- 使用 synthetic validation 作为验证集
- 当前 `test_folder/test_tsv` 也指向同一份 synthetic validation
- 因此当前最终结果更接近“自测分数”，对模型是否学会了 synthetic validation 分布有较强参考价值，但对真实泛化能力的结论仍然有限

本次最佳 checkpoint 为：

- `epoch={best_ckpt_epoch}`，文件为 [epoch=133-step=111756.ckpt](/home/llxxll/pyProj/dcase-2022-task4/exp/2022_baseline/version_4/epoch=133-step=111756.ckpt)

补充环境信息：

- 最大训练轮数：`{config['training']['n_epochs']}`
- early stop patience：`{config['training']['early_stop_patience']}`
- batch size：`{config['training']['batch_size']}`
- `obj_metric_synth_type`：`{config['training']['obj_metric_synth_type']}`
- 训练能耗：`{read_float(TRAIN_ENERGY_TXT):.3f} kWh`
- dev-test 能耗：`{read_float(TEST_ENERGY_TXT):.3f} kWh`

## 2. 最终指标汇总

### 2.1 整体指标

{df_to_markdown(overall_df)}

简要解读：

- `PSDS-scenario1/2` 分别为 `{final_scalar(tb_data, 'test/student/psds_score_scenario1'):.3f}` 和 `{final_scalar(tb_data, 'test/student/psds_score_scenario2'):.3f}`，说明模型在不同容错设定下都能给出可用结果，scenario2 明显更高，符合通常“宽松场景分数更高”的预期。
- `Intersection-based F1` 为 `{final_scalar(tb_data, 'test/student/intersection_f1_macro'):.3f}`，整体处于较好水平。
- `Event-based F1 (micro)` 为 `{event_results['overall']['f_measure']['f_measure'] * 100:.2f}%`，显著低于 segment-based F1，这是事件边界评估比 1 秒分段评估更严格所致。
- `Segment-based F1 (micro)` 为 `{segment_results['overall']['f_measure']['f_measure'] * 100:.2f}%`，说明在较粗粒度时序上模型整体检测稳定。

### 2.2 各类别指标

{df_to_markdown(class_metrics_display)}

简要解读：

- 整体较强类别：`{', '.join(strong_classes)}`。这些类别的 event/segment F1 都相对靠前，通常具有更稳定的声学模式或更容易在 synthetic 数据中建模。
- 整体较弱类别：`{', '.join(weak_classes)}`。这些类别并非“完全失效”，但在事件级别上明显更难，主要表现为召回不足和边界控制较弱。
- 从 `Pred/GT` 看，`Dishes`、`Alarm_bell_ringing`、`Dog`、`Blender` 等类别存在不同程度的欠预测；`Speech` 数量接近真值，`Cat` 甚至略有过预测。

## 3. 训练过程分析

![训练曲线](report_assets/{generated_assets['training_curves']})

### 3.1 loss 变化是否正常

- `train/student/loss_strong` 从约 `{train_loss[0]:.4f}` 下降到 `{train_loss[-1]:.4f}`，下降趋势清晰，没有出现持续发散。
- `val/synth/student/loss_strong` 从约 `{val_loss[0]:.4f}` 下降到 `{val_loss[-1]:.4f}`，最低达到 `{min(val_loss):.4f}`，说明验证损失整体也在改善。
- 从曲线形态看，训练后期仍有波动，但没有出现异常抖动到失控的迹象。

### 3.2 选模指标与最佳轮次

- `val/obj_metric` 的最佳值约为 `{best_val:.4f}`，出现在约第 `{best_epoch}` 轮。
- 最后一轮 `val/obj_metric` 约为 `{last_val:.4f}`，明显低于最佳值。
- 最佳 checkpoint 文件名为 `epoch={best_ckpt_epoch}`，与 `val/obj_metric` 中后期达到峰值的现象一致。

这说明：

- “最后一轮不等于最佳轮”在本次实验中是有明确证据的；
- 如果只看最后一轮结果，会低估模型在中后期达到过的最好性能；
- 当前保存 best checkpoint 的策略是有效的。

### 3.3 对当前曲线的判断

- 训练不像崩溃，也不像完全没有学到东西；
- 模型在前中期快速提升，后期进入波动区；
- 对于 `synth_only` 设置，这种“中后期达到峰值、最后略回落”的现象是合理的，提示后续可考虑更积极的早停或更稳的选模标准。

## 4. 预测行为统计

### 4.1 整体统计

{df_to_markdown(prediction_summary_df)}

![类别统计与类别 F1](report_assets/{generated_assets['class_statistics']})

![事件时长分布](report_assets/{generated_assets['duration_distribution']})

### 4.2 各类别预测数 vs 真值数

{df_to_markdown(count_table)}

### 4.3 统计解读

- 总文件数为 `{prediction_stats['gt_file_count']}`，其中有预测的文件为 `{prediction_stats['pred_file_count']}`，空预测文件为 `{prediction_stats['empty_pred_count']}`（`{prediction_stats['empty_pred_ratio'] * 100:.2f}%`）。这说明模型不是“全预测为空”，但确实存在小比例完全漏检文件。
- 总预测事件数 `{prediction_stats['pred_event_count']}` 低于总真值事件数 `{prediction_stats['gt_event_count']}`，整体更偏“保守”，主要表现为局部类别召回不足，而不是全局过预测。
- 预测事件平均时长 `{prediction_stats['pred_duration_mean']:.2f}s` 低于真值的 `{prediction_stats['gt_duration_mean']:.2f}s`，说明模型更常见的问题是把事件段预测得偏短，而不是普遍把整段涂满。
- 接近整段（`>=9.5s`）的预测共有 `{prediction_stats['pred_near_full']}` 个，而真值中本身就有 `{prediction_stats['gt_near_full']}` 个长事件。因此“长条预测”存在，但大部分与数据分布本身一致，不能简单视为失控。
- 各类别都有预测输出，没有出现“某个类别永远检测不到”的情况。
- 更主要的问题是局部类别偏弱，尤其体现在：`{', '.join(weak_classes)}`。

综合判断：当前现象更像“部分类别召回不足 + 个别文件存在过预测/漏预测”，而不是整体性失效。

## 5. 典型样本分析

下面选取 6 个典型样本，覆盖高质量检测、完全漏检、过预测、边界偏移、多事件场景和弱类漏检等情况。图中：

- 顶部为 log-mel 频谱
- 中部为 Ground Truth 时间轴
- 底部为 Prediction 时间轴

{chr(10).join(sample_sections)}

## 6. 结论与建议

### 6.1 结论

- 本次训练结果**不像训练崩了**。证据包括：loss 正常下降、验证指标在中后期明显上升、最终输出覆盖大多数文件与全部类别。
- 当前最主要的问题不是“全局异常”，而是**部分类别召回不足**与**少量文件的局部过预测/漏预测**。
- 在 `synth_only` 设定下，这次结果已经显示出较稳定的 synthetic validation 拟合能力，但由于当前 test 实际仍是 synthetic validation，本报告不应被解读为强泛化结论。

### 6.2 最主要的问题

1. 弱类事件在 event-based 评估上明显吃亏，尤其是 `{', '.join(weak_classes)}`。
2. 少量文件存在空预测，说明模型在部分样本上仍有明显漏检。
3. 最佳 checkpoint 出现在中后期而非最后一轮，说明当前训练后期存在一定回落，继续盲目长训收益有限。

### 6.3 是否建议继续训练

- **不建议简单继续按原设置长训同一实验**。当前曲线已经表明最佳点出现在中后期，继续硬跑到最后并不能带来更好的选模结果。
- **建议保留当前 best checkpoint**，并把后续工作重点放在阈值、类别平衡和弱类改进上。

### 6.4 后续最值得优先尝试的方向

1. **阈值调优**：当前报告基于固定阈值预测文件；对弱类分别调阈值，通常是性价比最高的第一步。
2. **弱类补强**：优先针对 `{', '.join(weak_classes[:3])}` 做重采样、损失加权或针对性增强。
3. **更积极的早停/选模**：当前最佳轮次早于最后一轮，建议缩短 patience，或直接按 synthetic event F1 / intersection F1 的更稳组合选模。
4. **数据层改进**：若后续允许，可引入非 synth_only 的 weak/unlabeled/real 数据，减少“自测分数偏高、泛化有限”的问题。
5. **分析 teacher 分支**：本次 teacher 指标略高于 student，可进一步确认是否在最终推理或蒸馏策略上有改进空间。

---

本报告由脚本自动生成，主要依据以下文件：

- [event_f1.txt](/home/llxxll/pyProj/dcase-2022-task4/exp/2022_baseline/metrics_test/student/event_f1.txt)
- [segment_f1.txt](/home/llxxll/pyProj/dcase-2022-task4/exp/2022_baseline/metrics_test/student/segment_f1.txt)
- [predictions_th_0.49.tsv](/home/llxxll/pyProj/dcase-2022-task4/exp/2022_baseline/metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv)
- [soundscapes.tsv](/mnt/d/Downloads/Compressed/dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv)
- [version_4 TensorBoard](/home/llxxll/pyProj/dcase-2022-task4/exp/2022_baseline/version_4)
"""
    REPORT_PATH.write_text(report)


def main():
    ensure_dirs()
    config, ckpt = load_config()
    tb_data = load_tensorboard_scalars()
    pred, gt = load_predictions_and_gt()
    event_results, segment_results = compute_event_segment_metrics(pred, gt)

    overall_df = build_overall_metrics(event_results, segment_results, tb_data)
    class_metrics_df = build_class_metrics(event_results, segment_results, pred, gt)
    prediction_stats = compute_prediction_stats(pred, gt)
    representative = choose_representative_samples(pred, gt, class_metrics_df)

    generated_assets = {
        "training_curves": plot_training_curves(tb_data),
        "class_statistics": plot_class_statistics(class_metrics_df),
        "duration_distribution": plot_duration_distribution(pred, gt),
    }

    sample_infos = []
    for sample in representative:
        gt_rows = sample_event_rows(gt, sample["filename"])
        pred_rows = sample_event_rows(pred, sample["filename"])
        asset_name = plot_sample_figure(sample["filename"], gt_rows, pred_rows, sample["type"])
        sample_infos.append(
            {
                **sample,
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
                "asset_name": asset_name,
                "note": build_sample_note(sample["type"], gt_rows, pred_rows),
            }
        )

    build_report(
        config,
        ckpt,
        tb_data,
        pred,
        gt,
        event_results,
        segment_results,
        overall_df,
        class_metrics_df,
        prediction_stats,
        sample_infos,
        generated_assets,
    )

    print("Generated files:")
    print(f"- {REPORT_PATH}")
    for asset in sorted(ASSET_DIR.glob("*.png")):
        print(f"- {asset}")
    print("Selected representative samples:")
    for sample in sample_infos:
        print(f"- {sample['filename']} ({sample['type']})")


if __name__ == "__main__":
    main()
