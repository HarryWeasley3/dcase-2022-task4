import math
import os
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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/home/llxxll/pyProj/dcase-2022-task4")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local.classes_dict import classes_labels

REPORT_DIR = ROOT / "BEATs-crnn-fusion-baseline"
ASSET_DIR = REPORT_DIR / "report_assets"
REPORT_PATH = REPORT_DIR / "training_result_report.md"

TRAIN_TB = (
    ROOT / "exp/2022_baseline/version_20/events.out.tfevents.1774712307.HarryWeasley.2344.0"
)
TEST_TB = (
    ROOT / "exp/2022_baseline/version_20/events.out.tfevents.1774750409.HarryWeasley.2344.1"
)
BEST_CKPT = ROOT / "exp/2022_baseline/version_20/epoch=34-step=21875.ckpt"
LAST_CKPT = ROOT / "exp/2022_baseline/version_20/last.ckpt"
CONFIG_PATH = ROOT / "confs/crnn_beats_late_fusion_synth_only.yaml"
EVENT_F1_TXT = ROOT / "exp/2022_baseline/metrics_test/student/event_f1.txt"
SEGMENT_F1_TXT = ROOT / "exp/2022_baseline/metrics_test/student/segment_f1.txt"
GROUND_TRUTH_TSV = Path(
    "/mnt/d/Downloads/Compressed/dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv"
)
VAL_AUDIO_16K_DIR = (
    ROOT / "runtime_data/dcase_synth/audio/validation/synthetic21_validation/soundscapes_16k"
)
VAL_AUDIO_ORIG_DIR = Path(
    "/mnt/d/Downloads/Compressed/dcase_synth/audio/validation/synthetic21_validation/soundscapes"
)
CRNN_REPORT_PATH = ROOT / "CRNN-baseline/training_result_report.md"
BEATS_REPORT_PATH = ROOT / "BEATs-baseline/training_result_report.md"

FEAT_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 256,
    "n_mels": 128,
    "f_min": 0,
    "f_max": 8000,
}

REFERENCE_OVERVIEW = [
    {
        "模型": "CRNN baseline",
        "PSDS1": 0.356,
        "PSDS2": 0.578,
        "Intersection F1": 0.650,
        "Event F1 macro": 43.42,
        "Segment F1 macro": 71.25,
    },
    {
        "模型": "Frozen BEATs baseline",
        "PSDS1": 0.001,
        "PSDS2": 0.051,
        "Intersection F1": 0.432,
        "Event F1 macro": 8.58,
        "Segment F1 macro": 45.74,
    },
]

CRNN_PER_CLASS_EVENT = {
    "Alarm_bell_ringing": 21.07,
    "Blender": 43.10,
    "Cat": 29.86,
    "Dishes": 28.57,
    "Dog": 32.69,
    "Electric_shaver_toothbrush": 48.35,
    "Frying": 65.94,
    "Running_water": 49.47,
    "Speech": 46.86,
    "Vacuum_cleaner": 68.27,
}

CRNN_PER_CLASS_SEGMENT = {
    "Alarm_bell_ringing": 64.04,
    "Blender": 63.83,
    "Cat": 73.48,
    "Dishes": 50.55,
    "Dog": 59.67,
    "Electric_shaver_toothbrush": 84.23,
    "Frying": 83.89,
    "Running_water": 71.40,
    "Speech": 80.20,
    "Vacuum_cleaner": 81.20,
}

SAMPLE_SPECS = [
    {
        "filename": "355.wav",
        "type": "长持续类检测较好",
        "plot_type": "Good long-duration detection",
        "reason": "Frying 长持续事件几乎完整命中，适合展示融合模型已经恢复到正常工作状态。",
        "commentary": "这是当前 fusion 模型表现最稳的一类：长持续、纹理稳定、边界基本连续。",
    },
    {
        "filename": "1088.wav",
        "type": "弱类部分恢复但仍漏检",
        "plot_type": "Partial recovery on weak class",
        "reason": "Cat + Speech 双事件中，Speech 已经恢复，但 Cat 仍漏检，能体现“恢复正常但弱类仍吃亏”。",
        "commentary": "相较单独 frozen BEATs 的空预测，这里至少把 Speech 报出来了，但 Cat 仍然没能恢复，说明融合帮助有限且偏向主类。",
    },
    {
        "filename": "234.wav",
        "type": "长持续类碎片化与边界偏移",
        "plot_type": "Fragmentation and boundary shift",
        "reason": "Vacuum_cleaner 被切成多段，同时 Speech 只覆盖中间部分，适合体现边界与后处理问题。",
        "commentary": "融合没有完全塌掉，但对长持续类的完整覆盖仍不够稳定，表现为切段和漏掉前后边界。",
    },
    {
        "filename": "1312.wav",
        "type": "多事件场景明显欠检",
        "plot_type": "Multi-event under-detection",
        "reason": "Dishes + Speech 的复杂场景只留下少量 Speech 片段，能展示 fusion 在复杂场景里并没有明显胜过 CRNN。",
        "commentary": "这类样本最能说明当前 late fusion 还没有把多事件建模能力真正拉起来，尤其 Dishes 仍然弱。",
    },
    {
        "filename": "1195.wav",
        "type": "Dog 弱类完全漏检",
        "plot_type": "Weak-class miss: Dog",
        "reason": "Dog 长事件完全无预测，是当前弱类召回不足的直接证据。",
        "commentary": "Dog 仍接近 CRNN 明显偏弱的那一侧，说明融合没有实质解决动物类表征问题。",
    },
    {
        "filename": "1278.wav",
        "type": "设备类混淆与长段偏置",
        "plot_type": "Long-duration confusion",
        "reason": "Blender + Speech 被预测成接近整段的 Electric_shaver_toothbrush，体现长持续设备类的语义混淆。",
        "commentary": "这类错误说明 BEATs 分支带来了一些粗粒度语义覆盖，但没有稳定转化成精确的类别边界判别。",
    },
]


def ensure_dirs():
    REPORT_DIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False


def safe_pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.00%"
    return f"{value * 100:.2f}%"


def safe_num(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{0:.{digits}f}"
    return f"{value:.{digits}f}"


def df_to_markdown(df):
    display_df = df.copy()
    display_df.columns = [str(c) for c in display_df.columns]
    for col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else str(x))

    widths = {}
    for col in display_df.columns:
        widths[col] = (
            max(len(col), *(len(v) for v in display_df[col].tolist()))
            if len(display_df)
            else len(col)
        )

    header = "| " + " | ".join(col.ljust(widths[col]) for col in display_df.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in display_df.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in display_df.columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def load_tb(path):
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def scalar_values(ea, tag):
    return [item.value for item in ea.Scalars(tag)] if tag in ea.Tags().get("scalars", []) else []


def locate_prediction_tsv():
    candidates = sorted(
        (
            ROOT / "exp/2022_baseline/metrics_test/student/scenario1"
        ).glob("predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not locate version_20 prediction TSV for threshold 0.49.")
    return candidates[0]


def load_main_artifacts():
    config = load_yaml(CONFIG_PATH)
    ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
    train_tb = load_tb(TRAIN_TB)
    test_tb = load_tb(TEST_TB)
    prediction_tsv = locate_prediction_tsv()
    pred = pd.read_csv(prediction_tsv, sep="\t")
    gt = pd.read_csv(GROUND_TRUTH_TSV, sep="\t")
    pred["duration"] = pred["offset"] - pred["onset"]
    gt["duration"] = gt["offset"] - gt["onset"]
    return config, ckpt, train_tb, test_tb, pred, gt, prediction_tsv


def _resolve_class_label(raw_label):
    clean = raw_label.replace(".", "").replace("_", "").lower()
    for label in classes_labels.keys():
        label_clean = label.replace("_", "").lower()
        if label_clean.startswith(clean) or clean.startswith(label_clean[: max(3, len(clean))]):
            return label
    raise KeyError(f"Could not resolve class label from '{raw_label}'")


def parse_metric_txt(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    overall = {}
    class_avg = {}
    class_wise = {}
    section = None
    in_class_wise = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Overall metrics"):
            section = "overall"
            continue
        if stripped.startswith("Class-wise average metrics"):
            section = "class_avg"
            continue
        if stripped.startswith("Class-wise metrics"):
            in_class_wise = True
            continue

        if not stripped:
            continue

        if section in {"overall", "class_avg"}:
            if "F-measure (F1)" in stripped:
                value = float(stripped.split(":")[-1].strip().rstrip(" %")) / 100.0
                if section == "overall":
                    overall["f1"] = value
                else:
                    class_avg["f1"] = value
            elif stripped.startswith("Precision"):
                value = float(stripped.split(":")[-1].strip().rstrip(" %")) / 100.0
                if section == "overall":
                    overall["precision"] = value
                else:
                    class_avg["precision"] = value
            elif stripped.startswith("Recall"):
                value = float(stripped.split(":")[-1].strip().rstrip(" %")) / 100.0
                if section == "overall":
                    overall["recall"] = value
                else:
                    class_avg["recall"] = value

        if in_class_wise and "|" in stripped and "%" in stripped and not stripped.startswith(("Event label", "------------")):
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) < 5:
                continue
            raw_label = parts[0]
            counts = parts[1].split()
            scores = parts[2].split()
            if len(counts) < 2 or len(scores) < 3:
                continue
            label = _resolve_class_label(raw_label)
            class_wise[label] = {
                "Nref": int(counts[0]),
                "Nsys": int(counts[1]),
                "f1": float(scores[0].rstrip("%")) / 100.0,
                "precision": float(scores[1].rstrip("%")) / 100.0,
                "recall": float(scores[2].rstrip("%")) / 100.0,
            }

    return {"overall": overall, "class_avg": class_avg, "class_wise": class_wise}


def overall_metrics_table(event_metrics, segment_metrics, test_tb):
    rows = [
        ("PSDS-scenario1", safe_num(scalar_values(test_tb, "test/student/psds_score_scenario1")[-1])),
        ("PSDS-scenario2", safe_num(scalar_values(test_tb, "test/student/psds_score_scenario2")[-1])),
        ("Intersection-based F1", safe_num(scalar_values(test_tb, "test/student/intersection_f1_macro")[-1])),
        ("Event-based F1 (macro)", safe_pct(event_metrics["class_avg"]["f1"])),
        ("Event-based F1 (micro)", safe_pct(event_metrics["overall"]["f1"])),
        ("Segment-based F1 (macro)", safe_pct(segment_metrics["class_avg"]["f1"])),
        ("Segment-based F1 (micro)", safe_pct(segment_metrics["overall"]["f1"])),
    ]
    return pd.DataFrame(rows, columns=["指标", "数值"])


def classify_status(event_score):
    if event_score >= 0.55:
        return "较强"
    if event_score >= 0.40:
        return "中等"
    return "较弱"


def per_class_table(event_metrics, segment_metrics, pred, gt):
    gt_counts = gt["event_label"].value_counts()
    pred_counts = pred["event_label"].value_counts()
    rows = []
    for label in classes_labels.keys():
        event_score = float(event_metrics["class_wise"][label]["f1"])
        segment_score = float(segment_metrics["class_wise"][label]["f1"])
        gt_n = int(gt_counts.get(label, 0))
        pred_n = int(pred_counts.get(label, 0))
        rows.append(
            {
                "类别": label,
                "GT事件数": gt_n,
                "Pred事件数": pred_n,
                "Pred/GT": f"{(pred_n / gt_n):.2f}" if gt_n else "NA",
                "Event F1": safe_pct(event_score),
                "Segment F1": safe_pct(segment_score),
                "分组": classify_status(event_score),
                "_event": event_score,
                "_segment": segment_score,
            }
        )
    return pd.DataFrame(rows)


def compute_behavior_stats(pred, gt):
    gt_files = sorted(gt["filename"].unique())
    pred_files = sorted(pred["filename"].unique())
    empty_files = sorted(set(gt_files) - set(pred_files))

    class_count_df = pd.DataFrame(
        {
            "gt_count": gt["event_label"].value_counts().sort_index(),
            "pred_count": pred["event_label"].value_counts().sort_index(),
        }
    ).fillna(0).astype(int)
    class_count_df["pred_minus_gt"] = class_count_df["pred_count"] - class_count_df["gt_count"]

    pred_long = pred[pred["duration"] >= 9.0]
    long_bias_df = (
        pred.groupby("event_label")["duration"]
        .mean()
        .rename("pred_mean_duration")
        .to_frame()
        .join(
            pred_long["event_label"].value_counts().rename("pred_ge_9s").to_frame(),
            how="left",
        )
        .fillna(0)
        .sort_values(["pred_ge_9s", "pred_mean_duration"], ascending=False)
        .reset_index()
        .rename(columns={"index": "类别", "event_label": "类别"})
    )
    long_bias_df["pred_mean_duration"] = long_bias_df["pred_mean_duration"].map(lambda x: f"{x:.2f}s")
    long_bias_df["pred_ge_9s"] = long_bias_df["pred_ge_9s"].astype(int)

    file_stats = (
        gt.groupby("filename")
        .size()
        .rename("gt_n")
        .to_frame()
        .join(pred.groupby("filename").size().rename("pred_n"), how="left")
        .fillna(0)
        .astype(int)
    )
    file_stats["delta"] = file_stats["pred_n"] - file_stats["gt_n"]
    fragmented_files = int(((file_stats["pred_n"] >= 3) & (file_stats["delta"] >= 2)).sum())

    summary_rows = [
        ("总文件数", len(gt_files)),
        ("有预测文件数", len(pred_files)),
        ("空预测文件数", len(empty_files)),
        ("空预测比例", f"{len(empty_files) / len(gt_files) * 100:.2f}%"),
        ("总真值事件数", len(gt)),
        ("总预测事件数", len(pred)),
        ("真值平均事件时长", f"{gt['duration'].mean():.2f}s"),
        ("预测平均事件时长", f"{pred['duration'].mean():.2f}s"),
        ("预测中 >=8s 长段数", int((pred["duration"] >= 8.0).sum())),
        ("预测中 >=9s 长段数", int((pred["duration"] >= 9.0).sum())),
        ("疑似碎片化过预测文件数", fragmented_files),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["统计项", "数值"])

    return {
        "summary_df": summary_df,
        "class_count_df": class_count_df.reset_index().rename(columns={"index": "类别", "event_label": "类别"}),
        "long_bias_df": long_bias_df,
        "empty_files": empty_files,
        "file_stats": file_stats,
        "fragmented_files": fragmented_files,
    }


def comparison_overview_table(overall_df):
    fusion_row = {
        "模型": "CRNN + BEATs late fusion",
        "PSDS1": float(overall_df.iloc[0, 1]),
        "PSDS2": float(overall_df.iloc[1, 1]),
        "Intersection F1": float(overall_df.iloc[2, 1]),
        "Event F1 macro": float(overall_df.iloc[3, 1].rstrip("%")),
        "Segment F1 macro": float(overall_df.iloc[5, 1].rstrip("%")),
    }
    rows = REFERENCE_OVERVIEW + [fusion_row]
    df = pd.DataFrame(rows)
    df["PSDS1"] = df["PSDS1"].map(lambda x: f"{x:.3f}")
    df["PSDS2"] = df["PSDS2"].map(lambda x: f"{x:.3f}")
    df["Intersection F1"] = df["Intersection F1"].map(lambda x: f"{x:.3f}")
    df["Event F1 macro"] = df["Event F1 macro"].map(lambda x: f"{x:.2f}%")
    df["Segment F1 macro"] = df["Segment F1 macro"].map(lambda x: f"{x:.2f}%")
    return df


def fusion_vs_crnn_per_class_table(per_class_df):
    rows = []
    for _, row in per_class_df.iterrows():
        label = row["类别"]
        fusion_event = row["_event"] * 100
        fusion_segment = row["_segment"] * 100
        crnn_event = CRNN_PER_CLASS_EVENT[label]
        crnn_segment = CRNN_PER_CLASS_SEGMENT[label]
        rows.append(
            {
                "类别": label,
                "Fusion Event F1": f"{fusion_event:.2f}%",
                "CRNN Event F1": f"{crnn_event:.2f}%",
                "差值(Event)": f"{fusion_event - crnn_event:+.2f}pp",
                "Fusion Segment F1": f"{fusion_segment:.2f}%",
                "CRNN Segment F1": f"{crnn_segment:.2f}%",
                "差值(Segment)": f"{fusion_segment - crnn_segment:+.2f}pp",
            }
        )
    return pd.DataFrame(rows)


def plot_training_curves(train_tb):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    train_loss = scalar_values(train_tb, "train/student/loss_strong")
    val_loss = scalar_values(train_tb, "val/synth/student/loss_strong")
    val_obj = scalar_values(train_tb, "val/obj_metric")
    val_inter = scalar_values(train_tb, "val/synth/student/intersection_f1_macro")
    val_event = scalar_values(train_tb, "val/synth/student/event_f1_macro")

    if train_loss:
        axes[0, 0].plot(train_loss, alpha=0.25, color="#999999")
        window = 20
        smoothed = [np.mean(train_loss[max(0, i - window + 1) : i + 1]) for i in range(len(train_loss))]
        axes[0, 0].plot(smoothed, color="#cc3311")
        axes[0, 0].set_title("train/student/loss_strong")
        axes[0, 0].set_xlabel("step")

    if val_loss:
        best_idx = int(np.argmin(val_loss))
        axes[0, 1].plot(val_loss, marker="o", color="#0077bb")
        axes[0, 1].axvline(best_idx, linestyle="--", color="#cc3311", label=f"best idx={best_idx}")
        axes[0, 1].legend()
        axes[0, 1].set_title("val/synth/student/loss_strong")
        axes[0, 1].set_xlabel("validation epoch")

    if val_obj:
        best_idx = int(np.argmax(val_obj))
        axes[0, 2].plot(val_obj, marker="o", color="#009988")
        axes[0, 2].axvline(best_idx, linestyle="--", color="#cc3311", label=f"best idx={best_idx}")
        axes[0, 2].legend()
        axes[0, 2].set_title("val/obj_metric")
        axes[0, 2].set_xlabel("validation epoch")

    if val_inter:
        axes[1, 0].plot(val_inter, marker="o", color="#ee7733")
        axes[1, 0].set_title("val/synth/student/intersection_f1_macro")
        axes[1, 0].set_xlabel("validation epoch")

    if val_event:
        axes[1, 1].plot(val_event, marker="o", color="#332288")
        axes[1, 1].set_title("val/synth/student/event_f1_macro")
        axes[1, 1].set_xlabel("validation epoch")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        "\n".join(
            [
                f"best obj idx: {int(np.argmax(val_obj)) if val_obj else 'NA'}",
                f"best obj: {max(val_obj):.4f}" if val_obj else "best obj: NA",
                f"best event idx: {int(np.argmax(val_event)) if val_event else 'NA'}",
                f"best event: {max(val_event):.4f}" if val_event else "best event: NA",
                f"best val loss idx: {int(np.argmin(val_loss)) if val_loss else 'NA'}",
                f"best val loss: {min(val_loss):.4f}" if val_loss else "best val loss: NA",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
    )

    out_path = ASSET_DIR / "training_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_counts(class_count_df):
    labels = class_count_df["类别"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - width / 2, class_count_df["gt_count"], width=width, label="GT")
    ax.bar(x + width / 2, class_count_df["pred_count"], width=width, label="Prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("event count")
    ax.set_title("GT vs predicted event counts by class")
    ax.legend()
    out_path = ASSET_DIR / "class_count_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_f1(per_class_df):
    labels = per_class_df["类别"].tolist()
    event_scores = per_class_df["_event"].to_numpy() * 100
    segment_scores = per_class_df["_segment"].to_numpy() * 100
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - width / 2, event_scores, width=width, label="Event F1")
    ax.bar(x + width / 2, segment_scores, width=width, label="Segment F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1 (%)")
    ax.set_title("Fusion event / segment F1 by class")
    ax.legend()
    out_path = ASSET_DIR / "class_f1_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fusion_vs_crnn(per_class_df):
    labels = per_class_df["类别"].tolist()
    fusion_scores = per_class_df["_event"].to_numpy() * 100
    crnn_scores = np.array([CRNN_PER_CLASS_EVENT[label] for label in labels])
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - width / 2, crnn_scores, width=width, label="CRNN Event F1")
    ax.bar(x + width / 2, fusion_scores, width=width, label="Fusion Event F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Event F1 (%)")
    ax.set_title("Fusion vs CRNN per-class event F1")
    ax.legend()
    out_path = ASSET_DIR / "fusion_vs_crnn_event_f1.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_duration_distribution(pred, gt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    bins = np.linspace(0, 10, 41)
    axes[0].hist(gt["duration"], bins=bins, alpha=0.6, label="GT")
    axes[0].hist(pred["duration"], bins=bins, alpha=0.6, label="Prediction")
    axes[0].set_xlabel("duration (s)")
    axes[0].set_title("Event duration distribution")
    axes[0].legend()

    axes[1].boxplot([gt["duration"], pred["duration"]], labels=["GT", "Prediction"], showfliers=False)
    axes[1].set_ylabel("duration (s)")
    axes[1].set_title("Event duration boxplot")

    out_path = ASSET_DIR / "duration_distribution.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_long_bias(long_bias_df):
    display_df = long_bias_df.head(6).copy()
    fig, ax1 = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(display_df))
    ax1.bar(x, display_df["pred_ge_9s"], color="#ee7733")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_df["类别"], rotation=25, ha="right")
    ax1.set_ylabel("count of predictions >= 9s")
    ax1.set_title("Long-duration prediction bias (top 6 classes)")

    out_path = ASSET_DIR / "long_duration_bias.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def find_audio_path(filename):
    candidate = VAL_AUDIO_16K_DIR / filename
    if candidate.exists():
        return candidate
    candidate = VAL_AUDIO_ORIG_DIR / filename
    if candidate.exists():
        return candidate
    return None


def sample_rows(df, filename):
    rows = df[df["filename"] == filename][["event_label", "onset", "offset"]].copy()
    return rows.sort_values(["onset", "offset", "event_label"]).reset_index(drop=True)


def format_events(rows):
    if rows.empty:
        return "无预测"
    return "<br>".join(
        f"{row.event_label} ({row.onset:.3f}-{row.offset:.3f}s)"
        for row in rows.itertuples(index=False)
    )


def plot_sample_figure(filename, sample_type, gt_rows, pred_rows):
    audio_path = find_audio_path(filename)
    duration = max(
        10.0,
        float(gt_rows["offset"].max()) if not gt_rows.empty else 0.0,
        float(pred_rows["offset"].max()) if not pred_rows.empty else 0.0,
    )

    row_labels = sorted(set(gt_rows["event_label"]).union(set(pred_rows["event_label"])))
    if not row_labels:
        row_labels = list(classes_labels.keys())[:1]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.1, 1.1, 1.1], hspace=0.25)
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
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=FEAT_CONFIG["n_fft"],
            win_length=FEAT_CONFIG["win_length"],
            hop_length=FEAT_CONFIG["hop_length"],
            f_min=FEAT_CONFIG["f_min"],
            f_max=FEAT_CONFIG["f_max"],
            n_mels=FEAT_CONFIG["n_mels"],
            power=1.0,
        )(audio.unsqueeze(0))
        mel = torchaudio.transforms.AmplitudeToDB(stype="amplitude")(mel).squeeze(0).numpy()
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

    cmap = matplotlib.colormaps.get_cmap("tab10")
    color_lookup = {label: cmap(i % 10) for i, label in enumerate(classes_labels.keys())}

    def draw_axis(ax, rows, ylabel):
        label_to_y = {label: idx for idx, label in enumerate(row_labels)}
        for label in row_labels:
            ax.axhline(label_to_y[label], color="#e0e0e0", linewidth=0.6, zorder=0)
        if rows.empty:
            ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
        else:
            for row in rows.itertuples(index=False):
                y = label_to_y[row.event_label] - 0.35
                ax.broken_barh(
                    [(row.onset, row.offset - row.onset)],
                    (y, 0.7),
                    facecolors=color_lookup[row.event_label],
                    edgecolors="black",
                    linewidth=0.5,
                )
        ax.set_yticks(list(label_to_y.values()))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, duration)

    draw_axis(ax_gt, gt_rows, "GT")
    draw_axis(ax_pred, pred_rows, "Pred")
    ax_pred.set_xlabel("Time (s)")
    ax_spec.set_title(f"{filename} | {sample_type}")

    out_path = ASSET_DIR / f"sample_{filename.replace('.wav', '')}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_report(
    config,
    ckpt,
    train_tb,
    test_tb,
    pred,
    gt,
    prediction_tsv,
    event_metrics,
    segment_metrics,
    overall_df,
    per_class_df,
    behavior_stats,
    comparison_df,
    fusion_vs_crnn_df,
    sample_details,
    asset_paths,
):
    train_loss = scalar_values(train_tb, "train/student/loss_strong")
    val_loss = scalar_values(train_tb, "val/synth/student/loss_strong")
    val_obj = scalar_values(train_tb, "val/obj_metric")
    val_inter = scalar_values(train_tb, "val/synth/student/intersection_f1_macro")
    val_event = scalar_values(train_tb, "val/synth/student/event_f1_macro")

    best_epoch = int(ckpt.get("epoch", -1))
    best_step = int(ckpt.get("global_step", -1))
    best_local_idx = int(np.argmax(val_obj)) if val_obj else -1

    strong_classes = per_class_df[per_class_df["分组"] == "较强"]["类别"].tolist()
    mid_classes = per_class_df[per_class_df["分组"] == "中等"]["类别"].tolist()
    weak_classes = per_class_df[per_class_df["分组"] == "较弱"]["类别"].tolist()

    config_summary = pd.DataFrame(
        [
            ("实验设置", "CRNN + BEATs late fusion baseline"),
            ("评估对象", "student"),
            ("model_type", config["model"]["model_type"]),
            ("fusion type", config["model"]["fusion"]["fusion_type"]),
            ("align method", config["model"]["fusion"]["align_method"]),
            ("BEATs freeze", config["model"]["beats"]["freeze"]),
            ("decoder temporal", "BiGRU + strong/weak heads"),
            ("数据划分", "synthetic train + synthetic validation"),
            ("test 是否独立", "否，当前 test 与 synthetic validation 为同一套数据"),
            ("best checkpoint", f"epoch={best_epoch}, step={best_step}"),
            ("prediction TSV", str(prediction_tsv.relative_to(ROOT))),
        ],
        columns=["项目", "说明"],
    )

    train_analysis_df = pd.DataFrame(
        [
            ("train/student/loss_strong", f"{train_loss[0]:.4f}", f"{train_loss[-1]:.4f}", f"{min(train_loss):.4f}"),
            ("val/synth/student/loss_strong", f"{val_loss[0]:.4f}", f"{val_loss[-1]:.4f}", f"{min(val_loss):.4f}"),
            ("val/obj_metric", f"{val_obj[0]:.4f}", f"{val_obj[-1]:.4f}", f"{max(val_obj):.4f}"),
            ("val/synth/student/intersection_f1_macro", f"{val_inter[0]:.4f}", f"{val_inter[-1]:.4f}", f"{max(val_inter):.4f}"),
            ("val/synth/student/event_f1_macro", f"{val_event[0]:.4f}", f"{val_event[-1]:.4f}", f"{max(val_event):.4f}"),
        ],
        columns=["曲线", "起始值", "最终值", "最佳值"],
    )

    per_class_display = per_class_df.drop(columns=["_event", "_segment"]).copy()
    count_display = behavior_stats["class_count_df"].copy()
    count_display.columns = ["类别", "GT事件数", "Pred事件数", "Pred-GT"]
    long_bias_display = behavior_stats["long_bias_df"].copy()
    long_bias_display.columns = ["类别", "平均预测时长", ">=9s 预测段数"]

    md = []
    md.append("# CRNN + BEATs Late Fusion Baseline 训练结果分析报告")
    md.append("")
    md.append("## 目录")
    md.append("- [实验概况](#实验概况)")
    md.append("- [最终指标汇总](#最终指标汇总)")
    md.append("- [训练过程与选模分析](#训练过程与选模分析)")
    md.append("- [预测行为统计](#预测行为统计)")
    md.append("- [典型样本分析](#典型样本分析)")
    md.append("- [结论与讨论](#结论与讨论)")
    md.append("- [后续建议](#后续建议)")
    md.append("")

    md.append("## 实验概况")
    md.append("")
    md.append(df_to_markdown(config_summary))
    md.append("")
    md.append(
        "本次实验属于 `CRNN + BEATs late fusion baseline`：一路使用 CRNN 的 CNN branch 提取时频局部特征，"
        "另一路使用冻结的 BEATs 提取 frame-level 表征；随后将 BEATs 特征先对齐到 CNN 时间长度，再做 concat 融合，"
        "经 Merge MLP 后进入共享的 BiGRU + strong/weak 分类头。"
    )
    md.append("")
    md.append(
        "当前最终评估对象仍以 `student` 为主。由于 `test_folder/test_tsv` 仍指向 synthetic validation，"
        "所以下面的结果更偏“自测分数”，适合判断 fusion 是否跑通以及它相对 CRNN/BEATs 的变化方向，但不能直接视为真实泛化结论。"
    )
    md.append("")

    md.append("## 最终指标汇总")
    md.append("")
    md.append(df_to_markdown(overall_df))
    md.append("")
    md.append("### 跨模型整体对照")
    md.append("")
    md.append(df_to_markdown(comparison_df))
    md.append("")
    md.append(df_to_markdown(per_class_display))
    md.append("")
    md.append(f"![Fusion vs CRNN per-class event F1](report_assets/{asset_paths['fusion_vs_crnn'].name})")
    md.append("")
    md.append(df_to_markdown(fusion_vs_crnn_df))
    md.append("")
    md.append(
        f"从本次 fusion 模型自身看，较强类别主要是 `{', '.join(strong_classes)}`，"
        f"中等类别主要是 `{', '.join(mid_classes)}`，"
        f"较弱类别则集中在 `{', '.join(weak_classes)}`。"
    )
    md.append("")
    md.append(
        "和单独 frozen BEATs 相比，这次 late fusion 已经明显恢复到“正常工作”的状态："
        "PSDS、event F1、segment F1 都大幅回升，之前大面积的类别塌缩基本被解除。"
    )
    md.append("")
    md.append(
        "但和 CRNN baseline 相比，这次还没有形成明显超越。整体上，fusion 的 `PSDS1/PSDS2` 与 `Intersection F1` 仍低于 CRNN，"
        "虽然 `Running_water`、`Blender`、`Cat` 这类类别有局部改善，但 `Dishes`、`Dog` 等弱类仍然偏弱。"
        "这正是“融合有效但收益有限”的核心证据：它不是没学到，而是还没有把 BEATs 的信息稳定转化成对复杂类别和边界更强的检测能力。"
    )
    md.append("")

    md.append("## 训练过程与选模分析")
    md.append("")
    md.append(f"![训练曲线](report_assets/{asset_paths['training_curves'].name})")
    md.append("")
    md.append(df_to_markdown(train_analysis_df))
    md.append("")
    md.append(
        f"训练过程整体是正常收敛的：`train/student/loss_strong` 在 step 级别存在波动，但最低达到 {min(train_loss):.4f}；"
        f"`val/synth/student/loss_strong` 从 {val_loss[0]:.4f} 下降到 {val_loss[-1]:.4f}，最低达到 {min(val_loss):.4f}。"
    )
    md.append("")
    md.append(
        f"`val/obj_metric` 在 version_20 这段续训日志中的局部索引 `{best_local_idx}` 达到峰值 {max(val_obj):.4f}；"
        f"对应的全局最佳 checkpoint 正是 `epoch={best_epoch}, step={best_step}`。"
    )
    md.append("")
    md.append(
        "这说明最佳 checkpoint 选在合理位置，而且真正最佳点大约出现在全局 20 轮左右之后不久。"
        "继续训练到 40 epoch 后，验证指标主要表现为平台震荡，而不是持续稳步上涨。"
    )
    md.append("")
    md.append(
        "这里需要特别说明：在 `synth_only` 下，`val/obj_metric` 实际上等于 `val/synth/student/intersection_f1_macro`。"
        "所以它更强调“预测区间大致重合”，而不是更严格的事件边界质量。"
        "这也是为什么 best checkpoint 主要由 intersection 指标驱动，而不是 event-based F1 驱动。"
    )
    md.append("")
    md.append(
        f"不过这次并不是单看 `obj_metric`：`val/synth/student/event_f1_macro` 的最佳值 {max(val_event):.4f} 也出现在同一轮附近，"
        "而且后面继续训练并没有明显再抬高这项指标。结合 `val loss` 一起看，这更像是正常平台期，而不是再训很久还能显著涨分的状态。"
    )
    md.append("")

    md.append("## 预测行为统计")
    md.append("")
    md.append(df_to_markdown(behavior_stats["summary_df"]))
    md.append("")
    md.append(f"![类别事件数对比](report_assets/{asset_paths['class_counts'].name})")
    md.append("")
    md.append(f"![类别 F1 对比](report_assets/{asset_paths['class_f1'].name})")
    md.append("")
    md.append(f"![事件时长分布](report_assets/{asset_paths['duration_dist'].name})")
    md.append("")
    md.append(f"![长时段偏置](report_assets/{asset_paths['long_bias'].name})")
    md.append("")
    md.append(df_to_markdown(count_display))
    md.append("")
    md.append(df_to_markdown(long_bias_display.head(6)))
    md.append("")
    md.append(
        f"当前系统已经摆脱了单独 frozen BEATs 时那种“大面积类别塌缩”的状态。"
        f"从 2500 个文件中，有预测的文件达到 {len(pred['filename'].unique())} 个，空预测文件 {len(behavior_stats['empty_files'])} 个，"
        f"空预测比例只有 {behavior_stats['summary_df'].iloc[3,1]}，明显比 frozen BEATs 单模型更健康。"
    )
    md.append("")
    md.append(
        f"但它仍没有明显超过 CRNN baseline：总预测事件数为 {len(pred)}，低于真值 {len(gt)}，"
        "说明系统整体仍偏保守，弱类召回不足仍在。尤其 `Dishes`、`Dog`、`Alarm_bell_ringing` 仍呈现明显的欠预测。"
    )
    md.append("")
    md.append(
        f"长时段偏置依然存在。预测中 `>=9s` 的长段有 {behavior_stats['summary_df'].iloc[9,1]} 个，"
        "主要集中在 `Vacuum_cleaner`、`Frying`、`Running_water` 和 `Blender` 这类更容易被建模为持续纹理的类别上。"
        "这说明融合主要改善了长持续类和设备类的粗粒度覆盖，但对复杂短事件和多事件场景帮助有限。"
    )
    md.append("")
    md.append(
        f"同时，疑似碎片化过预测文件数为 {behavior_stats['fragmented_files']} 个，"
        "说明系统虽然不再像 frozen BEATs 那样大面积空预测，但在部分长持续类上仍会出现切段和边界偏移。"
    )
    md.append("")

    md.append("## 典型样本分析")
    md.append("")
    for sample in sample_details:
        md.append(f"### {sample['filename']} | {sample['type']}")
        md.append("")
        md.append(f"![{sample['filename']}](report_assets/{sample['asset_name']})")
        md.append("")
        md.append(f"- 代表性：{sample['reason']}")
        md.append(f"- 真值事件：{format_events(sample['gt_rows'])}")
        md.append(f"- 预测事件：{format_events(sample['pred_rows'])}")
        md.append(f"- 简短点评：{sample['commentary']}")
        md.append("")

    md.append("## 结论与讨论")
    md.append("")
    md.append(
        "这次训练是正常的，late fusion 也确实已经跑通。无论从训练曲线、最终指标，还是典型样本来看，"
        "它都不是 bug、不是 NaN、不是全空预测，而是一版能稳定工作的融合 baseline。"
    )
    md.append("")
    md.append(
        "和单独 frozen BEATs 相比，这次结果已经明显更好：类别覆盖恢复、PSDS 回升、event/segment 指标大幅改善，"
        "说明将 CRNN branch 和 BEATs branch 在模型内部做 late fusion 的方向本身是成立的。"
    )
    md.append("")
    md.append(
        "但它还没有明显优于 CRNN baseline。当前最主要的问题不是“模型没学到”，而是“融合增益有限”："
        "BEATs 分支带来了一些对长持续设备类的补充，但还没有稳定提升复杂多事件、弱类、动物类和边界精度。"
    )
    md.append("")
    md.append(
        "这也是为什么它更像“CRNN 为主，BEATs 帮一点”。融合有效，说明信息确实进来了；收益有限，说明这些额外信息还没有被共享 temporal/classification head 充分转化。"
    )
    md.append("")
    md.append(
        f"至于为什么最佳轮在 20 左右就出现：从 `val/obj_metric`、`val/synth/student/event_f1_macro` 和 `val/synth/student/loss_strong` 三条曲线一起看，"
        "它们都在同一阶段附近达到最佳或接近最佳，后续继续训练到 40 epoch 主要是平台震荡，而不是继续稳定变好。"
        "因此这次结果足以作为一版“成功跑通、结果正常的 late fusion baseline”，但还不值得继续盲目长训堆 epoch。"
    )
    md.append("")

    md.append("## 后续建议")
    md.append("")
    md.append("1. 以后把 `epoch≈20` 作为 late fusion 的经验上限，先做早停或缩短训练，不要继续盲目长训到 40。")
    md.append("2. 优先针对 `Dishes / Dog / Alarm_bell_ringing / Cat` 做阈值与类不平衡分析，因为这些类别仍是 fusion 的主要短板。")
    md.append("3. 检查融合前后的尺度归一化与 feature scale，对 `cnn_feat` 与 `beats_feat` 加更明确的 normalization/projection，避免一边主导融合。")
    md.append("4. 做一页式 `CRNN vs Fusion` 对比，把每类 Event/Segment F1、Pred/GT 和典型样本并排放出来，尽快判断 late fusion 是否值得继续深挖。")
    md.append("5. 如果后续扩展资源有限，可以先评估 posterior fusion、WavLM 或部分解冻 BEATs，而不是继续单纯增加 late fusion 训练轮数。")
    md.append("")

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")


def main():
    ensure_dirs()
    config, ckpt, train_tb, test_tb, pred, gt, prediction_tsv = load_main_artifacts()
    event_metrics = parse_metric_txt(EVENT_F1_TXT)
    segment_metrics = parse_metric_txt(SEGMENT_F1_TXT)
    overall_df = overall_metrics_table(event_metrics, segment_metrics, test_tb)
    per_class_df = per_class_table(event_metrics, segment_metrics, pred, gt)
    behavior_stats = compute_behavior_stats(pred, gt)
    comparison_df = comparison_overview_table(overall_df)
    fusion_vs_crnn_df = fusion_vs_crnn_per_class_table(per_class_df)

    asset_paths = {
        "training_curves": plot_training_curves(train_tb),
        "class_counts": plot_class_counts(behavior_stats["class_count_df"]),
        "class_f1": plot_class_f1(per_class_df),
        "fusion_vs_crnn": plot_fusion_vs_crnn(per_class_df),
        "duration_dist": plot_duration_distribution(pred, gt),
        "long_bias": plot_long_bias(behavior_stats["long_bias_df"]),
    }

    sample_details = []
    for spec in SAMPLE_SPECS:
        gt_rows = sample_rows(gt, spec["filename"])
        pred_rows = sample_rows(pred, spec["filename"])
        asset = plot_sample_figure(spec["filename"], spec["plot_type"], gt_rows, pred_rows)
        sample_details.append(
            {
                "filename": spec["filename"],
                "type": spec["type"],
                "reason": spec["reason"],
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
                "asset_name": asset.name,
                "commentary": spec["commentary"],
            }
        )

    build_report(
        config,
        ckpt,
        train_tb,
        test_tb,
        pred,
        gt,
        prediction_tsv,
        event_metrics,
        segment_metrics,
        overall_df,
        per_class_df,
        behavior_stats,
        comparison_df,
        fusion_vs_crnn_df,
        sample_details,
        asset_paths,
    )

    print("Generated files:")
    print(f"- {REPORT_PATH}")
    for path in sorted(ASSET_DIR.glob("*.png")):
        print(f"- {path}")
    print("Selected samples:")
    for sample in sample_details:
        print(f"- {sample['filename']} ({sample['type']})")
    print("Completed:")
    print("- Final metrics summary")
    print("- Training curve analysis")
    print("- Prediction behavior statistics")
    print("- Typical sample visualizations")
    print("- Conclusions and recommendations")
    print("Not completed:")
    print("- No independent external test set analysis; current test remains synthetic validation.")


if __name__ == "__main__":
    main()
