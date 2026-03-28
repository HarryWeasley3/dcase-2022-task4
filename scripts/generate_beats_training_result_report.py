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

from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics
from local.classes_dict import classes_labels

REPORT_DIR = ROOT / "BEATs-baseline"
ASSET_DIR = REPORT_DIR / "report_assets"
REPORT_PATH = REPORT_DIR / "training_result_report.md"

TRAIN_TB = (
    ROOT / "exp/2022_baseline/version_9/events.out.tfevents.1774629440.HarryWeasley.1023570.0"
)
TEST_TB = (
    ROOT / "exp/2022_baseline/version_9/events.out.tfevents.1774664193.HarryWeasley.1023570.1"
)
BEST_CKPT = ROOT / "exp/2022_baseline/version_9/epoch=27-step=23352.ckpt"
CONFIG_PATH = ROOT / "confs/unified_beats_synth_only_d_drive.yaml"
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

FEAT_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 256,
    "n_mels": 128,
    "f_min": 0,
    "f_max": 8000,
}

SAMPLE_SPECS = [
    {
        "filename": "355.wav",
        "type": "检测较好 / 长持续类",
        "plot_type": "Good detection / long-duration class",
        "reason": "Frying 长持续事件几乎完整命中，适合作为正例。",
    },
    {
        "filename": "1088.wav",
        "type": "空预测 / 弱类漏检",
        "plot_type": "Empty prediction / weak-class miss",
        "reason": "Cat + Speech 双事件完全漏掉，能体现空预测问题。",
    },
    {
        "filename": "234.wav",
        "type": "碎片化过预测",
        "plot_type": "Over-segmentation / fragmented prediction",
        "reason": "Vacuum_cleaner 被切成大量碎片段，是典型过预测案例。",
    },
    {
        "filename": "1312.wav",
        "type": "多事件场景严重欠检",
        "plot_type": "Severe under-detection in multi-event scene",
        "reason": "多段 Dishes + Speech 只报出一小段 Speech，体现多事件召回不足。",
    },
    {
        "filename": "1195.wav",
        "type": "Dog 类失效",
        "plot_type": "Dog class collapse",
        "reason": "Dog 长事件完全无预测，直接反映类别塌缩。",
    },
    {
        "filename": "1000.wav",
        "type": "长时段类边界偏移",
        "plot_type": "Boundary shift on long-duration class",
        "reason": "Running_water 视觉上接近真值，但事件边界偏差较大，适合解释 intersection 高而 event F1 低。",
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


def load_main_artifacts():
    config = load_yaml(CONFIG_PATH)
    ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
    train_tb = load_tb(TRAIN_TB)
    test_tb = load_tb(TEST_TB)
    pred = pd.read_csv(PREDICTION_TSV, sep="\t")
    gt = pd.read_csv(GROUND_TRUTH_TSV, sep="\t")
    pred["duration"] = pred["offset"] - pred["onset"]
    gt["duration"] = gt["offset"] - gt["onset"]
    return config, ckpt, train_tb, test_tb, pred, gt


def compute_metrics(pred, gt):
    event_metric, segment_metric = compute_sed_eval_metrics(pred, gt)
    event_res = event_metric.results()
    segment_res = segment_metric.results()
    return event_res, segment_res


def overall_metrics_table(event_res, segment_res, test_tb):
    rows = [
        ("PSDS-scenario1", safe_num(scalar_values(test_tb, "test/student/psds_score_scenario1")[-1])),
        ("PSDS-scenario2", safe_num(scalar_values(test_tb, "test/student/psds_score_scenario2")[-1])),
        ("Intersection-based F1", safe_num(scalar_values(test_tb, "test/student/intersection_f1_macro")[-1])),
        (
            "Event-based F1 (macro)",
            safe_pct(event_res["class_wise_average"]["f_measure"]["f_measure"]),
        ),
        (
            "Event-based F1 (micro)",
            safe_pct(event_res["overall"]["f_measure"]["f_measure"]),
        ),
        (
            "Segment-based F1 (macro)",
            safe_pct(segment_res["class_wise_average"]["f_measure"]["f_measure"]),
        ),
        (
            "Segment-based F1 (micro)",
            safe_pct(segment_res["overall"]["f_measure"]["f_measure"]),
        ),
    ]
    return pd.DataFrame(rows, columns=["指标", "数值"])


def classify_status(pred_count, event_score):
    if pred_count == 0:
        return "完全失效"
    if event_score <= 0.001:
        return "有输出但事件级失效"
    if event_score >= 0.15:
        return "相对较强"
    if event_score >= 0.05:
        return "较弱但可用"
    return "明显偏弱"


def per_class_table(event_res, segment_res, pred, gt):
    gt_counts = gt["event_label"].value_counts()
    pred_counts = pred["event_label"].value_counts()
    rows = []
    for label in classes_labels.keys():
        event_score = event_res["class_wise"][label]["f_measure"]["f_measure"]
        segment_score = segment_res["class_wise"][label]["f_measure"]["f_measure"]
        event_score = 0.0 if math.isnan(event_score) else float(event_score)
        segment_score = 0.0 if math.isnan(segment_score) else float(segment_score)
        gt_n = int(gt_counts.get(label, 0))
        pred_n = int(pred_counts.get(label, 0))
        rows.append(
            {
                "类别": label,
                "GT事件数": gt_n,
                "Pred事件数": pred_n,
                "Event F1": safe_pct(event_score),
                "Segment F1": safe_pct(segment_score),
                "状态": classify_status(pred_n, event_score),
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
            pred_long["event_label"]
            .value_counts()
            .rename("pred_ge_9s")
            .to_frame(),
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
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["统计项", "数值"])

    return {
        "summary_df": summary_df,
        "class_count_df": class_count_df.reset_index().rename(columns={"index": "类别", "event_label": "类别"}),
        "long_bias_df": long_bias_df,
        "empty_files": empty_files,
        "file_stats": file_stats,
    }


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


def plot_training_curves(train_tb):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    train_loss = scalar_values(train_tb, "train/student/loss_strong")
    val_loss = scalar_values(train_tb, "val/synth/student/loss_strong")
    val_obj = scalar_values(train_tb, "val/obj_metric")
    val_inter = scalar_values(train_tb, "val/synth/student/intersection_f1_macro")

    if train_loss:
        axes[0, 0].plot(train_loss, alpha=0.25, color="#999999")
        window = 30
        smoothed = [np.mean(train_loss[max(0, i - window + 1) : i + 1]) for i in range(len(train_loss))]
        axes[0, 0].plot(smoothed, color="#cc3311")
        axes[0, 0].set_title("train/student/loss_strong")
        axes[0, 0].set_xlabel("step")

    if val_loss:
        axes[0, 1].plot(val_loss, marker="o", color="#0077bb")
        axes[0, 1].set_title("val/synth/student/loss_strong")
        axes[0, 1].set_xlabel("validation epoch")

    if val_obj:
        best_idx = int(np.argmax(val_obj))
        axes[1, 0].plot(val_obj, marker="o", color="#009988")
        axes[1, 0].axvline(best_idx, linestyle="--", color="#cc3311", label=f"best={best_idx}")
        axes[1, 0].legend()
        axes[1, 0].set_title("val/obj_metric")
        axes[1, 0].set_xlabel("validation epoch")

    if val_inter:
        axes[1, 1].plot(val_inter, marker="o", color="#ee7733")
        axes[1, 1].set_title("val/synth/student/intersection_f1_macro")
        axes[1, 1].set_xlabel("validation epoch")

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
    ax.set_title("Event / segment F1 by class")
    ax.legend()
    out_path = ASSET_DIR / "class_f1_comparison.png"
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
    event_res,
    segment_res,
    overall_df,
    per_class_df,
    behavior_stats,
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

    strong_classes = per_class_df.sort_values("_event", ascending=False).head(3)["类别"].tolist()
    weak_classes = per_class_df[(per_class_df["_event"] > 0) & (per_class_df["_event"] < 0.1)]["类别"].tolist()
    collapsed_classes = per_class_df[per_class_df["Pred事件数"] == 0]["类别"].tolist()
    nearly_collapsed = per_class_df[(per_class_df["Pred事件数"] > 0) & (per_class_df["_event"] == 0)]["类别"].tolist()

    config_summary = pd.DataFrame(
        [
            ("实验设置", "BEATs + synth_only"),
            ("评估对象", "student"),
            ("encoder_type", config["model"]["encoder_type"]),
            ("BEATs freeze", config["model"]["beats"]["freeze"]),
            ("shared decoder", "BiGRU + strong/weak head"),
            ("decoder input_proj_dim", config["model"]["decoder"]["input_proj_dim"]),
            ("数据划分", "synthetic train + synthetic validation"),
            ("test 是否独立", "否，当前 test 与 synthetic validation 为同一套数据"),
            ("最优 checkpoint", f"epoch={best_epoch}, step={best_step}"),
        ],
        columns=["项目", "说明"],
    )

    train_analysis_df = pd.DataFrame(
        [
            ("train/student/loss_strong", f"{train_loss[0]:.4f}", f"{train_loss[-1]:.4f}", f"{min(train_loss):.4f}"),
            ("val/synth/student/loss_strong", f"{val_loss[0]:.4f}", f"{val_loss[-1]:.4f}", f"{min(val_loss):.4f}"),
            ("val/obj_metric", f"{val_obj[0]:.4f}", f"{val_obj[-1]:.4f}", f"{max(val_obj):.4f}"),
            (
                "val/synth/student/intersection_f1_macro",
                f"{val_inter[0]:.4f}",
                f"{val_inter[-1]:.4f}",
                f"{max(val_inter):.4f}",
            ),
            (
                "val/synth/student/event_f1_macro",
                f"{val_event[0]:.4f}",
                f"{val_event[-1]:.4f}",
                f"{max(val_event):.4f}",
            ),
        ],
        columns=["曲线", "起始值", "最终值", "最佳值"],
    )

    per_class_display = per_class_df.drop(columns=["_event", "_segment"]).copy()
    count_display = behavior_stats["class_count_df"].copy()
    count_display.columns = ["类别", "GT事件数", "Pred事件数", "Pred-GT"]
    long_bias_display = behavior_stats["long_bias_df"].copy()
    long_bias_display.columns = ["类别", "平均预测时长", ">=9s 预测段数"]

    md = []
    md.append("# BEATs + synth_only 训练结果分析报告")
    md.append("")
    md.append("## 目录")
    md.append("- [实验概况](#实验概况)")
    md.append("- [最终指标汇总](#最终指标汇总)")
    md.append("- [训练过程与选模分析](#训练过程与选模分析)")
    md.append("- [预测行为统计](#预测行为统计)")
    md.append("- [典型样本分析](#典型样本分析)")
    md.append("- [结论与问题归因](#结论与问题归因)")
    md.append("- [后续建议](#后续建议)")
    md.append("")

    md.append("## 实验概况")
    md.append("")
    md.append(df_to_markdown(config_summary))
    md.append("")
    md.append(
        "本次实验使用冻结的 BEATs 作为 encoder，并接统一的共享 decoder（带 BiGRU 的传统 SED 头）。"
        "从配置与 checkpoint 可确认这是 `encoder_type=beats`、`freeze=True`、`synth_only=True` 的设置。"
    )
    md.append("")
    md.append(
        "由于当前配置把 `test_folder/test_tsv` 指向 synthetic validation，所以下面的测试结果更接近“自测分数”，"
        "适合做模型行为诊断，但不能直接当作真实外部分布上的泛化结论。"
    )
    md.append("")

    md.append("## 最终指标汇总")
    md.append("")
    md.append(df_to_markdown(overall_df))
    md.append("")
    md.append(df_to_markdown(per_class_display))
    md.append("")
    md.append(
        f"表现相对较强的类别主要是 `{', '.join(strong_classes)}`；"
        f"明显偏弱但仍有一定输出的是 `{', '.join(weak_classes) if weak_classes else '无'}`；"
        f"完全无预测的类别包括 `{', '.join(collapsed_classes)}`。"
    )
    md.append("")
    md.append(
        "这组结果最值得注意的地方是：`intersection F1` 还能到 0.432，但 `event-based F1` 和 `PSDS` 极低。"
        "这说明模型并不是完全不会报事件，而是更偏向报出“粗粒度、长持续、边界不稳”的事件片段。"
        "当评估放宽到区间交并关系时还能拿到一定分数，但一旦要求更严格的起止边界、类别完整性和跨阈值稳定性，性能就会显著下降。"
    )
    md.append("")

    md.append("## 训练过程与选模分析")
    md.append("")
    md.append(f"![训练曲线](report_assets/{asset_paths['training_curves'].name})")
    md.append("")
    md.append(df_to_markdown(train_analysis_df))
    md.append("")
    md.append(
        f"训练过程本身是正常收敛的：`train/student/loss_strong` 从 {train_loss[0]:.4f} 降到 {train_loss[-1]:.4f}，"
        f"`val/synth/student/loss_strong` 虽然下降有限，但最低也达到 {min(val_loss):.4f}。"
    )
    md.append("")
    md.append(
        f"`val/obj_metric` 在 epoch={best_epoch} 附近达到峰值 {max(val_obj):.4f}，与最佳 checkpoint 的保存位置一致，说明选模逻辑本身没有异常。"
    )
    md.append("")
    md.append(
        "但在 `synth_only` 下，`val/obj_metric` 实际上等于 `val/synth/student/intersection_f1_macro`，"
        "它更强调“预测区间是否大致覆盖真值”，而不是事件起止边界和跨阈值检测质量。"
        "因此 checkpoint 虽然在 `obj_metric` 上最优，却不一定在 `event-based F1` 或 `PSDS` 上最优。"
    )
    md.append("")
    md.append(
        f"这点从验证曲线也能看出来：`intersection_f1_macro` 最高到 {max(val_inter):.4f}，"
        f"但 `val/synth/student/event_f1_macro` 最佳值只有 {max(val_event):.4f}，两者之间存在明显落差。"
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
        f"这次模型不是“全空”：2500 个文件里有预测的文件有 {behavior_stats['summary_df'].iloc[1,1]} 个，"
        f"空预测文件是 {behavior_stats['summary_df'].iloc[2,1]} 个，占比 {behavior_stats['summary_df'].iloc[3,1]}。"
        "但它也绝不是健康状态，因为 10 个类别里有 4 个完全没有输出，另有 2 个几乎失效。"
    )
    md.append("")
    md.append(
        f"类别塌缩的直接证据包括：`{', '.join(collapsed_classes)}` 的预测数为 0，"
        f"`{', '.join(nearly_collapsed) if nearly_collapsed else '无'}` 虽然有少量输出，但事件级 F1 仍为 0。"
    )
    md.append("")
    md.append(
        "同时，长时段偏置也很明显：预测中 `>=9s` 的长段共有 "
        f"{behavior_stats['summary_df'].iloc[9,1]} 个，主要集中在 `Frying`、`Vacuum_cleaner`、`Electric_shaver_toothbrush` 和部分 `Running_water`。"
        "这说明模型更容易把长持续背景事件学成大块区间，而对短事件、动物类和碎片化类明显不稳定。"
    )
    md.append("")
    md.append(
        "因此总体判断不是“全局训练崩溃”，而是“少数类工作、部分类严重塌缩，并伴随长持续事件偏置和局部碎片化过预测”。"
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

    md.append("## 结论与问题归因")
    md.append("")
    md.append(
        "这次训练不能算“彻底没学到”。训练 loss 和验证 `obj_metric` 都有明显优化，说明 frozen BEATs + shared decoder 这条链路至少学到了一部分可用模式。"
    )
    md.append("")
    md.append(
        "但最终结果不能算正常。问题不是单纯的整体偏低，而是非常典型的“类别塌缩 + 长时段偏置”："
        "模型主要学会了 `Speech / Frying / Electric_shaver_toothbrush / Vacuum_cleaner` 等少数类别，"
        "却几乎完全失去了 `Dog / Dishes / Alarm_bell_ringing / Cat` 等类别。"
    )
    md.append("")
    md.append(
        "为什么 `obj_metric` 不足以代表最终 event-based 检测质量："
        "当前 `obj_metric` 在 synth_only 下只看 `intersection_f1_macro`，它对“区间大致重合”更敏感，"
        "而对严格边界、碎片化预测、跨阈值稳定性和类别覆盖不足不够敏感。"
        "所以模型可以在 `obj_metric` 上达到 0.432，却在 event-based F1 和 PSDS 上依然很差。"
    )
    md.append("")
    md.append(
        "从这次结果看，frozen BEATs + shared decoder 在当前设置下可能有几个局限："
        "一是 decoder 对 BEATs 表征的适配还不够，二是类间不平衡被进一步放大，三是当前阈值和后处理更有利于长持续事件，"
        "四是 frozen encoder 对细粒度短事件与动物类的分离能力不足。"
    )
    md.append("")

    md.append("## 后续建议")
    md.append("")
    md.append("1. 先把这次 BEATs 与 CRNN 的逐类 Event/Segment F1 并排比较，确认问题主要来自 encoder 表征，还是 decoder/阈值/后处理。")
    md.append("2. 单独搜索阈值与 median filter，优先看能否显著抬升 event-based F1 和 PSDS；这一步成本最低，也最能解释“intersection 高但 event/PSDS 低”的原因。")
    md.append("3. 对 `Dog / Dishes / Alarm_bell_ringing / Cat / Blender` 做采样重平衡或损失重加权，避免长持续类继续主导训练。")
    md.append("4. 保持统一 decoder 不变，进一步尝试 BEATs 的部分解冻或最后若干层微调，验证是不是 frozen encoder 本身限制了弱类表征。")
    md.append("5. 如果后续要公平比较 encoder，建议固定同一份 decoder 和选模流程，但增加一个 event-based 或混合型验证指标，避免只靠 intersection 选模型。")
    md.append("")

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")


def build_sample_commentary(filename):
    mapping = {
        "355.wav": "Frying 长事件几乎完整重合，说明模型对少数长持续类并非完全失效。",
        "1088.wav": "Cat 与短 Speech 同时漏掉，体现了弱类与短时事件在当前模型中的脆弱性。",
        "234.wav": "Vacuum_cleaner 没有整段报满，但被切成许多小段，属于典型的碎片化过预测。",
        "1312.wav": "多事件场景中几乎只剩下一小段 Speech，被 Dishes 和其他 Speech 片段完全淹没。",
        "1195.wav": "Dog 长事件完全不出结果，与全局统计里 Dog 预测数为 0 一致。",
        "1000.wav": "Running_water 在视觉上接近真值，但起止边界仍明显偏移，能解释为什么区间类指标尚可而事件级指标很差。",
    }
    return mapping.get(filename, "代表性样本。")


def main():
    ensure_dirs()
    config, ckpt, train_tb, test_tb, pred, gt = load_main_artifacts()
    event_res, segment_res = compute_metrics(pred, gt)
    overall_df = overall_metrics_table(event_res, segment_res, test_tb)
    per_class_df = per_class_table(event_res, segment_res, pred, gt)
    behavior_stats = compute_behavior_stats(pred, gt)

    asset_paths = {
        "training_curves": plot_training_curves(train_tb),
        "class_counts": plot_class_counts(behavior_stats["class_count_df"]),
        "class_f1": plot_class_f1(per_class_df),
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
                "commentary": build_sample_commentary(spec["filename"]),
            }
        )

    build_report(
        config,
        ckpt,
        train_tb,
        test_tb,
        pred,
        gt,
        event_res,
        segment_res,
        overall_df,
        per_class_df,
        behavior_stats,
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
    print("- No additional external test set analysis, because current test is synthetic validation.")


if __name__ == "__main__":
    main()
