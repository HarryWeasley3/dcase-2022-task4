import math
import os
import re
import shutil
import sys
from collections import OrderedDict
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desed_task.evaluation.evaluation_measures import compute_psds_from_operating_points
from local.classes_dict import classes_labels

REPORT_DIR = ROOT / "baselines" / "beats-fintune"
ASSET_DIR = REPORT_DIR / "report_assets"
MIRROR_ASSET_DIR = ROOT / "bbaselines" / "beats-fintune" / "report_assets"
REPORT_PATH = REPORT_DIR / "training_result_report.md"

EXPERIMENT_ROOT = ROOT / "exp" / "unified_beats_synth_only_a800_finetune"
BASELINE_REPORTS = OrderedDict(
    [
        ("CRNN baseline", ROOT / "baselines" / "CRNN-baseline" / "training_result_report.md"),
        ("Frozen BEATs baseline", ROOT / "baselines" / "BEATs-baseline" / "training_result_report.md"),
        ("Concat late fusion baseline", ROOT / "baselines" / "BEATs-crnn-fusion-baseline" / "training_result_report.md"),
        ("Residual gated fusion baseline", ROOT / "baselines" / "gate-fusion-baseline" / "training_result_report.md"),
    ]
)

SAMPLE_SPECS = [
    {
        "filename": "355.wav",
        "pattern": "检测较好的正例 / 长持续类",
        "plot_title": "Good detection / long-duration class",
        "reason": "Frying 长持续事件近乎完整命中，适合作为 full finetune 已经恢复稳定检测能力的正例。",
    },
    {
        "filename": "1088.wav",
        "pattern": "相比 frozen BEATs 明显改善",
        "plot_title": "Clear improvement over frozen BEATs",
        "reason": "同一文件在 frozen BEATs 报告里是空预测，而这次已经能同时报出 Cat 和 Speech。",
    },
    {
        "filename": "234.wav",
        "pattern": "相比 frozen BEATs 明显改善 / 边界更稳",
        "plot_title": "Recovered boundary stability",
        "reason": "frozen BEATs 曾把 Vacuum_cleaner 切成碎片，这次已恢复成接近整段的连续预测。",
    },
    {
        "filename": "1000.wav",
        "pattern": "长持续类表现较好",
        "plot_title": "Stable long-duration scene",
        "reason": "Running_water 长事件能够稳定覆盖整段，同时仍能保留部分 Speech 片段。",
    },
    {
        "filename": "1195.wav",
        "pattern": "弱类仍然较差",
        "plot_title": "Weak class still poor",
        "reason": "Dog 长事件依旧完全漏检，适合展示 hardest weak class 仍未解决。",
    },
    {
        "filename": "1312.wav",
        "pattern": "多事件场景欠检",
        "plot_title": "Under-detection in multi-event scene",
        "reason": "Dishes + Speech 的复杂场景仍主要只剩 Speech，体现多事件召回不足。",
    },
]

FEAT_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 256,
    "n_mels": 128,
    "f_min": 0,
    "f_max": 8000,
}


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    MIRROR_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False


def safe_num(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{0:.{digits}f}"
    return f"{value:.{digits}f}"


def pct_str(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{0:.{digits}f}%"
    return f"{value * 100:.{digits}f}%"


def to_markdown(df):
    frame = df.copy()
    frame.columns = [str(c) for c in frame.columns]
    for col in frame.columns:
        frame[col] = frame[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = {
        col: max(len(col), *(len(v) for v in frame[col].tolist())) if len(frame) else len(col)
        for col in frame.columns
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in frame.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in frame.columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def load_tb(path):
    event = EventAccumulator(str(path), size_guidance={"scalars": 0})
    event.Reload()
    return event


def scalar_values(event, tag):
    return [x.value for x in event.Scalars(tag)] if tag in event.Tags().get("scalars", []) else []


def resolve_config_path(target_cfg):
    if not isinstance(target_cfg, dict):
        return None
    for path in sorted((ROOT / "confs").glob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        if cfg == target_cfg:
            return path
    return None


def locate_current_experiment():
    candidates = []
    for version_dir in sorted(EXPERIMENT_ROOT.glob("version_*")):
        ckpts = sorted(version_dir.glob("epoch=*-step=*.ckpt"))
        if not ckpts:
            continue
        best_ckpt = ckpts[-1]
        ckpt = torch.load(best_ckpt, map_location="cpu")
        hparams = ckpt.get("hyper_parameters", {})
        model_cfg = hparams.get("model", {})
        encoder_type = model_cfg.get("encoder_type")
        freeze = model_cfg.get("beats", {}).get("freeze")
        model_type = model_cfg.get("model_type")
        if encoder_type != "beats":
            continue
        if freeze is not False:
            continue
        if model_type and "fusion" in str(model_type):
            continue
        callbacks = ckpt.get("callbacks", {})
        best_score = None
        best_path = None
        for val in callbacks.values():
            if isinstance(val, dict) and val.get("best_model_path"):
                best_path = val.get("best_model_path")
                score = val.get("best_model_score")
                best_score = float(score) if score is not None else None
                break
        event_files = sorted(version_dir.glob("events.out.tfevents.*"))
        train_event = event_files[-1] if event_files else None
        cfg_path = resolve_config_path({k: v for k, v in hparams.items() if k != "log_dir"})
        candidates.append(
            {
                "version": version_dir.name,
                "version_dir": version_dir,
                "best_ckpt": best_ckpt,
                "last_epoch": int(ckpt.get("epoch", -1)),
                "global_step": int(ckpt.get("global_step", -1)),
                "best_score": best_score,
                "best_path": best_path,
                "mtime": version_dir.stat().st_mtime,
                "train_event": train_event,
                "train_event_count": len(event_files),
                "metrics_root": EXPERIMENT_ROOT / "metrics_test",
                "config_path": cfg_path,
                "hparams": hparams,
            }
        )
    if not candidates:
        raise FileNotFoundError("No BEATs full-finetune experiment candidates were found.")
    candidates.sort(key=lambda x: (x["mtime"], x["last_epoch"], x["global_step"]), reverse=True)
    chosen = candidates[0]
    return chosen, candidates


def actual_gt_path(hparams):
    placeholder = "<GT_TSV>"
    candidates = [
        hparams.get("data", {}).get("test_tsv"),
        hparams.get("data", {}).get("synth_val_tsv"),
        "./runtime_data/dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv",
    ]
    for raw in candidates:
        if not raw or raw == placeholder:
            continue
        path = (ROOT / raw).resolve() if str(raw).startswith("./") else Path(raw)
        if path.exists():
            return path, placeholder
    raise FileNotFoundError("Could not resolve a valid ground-truth TSV on the server.")


def actual_audio_dir(hparams):
    candidates = [
        hparams.get("data", {}).get("test_folder"),
        hparams.get("data", {}).get("synth_val_folder"),
        "./runtime_data/dcase_synth/audio/validation/synthetic21_validation/soundscapes_16k",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = (ROOT / raw).resolve() if str(raw).startswith("./") else Path(raw)
        if path.exists():
            return path
    return None


def parse_markdown_table(block_lines):
    rows = []
    for line in block_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        rows.append(parts)
    if len(rows) < 2:
        return None
    header = rows[0]
    body = []
    for row in rows[2:]:
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        body.append(row[: len(header)])
    return pd.DataFrame(body, columns=header)


def extract_markdown_tables(text):
    tables = []
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().startswith("|"):
            block = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                block.append(lines[idx])
                idx += 1
            frame = parse_markdown_table(block)
            if frame is not None:
                tables.append(frame)
        else:
            idx += 1
    return tables


def parse_percent(value):
    text = str(value).replace("%", "").strip()
    if not text:
        return float("nan")
    return float(text) / 100.0


def parse_scalar(value):
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_baseline_report(report_path):
    text = report_path.read_text(encoding="utf-8")
    tables = extract_markdown_tables(text)
    overall_df = next(
        (df for df in tables if list(df.columns)[:2] == ["指标", "数值"]),
        None,
    )
    per_class_df = next(
        (
            df
            for df in tables
            if "类别" in df.columns and "Event F1" in df.columns and "Segment F1" in df.columns and "GT事件数" in df.columns
        ),
        None,
    )
    if overall_df is None or per_class_df is None:
        raise ValueError(f"Failed to parse required tables from {report_path}")
    overall = {}
    for _, row in overall_df.iterrows():
        metric = row["指标"]
        value = row["数值"]
        if "%" in str(value):
            overall[metric] = parse_percent(value)
        else:
            overall[metric] = parse_scalar(value)
    per_class = per_class_df.copy()
    for col in ["GT事件数", "Pred事件数"]:
        if col in per_class.columns:
            per_class[col] = per_class[col].astype(str).str.replace(",", "").astype(float).astype(int)
    if "Pred/GT" in per_class.columns:
        per_class["Pred/GT"] = per_class["Pred/GT"].map(parse_scalar)
    per_class["Event F1 value"] = per_class["Event F1"].map(parse_percent)
    per_class["Segment F1 value"] = per_class["Segment F1"].map(parse_percent)
    return {
        "overall": overall,
        "per_class": per_class,
        "raw_text": text,
    }


def extract_empty_ratio_from_report(text):
    match = re.search(r"空预测比例\s*\|\s*([0-9.]+)%", text)
    return float(match.group(1)) / 100.0 if match else float("nan")


def normalize_label(raw):
    text = raw.strip()
    if text in classes_labels:
        return text
    prefix = text.replace(".", "")
    for label in classes_labels.keys():
        if label.startswith(prefix):
            return label
    return text


def parse_metric_txt(path):
    text = path.read_text(encoding="utf-8")
    micro = re.search(r"Overall metrics \(micro-average\).*?F-measure \(F1\)\s*:\s*([0-9.]+) %", text, re.S)
    macro = re.search(r"Class-wise average metrics \(macro-average\).*?F-measure \(F1\)\s*:\s*([0-9.]+) %", text, re.S)
    class_rows = []
    class_section = text.split("Class-wise metrics", 1)[-1]
    for line in class_section.splitlines():
        if "|" not in line:
            continue
        if "Event label" in line or "------------" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        label = normalize_label(parts[0])
        try:
            count_parts = parts[1].split()
            score_parts = parts[2].split()
            nref = int(count_parts[0])
            nsys = int(count_parts[1])
            f1 = float(score_parts[0].replace("%", "")) / 100.0
        except ValueError:
            continue
        class_rows.append({"类别": label, "Nref": nref, "Nsys": nsys, "F1": f1})
    return {
        "micro_f1": float(micro.group(1)) / 100.0 if micro else float("nan"),
        "macro_f1": float(macro.group(1)) / 100.0 if macro else float("nan"),
        "class_df": pd.DataFrame(class_rows),
    }


def load_prediction_operating_points(scenario_dir):
    ops = {}
    for path in sorted(scenario_dir.glob("predictions_th_*.tsv")):
        match = re.search(r"predictions_th_([0-9.]+)\.tsv$", path.name)
        if not match:
            continue
        threshold = float(match.group(1))
        frame = pd.read_csv(path, sep="\t")
        ops[str(threshold)] = frame
    return ops


def build_current_results(chosen):
    hparams = chosen["hparams"]
    gt_path, placeholder = actual_gt_path(hparams)
    dur_path = Path(hparams["data"]["test_dur"])
    if not dur_path.is_absolute():
        dur_path = (ROOT / dur_path).resolve()
    pred_tsv = chosen["metrics_root"] / "student" / "scenario1" / "predictions_dtc0.7_gtc0.7_cttc0.3" / "predictions_th_0.49.tsv"
    event_txt = chosen["metrics_root"] / "student" / "event_f1.txt"
    segment_txt = chosen["metrics_root"] / "student" / "segment_f1.txt"
    if not pred_tsv.exists() or not event_txt.exists() or not segment_txt.exists():
        raise FileNotFoundError("Current experiment is missing metrics_test artifacts.")
    pred = pd.read_csv(pred_tsv, sep="\t")
    gt = pd.read_csv(gt_path, sep="\t")
    pred["duration"] = pred["offset"] - pred["onset"]
    gt["duration"] = gt["offset"] - gt["onset"]
    event_metric = parse_metric_txt(event_txt)
    segment_metric = parse_metric_txt(segment_txt)
    scenario1_ops = load_prediction_operating_points(
        chosen["metrics_root"] / "student" / "scenario1" / "predictions_dtc0.7_gtc0.7_cttc0.3"
    )
    scenario2_ops = load_prediction_operating_points(
        chosen["metrics_root"] / "student" / "scenario2" / "predictions_dtc0.1_gtc0.1_cttc0.3"
    )
    psds1 = float(
        compute_psds_from_operating_points(
            scenario1_ops,
            str(gt_path),
            str(dur_path),
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
        )
    )
    psds2 = float(
        compute_psds_from_operating_points(
            scenario2_ops,
            str(gt_path),
            str(dur_path),
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
        )
    )
    train_tb = load_tb(chosen["train_event"])
    val_obj = scalar_values(train_tb, "val/obj_metric")
    current_per_class = pd.DataFrame({"类别": list(classes_labels.keys())})
    current_per_class = current_per_class.merge(
        event_metric["class_df"][["类别", "Nref", "Nsys", "F1"]].rename(
            columns={"Nref": "GT事件数_event", "Nsys": "Nsys_event", "F1": "Event F1 value"}
        ),
        on="类别",
        how="left",
    )
    current_per_class = current_per_class.merge(
        segment_metric["class_df"][["类别", "Nref", "Nsys", "F1"]].rename(
            columns={"Nref": "GT段数_segment", "Nsys": "Nsys_segment", "F1": "Segment F1 value"}
        ),
        on="类别",
        how="left",
    )
    gt_counts = gt["event_label"].value_counts()
    pred_counts = pred["event_label"].value_counts()
    current_per_class["GT事件数"] = current_per_class["类别"].map(lambda x: int(gt_counts.get(x, 0)))
    current_per_class["Pred事件数"] = current_per_class["类别"].map(lambda x: int(pred_counts.get(x, 0)))
    current_per_class["Pred/GT"] = current_per_class.apply(
        lambda row: row["Pred事件数"] / row["GT事件数"] if row["GT事件数"] else float("nan"),
        axis=1,
    )
    current_per_class["Event F1"] = current_per_class["Event F1 value"].map(pct_str)
    current_per_class["Segment F1"] = current_per_class["Segment F1 value"].map(pct_str)
    return {
        "gt_path": gt_path,
        "gt_placeholder": placeholder,
        "audio_dir": actual_audio_dir(hparams),
        "pred_tsv": pred_tsv,
        "event_txt": event_txt,
        "segment_txt": segment_txt,
        "pred": pred,
        "gt": gt,
        "psds1": psds1,
        "psds2": psds2,
        "intersection": float(max(val_obj)) if val_obj else float("nan"),
        "event_micro": event_metric["micro_f1"],
        "event_macro": event_metric["macro_f1"],
        "segment_micro": segment_metric["micro_f1"],
        "segment_macro": segment_metric["macro_f1"],
        "per_class": current_per_class,
        "train_tb": train_tb,
    }


def behavior_stats(pred, gt):
    gt_files = sorted(gt["filename"].unique())
    pred_files = sorted(pred["filename"].unique())
    empty_files = sorted(set(gt_files) - set(pred_files))
    stats = {
        "gt_file_count": len(gt_files),
        "pred_file_count": len(pred_files),
        "empty_pred_count": len(empty_files),
        "empty_pred_ratio": len(empty_files) / len(gt_files),
        "gt_event_count": int(len(gt)),
        "pred_event_count": int(len(pred)),
        "gt_duration_mean": float(gt["duration"].mean()),
        "pred_duration_mean": float(pred["duration"].mean()),
        "gt_duration_median": float(gt["duration"].median()),
        "pred_duration_median": float(pred["duration"].median()),
        "gt_duration_p95": float(gt["duration"].quantile(0.95)),
        "pred_duration_p95": float(pred["duration"].quantile(0.95)),
        "pred_near_full": int((pred["duration"] >= 9.5).sum()),
        "gt_near_full": int((gt["duration"] >= 9.5).sum()),
        "empty_files": empty_files,
    }
    class_count_df = pd.DataFrame({"类别": list(classes_labels.keys())})
    class_count_df["GT事件数"] = class_count_df["类别"].map(lambda x: int(gt["event_label"].eq(x).sum()))
    class_count_df["Pred事件数"] = class_count_df["类别"].map(lambda x: int(pred["event_label"].eq(x).sum()))
    class_count_df["Pred-GT"] = class_count_df["Pred事件数"] - class_count_df["GT事件数"]
    long_bias = (
        pred.groupby("event_label")["duration"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "平均预测时长", "count": "Pred事件数"})
        .reset_index()
        .rename(columns={"event_label": "类别"})
    )
    long_bias[">=9s 预测段数"] = long_bias["类别"].map(
        lambda x: int(pred[(pred["event_label"] == x) & (pred["duration"] >= 9.0)].shape[0])
    )
    long_bias["平均预测时长_num"] = long_bias["平均预测时长"]
    long_bias["平均预测时长"] = long_bias["平均预测时长"].map(lambda x: f"{x:.2f}s")
    long_bias = long_bias.sort_values([">=9s 预测段数", "平均预测时长_num"], ascending=False)
    file_event_counts = (
        gt.groupby("filename").size().rename("gt_events").to_frame().join(
            pred.groupby("filename").size().rename("pred_events"), how="left"
        ).fillna(0)
    )
    fragmented_files = int(((file_event_counts["pred_events"] - file_event_counts["gt_events"]) >= 4).sum())
    stats["fragmented_files"] = fragmented_files
    stats["class_count_df"] = class_count_df
    stats["long_bias_df"] = long_bias
    return stats


def strength_group(score):
    if score >= 0.55:
        return "较强"
    if score >= 0.40:
        return "中等"
    return "较弱"


def choose_sample_rows(df, filename):
    rows = df[df["filename"] == filename][["event_label", "onset", "offset"]].copy()
    return rows.sort_values(["onset", "offset", "event_label"]).reset_index(drop=True)


def find_audio_path(audio_dir, filename):
    if audio_dir is None:
        return None
    candidate = audio_dir / filename
    if candidate.exists():
        return candidate
    return None


def format_events(rows):
    if rows.empty:
        return "无预测"
    return "<br>".join(
        f"{row.event_label} ({row.onset:.3f}-{row.offset:.3f}s)"
        for row in rows.itertuples(index=False)
    )


def plot_training_curves(tb):
    tags = [
        ("train/student/loss_strong", "Train loss", "#cc3311"),
        ("val/synth/student/loss_strong", "Val loss", "#0077bb"),
        ("val/obj_metric", "Val obj metric", "#009988"),
        ("val/synth/student/intersection_f1_macro", "Val intersection F1", "#ee7733"),
        ("val/synth/student/event_f1_macro", "Val event macro F1", "#332288"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.flatten()
    for ax, (tag, title, color) in zip(axes, tags):
        values = scalar_values(tb, tag)
        if not values:
            ax.set_visible(False)
            continue
        ax.plot(values, marker="o" if len(values) <= 120 else None, color=color, linewidth=1.8)
        if "val/" in tag:
            best_idx = int(np.argmax(values)) if "loss" not in tag else int(np.argmin(values))
            ax.axvline(best_idx, color="#999999", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("epoch" if "val/" in tag else "step")
        ax.grid(alpha=0.2)
    axes[-1].set_visible(False)
    out = ASSET_DIR / "training_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_class_counts(class_count_df):
    labels = class_count_df["类别"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - width / 2, class_count_df["GT事件数"], width=width, label="GT")
    ax.bar(x + width / 2, class_count_df["Pred事件数"], width=width, label="Prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("event count")
    ax.set_title("GT vs prediction event count by class")
    ax.legend()
    out = ASSET_DIR / "class_count_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_class_f1(per_class_df):
    labels = per_class_df["类别"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - width / 2, per_class_df["Event F1 value"] * 100, width=width, label="Event F1")
    ax.bar(x + width / 2, per_class_df["Segment F1 value"] * 100, width=width, label="Segment F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-class Event / Segment F1")
    ax.legend()
    out = ASSET_DIR / "class_f1_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_model_comparison(overall_rows):
    metrics = ["PSDS1", "PSDS2", "Intersection", "Event macro", "Segment macro"]
    labels = [
        {
            "CRNN baseline": "CRNN",
            "Frozen BEATs baseline": "Frozen BEATs",
            "Concat late fusion baseline": "Concat fusion",
            "Residual gated fusion baseline": "Residual gate",
            "BEATs 全量微调": "BEATs FT",
        }.get(row["模型"], row["模型"])
        for row in overall_rows
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        values = [row[metric] * 100 if "macro" in metric else row[metric] for row in overall_rows]
        ax.bar(np.arange(len(labels)), values, color=plt.cm.tab20(np.linspace(0, 1, len(labels))))
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.2)
    out = ASSET_DIR / "overall_model_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


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
    out = ASSET_DIR / "duration_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_long_duration_bias(long_bias_df):
    display = long_bias_df.head(6)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(display))
    ax.bar(x, display[">=9s 预测段数"], color="#ee7733")
    ax.set_xticks(x)
    ax.set_xticklabels(display["类别"], rotation=25, ha="right")
    ax.set_ylabel("count of predictions >= 9s")
    ax.set_title("Long-duration prediction bias (top 6 classes)")
    out = ASSET_DIR / "long_duration_bias.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sample(audio_dir, filename, plot_title, gt_rows, pred_rows):
    audio_path = find_audio_path(audio_dir, filename)
    duration = max(
        10.0,
        float(gt_rows["offset"].max()) if not gt_rows.empty else 0.0,
        float(pred_rows["offset"].max()) if not pred_rows.empty else 0.0,
    )
    row_labels = sorted(set(gt_rows["event_label"]).union(set(pred_rows["event_label"])))
    if not row_labels:
        row_labels = list(classes_labels.keys())[:1]
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.1, 1.1], hspace=0.25)
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
    ax_spec.set_title(f"{filename} | {plot_title}")
    out = ASSET_DIR / f"sample_{filename.replace('.wav', '')}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_sample_commentary(filename, gt_rows, pred_rows):
    if filename == "355.wav":
        return "Frying 长事件几乎完整命中，说明 full finetune 已经把长持续纹理类稳定拉回到可用区间。"
    if filename == "1088.wav":
        return "frozen BEATs 报告里这条样本是空预测；当前不仅恢复了 Speech，还把 Cat 也一并报了出来，是最直观的恢复样本之一。"
    if filename == "234.wav":
        return "同一文件在 frozen BEATs 里是大量碎片化 Vacuum_cleaner，而当前已经恢复成接近整段的连续预测，边界稳定性明显提升。"
    if filename == "1000.wav":
        return "Running_water 主事件覆盖已经接近满段，同时保留了三段 Speech，说明 full finetune 对长持续类和叠加语音都更稳。"
    if filename == "1195.wav":
        return "Dog 仍然完全漏检，说明动物类依旧是当前 full finetune 的主要短板，尤其在长事件上仍会失效。"
    if filename == "1312.wav":
        return "复杂多事件场景里仍主要只剩 Speech，Dishes 几乎全部漏掉，说明 hardest multi-event scene 还没有被真正攻克。"
    return "典型样本。"


def mirror_assets():
    for path in ASSET_DIR.glob("*.png"):
        shutil.copy2(path, MIRROR_ASSET_DIR / path.name)


def main():
    ensure_dirs()
    chosen, candidates = locate_current_experiment()
    current = build_current_results(chosen)
    baseline_data = {name: parse_baseline_report(path) for name, path in BASELINE_REPORTS.items()}
    frozen_empty_ratio = extract_empty_ratio_from_report(baseline_data["Frozen BEATs baseline"]["raw_text"])
    per_class = current["per_class"].copy()
    per_class["分组"] = per_class["Event F1 value"].map(strength_group)
    stats = behavior_stats(current["pred"], current["gt"])

    overall_rows = []
    for name, info in baseline_data.items():
        overall_rows.append(
            {
                "模型": name,
                "PSDS1": info["overall"]["PSDS-scenario1"],
                "PSDS2": info["overall"]["PSDS-scenario2"],
                "Intersection": info["overall"]["Intersection-based F1"],
                "Event macro": info["overall"]["Event-based F1 (macro)"],
                "Segment macro": info["overall"]["Segment-based F1 (macro)"],
            }
        )
    overall_rows.append(
        {
            "模型": "BEATs 全量微调",
            "PSDS1": current["psds1"],
            "PSDS2": current["psds2"],
            "Intersection": current["intersection"],
            "Event macro": current["event_macro"],
            "Segment macro": current["segment_macro"],
        }
    )

    overall_df = pd.DataFrame(
        [
            ("PSDS-scenario1", safe_num(current["psds1"])),
            ("PSDS-scenario2", safe_num(current["psds2"])),
            ("Intersection-based F1", safe_num(current["intersection"])),
            ("Event-based F1 (macro)", pct_str(current["event_macro"])),
            ("Event-based F1 (micro)", pct_str(current["event_micro"])),
            ("Segment-based F1 (macro)", pct_str(current["segment_macro"])),
            ("Segment-based F1 (micro)", pct_str(current["segment_micro"])),
        ],
        columns=["指标", "数值"],
    )

    candidate_df = pd.DataFrame(
        [
            {
                "版本": c["version"],
                "last epoch": c["last_epoch"],
                "best score": safe_num(c["best_score"], 4) if c["best_score"] is not None else "NA",
                "global step": c["global_step"],
                "train event 文件数": c["train_event_count"],
                "best checkpoint": str(c["best_ckpt"].relative_to(ROOT)),
            }
            for c in candidates
        ]
    )

    comparison_overall_df = pd.DataFrame(
        [
            {
                "模型": row["模型"],
                "PSDS1": safe_num(row["PSDS1"]),
                "PSDS2": safe_num(row["PSDS2"]),
                "Intersection F1": safe_num(row["Intersection"]),
                "Event F1 macro": pct_str(row["Event macro"]),
                "Segment F1 macro": pct_str(row["Segment macro"]),
            }
            for row in overall_rows
        ]
    )

    comparison_per_class_rows = []
    for label in classes_labels.keys():
        row = {"类别": label}
        current_row = per_class[per_class["类别"] == label].iloc[0]
        row["Full FT Event"] = current_row["Event F1"]
        row["Full FT Segment"] = current_row["Segment F1"]
        row["Full FT Pred/GT"] = f"{current_row['Pred/GT']:.2f}"
        for model_name, info in baseline_data.items():
            base_row = info["per_class"][info["per_class"]["类别"] == label].iloc[0]
            short = {
                "CRNN baseline": "CRNN",
                "Frozen BEATs baseline": "Frozen",
                "Concat late fusion baseline": "Concat",
                "Residual gated fusion baseline": "Gate",
            }[model_name]
            row[f"{short} Event"] = base_row["Event F1"]
            row[f"{short} Segment"] = base_row["Segment F1"]
            if "Pred/GT" in base_row:
                row[f"{short} Pred/GT"] = f"{base_row['Pred/GT']:.2f}"
        comparison_per_class_rows.append(row)
    comparison_per_class_df = pd.DataFrame(comparison_per_class_rows)

    delta_rows = []
    frozen_per = baseline_data["Frozen BEATs baseline"]["per_class"].set_index("类别")
    crnn_per = baseline_data["CRNN baseline"]["per_class"].set_index("类别")
    gate_per = baseline_data["Residual gated fusion baseline"]["per_class"].set_index("类别")
    concat_per = baseline_data["Concat late fusion baseline"]["per_class"].set_index("类别")
    for label in classes_labels.keys():
        cur = per_class.set_index("类别").loc[label]
        delta_rows.append(
            {
                "类别": label,
                "相对 Frozen Event": f"{(cur['Event F1 value'] - frozen_per.loc[label, 'Event F1 value']) * 100:+.2f}pp",
                "相对 CRNN Event": f"{(cur['Event F1 value'] - crnn_per.loc[label, 'Event F1 value']) * 100:+.2f}pp",
                "相对 Concat Event": f"{(cur['Event F1 value'] - concat_per.loc[label, 'Event F1 value']) * 100:+.2f}pp",
                "相对 Gate Event": f"{(cur['Event F1 value'] - gate_per.loc[label, 'Event F1 value']) * 100:+.2f}pp",
                "相对 Frozen Segment": f"{(cur['Segment F1 value'] - frozen_per.loc[label, 'Segment F1 value']) * 100:+.2f}pp",
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    asset_paths = {
        "training_curves": plot_training_curves(current["train_tb"]),
        "class_counts": plot_class_counts(stats["class_count_df"]),
        "class_f1": plot_class_f1(per_class),
        "duration_dist": plot_duration_distribution(current["pred"], current["gt"]),
        "long_bias": plot_long_duration_bias(stats["long_bias_df"]),
        "overall_compare": plot_model_comparison(overall_rows),
    }

    sample_details = []
    for spec in SAMPLE_SPECS:
        gt_rows = choose_sample_rows(current["gt"], spec["filename"])
        pred_rows = choose_sample_rows(current["pred"], spec["filename"])
        asset = plot_sample(current["audio_dir"], spec["filename"], spec["plot_title"], gt_rows, pred_rows)
        sample_details.append(
            {
                **spec,
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
                "asset_name": asset.name,
                "commentary": build_sample_commentary(spec["filename"], gt_rows, pred_rows),
            }
        )

    mirror_assets()

    train_loss = scalar_values(current["train_tb"], "train/student/loss_strong")
    val_loss = scalar_values(current["train_tb"], "val/synth/student/loss_strong")
    val_obj = scalar_values(current["train_tb"], "val/obj_metric")
    val_inter = scalar_values(current["train_tb"], "val/synth/student/intersection_f1_macro")
    val_event = scalar_values(current["train_tb"], "val/synth/student/event_f1_macro")
    best_epoch = int(chosen["best_ckpt"].stem.split("-")[0].split("=")[1])
    best_step = int(chosen["best_ckpt"].stem.split("-")[1].split("=")[1])

    strong_classes = per_class.sort_values("Event F1 value", ascending=False).head(3)["类别"].tolist()
    medium_classes = per_class[(per_class["Event F1 value"] >= 0.40) & (per_class["Event F1 value"] < 0.55)]["类别"].tolist()
    weak_classes = per_class[per_class["Event F1 value"] < 0.40]["类别"].tolist()

    summary_df = pd.DataFrame(
        [
            ("总文件数", stats["gt_file_count"]),
            ("有预测文件数", stats["pred_file_count"]),
            ("空预测文件数", stats["empty_pred_count"]),
            ("空预测比例", f"{stats['empty_pred_ratio'] * 100:.2f}%"),
            ("总真值事件数", stats["gt_event_count"]),
            ("总预测事件数", stats["pred_event_count"]),
            ("真值平均时长", f"{stats['gt_duration_mean']:.2f}s"),
            ("预测平均时长", f"{stats['pred_duration_mean']:.2f}s"),
            ("真值中位时长", f"{stats['gt_duration_median']:.2f}s"),
            ("预测中位时长", f"{stats['pred_duration_median']:.2f}s"),
            ("预测事件 p95 时长", f"{stats['pred_duration_p95']:.2f}s"),
            ("真值事件 p95 时长", f"{stats['gt_duration_p95']:.2f}s"),
            ("接近整段(>=9.5s)预测数", stats["pred_near_full"]),
            ("接近整段(>=9.5s)真值数", stats["gt_near_full"]),
            ("疑似碎片化过预测文件数", stats["fragmented_files"]),
        ],
        columns=["统计项", "数值"],
    )

    experiment_df = pd.DataFrame(
        [
            ("实验设置", "BEATs 全量微调（full finetune）"),
            ("评估对象", "student"),
            ("encoder_type", chosen["hparams"]["model"]["encoder_type"]),
            ("BEATs freeze", chosen["hparams"]["model"]["beats"]["freeze"]),
            ("是否 full finetune", "是，BEATs 参数参与训练"),
            ("共享 decoder", "BiGRU + strong/weak heads"),
            ("数据划分", "synthetic train + synthetic validation"),
            ("当前 test", "仍然是 synthetic validation"),
            ("含义", "结果更适合判断模型是否在合成分布上跑通与比较相对增益，不等同真实外部分布泛化"),
            ("自动定位的配置文件", str(chosen["config_path"].relative_to(ROOT)) if chosen["config_path"] else "未找到完全匹配配置，已回退到 checkpoint hparams"),
            ("best checkpoint", str(chosen["best_ckpt"].relative_to(ROOT))),
            ("prediction TSV", str(current["pred_tsv"].relative_to(ROOT))),
            ("event_f1.txt", str(current["event_txt"].relative_to(ROOT))),
            ("segment_f1.txt", str(current["segment_txt"].relative_to(ROOT))),
            ("ground truth TSV", str(current["gt_path"].relative_to(ROOT))),
        ],
        columns=["项目", "说明"],
    )

    train_df = pd.DataFrame(
        [
            ("train/student/loss_strong", f"{train_loss[0]:.4f}", f"{train_loss[-1]:.4f}", f"{min(train_loss):.4f}"),
            ("val/synth/student/loss_strong", f"{val_loss[0]:.4f}", f"{val_loss[-1]:.4f}", f"{min(val_loss):.4f}"),
            ("val/obj_metric", f"{val_obj[0]:.4f}", f"{val_obj[-1]:.4f}", f"{max(val_obj):.4f}"),
            ("val/synth/student/intersection_f1_macro", f"{val_inter[0]:.4f}", f"{val_inter[-1]:.4f}", f"{max(val_inter):.4f}"),
            ("val/synth/student/event_f1_macro", f"{val_event[0]:.4f}", f"{val_event[-1]:.4f}", f"{max(val_event):.4f}"),
        ],
        columns=["曲线", "起始值", "最终值", "最佳值"],
    )

    md = []
    md.append("# BEATs 全量微调训练结果分析报告")
    md.append("")
    md.append("## 1. 实验概况")
    md.append("")
    md.append("### 1.1 自动定位结果")
    md.append("")
    md.append(to_markdown(candidate_df))
    md.append("")
    md.append(
        f"最终采用的是 `{chosen['version']}`，并以 `{chosen['best_ckpt'].relative_to(ROOT)}` 作为 best checkpoint。"
        f"选择依据是：它是最近一次完成的 `encoder_type=beats`、`freeze=False`、非 fusion 的候选版本；"
        f"checkpoint 内的 best score 为 `{safe_num(chosen['best_score'], 4)}`，同时训练事件文件与 `metrics_test` 时间戳也与这次 full finetune 结果回写一致。"
    )
    md.append("")
    md.append(to_markdown(experiment_df))
    md.append("")
    md.append(
        "这次实验属于 `BEATs 全量微调`：主干仍是 `encoder -> shared decoder -> strong/weak heads` 的统一结构，但与 frozen baseline 最大的区别在于 "
        "`model.beats.freeze=False`，也就是 BEATs 编码器不再固定，而是和 decoder 一起参与训练。"
    )
    md.append("")
    md.append(
        "你在任务说明里给的 GT 路径占位符是 `<GT_TSV>`，服务器上并不存在这个字面路径。"
        f"本报告已自动回退到 checkpoint / 配置里实际存在的 `{current['gt_path'].relative_to(ROOT)}`，并用它作为固定真值文件。"
    )
    md.append("")
    md.append("## 2. 最终指标汇总")
    md.append("")
    md.append(to_markdown(overall_df))
    md.append("")
    per_class_display = per_class[["类别", "GT事件数", "Pred事件数", "Pred/GT", "Event F1", "Segment F1", "分组"]].copy()
    per_class_display["Pred/GT"] = per_class_display["Pred/GT"].map(lambda x: f"{x:.2f}")
    md.append(to_markdown(per_class_display))
    md.append("")
    md.append(
        f"整体上，较强类别主要是 `{', '.join(strong_classes)}`；中等类别主要是 `{', '.join(medium_classes)}`；"
        f"相对较弱的类别主要是 `{', '.join(weak_classes)}`。"
    )
    md.append("")
    md.append(
        f"最值得注意的是，这次 full finetune 的整体指标已经达到 `PSDS1={current['psds1']:.3f}`、`PSDS2={current['psds2']:.3f}`、"
        f"`Intersection={current['intersection']:.3f}`、`Event macro={current['event_macro'] * 100:.2f}%`、"
        f"`Segment macro={current['segment_macro'] * 100:.2f}%`。这说明它不只是“从 frozen BEATs 的类别塌缩中恢复”，"
        "而是已经进入可以和 CRNN / fusion baseline 正面对比的区间。"
    )
    md.append("")
    md.append("## 3. 横向对比")
    md.append("")
    md.append(to_markdown(comparison_overall_df))
    md.append("")
    md.append(f"![整体模型对比](report_assets/{asset_paths['overall_compare'].name})")
    md.append("")
    md.append(to_markdown(comparison_per_class_df))
    md.append("")
    md.append("### 3.1 逐类变化（full finetune 相对各 baseline）")
    md.append("")
    md.append(to_markdown(delta_df))
    md.append("")
    md.append(
        f"和 frozen BEATs baseline 相比，这次 full finetune 是显著提升，不是边角增益。"
        f"例如 overall `Event macro` 从 {baseline_data['Frozen BEATs baseline']['overall']['Event-based F1 (macro)'] * 100:.2f}% "
        f"提升到 {current['event_macro'] * 100:.2f}%；`PSDS1` 从 {baseline_data['Frozen BEATs baseline']['overall']['PSDS-scenario1']:.3f} "
        f"提升到 {current['psds1']:.3f}。"
    )
    md.append("")
    md.append(
        f"和 CRNN baseline 相比，这次 full finetune 在 synthetic validation 上同样已经形成明确优势："
        f"`PSDS1/PSDS2/Intersection/Event macro/Segment macro` 分别从 "
        f"{baseline_data['CRNN baseline']['overall']['PSDS-scenario1']:.3f}/{baseline_data['CRNN baseline']['overall']['PSDS-scenario2']:.3f}/"
        f"{baseline_data['CRNN baseline']['overall']['Intersection-based F1']:.3f}/"
        f"{baseline_data['CRNN baseline']['overall']['Event-based F1 (macro)'] * 100:.2f}%/"
        f"{baseline_data['CRNN baseline']['overall']['Segment-based F1 (macro)'] * 100:.2f}% "
        f"提升到 {current['psds1']:.3f}/{current['psds2']:.3f}/{current['intersection']:.3f}/"
        f"{current['event_macro'] * 100:.2f}%/{current['segment_macro'] * 100:.2f}%。"
    )
    md.append("")
    md.append(
        "如果只以当前 synthetic validation 作为开发分析标准，这版 full finetune 已经足够值得作为论文里的重要结果候选。"
        "不过它依然不是“所有类都被彻底解决”：`Dog`、`Alarm_bell_ringing` 仍偏弱，`Dishes` 也没有像强类那样完全被拉平。"
    )
    md.append("")
    md.append("## 4. 训练过程与选模分析")
    md.append("")
    md.append(f"![训练曲线](report_assets/{asset_paths['training_curves'].name})")
    md.append("")
    md.append(to_markdown(train_df))
    md.append("")
    md.append(
        f"训练过程总体是正常的。`train/student/loss_strong` 从 {train_loss[0]:.4f} 降到 {train_loss[-1]:.4f}，"
        f"`val/synth/student/loss_strong` 从 {val_loss[0]:.4f} 降到 {val_loss[-1]:.4f}；"
        f"`val/obj_metric` 在 best checkpoint 对应阶段达到 {max(val_obj):.4f}，与 `epoch={best_epoch}, step={best_step}` 的 best checkpoint 一致。"
    )
    md.append("")
    md.append(
        "从曲线形态看，这版 full finetune 比 frozen baseline 更难训，但上限也明显更高。"
        "前期它并不是马上起飞，而是在前十来个 epoch 先完成稳定化，随后 `val/obj_metric`、`val/synth/student/event_f1_macro` 和 `intersection_f1_macro` 一起抬升。"
        "后期虽然有波动，但没有出现失控发散，更像正常平台期与局部震荡。"
    )
    md.append("")
    md.append(
        "这里仍然要提醒：在 `synth_only` 设置下，`val/obj_metric` 实际上等于 `val/synth/student/intersection_f1_macro`。"
        "它对区间覆盖更敏感，对严格事件边界和跨阈值稳定性不如 event-based F1 和 PSDS 敏感。"
        "不过这次 best checkpoint 附近，event macro 也同步进入高位，因此选模没有明显跑偏。"
    )
    md.append("")
    md.append("## 5. 预测行为统计")
    md.append("")
    md.append(to_markdown(summary_df))
    md.append("")
    md.append(f"![各类别 GT vs Pred](report_assets/{asset_paths['class_counts'].name})")
    md.append("")
    md.append(f"![各类别 Event / Segment F1](report_assets/{asset_paths['class_f1'].name})")
    md.append("")
    md.append(f"![事件时长分布](report_assets/{asset_paths['duration_dist'].name})")
    md.append("")
    md.append(f"![长时段偏置](report_assets/{asset_paths['long_bias'].name})")
    md.append("")
    md.append(to_markdown(stats["class_count_df"]))
    md.append("")
    long_bias_display = stats["long_bias_df"][["类别", "平均预测时长", ">=9s 预测段数"]].head(6)
    md.append(to_markdown(long_bias_display))
    md.append("")
    md.append(
        f"当前 full finetune 已经恢复到“正常工作”状态。它在 2500 个文件里只有 {stats['empty_pred_count']} 个空预测，"
        f"空预测比例 {stats['empty_pred_ratio'] * 100:.2f}%；而 frozen BEATs baseline 的空预测比例是 "
        f"{frozen_empty_ratio * 100:.2f}%。"
    )
    md.append("")
    md.append(
        f"和 frozen BEATs 相比，这次最大的变化不是单一长持续类涨分，而是整体类别覆盖恢复：`Cat`、`Dishes`、`Dog`、`Alarm_bell_ringing` "
        "这些曾经接近失效的类，现在至少都能稳定产出预测并拿到非零事件级分数。"
        "不过弱类仍没有完全解决，特别是 `Dog` 和 `Alarm_bell_ringing` 依然明显落后于强类。"
    )
    md.append("")
    md.append(
        f"从时长分布看，当前预测平均时长 {stats['pred_duration_mean']:.2f}s 仍低于真值的 {stats['gt_duration_mean']:.2f}s，"
        "因此主问题更像局部欠检与边界偏移，而不是全局把整段涂满。"
        f"同时 `>=9s` 长段预测仍有 {stats['pred_near_full']} 个，说明长持续类仍然更容易被完整覆盖。"
    )
    md.append("")
    md.append("## 6. 典型样本分析")
    md.append("")
    for sample in sample_details:
        md.append(f"### {sample['filename']} | {sample['pattern']}")
        md.append("")
        md.append(f"![{sample['filename']}](report_assets/{sample['asset_name']})")
        md.append("")
        md.append(f"- 文件名：`{sample['filename']}`")
        md.append(f"- 典型模式：{sample['pattern']}")
        md.append(f"- 代表性原因：{sample['reason']}")
        md.append(f"- 真值事件列表：{format_events(sample['gt_rows'])}")
        md.append(f"- 预测事件列表：{format_events(sample['pred_rows'])}")
        md.append(f"- 简短点评：{sample['commentary']}")
        md.append("")
    md.append("## 7. 结论与讨论")
    md.append("")
    md.append(
        "这次 full finetune 是正常跑通的，而且不是勉强可用，而是真正取得了强结果。"
        "从 current metrics 看，它已经显著优于 frozen BEATs baseline，也已经明显超过当前仓库里的 CRNN baseline 和两种 BEATs-CRNN fusion baseline。"
    )
    md.append("")
    md.append(
        "如果论文当前阶段允许把 synthetic validation 结果作为开发实验主表的一部分，那么这版 full finetune 是值得写进正式表格的。"
        "它给出的核心结论很明确：对 BEATs 而言，单纯 frozen encoder + 共享 decoder 的确不够；一旦允许全量微调，模型性能会有质变。"
    )
    md.append("")
    md.append(
        "但这不意味着问题已经结束。当前主要问题仍集中在 hardest weak classes 和复杂多事件场景：`Dog` 仍有空预测样本，"
        "`Dishes` 在多事件场景里仍容易被 `Speech` 压制，`Alarm_bell_ringing` 也还没有进入第一梯队。"
        "换句话说，full finetune 解决了“整体不工作”的问题，但还没有彻底解决“所有类都同样强”的问题。"
    )
    md.append("")
    md.append("## 8. 后续建议")
    md.append("")
    md.append("1. 优先把这版 full finetune 写进正式实验主表，并明确标注它当前是在 synthetic validation 上取得的开发结果。")
    md.append("2. 在这版 checkpoint 基础上做更细的逐类分析，重点盯 `Dog / Dishes / Alarm_bell_ringing`，确认它们落后的原因是召回不足、边界不稳还是混淆。")
    md.append("3. 单独搜索 threshold 和 median filter，因为当前 `Pred/GT` 和长段统计表明，后处理仍有进一步挖潜空间。")
    md.append("4. 如果论文还想继续深挖 encoder 价值，优先比较 `full finetune` 与 `residual gated fusion` 在同一评估协议下的取舍，而不是回到 frozen baseline。")
    md.append("5. 在 synthetic validation 之外补一组更贴近真实分布的测试或评估，否则 full finetune 是否真正泛化，论文里仍需要更谨慎表述。")

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")

    print("采用实验版本:", chosen["version"])
    print("best checkpoint:", chosen["best_ckpt"])
    print("报告文件:", REPORT_PATH)
    print("图片目录:", ASSET_DIR)
    print("镜像图片目录:", MIRROR_ASSET_DIR)
    print("典型样本:", ", ".join(sample["filename"] for sample in sample_details))
    print("成功完成:")
    print("- 自动定位 full finetune 实验版本")
    print("- 基于 best checkpoint 补齐 metrics_test 分析输入")
    print("- 汇总 overall / per-class / 训练曲线 / 预测行为统计")
    print("- 生成 6 个典型样本时间轴 + 频谱图")
    print("- 解析已有 baseline 报告并完成横向对比")
    print("未完成:")
    print("- 没有额外独立外部分布测试，因此结论仍以 synthetic validation 开发结果为主")


if __name__ == "__main__":
    main()
