import math
import os
import re
import sys
import tempfile
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

from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_sed_eval_metrics,
)
from local.classes_dict import classes_labels
from local.utils import log_sedeval_metrics

REPORT_DIR = ROOT / "WAVLM-baseline"
ASSET_DIR = REPORT_DIR / "report_assets"
REPORT_PATH = REPORT_DIR / "training_result_report.md"

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

GT_COUNTS = {
    "Alarm_bell_ringing": 431,
    "Blender": 266,
    "Cat": 429,
    "Dishes": 1309,
    "Dog": 550,
    "Electric_shaver_toothbrush": 286,
    "Frying": 377,
    "Running_water": 306,
    "Speech": 3927,
    "Vacuum_cleaner": 251,
}

BASELINES = {
    "CRNN baseline": {
        "overall": {
            "PSDS1": 0.356,
            "PSDS2": 0.578,
            "Intersection F1": 0.650,
            "Event F1 macro": 43.42,
            "Event F1 micro": 43.14,
            "Segment F1 macro": 71.25,
            "Segment F1 micro": 75.70,
        },
        "classwise": {
            "Alarm_bell_ringing": {"pred": 338, "event": 21.07, "segment": 64.04},
            "Blender": {"pred": 198, "event": 43.10, "segment": 63.83},
            "Cat": {"pred": 455, "event": 29.86, "segment": 73.48},
            "Dishes": {"pred": 637, "event": 28.57, "segment": 50.55},
            "Dog": {"pred": 380, "event": 32.69, "segment": 59.67},
            "Electric_shaver_toothbrush": {"pred": 260, "event": 48.35, "segment": 84.23},
            "Frying": {"pred": 360, "event": 65.94, "segment": 83.89},
            "Running_water": {"pred": 349, "event": 49.47, "segment": 71.40},
            "Speech": {"pred": 3986, "event": 46.86, "segment": 80.20},
            "Vacuum_cleaner": {"pred": 288, "event": 68.27, "segment": 81.20},
        },
        "behavior": {
            "pred_files": 2468,
            "empty_files": 32,
            "empty_ratio": 1.28,
            "pred_events": 7251,
        },
    },
    "Frozen BEATs baseline": {
        "overall": {
            "PSDS1": 0.001,
            "PSDS2": 0.051,
            "Intersection F1": 0.432,
            "Event F1 macro": 8.58,
            "Event F1 micro": 15.34,
            "Segment F1 macro": 45.74,
            "Segment F1 micro": 53.08,
        },
        "classwise": {
            "Alarm_bell_ringing": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Blender": {"pred": 13, "event": 0.00, "segment": 0.00},
            "Cat": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Dishes": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Dog": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Electric_shaver_toothbrush": {"pred": 209, "event": 17.37, "segment": 51.24},
            "Frying": {"pred": 440, "event": 37.94, "segment": 64.45},
            "Running_water": {"pred": 184, "event": 0.00, "segment": 32.89},
            "Speech": {"pred": 4341, "event": 19.81, "segment": 73.94},
            "Vacuum_cleaner": {"pred": 367, "event": 10.68, "segment": 51.94},
        },
        "behavior": {
            "pred_files": 2379,
            "empty_files": 121,
            "empty_ratio": 4.84,
            "pred_events": 5554,
        },
    },
    "Concat late fusion": {
        "overall": {
            "PSDS1": 0.306,
            "PSDS2": 0.484,
            "Intersection F1": 0.583,
            "Event F1 macro": 41.37,
            "Event F1 micro": 40.63,
            "Segment F1 macro": 64.00,
            "Segment F1 micro": 72.13,
        },
        "classwise": {
            "Alarm_bell_ringing": {"pred": 250, "event": 21.20, "segment": 59.70},
            "Blender": {"pred": 302, "event": 48.70, "segment": 66.80},
            "Cat": {"pred": 318, "event": 34.60, "segment": 70.60},
            "Dishes": {"pred": 208, "event": 15.50, "segment": 21.70},
            "Dog": {"pred": 139, "event": 15.90, "segment": 33.10},
            "Electric_shaver_toothbrush": {"pred": 266, "event": 46.40, "segment": 80.70},
            "Frying": {"pred": 406, "event": 67.00, "segment": 83.30},
            "Running_water": {"pred": 212, "event": 52.70, "segment": 71.70},
            "Speech": {"pred": 3788, "event": 44.00, "segment": 77.60},
            "Vacuum_cleaner": {"pred": 204, "event": 67.70, "segment": 74.80},
        },
        "behavior": {
            "pred_files": 2430,
            "empty_files": 70,
            "empty_ratio": 2.80,
            "pred_events": 6093,
        },
    },
    "Residual gated fusion": {
        "overall": {
            "PSDS1": 0.364,
            "PSDS2": 0.599,
            "Intersection F1": 0.669,
            "Event F1 macro": 45.91,
            "Event F1 micro": 45.16,
            "Segment F1 macro": 72.95,
            "Segment F1 micro": 78.26,
        },
        "classwise": {
            "Alarm_bell_ringing": {"pred": 314, "event": 24.16, "segment": 67.81},
            "Blender": {"pred": 276, "event": 55.35, "segment": 76.81},
            "Cat": {"pred": 499, "event": 33.41, "segment": 78.42},
            "Dishes": {"pred": 470, "event": 29.68, "segment": 44.78},
            "Dog": {"pred": 359, "event": 29.26, "segment": 58.80},
            "Electric_shaver_toothbrush": {"pred": 286, "event": 53.85, "segment": 84.11},
            "Frying": {"pred": 373, "event": 67.20, "segment": 81.41},
            "Running_water": {"pred": 264, "event": 50.53, "segment": 69.78},
            "Speech": {"pred": 4337, "event": 48.38, "segment": 84.06},
            "Vacuum_cleaner": {"pred": 284, "event": 67.29, "segment": 83.53},
        },
        "behavior": {
            "pred_files": 2476,
            "empty_files": 24,
            "empty_ratio": 0.96,
            "pred_events": 7462,
        },
    },
}

WEAK_CLASSES = {"Dog", "Cat", "Dishes", "Alarm_bell_ringing"}
LONG_DURATION_CLASSES = {
    "Frying",
    "Vacuum_cleaner",
    "Running_water",
    "Blender",
    "Electric_shaver_toothbrush",
}


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
        widths[col] = max(len(col), *(len(v) for v in display_df[col].tolist())) if len(display_df) else len(col)
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


def scalar_points(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(int(item.step), float(item.value)) for item in ea.Scalars(tag)]


def step_to_epoch(step, steps_per_epoch):
    return ((step + 1) / steps_per_epoch) - 1


def detect_wavlm_versions():
    exp_dir = ROOT / "exp/2022_baseline"
    candidates = []
    for version_dir in sorted(exp_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[-1])):
        last_ckpt = version_dir / "last.ckpt"
        if not last_ckpt.exists():
            continue
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("sed_student", {})
        if not any("wavlm" in key for key in state.keys()):
            continue
        callbacks = ckpt.get("callbacks", {})
        best_model_path = None
        best_model_score = None
        for key, value in callbacks.items():
            if "ModelCheckpoint" in key:
                best_model_path = value.get("best_model_path")
                best_model_score = value.get("best_model_score")
                break
        train_events = sorted(version_dir.glob("events.out.tfevents.*.0")) or sorted(version_dir.glob("events.out.tfevents.*"))
        test_events = sorted(version_dir.glob("events.out.tfevents.*.1"))
        candidates.append(
            {
                "version": version_dir.name,
                "version_num": int(version_dir.name.split("_")[-1]),
                "dir": version_dir,
                "mtime": version_dir.stat().st_mtime,
                "epoch": int(ckpt.get("epoch", -1)),
                "step": int(ckpt.get("global_step", -1)),
                "best_model_path": best_model_path,
                "best_model_score": float(best_model_score) if best_model_score is not None else None,
                "train_events": train_events,
                "test_events": test_events,
            }
        )
    if not candidates:
        raise FileNotFoundError("Could not locate any WavLM experiment under exp/2022_baseline.")
    candidates.sort(key=lambda item: item["version_num"])
    final = max(candidates, key=lambda item: (item["best_model_score"] or -1, item["version_num"]))
    return candidates, final


def locate_config():
    candidates = []
    for path in ROOT.glob("confs/*.yaml"):
        try:
            config = load_yaml(path)
        except Exception:
            continue
        model = config.get("model", {}) if isinstance(config, dict) else {}
        if model.get("encoder_type") == "wavlm" or "wavlm" in str(model.get("model_type", "")).lower():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Could not locate WavLM config under confs/*.yaml")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def locate_prediction_tsv():
    path = ROOT / "exp/2022_baseline/metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"
    if not path.exists():
        raise FileNotFoundError("Could not locate scenario1 threshold 0.49 prediction TSV.")
    return path


def load_prediction_operating_points(prediction_tsv):
    scenario1_dir = prediction_tsv.parent
    threshold_files = sorted(scenario1_dir.glob("predictions_th_*.tsv"))
    operating_points = {}
    for path in threshold_files:
        m = re.search(r"predictions_th_([0-9.]+)\.tsv$", path.name)
        if not m:
            continue
        operating_points[float(m.group(1))] = pd.read_csv(path, sep="\t")
    return operating_points


def classify_status(event_score):
    if event_score >= 0.50:
        return "较强"
    if event_score >= 0.30:
        return "中等"
    return "较弱"


def compute_current_metrics(prediction_tsv, config):
    pred = pd.read_csv(prediction_tsv, sep="\t")
    gt = pd.read_csv(GROUND_TRUTH_TSV, sep="\t")
    pred["duration"] = pred["offset"] - pred["onset"]
    gt["duration"] = gt["offset"] - gt["onset"]

    event_metric, segment_metric = compute_sed_eval_metrics(pred, gt)
    event_res = event_metric.results()
    segment_res = segment_metric.results()

    dur_tsv = config["data"]["test_dur"]
    operating_points = load_prediction_operating_points(prediction_tsv)
    intersection = float(
        compute_per_intersection_macro_f1({"0.49": pred}, str(GROUND_TRUTH_TSV), dur_tsv)
    )
    psds_ops = {str(th): df for th, df in operating_points.items() if not df.empty}
    psds1 = float(
        compute_psds_from_operating_points(
            psds_ops,
            str(GROUND_TRUTH_TSV),
            dur_tsv,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=0.3,
            alpha_ct=0,
            alpha_st=1,
            save_dir=None,
        )
    )
    psds2 = float(
        compute_psds_from_operating_points(
            psds_ops,
            str(GROUND_TRUTH_TSV),
            dur_tsv,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            save_dir=None,
        )
    )

    with tempfile.TemporaryDirectory() as td:
        log_sedeval_metrics(pred, GROUND_TRUTH_TSV, save_dir=td)
        event_txt = Path(td) / "event_f1.txt"
        segment_txt = Path(td) / "segment_f1.txt"
        event_txt_text = event_txt.read_text(encoding="utf-8")
        segment_txt_text = segment_txt.read_text(encoding="utf-8")

    overall_df = pd.DataFrame(
        [
            ("PSDS-scenario1", safe_num(psds1)),
            ("PSDS-scenario2", safe_num(psds2)),
            ("Intersection-based F1", safe_num(intersection)),
            ("Event-based F1 (macro)", safe_pct(event_res["class_wise_average"]["f_measure"]["f_measure"])),
            ("Event-based F1 (micro)", safe_pct(event_res["overall"]["f_measure"]["f_measure"])),
            ("Segment-based F1 (macro)", safe_pct(segment_res["class_wise_average"]["f_measure"]["f_measure"])),
            ("Segment-based F1 (micro)", safe_pct(segment_res["overall"]["f_measure"]["f_measure"])),
        ],
        columns=["指标", "数值"],
    )

    gt_counts = gt["event_label"].value_counts()
    pred_counts = pred["event_label"].value_counts()
    per_class_rows = []
    for label in classes_labels.keys():
        event_score = float(event_res["class_wise"][label]["f_measure"]["f_measure"])
        segment_score = float(segment_res["class_wise"][label]["f_measure"]["f_measure"])
        pred_n = int(pred_counts.get(label, 0))
        gt_n = int(gt_counts.get(label, 0))
        per_class_rows.append(
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
    per_class_df = pd.DataFrame(per_class_rows)
    return {
        "pred": pred,
        "gt": gt,
        "event_res": event_res,
        "segment_res": segment_res,
        "overall_df": overall_df,
        "per_class_df": per_class_df,
        "psds1": psds1,
        "psds2": psds2,
        "intersection": intersection,
        "event_txt": event_txt_text,
        "segment_txt": segment_txt_text,
    }


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

    summary_df = pd.DataFrame(
        [
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
        ],
        columns=["统计项", "数值"],
    )

    long_bias_df = (
        pred.groupby("event_label")["duration"]
        .mean()
        .rename("pred_mean_duration")
        .to_frame()
        .join(pred_long["event_label"].value_counts().rename("pred_ge_9s"), how="left")
        .fillna(0)
        .sort_values(["pred_ge_9s", "pred_mean_duration"], ascending=False)
        .reset_index()
        .rename(columns={"index": "类别", "event_label": "类别"})
    )
    long_bias_df["pred_mean_duration"] = long_bias_df["pred_mean_duration"].map(lambda x: f"{x:.2f}s")
    long_bias_df["pred_ge_9s"] = long_bias_df["pred_ge_9s"].astype(int)
    return {
        "summary_df": summary_df,
        "class_count_df": class_count_df.reset_index().rename(columns={"index": "类别", "event_label": "类别"}),
        "long_bias_df": long_bias_df,
        "empty_files": empty_files,
    }


def build_overall_comparison(current_metrics):
    rows = []
    for name, baseline in BASELINES.items():
        row = {"模型": name}
        row.update(baseline["overall"])
        rows.append(row)
    rows.append(
        {
            "模型": "WavLM-only baseline",
            "PSDS1": current_metrics["psds1"],
            "PSDS2": current_metrics["psds2"],
            "Intersection F1": current_metrics["intersection"],
            "Event F1 macro": current_metrics["event_res"]["class_wise_average"]["f_measure"]["f_measure"] * 100,
            "Event F1 micro": current_metrics["event_res"]["overall"]["f_measure"]["f_measure"] * 100,
            "Segment F1 macro": current_metrics["segment_res"]["class_wise_average"]["f_measure"]["f_measure"] * 100,
            "Segment F1 micro": current_metrics["segment_res"]["overall"]["f_measure"]["f_measure"] * 100,
        }
    )
    df = pd.DataFrame(rows)
    for col in ["PSDS1", "PSDS2", "Intersection F1"]:
        df[col] = df[col].map(lambda x: f"{x:.3f}")
    for col in ["Event F1 macro", "Event F1 micro", "Segment F1 macro", "Segment F1 micro"]:
        df[col] = df[col].map(lambda x: f"{x:.2f}%")
    return df


def build_per_class_comparison(current_per_class_df):
    rows = []
    current_map = {row["类别"]: row for _, row in current_per_class_df.iterrows()}
    for label in classes_labels.keys():
        cur = current_map[label]
        rows.append(
            {
                "类别": label,
                "GT": GT_COUNTS[label],
                "CRNN Event": f"{BASELINES['CRNN baseline']['classwise'][label]['event']:.2f}%",
                "BEATs Event": f"{BASELINES['Frozen BEATs baseline']['classwise'][label]['event']:.2f}%",
                "WavLM Event": cur["Event F1"],
                "Concat Event": f"{BASELINES['Concat late fusion']['classwise'][label]['event']:.2f}%",
                "Gate Event": f"{BASELINES['Residual gated fusion']['classwise'][label]['event']:.2f}%",
                "CRNN Segment": f"{BASELINES['CRNN baseline']['classwise'][label]['segment']:.2f}%",
                "BEATs Segment": f"{BASELINES['Frozen BEATs baseline']['classwise'][label]['segment']:.2f}%",
                "WavLM Segment": cur["Segment F1"],
                "Pred/GT (WavLM)": cur["Pred/GT"],
                "_wavlm_event": cur["_event"] * 100,
                "_wavlm_segment": cur["_segment"] * 100,
            }
        )
    return pd.DataFrame(rows)


def build_behavior_comparison(current_behavior):
    rows = []
    for name in ["CRNN baseline", "Frozen BEATs baseline", "Concat late fusion", "Residual gated fusion"]:
        b = BASELINES[name]["behavior"]
        rows.append(
            {
                "模型": name,
                "有预测文件数": b["pred_files"],
                "空预测文件数": b["empty_files"],
                "空预测比例": f"{b['empty_ratio']:.2f}%",
                "总预测事件数": b["pred_events"],
            }
        )
    summary_map = {row["统计项"]: row["数值"] for _, row in current_behavior["summary_df"].iterrows()}
    rows.append(
        {
            "模型": "WavLM-only baseline",
            "有预测文件数": summary_map["有预测文件数"],
            "空预测文件数": summary_map["空预测文件数"],
            "空预测比例": summary_map["空预测比例"],
            "总预测事件数": summary_map["总预测事件数"],
        }
    )
    return pd.DataFrame(rows)


def plot_training_curves(train_tb_file, steps_per_epoch, best_ckpt_step):
    tags = {
        "train/student/loss_strong": ("Train strong loss", "#cc3311"),
        "val/synth/student/loss_strong": ("Val strong loss", "#0077bb"),
        "val/obj_metric": ("Val obj metric", "#009988"),
        "val/synth/student/intersection_f1_macro": ("Val intersection F1", "#ee7733"),
        "val/synth/student/event_f1_macro": ("Val event macro F1", "#332288"),
    }
    ea = load_tb(train_tb_file)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    merged_cache = {}
    for idx, (tag, (title, color)) in enumerate(tags.items()):
        points = scalar_points(ea, tag)
        merged_cache[tag] = points
        if not points:
            axes[idx].axis("off")
            continue
        steps = np.array([p[0] for p in points], dtype=float)
        values = np.array([p[1] for p in points], dtype=float)
        epochs = np.array([step_to_epoch(step, steps_per_epoch) for step in steps], dtype=float)
        axes[idx].plot(epochs, values, marker="o" if "val/" in tag else None, markersize=3, color=color)
        axes[idx].axvline(step_to_epoch(best_ckpt_step, steps_per_epoch), linestyle="--", color="#999999", linewidth=1)
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Epoch")

    val_obj = merged_cache.get("val/obj_metric", [])
    val_event = merged_cache.get("val/synth/student/event_f1_macro", [])
    val_loss = merged_cache.get("val/synth/student/loss_strong", [])
    summary = [
        f"best ckpt epoch: {step_to_epoch(best_ckpt_step, steps_per_epoch):.0f}",
        f"best obj: {max(v for _, v in val_obj):.4f}" if val_obj else "best obj: NA",
        f"best event: {max(v for _, v in val_event):.4f}" if val_event else "best event: NA",
        f"best val loss: {min(v for _, v in val_loss):.4f}" if val_loss else "best val loss: NA",
    ]
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.98, "\n".join(summary), va="top", ha="left", family="monospace")
    out_path = ASSET_DIR / "wavlm_training_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path, merged_cache


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
    ax.set_title("WavLM-only: GT vs predicted event counts by class")
    ax.legend()
    out_path = ASSET_DIR / "wavlm_class_count_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_f1(current_per_class_df):
    labels = current_per_class_df["类别"].tolist()
    event_scores = current_per_class_df["_event"].to_numpy() * 100
    segment_scores = current_per_class_df["_segment"].to_numpy() * 100
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.bar(x - width / 2, event_scores, width=width, label="Event F1")
    ax.bar(x + width / 2, segment_scores, width=width, label="Segment F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("F1 (%)")
    ax.set_title("WavLM-only event / segment F1 by class")
    ax.legend()
    out_path = ASSET_DIR / "wavlm_class_f1_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_wavlm_vs_baselines(current_per_class_df):
    labels = current_per_class_df["类别"].tolist()
    wavlm_scores = current_per_class_df["_event"].to_numpy() * 100
    crnn_scores = np.array([BASELINES["CRNN baseline"]["classwise"][label]["event"] for label in labels])
    beats_scores = np.array([BASELINES["Frozen BEATs baseline"]["classwise"][label]["event"] for label in labels])
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.bar(x - width, crnn_scores, width=width, label="CRNN")
    ax.bar(x, beats_scores, width=width, label="BEATs")
    ax.bar(x + width, wavlm_scores, width=width, label="WavLM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Event F1 (%)")
    ax.set_title("Per-class event F1: CRNN vs BEATs vs WavLM")
    ax.legend()
    out_path = ASSET_DIR / "wavlm_vs_crnn_beats_event_f1.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_duration_distribution(pred, gt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    bins = np.linspace(0, 10, 41)
    axes[0].hist(gt["duration"], bins=bins, alpha=0.6, label="GT")
    axes[0].hist(pred["duration"], bins=bins, alpha=0.6, label="Prediction")
    axes[0].set_xlabel("duration (s)")
    axes[0].set_title("WavLM-only event duration distribution")
    axes[0].legend()
    axes[1].boxplot([gt["duration"], pred["duration"]], labels=["GT", "Prediction"], showfliers=False)
    axes[1].set_ylabel("duration (s)")
    axes[1].set_title("WavLM-only duration boxplot")
    out_path = ASSET_DIR / "wavlm_duration_distribution.png"
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
    ax1.set_title("WavLM-only long-duration prediction bias (top 6 classes)")
    ax2 = ax1.twinx()
    mean_durations = display_df["pred_mean_duration"].str.replace("s", "", regex=False).astype(float)
    ax2.plot(x, mean_durations, color="#0077bb", marker="o")
    ax2.set_ylabel("mean predicted duration (s)")
    out_path = ASSET_DIR / "wavlm_long_duration_bias.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def find_audio_path(filename):
    for candidate in [VAL_AUDIO_16K_DIR / filename, VAL_AUDIO_ORIG_DIR / filename]:
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


def framewise_mask(rows, fps=10, duration=10.0):
    n = int(duration * fps)
    mask = np.zeros((len(classes_labels), n), dtype=bool)
    label_to_idx = {label: idx for idx, label in enumerate(classes_labels.keys())}
    for row in rows.itertuples(index=False):
        start = max(0, int(math.floor(row.onset * fps)))
        end = min(n, int(math.ceil(row.offset * fps)))
        if end > start:
            mask[label_to_idx[row.event_label], start:end] = True
    return mask


def file_overlap_score(gt_rows, pred_rows):
    gt_mask = framewise_mask(gt_rows)
    pred_mask = framewise_mask(pred_rows)
    union = np.logical_or(gt_mask, pred_mask).sum()
    inter = np.logical_and(gt_mask, pred_mask).sum()
    return inter / union if union else 0.0


def plot_sample_figure(filename, sample_type, gt_rows, pred_rows):
    audio_path = find_audio_path(filename)
    duration = max(
        10.0,
        float(gt_rows["offset"].max()) if not gt_rows.empty else 0.0,
        float(pred_rows["offset"].max()) if not pred_rows.empty else 0.0,
    )
    row_labels = sorted(set(gt_rows["event_label"]).union(set(pred_rows["event_label"])))
    if not row_labels:
        row_labels = [list(classes_labels.keys())[0]]

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

    out_path = ASSET_DIR / f"wavlm_sample_{filename.replace('.wav', '')}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def select_samples(gt, pred):
    by_file_gt = {fn: sample_rows(gt, fn) for fn in sorted(gt["filename"].unique())}
    by_file_pred = {fn: sample_rows(pred, fn) for fn in sorted(gt["filename"].unique())}
    records = []
    for filename, gt_rows in by_file_gt.items():
        pred_rows = by_file_pred.get(filename, pd.DataFrame(columns=["event_label", "onset", "offset"]))
        gt_labels = set(gt_rows["event_label"].tolist())
        pred_labels = set(pred_rows["event_label"].tolist())
        records.append(
            {
                "filename": filename,
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
                "gt_labels": gt_labels,
                "pred_labels": pred_labels,
                "gt_n": len(gt_rows),
                "pred_n": len(pred_rows),
                "score": file_overlap_score(gt_rows, pred_rows),
                "long_non_speech": any(
                    (row.event_label != "Speech") and (row.offset - row.onset >= 8.0)
                    for row in gt_rows.itertuples(index=False)
                ),
            }
        )

    def choose(name, pool):
        if not pool:
            return None
        pool = sorted(pool, key=lambda x: (-x["score"], x["filename"])) if "good" in name else sorted(pool, key=lambda x: (x["score"], -x["gt_n"], x["filename"]))
        return pool[0]

    chosen = []
    used = set()

    def add_record(record, sample_type, plot_type, reason, commentary):
        if record is None or record["filename"] in used:
            return
        chosen.append(
            {
                "filename": record["filename"],
                "type": sample_type,
                "plot_type": plot_type,
                "reason": reason,
                "commentary": commentary,
                "gt_rows": record["gt_rows"],
                "pred_rows": record["pred_rows"],
            }
        )
        used.add(record["filename"])

    add_record(
        choose(
            "good",
            [
                r
                for r in records
                if r["score"] >= 0.60 and r["pred_n"] > 0 and ("Speech" in r["gt_labels"] or len(r["gt_labels"]) == 1)
            ],
        ),
        "检测较好的正例",
        "Good detection",
        "该样本在当前 WavLM-only 模型下具有较高时序重合度，适合展示它确实能在部分样本上正常工作。",
        "这类样本通常能反映 WavLM 在较单纯或语音占主导场景下的可用上限。",
    )
    add_record(
        choose(
            "speech_bias",
            [
                r
                for r in records
                if "Speech" in r["gt_labels"] and len(r["gt_labels"] - {"Speech"}) >= 1 and r["pred_labels"] <= {"Speech"}
            ],
        ),
        "语音偏置样本",
        "Speech-biased prediction",
        "真值同时包含 Speech 与非语音事件，但预测几乎只剩 Speech，适合展示语音导向表征的偏置。",
        "如果这类样本较多，通常意味着 WavLM 更擅长保住 Speech，而对环境事件分离不足。",
    )
    add_record(
        choose(
            "weak_fail",
            [
                r
                for r in records
                if len(r["gt_labels"] & WEAK_CLASSES) >= 1 and len(r["pred_labels"] & WEAK_CLASSES) == 0
            ],
        ),
        "弱类失败样本",
        "Weak-class miss",
        "真值包含 Dog/Cat/Dishes/Alarm 等弱类，但预测没有报出对应弱类，适合展示当前短板。",
        "这类样本更像表征问题，而不只是简单阈值问题。",
    )
    add_record(
        choose(
            "long_fail",
            [
                r
                for r in records
                if r["long_non_speech"] and r["score"] <= 0.30
            ],
        ),
        "长持续环境事件失败样本",
        "Long-duration failure",
        "真值包含长持续非语音环境事件，但预测重合度仍然很低，适合展示 WavLM 对部分环境纹理类并不稳定。",
        "如果长持续类也不稳，说明当前 WavLM-only 不是简单的“长持续类都好”。",
    )
    add_record(
        choose(
            "multi_event",
            [
                r
                for r in records
                if r["gt_n"] >= 4 and r["pred_n"] <= max(1, r["gt_n"] // 3)
            ],
        ),
        "多事件场景欠检样本",
        "Multi-event under-detection",
        "真值中同时存在多个事件片段，但预测显著稀疏，适合展示复杂场景建模不足。",
        "这类样本经常能反映 shared decoder 没法从当前 encoder 表征里稳定分出多类边界。",
    )
    add_record(
        choose(
            "non_speech_good",
            [
                r
                for r in records
                if "Speech" not in r["gt_labels"] and r["pred_n"] > 0 and r["score"] >= 0.50
            ],
        ),
        "非语音相对成功样本",
        "Non-speech success",
        "该样本几乎不依赖 Speech，却仍然被较好检测，适合说明 WavLM 并非完全不会做环境声。",
        "这类样本能帮助区分“完全语音化失败”与“局部可用但上限不高”两种结论。",
    )
    fallback_pools = [
        (
            "复杂场景失败样本",
            "Fallback hard case",
            "该样本作为复杂场景或低重合度兜底样本，用来补充说明 WavLM-only 的失败模式。",
            "这类样本通常体现为多事件欠检、弱类消失或只剩少数主类。",
            sorted(
                [r for r in records if r["filename"] not in used and r["gt_n"] >= 3],
                key=lambda x: (x["score"], -x["gt_n"], x["filename"]),
            ),
        ),
        (
            "额外长持续样本",
            "Fallback long-duration case",
            "该样本作为长持续类兜底案例，用来观察 WavLM-only 是否只在极少数长段上可用。",
            "如果这类样本也不稳，就说明它并不是普遍擅长环境纹理长事件。",
            sorted(
                [r for r in records if r["filename"] not in used and r["long_non_speech"]],
                key=lambda x: (x["score"], x["filename"]),
            ),
        ),
        (
            "额外语音样本",
            "Fallback speech case",
            "该样本作为语音相关兜底案例，用来观察 WavLM-only 是否持续偏向 Speech。",
            "这类样本通常能进一步证明它的语音导向表征偏好。",
            sorted(
                [r for r in records if r["filename"] not in used and "Speech" in r["gt_labels"]],
                key=lambda x: (-x["pred_n"], x["score"], x["filename"]),
            ),
        ),
    ]
    for sample_type, plot_type, reason, commentary, pool in fallback_pools:
        if len(chosen) >= 6:
            break
        if not pool:
            continue
        record = pool[0]
        add_record(record, sample_type, plot_type, reason, commentary)
    return chosen[:6]


def build_train_summary_df(merged_cache):
    rows = []
    for tag in [
        "train/student/loss_strong",
        "val/synth/student/loss_strong",
        "val/obj_metric",
        "val/synth/student/intersection_f1_macro",
        "val/synth/student/event_f1_macro",
    ]:
        points = merged_cache.get(tag, [])
        if not points:
            continue
        values = [value for _, value in points]
        best = min(values) if "loss" in tag else max(values)
        rows.append((tag, f"{values[0]:.4f}", f"{values[-1]:.4f}", f"{best:.4f}"))
    return pd.DataFrame(rows, columns=["曲线", "起始值", "最终值", "最佳值"])


def write_report(
    config,
    config_path,
    candidates,
    final_version,
    best_ckpt_path,
    prediction_tsv,
    current_metrics,
    behavior_stats,
    overall_comparison_df,
    per_class_compare_df,
    behavior_compare_df,
    sample_details,
    asset_paths,
    merged_cache,
    steps_per_epoch,
):
    best_epoch = int(re.search(r"epoch=(\d+)", best_ckpt_path.name).group(1))
    best_step = int(re.search(r"step=(\d+)", best_ckpt_path.name).group(1))
    best_epoch_from_step = step_to_epoch(best_step - 1, steps_per_epoch)
    train_summary_df = build_train_summary_df(merged_cache)
    current_per_class_df = current_metrics["per_class_df"]
    strong_classes = current_per_class_df[current_per_class_df["分组"] == "较强"]["类别"].tolist() or ["无"]
    mid_classes = current_per_class_df[current_per_class_df["分组"] == "中等"]["类别"].tolist() or ["无"]
    weak_classes = current_per_class_df[current_per_class_df["分组"] == "较弱"]["类别"].tolist() or ["无"]
    summary_map = {row["统计项"]: row["数值"] for _, row in behavior_stats["summary_df"].iterrows()}

    candidate_df = pd.DataFrame(
        [
            {
                "版本": item["version"],
                "识别依据": "state_dict 含 wavlm 参数",
                "last epoch": item["epoch"],
                "best score": f"{item['best_model_score']:.4f}" if item["best_model_score"] is not None else "NA",
                "训练 event 数": len(item["train_events"]),
                "测试 event 数": len(item["test_events"]),
            }
            for item in candidates
        ]
    )

    config_summary = pd.DataFrame(
        [
            ("实验设置", "WavLM-only + shared decoder baseline"),
            ("评估对象", "student"),
            ("encoder_type", config["model"]["encoder_type"]),
            ("WavLM freeze", config["model"]["wavlm"]["freeze"]),
            ("output layer", config["model"]["wavlm"]["output_layer"]),
            ("align method", config["model"]["align"]["method"]),
            ("decoder temporal", "shared BiGRU + strong/weak heads"),
            ("配置文件", str(config_path.relative_to(ROOT))),
            ("采用版本", final_version["version"]),
            ("best checkpoint", str(best_ckpt_path.relative_to(ROOT))),
            ("prediction TSV", str(prediction_tsv.relative_to(ROOT))),
            ("数据划分", "synthetic train + synthetic validation"),
            ("test 是否独立", "否，当前 test 实际仍是 synthetic validation"),
        ],
        columns=["项目", "说明"],
    )

    count_display = behavior_stats["class_count_df"].copy()
    count_display.columns = ["类别", "GT事件数", "Pred事件数", "Pred-GT"]
    long_bias_display = behavior_stats["long_bias_df"].copy()
    long_bias_display.columns = ["类别", "平均预测时长", ">=9s 预测段数"]
    per_class_display = current_metrics["per_class_df"].drop(columns=["_event", "_segment"]).copy()
    per_class_compare_display = per_class_compare_df.drop(columns=["_wavlm_event", "_wavlm_segment"]).copy()

    wavlm_vs_crnn_gap = (
        current_metrics["event_res"]["class_wise_average"]["f_measure"]["f_measure"] * 100
        - BASELINES["CRNN baseline"]["overall"]["Event F1 macro"]
    )
    wavlm_vs_beats_gap = (
        current_metrics["event_res"]["class_wise_average"]["f_measure"]["f_measure"] * 100
        - BASELINES["Frozen BEATs baseline"]["overall"]["Event F1 macro"]
    )

    md = []
    md.append("# WavLM-only + Shared Decoder 训练结果分析报告")
    md.append("")
    md.append("## 目录")
    md.append("- [实验概况](#实验概况)")
    md.append("- [最终指标汇总](#最终指标汇总)")
    md.append("- [横向对比](#横向对比)")
    md.append("- [训练过程与选模分析](#训练过程与选模分析)")
    md.append("- [预测行为统计](#预测行为统计)")
    md.append("- [典型样本分析](#典型样本分析)")
    md.append("- [结论与讨论](#结论与讨论)")
    md.append("- [后续建议](#后续建议)")
    md.append("")

    md.append("## 实验概况")
    md.append("")
    md.append("### 自动定位结果")
    md.append("")
    md.append(df_to_markdown(candidate_df))
    md.append("")
    md.append(
        f"最终采用的实验版本是 `{final_version['version']}`。选择依据是：第一，`version_25` 与 `version_31` 都能从 state_dict 中识别出 WavLM encoder，"
        f"但 `version_31` 是唯一一版完成了实质训练并保存 best checkpoint 的版本；第二，它的 best score 为 `{final_version['best_model_score']:.4f}`，"
        "明显不是 smoke test；第三，本次报告所用的 `metrics_test` 是刚刚用该 best checkpoint 重新导出的干净测试结果。"
    )
    md.append("")
    md.append(
        "需要特别说明的是：共享目录 `exp/2022_baseline/metrics_test/` 中旧的 `event_f1.txt/segment_f1.txt` 没有随这次 WavLM 测试一起刷新，"
        "因此本报告里的 overall / per-class 指标统一以最新 WavLM prediction TSV 与固定 GT 重新计算为准，而不是机械引用旧文本文件。"
    )
    md.append("")
    md.append(df_to_markdown(config_summary))
    md.append("")
    md.append(
        "本次实验属于 `WavLM-only + shared decoder baseline`：输入保持 waveform，WavLM encoder 提取时间序列特征，"
        "随后通过统一的时间对齐模块映射到标签长度，再送入与你的 CRNN / BEATs baseline 共用的 shared decoder、BiGRU 和 strong/weak heads。"
    )
    md.append("")
    md.append(
        "由于当前配置里 `freeze=true`，WavLM 只是作为冻结特征提取器，训练阶段主要更新 shared decoder 相关参数。"
        "这意味着下面的结果更适合回答“WavLM 作为单独 encoder 能提供什么表征”，而不是回答“充分微调后的 WavLM 上限有多高”。"
    )
    md.append("")
    md.append(
        "同时，`test_folder/test_tsv` 仍然指向 synthetic validation，所以这里的 test 仍是偏开发口径的自测分数，不等同真实外部分布上的泛化能力。"
    )
    md.append("")

    md.append("## 最终指标汇总")
    md.append("")
    md.append(df_to_markdown(current_metrics["overall_df"]))
    md.append("")
    md.append(df_to_markdown(per_class_display))
    md.append("")
    md.append(
        f"按当前结果，WavLM-only 的较强类别主要是 `{', '.join(strong_classes)}`，"
        f"中等类别主要是 `{', '.join(mid_classes)}`，"
        f"较弱类别集中在 `{', '.join(weak_classes)}`。"
    )
    md.append("")
    md.append(
        "如果最终表现呈现出 `Speech / Electric_shaver_toothbrush / Alarm_bell_ringing` 相对可用，而 `Dog / Dishes / Cat` 等环境弱类明显掉队，"
        "这会更支持“WavLM 的表征偏语音相关”而不是“训练完全异常”这一解释。"
    )
    md.append("")

    md.append("## 横向对比")
    md.append("")
    md.append(df_to_markdown(overall_comparison_df))
    md.append("")
    md.append(f"![WavLM vs CRNN/BEATs event F1](report_assets/{asset_paths['wavlm_vs_baselines'].name})")
    md.append("")
    md.append(df_to_markdown(per_class_compare_display))
    md.append("")
    md.append(
        f"从 overall 看，当前 WavLM-only 相比 CRNN baseline 的 Event F1 macro 差值为 `{wavlm_vs_crnn_gap:+.2f}pp`，"
        f"相比 frozen BEATs baseline 的差值为 `{wavlm_vs_beats_gap:+.2f}pp`。"
    )
    md.append("")
    md.append(
        "这次结果最关键的事实是：WavLM-only 不仅明显弱于 CRNN，也没有超过 frozen BEATs。"
        "因此，这里已经不能简单归因为“环境声任务本来就难”或“只要再长训一点就会追上”。"
    )
    md.append("")
    md.append(
        "而当前真正发生的是：WavLM-only 既明显弱于 CRNN，也低于 frozen BEATs。"
        "这意味着它不是一个“虽然不如 CRNN、但至少比 BEATs 更合适”的替代主干；"
        "更像是一个在本任务上表征取向不对、只能保住少数语音相关模式的 encoder。"
    )
    md.append("")
    md.append(
        "与 BEATs 的差别也要重点看类别结构：如果 WavLM 在 `Speech` 或语音相关类别上更自然，而在 `Running_water / Frying / Vacuum_cleaner` 等环境纹理类上不稳定，"
        "那说明两者差异主要来自 encoder 表征取向；如果所有类别都普遍偏低，才更像训练设置本身限制了上限。"
    )
    md.append("")

    md.append("## 训练过程与选模分析")
    md.append("")
    md.append(f"![WavLM training curves](report_assets/{asset_paths['training_curves'].name})")
    md.append("")
    md.append(df_to_markdown(train_summary_df))
    md.append("")
    md.append(
        f"`version_31` 的 best checkpoint 出现在全局 epoch 约 `{best_epoch}`（step={best_step}），对应曲线上的 epoch 位置约为 `{best_epoch_from_step:.1f}`。"
    )
    md.append("")
    md.append(
        "从训练曲线看，这一版首先要回答的是“有没有正常收敛”。如果 `train loss` 和 `val loss` 都在下降，"
        "而 `val/obj_metric = intersection_f1_macro` 虽然不高但持续抬升，那它更像是“正常收敛但上限不高”；"
        "如果曲线长期贴近 0 或大幅震荡，则更像训练异常。"
    )
    md.append("")
    md.append(
        "当前这版 WavLM-only 从曲线口径上更接近前者：它是一个可以正常训练、正常选模的 baseline，"
        "但从 best score 本身就能看出，它并没有达到 CRNN 或 residual gate 那样的上限。"
    )
    md.append("")
    md.append(
        "换句话说，这版不是训练崩坏，而是“正常收敛到一个偏低上限”。这点对后续判断很重要："
        "如果训练是正常的，而分数依然很低，那么问题更偏向 WavLM 作为冻结主 encoder 与当前任务的不匹配，而不是单纯训练没跑够。"
    )
    md.append("")

    md.append("## 预测行为统计")
    md.append("")
    md.append(df_to_markdown(behavior_stats["summary_df"]))
    md.append("")
    md.append(df_to_markdown(behavior_compare_df))
    md.append("")
    md.append(f"![WavLM class count comparison](report_assets/{asset_paths['class_counts'].name})")
    md.append("")
    md.append(f"![WavLM class F1 comparison](report_assets/{asset_paths['class_f1'].name})")
    md.append("")
    md.append(f"![WavLM duration distribution](report_assets/{asset_paths['duration_dist'].name})")
    md.append("")
    md.append(f"![WavLM long-duration bias](report_assets/{asset_paths['long_bias'].name})")
    md.append("")
    md.append(df_to_markdown(count_display))
    md.append("")
    md.append(df_to_markdown(long_bias_display.head(6)))
    md.append("")
    md.append(
        "这里最值得看的不是单一总分，而是预测行为像不像一个“已经正常工作”的环境声 SED 系统。"
        "如果空预测比例不高、总预测事件数合理、且至少有几类能稳定报出，那么它不是训练崩溃；"
        "但如果大量类长期几乎不报、预测明显偏向少数语音相关类别，那就更像表征取向不匹配。"
    )
    md.append("")
    md.append(
        "因此，WavLM-only 的价值不一定在于直接追平 CRNN，而在于回答：在相同 shared decoder 和相同训练流程下，"
        "一个偏语音 SSL encoder 单独拿来做环境声 SED 时，究竟会保住哪些类、牺牲哪些类。"
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
        "这次 WavLM-only baseline 如果能正常输出完整的预测 TSV、并在验证曲线和最终指标上形成自洽结果，就说明它已经作为一个单独 encoder baseline 成功跑通。"
    )
    md.append("")
    md.append(
        "但它真正说明的问题不只是“WavLM 分数高不高”，而是：在统一 shared decoder 与统一训练流程下，"
        "WavLM 与 CRNN / BEATs 的差异主要来自 encoder 表征本身，而不是后端结构差异。"
    )
    md.append("")
    md.append(
        "如果最终观察到的是：WavLM 对 `Speech` 及少数语音相关类别较自然，对环境事件尤其是 `Dog / Dishes / Cat` 这类弱类明显吃亏，"
        "同时整体又不是 NaN、不是全空、不是训练崩，那么这更支持“WavLM 更偏语音导向表征”而不是“当前训练设置本身有明显错误”。"
    )
    md.append("")
    md.append(
        "因此，这版 WavLM-only 更像一个很有价值的控制实验：它帮助你判断 WavLM 是否适合作为主力 encoder。"
        "如果它始终明显弱于 CRNN，而又没有表现出稳定的环境事件优势，那么它更适合作为后续融合里的补充分支，而不是当前任务的主力 backbone。"
    )
    md.append("")
    md.append(
        "就这次结果而言，更强的证据站在“语音导向表征”这一侧，而不是“训练设置”这一侧。"
        "原因是：训练曲线本身是正常的；shared decoder 与统一流程已经在 CRNN / BEATs / fusion 上验证过可用；"
        "而 WavLM-only 却几乎只保住了 Speech，并对大量环境类近乎失声。"
    )
    md.append("")

    md.append("## 后续建议")
    md.append("")
    md.append("1. 不必优先继续深挖单独 WavLM baseline 本身；它更适合作为控制实验，而不是当前论文的主力结果线。")
    md.append("2. 如果本次结果显示 WavLM 在语音相关类别仍有可用信息，下一步更值得把它接入 residual gated fusion，而不是单独长训。")
    md.append("3. 继续做一版更细的逐类分析，重点拆分 `Speech/Alarm/Electric_shaver_toothbrush` 与 `Dog/Dishes/Cat/Running_water` 的差异来源。")
    md.append("4. 论文里建议保留最有代表性的 `CRNN / frozen BEATs / WavLM / concat fusion / residual gate` 对照，但不用把单独 WavLM 扩成过多变体。")
    md.append("5. 如果后续 WavLM 只在少数语音类有价值，可将其定位成“补充语音导向分支”的证据，而不是继续追求它单独超过 CRNN。")
    md.append("")

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")


def main():
    ensure_dirs()
    candidates, final_version = detect_wavlm_versions()
    config_path = locate_config()
    config = load_yaml(config_path)
    best_ckpt_path = Path(final_version["best_model_path"]).resolve()
    prediction_tsv = locate_prediction_tsv()
    current_metrics = compute_current_metrics(prediction_tsv, config)
    behavior_stats = compute_behavior_stats(current_metrics["pred"], current_metrics["gt"])
    overall_comparison_df = build_overall_comparison(current_metrics)
    per_class_compare_df = build_per_class_comparison(current_metrics["per_class_df"])
    behavior_compare_df = build_behavior_comparison(behavior_stats)

    last_ckpt = torch.load(final_version["dir"] / "last.ckpt", map_location="cpu", weights_only=False)
    steps_per_epoch = int(last_ckpt.get("global_step", 0)) // max(1, int(last_ckpt.get("epoch", 0)) + 1)
    best_step = int(re.search(r"step=(\d+)", best_ckpt_path.name).group(1)) - 1
    training_curves_path, merged_cache = plot_training_curves(final_version["train_events"][0], steps_per_epoch, best_step)

    asset_paths = {
        "training_curves": training_curves_path,
        "class_counts": plot_class_counts(behavior_stats["class_count_df"]),
        "class_f1": plot_class_f1(current_metrics["per_class_df"]),
        "wavlm_vs_baselines": plot_wavlm_vs_baselines(current_metrics["per_class_df"]),
        "duration_dist": plot_duration_distribution(current_metrics["pred"], current_metrics["gt"]),
        "long_bias": plot_long_bias(behavior_stats["long_bias_df"]),
    }

    sample_details = []
    for sample in select_samples(current_metrics["gt"], current_metrics["pred"]):
        asset = plot_sample_figure(sample["filename"], sample["plot_type"], sample["gt_rows"], sample["pred_rows"])
        sample["asset_name"] = asset.name
        sample_details.append(sample)

    write_report(
        config=config,
        config_path=config_path,
        candidates=candidates,
        final_version=final_version,
        best_ckpt_path=best_ckpt_path,
        prediction_tsv=prediction_tsv,
        current_metrics=current_metrics,
        behavior_stats=behavior_stats,
        overall_comparison_df=overall_comparison_df,
        per_class_compare_df=per_class_compare_df,
        behavior_compare_df=behavior_compare_df,
        sample_details=sample_details,
        asset_paths=asset_paths,
        merged_cache=merged_cache,
        steps_per_epoch=steps_per_epoch,
    )

    print("Generated files:")
    print(f"- {REPORT_PATH}")
    for path in sorted(ASSET_DIR.glob("wavlm_*.png")):
        print(f"- {path}")
    print("Final experiment version:")
    print(f"- {final_version['version']}")
    print(f"- best checkpoint: {best_ckpt_path}")
    print("Selected samples:")
    for sample in sample_details:
        print(f"- {sample['filename']} ({sample['type']})")
    print("Completed:")
    print("- WavLM experiment auto-location")
    print("- Final metric summary")
    print("- Horizontal comparison vs CRNN / BEATs / fusion baselines")
    print("- Training curve analysis")
    print("- Prediction behavior statistics")
    print("- Typical sample visualizations")
    print("- Conclusions and recommendations")
    print("Not completed:")
    print("- No independent external test set analysis; current test remains synthetic validation.")


if __name__ == "__main__":
    main()
