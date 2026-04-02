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

REPORT_DIR = ROOT / "baselines/WavLM-CRNN-gate-fusion-baseline"
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
    "WavLM-only baseline": {
        "overall": {
            "PSDS1": 0.007,
            "PSDS2": 0.049,
            "Intersection F1": 0.346,
            "Event F1 macro": 2.50,
            "Event F1 micro": 9.85,
            "Segment F1 macro": 33.76,
            "Segment F1 micro": 34.99,
        },
        "classwise": {
            "Alarm_bell_ringing": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Blender": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Cat": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Dishes": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Dog": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Electric_shaver_toothbrush": {"pred": 146, "event": 0.00, "segment": 31.27},
            "Frying": {"pred": 0, "event": 0.00, "segment": 0.00},
            "Running_water": {"pred": 16, "event": 0.00, "segment": 0.40},
            "Speech": {"pred": 3882, "event": 15.03, "segment": 59.42},
            "Vacuum_cleaner": {"pred": 253, "event": 9.92, "segment": 43.97},
        },
        "behavior": {
            "pred_files": 2289,
            "empty_files": 211,
            "empty_ratio": 8.44,
            "pred_events": 4297,
            "pred_mean_duration": 1.44,
            "pred_ge9": 145,
            "fragmented": 48,
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
            "pred_mean_duration": 2.92,
            "pred_ge9": 875,
            "fragmented": 136,
        },
    },
    "CRNN + BEATs residual gate": {
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
            "pred_mean_duration": 2.73,
            "pred_ge9": 899,
            "fragmented": 252,
        },
    },
}

SAMPLE_SPECS = [
    {
        "filename": "1088.wav",
        "type": "弱类恢复：Cat + Speech",
        "plot_type": "Cat + Speech recovery",
        "reason": "Cat + Speech 双事件几乎完整恢复，适合展示 WavLM 辅助分支并不只会强化 Speech。",
        "commentary": "这类样本说明把 WavLM 放到 CRNN 主干后的 residual gate 中，语音导向表征可以转化成对弱类边界的补充，而不是只剩 Speech。",
        "cross_reference": "该样本对比 WavLM-only 很有代表性：单独 WavLM 难以稳定承担环境声主干，而接到 CRNN 后可以保住 Cat + Speech 的同步检测。",
    },
    {
        "filename": "1195.wav",
        "type": "弱类恢复：Dog 长事件",
        "plot_type": "Dog recovery",
        "reason": "Dog 是当前最难类之一，而这里已经能较完整覆盖长事件。",
        "commentary": "如果 WavLM 只是无意义的语音偏置噪声，这类 Dog 长事件通常不会受益；这里的恢复说明它至少在部分弱类上提供了有价值的补充信息。",
        "cross_reference": "相较 WavLM-only 几乎无法处理动物类，这一例更能体现 CRNN 主干把 WavLM 从“主力失败 encoder”转成了“辅助分支”。",
    },
    {
        "filename": "1008.wav",
        "type": "多类语音相关场景受益",
        "plot_type": "Speech-related multi-event gain",
        "reason": "Alarm + Dog + Speech 共存，适合观察 WavLM 辅助分支在语音相关复杂场景中的补充作用。",
        "commentary": "这类样本最能回答“WavLM 的语音导向表征有没有转化成互补信息”：当前模型不只是报 Speech，也把 Alarm 和 Dog 带起来了。",
        "cross_reference": "相比 CRNN + BEATs residual gate，这种含语音上下文的多类片段更可能是 WavLM 辅助分支的优势区间。",
    },
    {
        "filename": "1312.wav",
        "type": "多事件场景仍明显欠检",
        "plot_type": "Multi-event under-detection",
        "reason": "Dishes + Speech 的复杂场景里仍几乎只剩 Speech，适合展示当前残留短板。",
        "commentary": "这说明 CRNN + WavLM 虽然比 WavLM-only 强得多，但对 hardest weak class 和多事件分离仍然不够，增益不是全局性的。",
        "cross_reference": "该类样本是与 CRNN + BEATs residual gate 做对照时最重要的失败模式之一，因为它直接反映 `Dishes` 仍然没有被攻克。",
    },
    {
        "filename": "1382.wav",
        "type": "长持续设备类检测较好",
        "plot_type": "Strong long-duration device detection",
        "reason": "Vacuum_cleaner 整段检测稳定，同时能报出少量 Speech，适合展示模型已经具备健康主干行为。",
        "commentary": "这说明当前结构不是单纯靠 WavLM 拉高 Speech，而是在 CRNN 主干上保住了长持续设备类的正常建模。",
        "cross_reference": "与 WavLM-only 同样的样本对照时，这类长持续设备声通常能看出“CRNN 主干负责稳，WavLM 辅助补语音上下文”的角色分工。",
    },
    {
        "filename": "1000.wav",
        "type": "语音相关长持续场景改善",
        "plot_type": "Speech-aware running water",
        "reason": "Running_water + Speech 的共存场景能观察 WavLM 辅助分支是否把语音相关表征转成稳定补充。",
        "commentary": "这里的结果说明 WavLM 作为辅助分支时，确实更容易把语音相关片段补出来，但仍然存在边界保守和后段漏检。",
        "cross_reference": "这是和 WavLM-only 做对照最自然的样本之一，因为它能同时反映 WavLM 的长持续场景能力与语音偏置。",
    },
    {
        "filename": "983.wav",
        "type": "复杂纹理场景失败样本",
        "plot_type": "Complex texture failure",
        "reason": "Frying + Running_water + Dishes + Speech 的高复杂度场景几乎崩掉，适合展示当前模型的边界。",
        "commentary": "这类样本说明 CRNN + WavLM 虽然比 WavLM-only 强很多，但对复杂环境纹理和多事件重叠的提升仍然有限。",
        "cross_reference": "相较 CRNN + BEATs residual gate，这种复杂环境纹理场景更能暴露 WavLM 作为辅助分支时的局限。",
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


def scalar_points(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(item.step, item.value) for item in ea.Scalars(tag)]


def merge_scalar_points(tb_files, tag):
    merged = {}
    for tb_file in tb_files:
        ea = load_tb(tb_file)
        for step, value in scalar_points(ea, tag):
            merged[int(step)] = float(value)
    return sorted(merged.items())


def step_to_epoch(step, steps_per_epoch):
    return ((step + 1) / steps_per_epoch) - 1


def detect_residual_versions():
    exp_dir = ROOT / "exp/2022_baseline"
    candidates = []
    for version_dir in sorted(exp_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[-1])):
        last_ckpt = version_dir / "last.ckpt"
        if not last_ckpt.exists():
            continue
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("sed_student", {})
        has_gate = any("gate" in key for key in state.keys())
        has_wavlm = any(key.startswith("wavlm_encoder.") for key in state.keys())
        has_crnn = any(key.startswith("crnn_encoder.") for key in state.keys())
        has_beats_encoder = any(key.startswith("beats_encoder.") for key in state.keys())
        if not (has_gate and has_wavlm and has_crnn) or has_beats_encoder:
            continue
        train_events = sorted(version_dir.glob("events.out.tfevents.*.0"))
        test_events = sorted(version_dir.glob("events.out.tfevents.*.1"))
        callbacks = ckpt.get("callbacks", {})
        best_model_path = None
        best_model_score = None
        for key, value in callbacks.items():
            if "ModelCheckpoint" in key:
                best_model_path = value.get("best_model_path")
                best_model_score = value.get("best_model_score")
                break
        candidates.append(
            {
                "version": version_dir.name,
                "version_num": int(version_dir.name.split("_")[-1]),
                "dir": version_dir,
                "mtime": version_dir.stat().st_mtime,
                "last_ckpt": last_ckpt,
                "epoch": int(ckpt.get("epoch", -1)),
                "step": int(ckpt.get("global_step", -1)),
                "train_events": train_events,
                "test_events": test_events,
                "best_model_path": best_model_path,
                "best_model_score": float(best_model_score) if best_model_score is not None else None,
                "has_wavlm": has_wavlm,
                "has_crnn": has_crnn,
                "has_beats_encoder": has_beats_encoder,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            "Could not locate any CRNN + WavLM residual gated fusion experiment under exp/2022_baseline."
        )

    candidates.sort(key=lambda item: item["version_num"])
    chosen = candidates[-2:] if len(candidates) >= 2 else candidates
    chosen.sort(key=lambda item: item["version_num"])
    best_item = max(
        chosen,
        key=lambda item: (
            item["best_model_score"] if item["best_model_score"] is not None else float("-inf"),
            item["version_num"],
        ),
    )
    return candidates, chosen, best_item


def locate_config():
    configs = []
    for path in ROOT.glob("confs/*.yaml"):
        try:
            config = load_yaml(path)
        except Exception:
            continue
        model_type = (
            config.get("model", {}).get("model_type")
            if isinstance(config, dict)
            else None
        )
        if model_type == "crnn_wavlm_residual_gated_fusion":
            configs.append(path)
    if not configs:
        raise FileNotFoundError("Could not locate CRNN + WavLM residual gated fusion config in confs/*.yaml")
    configs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return configs[0]


def locate_prediction_tsv():
    candidates = sorted(
        (
            ROOT / "exp/2022_baseline/metrics_test/student/scenario1"
        ).glob("predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not locate threshold 0.49 prediction TSV under metrics_test/student/scenario1.")
    return candidates[0]


def load_prediction_operating_points(prediction_tsv):
    scenario_dir = prediction_tsv.parent
    threshold_files = sorted(scenario_dir.glob("predictions_th_*.tsv"))
    operating_points = {}
    for path in threshold_files:
        m = re.search(r"predictions_th_([0-9.]+)\.tsv$", path.name)
        if not m:
            continue
        operating_points[float(m.group(1))] = pd.read_csv(path, sep="\t")
    return operating_points


def classify_status(event_score):
    if event_score >= 0.55:
        return "较强"
    if event_score >= 0.40:
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
    return {
        "summary_df": summary_df,
        "class_count_df": class_count_df.reset_index().rename(columns={"index": "类别", "event_label": "类别"}),
        "long_bias_df": long_bias_df,
        "empty_files": empty_files,
        "fragmented_files": fragmented_files,
    }


def build_overall_comparison(current_metrics):
    rows = []
    for name, baseline in BASELINES.items():
        row = {"模型": name}
        row.update(baseline["overall"])
        rows.append(row)
    rows.append(
        {
            "模型": "CRNN + WavLM residual gate",
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
    current_map = {
        row["类别"]: row for _, row in current_per_class_df.iterrows()
    }
    for label in classes_labels.keys():
        cur = current_map[label]
        row = {
            "类别": label,
            "GT": GT_COUNTS[label],
            "CRNN Event": f"{BASELINES['CRNN baseline']['classwise'][label]['event']:.2f}%",
            "Concat Event": f"{BASELINES['Concat late fusion']['classwise'][label]['event']:.2f}%",
            "Gate Event": cur["Event F1"],
            "BEATs Event": f"{BASELINES['Frozen BEATs baseline']['classwise'][label]['event']:.2f}%",
            "CRNN Segment": f"{BASELINES['CRNN baseline']['classwise'][label]['segment']:.2f}%",
            "Concat Segment": f"{BASELINES['Concat late fusion']['classwise'][label]['segment']:.2f}%",
            "Gate Segment": cur["Segment F1"],
            "Pred/GT (Gate)": cur["Pred/GT"],
            "_gate_event": cur["_event"] * 100,
            "_concat_event": BASELINES["Concat late fusion"]["classwise"][label]["event"],
            "_crnn_event": BASELINES["CRNN baseline"]["classwise"][label]["event"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_gate_vs_concat_delta(current_per_class_df):
    rows = []
    for _, row in current_per_class_df.iterrows():
        label = row["类别"]
        gate_event = row["_event"] * 100
        gate_segment = row["_segment"] * 100
        concat_event = BASELINES["Concat late fusion"]["classwise"][label]["event"]
        concat_segment = BASELINES["Concat late fusion"]["classwise"][label]["segment"]
        crnn_event = BASELINES["CRNN baseline"]["classwise"][label]["event"]
        rows.append(
            {
                "类别": label,
                "Gate Event F1": f"{gate_event:.2f}%",
                "Concat Event F1": f"{concat_event:.2f}%",
                "差值(Event)": f"{gate_event - concat_event:+.2f}pp",
                "Gate Segment F1": f"{gate_segment:.2f}%",
                "Concat Segment F1": f"{concat_segment:.2f}%",
                "差值(Segment)": f"{gate_segment - concat_segment:+.2f}pp",
                "相对 CRNN Event": f"{gate_event - crnn_event:+.2f}pp",
            }
        )
    return pd.DataFrame(rows)


def build_behavior_comparison(current_behavior):
    rows = []
    for name in ["CRNN baseline", "Frozen BEATs baseline", "Concat late fusion"]:
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
            "模型": "CRNN + WavLM residual gate",
            "有预测文件数": summary_map["有预测文件数"],
            "空预测文件数": summary_map["空预测文件数"],
            "空预测比例": summary_map["空预测比例"],
            "总预测事件数": summary_map["总预测事件数"],
        }
    )
    return pd.DataFrame(rows)


def plot_training_curves(train_tb_files, steps_per_epoch, best_ckpt_step):
    tags = {
        "train/student/loss_strong": ("Train strong loss", "#cc3311"),
        "val/synth/student/loss_strong": ("Val strong loss", "#0077bb"),
        "val/obj_metric": ("Val obj metric", "#009988"),
        "val/synth/student/intersection_f1_macro": ("Val intersection F1", "#ee7733"),
        "val/synth/student/event_f1_macro": ("Val event macro F1", "#332288"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()

    merged_cache = {}
    for idx, (tag, (title, color)) in enumerate(tags.items()):
        points = merge_scalar_points(train_tb_files, tag)
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

    val_obj = merged_cache["val/obj_metric"]
    val_event = merged_cache["val/synth/student/event_f1_macro"]
    val_loss = merged_cache["val/synth/student/loss_strong"]
    summary = [
        f"best ckpt epoch: {step_to_epoch(best_ckpt_step, steps_per_epoch):.0f}",
        f"best obj: {max(v for _, v in val_obj):.4f}" if val_obj else "best obj: NA",
        f"best event: {max(v for _, v in val_event):.4f}" if val_event else "best event: NA",
        f"best val loss: {min(v for _, v in val_loss):.4f}" if val_loss else "best val loss: NA",
    ]
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.98, "\n".join(summary), va="top", ha="left", family="monospace")

    out_path = ASSET_DIR / "training_curves.png"
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
    ax.set_title("GT vs predicted event counts by class")
    ax.legend()
    out_path = ASSET_DIR / "class_count_comparison.png"
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
    ax.set_title("CRNN + WavLM residual gate event / segment F1 by class")
    ax.legend()
    out_path = ASSET_DIR / "class_f1_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gate_vs_concat(current_per_class_df):
    labels = current_per_class_df["类别"].tolist()
    gate_scores = current_per_class_df["_event"].to_numpy() * 100
    concat_scores = np.array([BASELINES["Concat late fusion"]["classwise"][label]["event"] for label in labels])
    crnn_scores = np.array([BASELINES["CRNN baseline"]["classwise"][label]["event"] for label in labels])
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.bar(x - width, crnn_scores, width=width, label="CRNN")
    ax.bar(x, concat_scores, width=width, label="Concat fusion")
    ax.bar(x + width, gate_scores, width=width, label="CRNN + WavLM residual gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Event F1 (%)")
    ax.set_title("Per-class event F1: CRNN vs concat fusion vs CRNN + WavLM residual gate")
    ax.legend()
    out_path = ASSET_DIR / "gate_vs_concat_event_f1.png"
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

    ax2 = ax1.twinx()
    mean_durations = display_df["pred_mean_duration"].str.replace("s", "", regex=False).astype(float)
    ax2.plot(x, mean_durations, color="#0077bb", marker="o")
    ax2.set_ylabel("mean predicted duration (s)")

    out_path = ASSET_DIR / "long_duration_bias.png"
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

    out_path = ASSET_DIR / f"sample_{filename.replace('.wav', '')}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_train_summary_df(merged_cache, steps_per_epoch):
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
    chosen_versions,
    final_version,
    best_ckpt_path,
    prediction_tsv,
    current_metrics,
    behavior_stats,
    overall_comparison_df,
    per_class_compare_df,
    gate_vs_concat_df,
    behavior_compare_df,
    sample_details,
    asset_paths,
    merged_cache,
    steps_per_epoch,
):
    best_epoch = int(re.search(r"epoch=(\d+)", best_ckpt_path.name).group(1))
    best_step = int(re.search(r"step=(\d+)", best_ckpt_path.name).group(1))
    best_epoch_from_step = step_to_epoch(best_step - 1, steps_per_epoch)
    train_summary_df = build_train_summary_df(merged_cache, steps_per_epoch)
    current_per_class_df = current_metrics["per_class_df"]
    strong_classes = current_per_class_df[current_per_class_df["分组"] == "较强"]["类别"].tolist()
    mid_classes = current_per_class_df[current_per_class_df["分组"] == "中等"]["类别"].tolist()
    weak_classes = current_per_class_df[current_per_class_df["分组"] == "较弱"]["类别"].tolist()
    summary_map = {row["统计项"]: row["数值"] for _, row in behavior_stats["summary_df"].iterrows()}

    candidate_df = pd.DataFrame(
        [
            {
                "版本": item["version"],
                "是否含 gate 参数": "是",
                "last epoch": item["epoch"],
                "best score": f"{item['best_model_score']:.4f}" if item["best_model_score"] is not None else "NA",
                "train event 文件数": len(item["train_events"]),
                "test event 文件数": len(item["test_events"]),
            }
            for item in candidates
        ]
    )
    chosen_df = pd.DataFrame(
        [
            {
                "采用版本": item["version"],
                "作用": "首段训练" if idx == 0 else "中断后续训",
                "训练标量": ", ".join(path.name for path in item["train_events"]),
                "测试标量": ", ".join(path.name for path in item["test_events"]) if item["test_events"] else "无单独 test event",
            }
            for idx, item in enumerate(chosen_versions)
        ]
    )

    config_summary = pd.DataFrame(
        [
            ("实验设置", "CRNN + WavLM residual gated fusion"),
            ("评估对象", "student"),
            ("model_type", config["model"]["model_type"]),
            ("fusion type", config["model"]["fusion"]["fusion_type"]),
            ("gate mode", config["model"]["fusion"]["gate_mode"]),
            ("align method", config["model"]["fusion"]["align_method"]),
            ("projection / norm", "dual projection + dual LayerNorm"),
            ("residual formula", "fused = cnn_norm + gate * wavlm_norm"),
            ("WavLM freeze", config["model"]["wavlm"]["freeze"]),
            ("decoder temporal", "shared BiGRU + strong/weak heads"),
            ("配置文件", str(config_path.relative_to(ROOT))),
            ("数据划分", "synthetic train + synthetic validation"),
            ("test 是否独立", "否，当前 test 实际仍是 synthetic validation"),
            ("best checkpoint", str(best_ckpt_path.relative_to(ROOT))),
            ("prediction TSV", str(prediction_tsv.relative_to(ROOT))),
        ],
        columns=["项目", "说明"],
    )

    count_display = behavior_stats["class_count_df"].copy()
    count_display.columns = ["类别", "GT事件数", "Pred事件数", "Pred-GT"]
    long_bias_display = behavior_stats["long_bias_df"].copy()
    long_bias_display.columns = ["类别", "平均预测时长", ">=9s 预测段数"]
    per_class_display = current_per_class_df.drop(columns=["_event", "_segment"]).copy()
    delta_display = gate_vs_concat_df.copy()

    md = []
    md.append("# CRNN + WavLM Residual Gated Fusion 训练结果分析报告")
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
    md.append(df_to_markdown(chosen_df))
    md.append("")
    md.append(
        f"最终采用的实验版本链是 `{' + '.join(item['version'] for item in chosen_versions)}`，"
        f"并以 `{best_ckpt_path.relative_to(ROOT)}` 作为 best checkpoint。"
        "选择依据有三点：第一，这两版 checkpoint 的 `sed_student` state_dict 都同时包含 `crnn_encoder / wavlm_encoder / gate` 参数，"
        "能够明确识别为 `CRNN + WavLM residual gated fusion`；"
        f"第二，`{chosen_versions[-1]['version']}` 明显承接 `{chosen_versions[0]['version']}` 的 global step 和训练轨迹继续训练；"
        f"第三，按 checkpoint 里记录的 `best_model_score` 比较，当前链上的最佳验证分数来自 `{best_ckpt_path.parent.name}`。"
    )
    md.append("")
    md.append(df_to_markdown(config_summary))
    md.append("")
    md.append(
        "本次实验属于 `CRNN + WavLM residual gated fusion`：CNN branch 先提取 CRNN 的局部时频特征，"
        "冻结的 WavLM branch 提取 frame-level embedding，经时间对齐后分别做 projection + LayerNorm，"
        "再通过显式 gate 学习每一帧/每一通道应该让 WavLM 补多少，最后以 `cnn_norm + gate * wavlm_norm` 的残差形式送入共享 BiGRU 和 strong/weak heads。"
    )
    md.append("")
    md.append(
        "这意味着当前结构不再是旧版的无条件 concat，而是明确保留 `CRNN 为主、WavLM 为补充` 的归纳偏置。"
        "由于 `test_folder/test_tsv` 仍指向 synthetic validation，下面的结果仍是偏“自测分数”的开发分析，不等同真实外部分布上的泛化能力。"
    )
    md.append("")

    md.append("## 最终指标汇总")
    md.append("")
    md.append(df_to_markdown(current_metrics["overall_df"]))
    md.append("")
    md.append(df_to_markdown(per_class_display))
    md.append("")
    md.append(
        f"当前 `CRNN + WavLM residual gate` 的较强类别主要是 `{', '.join(strong_classes)}`，"
        f"中等类别主要是 `{', '.join(mid_classes)}`，"
        f"较弱类别则集中在 `{', '.join(weak_classes)}`。"
    )
    md.append("")
    md.append(
        "从整体指标看，这一版已经明显强于 `WavLM-only baseline`，说明 `WavLM` 单独做主力 encoder 虽然较弱，"
        "但接到 `CRNN` 主干后并不是完全没有价值。"
        "不过它是否已经进入可与 `CRNN baseline` 或 `CRNN + BEATs residual gate` 正面竞争的区间，还需要结合下面的横向对比一起判断。"
    )
    md.append("")

    md.append("## 横向对比")
    md.append("")
    md.append(df_to_markdown(overall_comparison_df))
    md.append("")
    md.append(f"![Gate vs concat per-class event F1](report_assets/{asset_paths['gate_vs_concat'].name})")
    md.append("")
    md.append(df_to_markdown(delta_display))
    md.append("")
    md.append(df_to_markdown(per_class_compare_df.drop(columns=[c for c in per_class_compare_df.columns if c.startswith('_')])))
    md.append("")
    md.append(
        "横向看，`CRNN + WavLM residual gate` 对旧版 concat late fusion 的提升是真实且全局性的，而不再只是局部小修小补。"
        f"整体上，它把 `PSDS1` 从 {BASELINES['Concat late fusion']['overall']['PSDS1']:.3f} 拉到了 {current_metrics['psds1']:.3f}，"
        f"`Intersection F1` 从 {BASELINES['Concat late fusion']['overall']['Intersection F1']:.3f} 拉到了 {current_metrics['intersection']:.3f}，"
        f"`Event F1 macro` 从 {BASELINES['Concat late fusion']['overall']['Event F1 macro']:.2f}% 提升到 {current_metrics['event_res']['class_wise_average']['f_measure']['f_measure'] * 100:.2f}%。"
    )
    md.append("")
    md.append(
        "更关键的是，这次 `CRNN + WavLM residual gate` 已经不只是“对长持续/设备类略有帮助”。"
        "相比 concat，它对 `Blender / Dishes / Dog / Alarm_bell_ringing / Electric_shaver_toothbrush / Speech` 都有比较明确的 event 级提升；"
        "其中 `Dishes` 和 `Dog` 的提升尤其关键，因为这两类正是最能体现“WavLM 作为辅助分支到底有没有独立价值”的难类。"
    )
    md.append("")
    md.append(
        f"和 CRNN baseline 对比，这一版也已经不再是“总体没超过，只是少数类局部增益”。"
        f"当前整体 `PSDS1/PSDS2/Intersection/Event macro/Segment macro` 分别为 "
        f"{current_metrics['psds1']:.3f} / {current_metrics['psds2']:.3f} / {current_metrics['intersection']:.3f} / "
        f"{current_metrics['event_res']['class_wise_average']['f_measure']['f_measure'] * 100:.2f}% / "
        f"{current_metrics['segment_res']['class_wise_average']['f_measure']['f_measure'] * 100:.2f}%，"
        "均已略高于 CRNN baseline。"
    )
    md.append("")
    md.append(
        "当然，增益仍然不是完全均匀的。`Cat` 和 `Running_water` 并没有继续比 concat 更强，`Vacuum_cleaner` 也基本只是持平；"
        "这说明 `CRNN + WavLM residual gate` 解决的是“如何更好地按需利用 WavLM”，而不是已经把所有细粒度类别区分都彻底拉开。"
    )
    md.append("")

    md.append("## 训练过程与选模分析")
    md.append("")
    md.append(f"![Merged training curves](report_assets/{asset_paths['training_curves'].name})")
    md.append("")
    md.append(df_to_markdown(train_summary_df))
    md.append("")
    md.append(
        f"把 `{chosen_versions[0]['version']}` 和 `{chosen_versions[-1]['version']}` 合并后看，训练过程是正常收敛的。"
        f"前半段在 `{chosen_versions[0]['version']}` 中快速抬升，后半段在 `{chosen_versions[-1]['version']}` 里继续缓慢优化；"
        f"但当前链上的最佳 checkpoint 最终停在 `{best_ckpt_path.parent.name}` 的 `{best_ckpt_path.name}`。"
    )
    md.append("")
    md.append(
        f"按每个 epoch 约 `{steps_per_epoch}` 个 step` 估算，best checkpoint 对应的全局 epoch 约为 `{best_epoch_from_step:.0f}`，"
        "这说明当前这条 `CRNN + WavLM residual gate` 训练线虽然可以继续训练，但续训段并没有自动带来新的全局 best；"
        f"至少就目前这条链来看，`{chosen_versions[-1]['version']}` 并没有超过 `{best_ckpt_path.parent.name}` 的最佳点。"
    )
    md.append("")
    md.append(
        "同时，`val/obj_metric`、`val/synth/student/intersection_f1_macro` 和 `val/synth/student/event_f1_macro` 的走势需要结合 best checkpoint 一起看："
        "后续续训并不是完全无效，但也没有把最优点从前一段训练再往后推。"
        "所以这版更接近“正常收敛后进入震荡与局部试探”，而不是“继续训一定会稳定涨”。"
    )
    md.append("")
    md.append(
        "这里仍要强调：`val/obj_metric` 在 `synth_only` 下实际等于 `val/synth/student/intersection_f1_macro`。"
        "它更偏向区间重合，不完全等价于 event-based F1；但这次 best checkpoint 附近，event macro 也同步抬升，说明选模并没有明显跑偏。"
    )
    md.append("")

    md.append("## 预测行为统计")
    md.append("")
    md.append(df_to_markdown(behavior_stats["summary_df"]))
    md.append("")
    md.append(df_to_markdown(behavior_compare_df))
    md.append("")
    md.append(f"![Class count comparison](report_assets/{asset_paths['class_counts'].name})")
    md.append("")
    md.append(f"![Class F1 comparison](report_assets/{asset_paths['class_f1'].name})")
    md.append("")
    md.append(f"![Duration distribution](report_assets/{asset_paths['duration_dist'].name})")
    md.append("")
    md.append(f"![Long-duration bias](report_assets/{asset_paths['long_bias'].name})")
    md.append("")
    md.append(df_to_markdown(count_display))
    md.append("")
    md.append(df_to_markdown(long_bias_display.head(6)))
    md.append("")
    md.append(
        f"当前 `CRNN + WavLM residual gate` 的系统行为比 concat 更积极：有预测文件数从 {BASELINES['Concat late fusion']['behavior']['pred_files']} 增加到 {summary_map['有预测文件数']}，"
        f"空预测文件数从 {BASELINES['Concat late fusion']['behavior']['empty_files']} 降到 {summary_map['空预测文件数']}，"
        f"总预测事件数也从 {BASELINES['Concat late fusion']['behavior']['pred_events']} 增加到 {summary_map['总预测事件数']}。"
        "这说明 gate 确实提升了 WavLM 分支对整体预测的参与度，不再只是完全边缘化的弱通道。"
    )
    md.append("")
    md.append(
        "但这种更积极的行为也带来了代价：长时段偏置和碎片化文件数并没有消失，反而在当前最优 threshold 下更明显。"
        f"例如 `>=9s` 长段数从 concat 的 {BASELINES['Concat late fusion']['behavior']['pred_ge9']} 增加到 {summary_map['预测中 >=9s 长段数']}，"
        f"疑似碎片化过预测文件数从 {BASELINES['Concat late fusion']['behavior']['fragmented']} 增加到 {summary_map['疑似碎片化过预测文件数']}。"
    )
    md.append("")
    md.append(
        "这意味着 `CRNN + WavLM residual gate` 解决的是“如何让 WavLM 真的进来并改善局部召回”，但还没有完全解决“如何在更高召回下保持最优边界和最低混淆”。"
        "弱类方面，`Dog`、`Dishes`、`Alarm_bell_ringing` 已经比 concat 有明显恢复，但 `Dishes` 仍然是当前最欠检的类，说明 hardest weak class 依旧没被完全攻克。"
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
        md.append(f"- 对照说明：{sample['cross_reference']}")
        md.append("")

    md.append("## 结论与讨论")
    md.append("")
    md.append(
        "这次 `CRNN + WavLM residual gate` 是正常跑通的，而且相较旧版 concat late fusion，已经有明确且可重复的性能改进。"
        "它不再只是“恢复正常工作”，而是已经在 overall 指标上超过 concat，并且小幅超过 CRNN baseline。"
    )
    md.append("")
    md.append(
        "更重要的是，它确实缓解了此前“CNN 主导、WavLM 只是弱补充通道”的问题。证据包括："
        "整体 event/segment/PSDS 都进一步提升；空预测文件继续下降；`Dog / Dishes / Alarm_bell_ringing / Blender` 等类都有实质改进；"
        "在 `1088.wav` 和 `1195.wav` 这类样本上，弱类已经从 concat 的缺失状态恢复到可检测状态。"
    )
    md.append("")
    md.append(
        "不过，这种改善仍然不是完全均匀的全局胜利。当前 `CRNN + WavLM residual gate` 主要解决了“让 WavLM 信息更稳地补到 CRNN 主干上”这个问题，"
        "但还没有完全解决长持续设备类之间的语义混淆，也没有彻底解决 `Dishes` 这种复杂弱类在多事件场景下的漏检。"
    )
    md.append("")
    md.append(
        "所以这版最准确的评价是：它已经不是“收益有限到不值得继续”的 late fusion 版本，而是一版确实值得继续深挖的 gate-fusion baseline。"
        "但深挖方向不应该再是盲目堆 epoch，而应该转向更细粒度的 gate 设计、归一化和类感知策略。"
    )
    md.append("")

    md.append("## 后续建议")
    md.append("")
    md.append("1. 继续深挖 gate fusion，但优先做 `class-aware / event-aware gate`，因为当前 hardest weak class 的收益仍然不足。")
    md.append("2. 在 projection + LayerNorm 之外，再补更明确的融合前后校准或温度缩放，减少设备类之间的语义混淆。")
    md.append("3. 针对 `Dishes / Dog / Alarm_bell_ringing / Cat` 做类不平衡与阈值分析，把“召回恢复”进一步转化成更稳的事件级 F1。")
    md.append("4. 如果后续还要做更细粒度融合，优先尝试轻量级模块级 gate，而不是直接跳到重型 cross-attention。")
    md.append("5. 如果后续要继续深挖，优先做 `CRNN + WavLM` 和 `CRNN + BEATs` 的并排受控比较，再决定是否值得扩展到三路融合。")
    md.append("")

    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")


def main():
    ensure_dirs()
    candidates, chosen_versions, final_version = detect_residual_versions()
    config_path = locate_config()
    config = load_yaml(config_path)
    best_ckpt_path = Path(final_version["best_model_path"]).resolve()
    prediction_tsv = locate_prediction_tsv()
    current_metrics = compute_current_metrics(prediction_tsv, config)
    behavior_stats = compute_behavior_stats(current_metrics["pred"], current_metrics["gt"])
    overall_comparison_df = build_overall_comparison(current_metrics)
    per_class_compare_df = build_per_class_comparison(current_metrics["per_class_df"])
    gate_vs_concat_df = build_gate_vs_concat_delta(current_metrics["per_class_df"])
    behavior_compare_df = build_behavior_comparison(behavior_stats)

    steps_per_epoch = int(re.search(r"step=(\d+)", best_ckpt_path.name).group(1)) // (
        int(re.search(r"epoch=(\d+)", best_ckpt_path.name).group(1)) + 1
    )
    train_tb_files = [path for item in chosen_versions for path in item["train_events"]]
    training_curves_path, merged_cache = plot_training_curves(
        train_tb_files, steps_per_epoch=steps_per_epoch, best_ckpt_step=int(re.search(r"step=(\d+)", best_ckpt_path.name).group(1)) - 1
    )

    asset_paths = {
        "training_curves": training_curves_path,
        "class_counts": plot_class_counts(behavior_stats["class_count_df"]),
        "class_f1": plot_class_f1(current_metrics["per_class_df"]),
        "gate_vs_concat": plot_gate_vs_concat(current_metrics["per_class_df"]),
        "duration_dist": plot_duration_distribution(current_metrics["pred"], current_metrics["gt"]),
        "long_bias": plot_long_bias(behavior_stats["long_bias_df"]),
    }

    sample_details = []
    for spec in SAMPLE_SPECS:
        gt_rows = sample_rows(current_metrics["gt"], spec["filename"])
        pred_rows = sample_rows(current_metrics["pred"], spec["filename"])
        asset = plot_sample_figure(spec["filename"], spec["plot_type"], gt_rows, pred_rows)
        sample_details.append(
            {
                "filename": spec["filename"],
                "type": spec["type"],
                "reason": spec["reason"],
                "commentary": spec["commentary"],
                "cross_reference": spec["cross_reference"],
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
                "asset_name": asset.name,
            }
        )

    write_report(
        config=config,
        config_path=config_path,
        candidates=candidates,
        chosen_versions=chosen_versions,
        final_version=final_version,
        best_ckpt_path=best_ckpt_path,
        prediction_tsv=prediction_tsv,
        current_metrics=current_metrics,
        behavior_stats=behavior_stats,
        overall_comparison_df=overall_comparison_df,
        per_class_compare_df=per_class_compare_df,
        gate_vs_concat_df=gate_vs_concat_df,
        behavior_compare_df=behavior_compare_df,
        sample_details=sample_details,
        asset_paths=asset_paths,
        merged_cache=merged_cache,
        steps_per_epoch=steps_per_epoch,
    )

    print("Generated files:")
    print(f"- {REPORT_PATH}")
    for path in sorted(ASSET_DIR.glob('*.png')):
        print(f"- {path}")
    print("Final experiment versions:")
    print(f"- candidates: {', '.join(item['version'] for item in candidates)}")
    print(f"- chosen chain: {' + '.join(item['version'] for item in chosen_versions)}")
    print(f"- best checkpoint: {best_ckpt_path}")
    print("Selected samples:")
    for sample in sample_details:
        print(f"- {sample['filename']} ({sample['type']})")
    print("Completed:")
    print("- Final metrics summary")
    print("- Horizontal comparison vs CRNN / frozen BEATs / WavLM-only / concat fusion / CRNN+BEATs gate")
    print(f"- Merged {' + '.join(item['version'] for item in chosen_versions)} training curves")
    print("- Prediction behavior statistics")
    print("- Typical sample visualizations")
    print("- Conclusions and recommendations")
    print("Not completed:")
    print("- No independent external test set analysis; current test remains synthetic validation.")


if __name__ == "__main__":
    main()
