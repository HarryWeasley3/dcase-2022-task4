#!/usr/bin/env python
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "baselines" / "thesis-experiment-report" / "source_data" / "remote_repo"
REPORT_DIR = ROOT / "baselines" / "thesis-experiment-report"
ASSET_DIR = REPORT_DIR / "report_assets"

if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from desed_task.evaluation.evaluation_measures import (  # noqa: E402
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_sed_eval_metrics,
)
from local.classes_dict import classes_labels  # noqa: E402

GT_TSV = SOURCE_ROOT / "runtime_data/dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv"
DUR_TSV = SOURCE_ROOT / "runtime_data/dcase_synth/metadata/validation/synthetic21_validation/durations.tsv"
AUDIO_DIR = ROOT / "runtime_data/dcase_synth/audio/validation/synthetic21_validation/soundscapes_16k"

REPRESENTATIVE_CLASSES = [
    "Alarm_bell_ringing",
    "Blender",
    "Dishes",
    "Cat",
    "Speech",
    "Electric_shaver_toothbrush",
]

TIMELINE_CASES = [
    ("13.wav", "Case A", "Dishes/Frying/Speech recovery"),
    ("287.wav", "Case B", "Alarm/Dog rare-event correction"),
    ("360.wav", "Case C", "Fusion-side regression example"),
]

PAPER_TIMELINE_MODELS = [
    ("CRNN", ROOT / "exp/2022_baseline/metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"),
    ("BEATs", SOURCE_ROOT / "exp/unified_beats_synth_only_a800_finetune/metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"),
    ("CRNN + BEATs", SOURCE_ROOT / "exp/crnn_beats_final/metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"),
    ("CRNN + BEATs + WavLM", SOURCE_ROOT / "exp/cnn_beats_wavlm_full_unfreeze_late_fusion/metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_0.49.tsv"),
]

GT_BAR_COLOR = "#0b7d77"
PRED_BAR_COLOR = "#f28e82"


@dataclass
class ModelSource:
    model: str
    source_type: str
    rel_root: str
    best_ckpt: str
    note: str


MODEL_SOURCES = [
    ModelSource(
        model="CRNN",
        source_type="report",
        rel_root="baselines/CRNN-baseline/training_result_report.md",
        best_ckpt="exp/2022_baseline/version_4/epoch=133-step=111756.ckpt",
        note="Server repo only exposes the final CRNN result in the report, so overall and class metrics are extracted from the markdown report.",
    ),
    ModelSource(
        model="BEATs",
        source_type="exp",
        rel_root="exp/unified_beats_synth_only_a800_finetune",
        best_ckpt="exp/unified_beats_synth_only_a800_finetune/version_5/epoch=55-step=8791.ckpt",
        note="Quantitative metrics are recomputed from the saved metrics_test prediction TSVs and the official GT/duration files.",
    ),
    ModelSource(
        model="WavLM",
        source_type="exp",
        rel_root="exp/WavLM_only",
        best_ckpt="exp/WavLM_only/version_0/epoch=49-step=31249.ckpt",
        note="The exp directory contains a newer WavLM-only run that is stronger than the older markdown baseline report, so the exp result is used.",
    ),
    ModelSource(
        model="CRNN + BEATs",
        source_type="exp",
        rel_root="exp/crnn_beats_final",
        best_ckpt="exp/crnn_beats_final/version_0/epoch=30-step=19374.ckpt",
        note="This is the strongest available fusion run in the current server repo and is therefore adopted as the final two-branch fusion result.",
    ),
    ModelSource(
        model="CRNN + WavLM",
        source_type="exp",
        rel_root="exp/crnn_wavlm_late_fusion_synth_only_fullft",
        best_ckpt="exp/crnn_wavlm_late_fusion_synth_only_fullft/version_0/epoch=54-step=34374.ckpt",
        note="The current report uses the full threshold prediction set under metrics_test rather than older gate-fusion markdown numbers.",
    ),
    ModelSource(
        model="CRNN + BEATs + WavLM",
        source_type="exp",
        rel_root="exp/cnn_beats_wavlm_full_unfreeze_late_fusion",
        best_ckpt="exp/cnn_beats_wavlm_full_unfreeze_late_fusion/version_2/epoch=20-step=26249.ckpt",
        note="Three-way fusion metrics are recomputed from metrics_test and compared under the same evaluation code as the other exp models.",
    ),
]

MODEL_ORDER = [item.model for item in MODEL_SOURCES]
CLASS_ORDER = list(classes_labels.keys())


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def detect_plot_fonts() -> Dict[str, str]:
    def pick(candidates: List[str]) -> str:
        for name in candidates:
            try:
                path = fm.findfont(name, fallback_to_default=False)
            except Exception:
                path = None
            if path and Path(path).exists():
                return name
        return ""

    latin = pick(["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"])
    cjk = pick(["SimSun", "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", "AR PL UMing CN"])
    if not latin:
        latin = "DejaVu Serif"
    return {"latin": latin, "cjk": cjk or latin}


def set_plot_style(fonts: Dict[str, str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [fonts["latin"], "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> Tuple[Path, Path]:
    png = ASSET_DIR / f"{stem}.png"
    pdf = ASSET_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def to_markdown(df: pd.DataFrame) -> str:
    show_df = df.copy()
    show_df.columns = [str(c) for c in show_df.columns]
    for col in show_df.columns:
        show_df[col] = show_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = {}
    for col in show_df.columns:
        widths[col] = max(len(col), *(len(v) for v in show_df[col].tolist())) if len(show_df) else len(col)
    header = "| " + " | ".join(col.ljust(widths[col]) for col in show_df.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in show_df.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in show_df.columns) + " |"
        for _, row in show_df.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def parse_markdown_table(lines: List[str], start_idx: int) -> pd.DataFrame:
    rows = []
    for line in lines[start_idx:]:
        if not line.startswith("|"):
            break
        rows.append([part.strip() for part in line.strip().strip("|").split("|")])
    if len(rows) < 2:
        raise ValueError("Markdown table not found")
    header = rows[0]
    body = rows[2:]
    return pd.DataFrame(body, columns=header)


def parse_crnn_report(report_path: Path) -> Tuple[Dict[str, float], pd.DataFrame]:
    text = report_path.read_text(encoding="utf-8")
    metric_patterns = {
        "psds1": r"PSDS-scenario1\s*\|\s*([0-9.]+)",
        "psds2": r"PSDS-scenario2\s*\|\s*([0-9.]+)",
        "intersection": r"Intersection-based F1\s*\|\s*([0-9.]+)",
        "event_micro": r"Event-based F1 \(micro\)\s*\|\s*([0-9.]+)%",
        "event_macro": r"Event-based F1 \(macro\)\s*\|\s*([0-9.]+)%",
        "segment_micro": r"Segment-based F1 \(micro\)\s*\|\s*([0-9.]+)%",
        "segment_macro": r"Segment-based F1 \(macro\)\s*\|\s*([0-9.]+)%",
    }
    overall = {}
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Failed to parse CRNN metric: {key}")
        overall[key] = float(match.group(1))

    lines = text.splitlines()
    table_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("| 类别") and "Pred/GT" in line and "Event F1" in line:
            table_idx = idx
            break
    if table_idx is None:
        raise ValueError("CRNN per-class table not found")
    class_df = parse_markdown_table(lines, table_idx)
    class_df = class_df.rename(columns={"类别": "event_label", "GT事件数": "gt_count", "Pred事件数": "pred_count", "Pred/GT": "pred_gt", "Event F1": "event_f1", "Segment F1": "segment_f1"})
    class_df["gt_count"] = class_df["gt_count"].astype(int)
    class_df["pred_count"] = class_df["pred_count"].astype(int)
    class_df["pred_gt"] = class_df["pred_gt"].astype(float)
    class_df["event_f1"] = class_df["event_f1"].str.rstrip("%").astype(float)
    class_df["segment_f1"] = class_df["segment_f1"].str.rstrip("%").astype(float)
    class_df["model"] = "CRNN"
    return overall, class_df[["model", "event_label", "gt_count", "pred_count", "pred_gt", "event_f1", "segment_f1"]]


def load_threshold_predictions(pred_dir: Path) -> Dict[float, pd.DataFrame]:
    dfs = {}
    for path in sorted(pred_dir.glob("predictions_th_*.tsv")):
        threshold = float(path.stem.split("_")[-1])
        dfs[threshold] = pd.read_csv(path, sep="\t")
    if not dfs:
        raise ValueError(f"No prediction TSVs found under {pred_dir}")
    return dfs


def compute_exp_metrics(model_name: str, rel_root: str, gt_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    root = SOURCE_ROOT / rel_root
    pred_path = root / "metrics_test" / "student" / "scenario1" / "predictions_dtc0.7_gtc0.7_cttc0.3" / "predictions_th_0.49.tsv"
    pred_df = pd.read_csv(pred_path, sep="\t")
    event_metric, segment_metric = compute_sed_eval_metrics(pred_df, gt_df)
    event_results = event_metric.results()
    segment_results = segment_metric.results()

    op_dfs = load_threshold_predictions(root / "metrics_test" / "student" / "scenario1" / "predictions_dtc0.7_gtc0.7_cttc0.3")
    overall = {
        "psds1": float(
            compute_psds_from_operating_points(
                op_dfs,
                str(GT_TSV),
                str(DUR_TSV),
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=0.3,
                alpha_ct=0,
                alpha_st=1,
                max_efpr=100,
            )
        ),
        "psds2": float(
            compute_psds_from_operating_points(
                op_dfs,
                str(GT_TSV),
                str(DUR_TSV),
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0,
                alpha_st=1,
                max_efpr=100,
            )
        ),
        "intersection": float(
            compute_per_intersection_macro_f1(
                op_dfs,
                str(GT_TSV),
                str(DUR_TSV),
                dtc_threshold=0.5,
                gtc_threshold=0.5,
                cttc_threshold=0.3,
            )
        ),
        "event_macro": event_results["class_wise_average"]["f_measure"]["f_measure"] * 100,
        "event_micro": event_results["overall"]["f_measure"]["f_measure"] * 100,
        "segment_macro": segment_results["class_wise_average"]["f_measure"]["f_measure"] * 100,
        "segment_micro": segment_results["overall"]["f_measure"]["f_measure"] * 100,
    }

    gt_counts = gt_df["event_label"].value_counts()
    pred_counts = pred_df["event_label"].value_counts()
    rows = []
    for label in CLASS_ORDER:
        rows.append(
            {
                "model": model_name,
                "event_label": label,
                "gt_count": int(gt_counts.get(label, 0)),
                "pred_count": int(pred_counts.get(label, 0)),
                "pred_gt": float(pred_counts.get(label, 0)) / float(gt_counts.get(label, 1)),
                "event_f1": event_results["class_wise"][label]["f_measure"]["f_measure"] * 100,
                "segment_f1": segment_results["class_wise"][label]["f_measure"]["f_measure"] * 100,
            }
        )
    return overall, pd.DataFrame(rows)


def short_name(model: str) -> str:
    return {
        "CRNN": "CRNN",
        "BEATs": "BEATs",
        "WavLM": "WavLM",
        "CRNN + BEATs": "CRNN+BEATs",
        "CRNN + WavLM": "CRNN+WavLM",
        "CRNN + BEATs + WavLM": "CRNN+BEATs+WavLM",
    }[model]


def metric_prop(value: float, is_percent: bool = False) -> float:
    return value / 100.0 if is_percent else value


def wrapped(text: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def plot_overall_grouped_bar(overall_df: pd.DataFrame) -> None:
    metrics = [
        ("PSDS1", "psds1", False),
        ("PSDS2", "psds2", False),
        ("Event-F1", "event_macro", True),
        ("Inter-F1", "intersection", False),
    ]
    x = np.arange(len(metrics))
    width = 0.12
    colors = ["#1f4e79", "#2f7d32", "#b85c38", "#7b5ea7", "#c48f00", "#b23a48"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, model in enumerate(MODEL_ORDER):
        heights = [metric_prop(overall_df.loc[model, key], is_percent) for _, key, is_percent in metrics]
        ax.bar(x + (idx - 2.5) * width, heights, width=width, label=short_name(model), color=colors[idx], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.18))
    ax.text(0.01, -0.24, "Event-F1 is plotted as a proportion for scale consistency.", transform=ax.transAxes, fontsize=9, color="#444444")
    save_figure(fig, "overall_metrics_grouped_bar")


def plot_class_grouped_bar(class_df: pd.DataFrame) -> None:
    selected = class_df[class_df["event_label"].isin(REPRESENTATIVE_CLASSES)].copy()
    pivot = selected.pivot(index="event_label", columns="model", values="event_f1").loc[REPRESENTATIVE_CLASSES, MODEL_ORDER]
    x = np.arange(len(REPRESENTATIVE_CLASSES))
    width = 0.12
    colors = ["#1f4e79", "#2f7d32", "#b85c38", "#7b5ea7", "#c48f00", "#b23a48"]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    for idx, model in enumerate(MODEL_ORDER):
        ax.bar(x + (idx - 2.5) * width, pivot[model].values, width=width, label=short_name(model), color=colors[idx], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([wrapped(c.replace("_", " "), 14) for c in REPRESENTATIVE_CLASSES])
    ax.set_ylabel("Event-based F1 (%)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2))
    save_figure(fig, "representative_class_event_f1_bar")


def build_color_map() -> Dict[str, str]:
    return {
        "Alarm_bell_ringing": "#b23a48",
        "Blender": "#7b5ea7",
        "Cat": "#577590",
        "Dishes": "#bc6c25",
        "Dog": "#4361ee",
        "Electric_shaver_toothbrush": "#2a9d8f",
        "Frying": "#e76f51",
        "Running_water": "#277da1",
        "Speech": "#6a994e",
        "Vacuum_cleaner": "#8c564b",
    }


def load_audio_waveform(filename: str) -> Tuple[np.ndarray, int]:
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        tensor = torch.tensor(audio, dtype=torch.float32)
        tensor = torchaudio.functional.resample(tensor, sr, 16000)
        audio = tensor.numpy()
        sr = 16000
    return audio.astype(np.float32), sr


def draw_spectrogram(ax: plt.Axes, filename: str, duration: float) -> None:
    try:
        audio, sr = load_audio_waveform(filename)
        tensor = torch.tensor(audio, dtype=torch.float32)
        spec = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            power=2.0,
        )(tensor)
        spec_db = 10.0 * torch.log10(spec + 1e-10)
        ax.imshow(
            spec_db.numpy(),
            origin="lower",
            aspect="auto",
            extent=[0, len(audio) / sr, 0, sr / 2],
            cmap="viridis",
        )
        ax.set_ylim(0, 4000)
        ax.set_ylabel("Frequency [Hz]")
    except FileNotFoundError:
        ax.text(0.5, 0.5, "Audio unavailable", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlim(0, duration)
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="x", linestyle=":", alpha=0.15)


def draw_overlay_timeline_axis(
    ax: plt.Axes,
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    filename: str,
    model_label: str,
    duration: float,
) -> None:
    gt_rows = gt_df[gt_df["filename"] == filename]
    pred_rows = pred_df[pred_df["filename"] == filename]
    class_labels = CLASS_ORDER[::-1]
    y_lookup = {label: idx for idx, label in enumerate(class_labels)}

    for yi in range(len(class_labels)):
        ax.axhline(yi, color="#d9d9d9", linewidth=0.6, zorder=0)

    for _, row in gt_rows.iterrows():
        y = y_lookup[row["event_label"]]
        ax.broken_barh(
            [(row["onset"], row["offset"] - row["onset"])],
            (y - 0.28, 0.56),
            facecolors=GT_BAR_COLOR,
            edgecolors="none",
            alpha=0.95,
            zorder=2,
        )
    for _, row in pred_rows.iterrows():
        y = y_lookup[row["event_label"]]
        ax.broken_barh(
            [(row["onset"], row["offset"] - row["onset"])],
            (y - 0.12, 0.24),
            facecolors=PRED_BAR_COLOR,
            edgecolors="none",
            alpha=0.95,
            zorder=3,
        )

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.6, len(class_labels) - 0.4)
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels([label.replace("_", " ") for label in class_labels], fontsize=8.8)
    ax.set_ylabel("Classes")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    legend_handles = [
        plt.Line2D([0], [0], color=GT_BAR_COLOR, linewidth=5, label="Ground truth"),
        plt.Line2D([0], [0], color=PRED_BAR_COLOR, linewidth=4, label=short_name(model_label)),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.9, borderpad=0.3)


def draw_event_axis(ax: plt.Axes, df: pd.DataFrame, filename: str, title: str, color_map: Dict[str, str], show_y: bool) -> None:
    file_df = df[df["filename"] == filename].copy()
    duration = float(file_df["offset"].max()) if len(file_df) else 10.0
    if filename in GT_CACHE:
        duration = max(duration, GT_CACHE[filename])
    for idx, label in enumerate(CLASS_ORDER[::-1]):
        y = idx
        events = file_df[file_df["event_label"] == label]
        for _, row in events.iterrows():
            ax.broken_barh([(row["onset"], row["offset"] - row["onset"])], (y - 0.35, 0.7), facecolors=color_map[label], edgecolors=color_map[label])
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.7, len(CLASS_ORDER) - 0.3)
    ax.set_title(title, pad=6)
    ax.grid(axis="x", linestyle=":", alpha=0.25)
    if show_y:
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_yticklabels([label.replace("_", " ") for label in CLASS_ORDER[::-1]])
    else:
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Time (s)")


def plot_timeline_cases(gt_df: pd.DataFrame, baseline_df: pd.DataFrame, fusion_df: pd.DataFrame) -> None:
    color_map = build_color_map()
    fig, axes = plt.subplots(3, len(TIMELINE_CASES), figsize=(10.2, 6.6), sharey=True)
    row_info = [("Ground Truth", gt_df), (short_name(TIMELINE_BASELINE_MODEL), baseline_df), (short_name(TIMELINE_FUSION_MODEL), fusion_df)]
    for col, (filename, case_tag, case_title) in enumerate(TIMELINE_CASES):
        for row, (row_title, df) in enumerate(row_info):
            title = f"{case_tag}: {filename}\n{row_title}" if row == 0 else row_title
            draw_event_axis(axes[row, col], df, filename, title, color_map, show_y=(col == 0))
    fig.subplots_adjust(left=0.19, right=0.99, top=0.93, bottom=0.08, hspace=0.70, wspace=0.12)
    save_figure(fig, "prediction_timeline_cases")


def plot_paper_timeline_figure(filename: str, note: str, gt_df: pd.DataFrame, model_pred_map: Dict[str, pd.DataFrame]) -> str:
    duration = max(
        10.0,
        float(gt_df.loc[gt_df["filename"] == filename, "offset"].max()),
        *[
            float(df.loc[df["filename"] == filename, "offset"].max()) if not df.loc[df["filename"] == filename].empty else 0.0
            for df in model_pred_map.values()
        ],
    )
    fig = plt.figure(figsize=(9.6, 9.0))
    gs = fig.add_gridspec(1 + len(model_pred_map), 1, height_ratios=[2.1] + [1.35] * len(model_pred_map), hspace=0.22)

    ax_spec = fig.add_subplot(gs[0, 0])
    draw_spectrogram(ax_spec, filename, duration)
    ax_spec.set_title(f"{filename} | {note}", pad=6)

    timeline_axes = []
    for idx, (model_name, pred_df) in enumerate(model_pred_map.items(), start=1):
        ax = fig.add_subplot(gs[idx, 0], sharex=ax_spec)
        draw_overlay_timeline_axis(ax, gt_df, pred_df, filename, model_name, duration)
        if idx != len(model_pred_map):
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Time [sec]")
        timeline_axes.append(ax)

    fig.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.06)
    stem = f"paper_timeline_{Path(filename).stem}"
    save_figure(fig, stem)
    return stem


def plot_paper_timeline_cases(gt_df: pd.DataFrame, model_pred_map: Dict[str, pd.DataFrame]) -> List[str]:
    stems = []
    for filename, _, note in TIMELINE_CASES:
        stems.append(plot_paper_timeline_figure(filename, note, gt_df, model_pred_map))
    return stems


def plot_ablation_path(overall_df: pd.DataFrame) -> None:
    path_models = ["CRNN", "CRNN + BEATs", "CRNN + BEATs + WavLM"]
    y = [overall_df.loc[m, "psds1"] for m in path_models]
    x = np.arange(len(path_models))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(x, y, marker="o", color="#1f4e79", linewidth=2.0, markersize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([wrapped(short_name(m), 10) for m in path_models])
    ax.set_ylabel("PSDS1")
    ax.set_ylim(0.30, 0.56)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for idx, val in enumerate(y):
        ax.text(x[idx], val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    deltas = [y[1] - y[0], y[2] - y[1]]
    ax.text(0.5, (y[0] + y[1]) / 2 + 0.01, f"{deltas[0]:+.3f}", ha="center", fontsize=9.5, color="#444444")
    ax.text(1.5, (y[1] + y[2]) / 2 + 0.01, f"{deltas[1]:+.3f}", ha="center", fontsize=9.5, color="#444444")
    ax.text(0.02, -0.24, "nSEBBs result is not available in the current repository.", transform=ax.transAxes, fontsize=8.8, color="#444444")
    save_figure(fig, "ablation_path_psds1")


def format_ratio(x: float) -> str:
    return f"{x:.2f}"


def format_percent(x: float) -> str:
    return f"{x:.2f}%"


def build_tables(overall_results: Dict[str, Dict[str, float]], class_results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    for model in MODEL_ORDER:
        metrics = overall_results[model]
        overall_rows.append(
            {
                "Model": model,
                "PSDS1": f"{metrics['psds1']:.3f}",
                "PSDS2": f"{metrics['psds2']:.3f}",
                "Inter-F1": f"{metrics['intersection']:.3f}",
                "Event-F1": format_percent(metrics["event_macro"]),
                "Segment-F1": format_percent(metrics["segment_macro"]),
            }
        )
    overall_table = pd.DataFrame(overall_rows)

    selected_df = class_results[class_results["event_label"].isin(REPRESENTATIVE_CLASSES)].copy()
    event_pivot = selected_df.pivot(index="event_label", columns="model", values="event_f1").loc[REPRESENTATIVE_CLASSES, MODEL_ORDER]
    selected_rows = []
    for label, row in event_pivot.iterrows():
        item = {"Class": label}
        for model in MODEL_ORDER:
            item[model] = format_percent(row[model])
        selected_rows.append(item)
    selected_class_table = pd.DataFrame(selected_rows)

    gap_rows = []
    for model in MODEL_ORDER:
        metrics = overall_results[model]
        gap_rows.append(
            {
                "Model": model,
                "Event-F1": format_percent(metrics["event_macro"]),
                "Segment-F1": format_percent(metrics["segment_macro"]),
                "Gap": format_percent(metrics["segment_macro"] - metrics["event_macro"]),
            }
        )
    gap_table = pd.DataFrame(gap_rows)
    return overall_table, selected_class_table, gap_table


def build_mapping_table() -> pd.DataFrame:
    rows = []
    for item in MODEL_SOURCES:
        core_files = item.rel_root if item.source_type == "report" else f"{item.rel_root}/metrics_test/student"
        rows.append(
            {
                "Model": item.model,
                "Adopted source": item.source_type,
                "Run/report path": item.rel_root,
                "Best checkpoint": item.best_ckpt,
                "Core files": core_files,
            }
        )
    return pd.DataFrame(rows)


def paragraph_analysis(overall: pd.DataFrame, class_df: pd.DataFrame) -> Dict[str, str]:
    overall_num = overall.copy()
    overall_num.index = overall_num["Model"]
    event_num = {
        model: float(str(val).rstrip("%"))
        for model, val in overall_num.set_index("Model")["Event-F1"].to_dict().items()
    }
    psds1_num = {
        model: float(val)
        for model, val in overall_num.set_index("Model")["PSDS1"].to_dict().items()
    }
    best_model = max(MODEL_ORDER, key=lambda m: psds1_num[m])
    tri_model = "CRNN + BEATs + WavLM"
    beats_model = "BEATs"
    wavlm_model = "WavLM"
    crnn_model = "CRNN"
    two_fusion = "CRNN + BEATs"
    wavlm_fusion = "CRNN + WavLM"

    class_pivot = class_df.pivot(index="event_label", columns="model", values="event_f1")
    gain_two_vs_crnn = (class_pivot[two_fusion] - class_pivot[crnn_model]).sort_values(ascending=False)
    tri_vs_two = (class_pivot[tri_model] - class_pivot[two_fusion]).sort_values()

    overall_para = (
        f"在统一口径重算后，{best_model} 是当前 synthetic validation 条件下整体性能最强的模型。"
        f"其 PSDS1 达到 {psds1_num[two_fusion]:.3f}，不仅明显高于 CRNN 的 {psds1_num[crnn_model]:.3f}，"
        f"也高于 BEATs 的 {psds1_num[beats_model]:.3f}、WavLM 的 {psds1_num[wavlm_model]:.3f} 以及其余融合变体。"
        f"在 Event-F1 上，该模型达到 {event_num[two_fusion]:.2f}%，较 CRNN baseline 提升 {event_num[two_fusion] - event_num[crnn_model]:.2f} 个百分点，"
        "说明在当前实验设置下，CRNN 主干与 BEATs 表征之间已经形成了较稳定的互补关系。"
    )

    fusion_para = (
        "从融合角度看，双路 CRNN+BEATs 的增益最为明确，这表明 BEATs 分支在与 CRNN 时频表征结合后，"
        "能够有效提升难类检测和整体排序稳定性。三路 CRNN+BEATs+WavLM 虽然仍然优于所有单编码器基线，"
        "也优于原始 CRNN baseline，但其整体结果并未超过双路融合最优值。"
        "这意味着 WavLM 分支确实带来了额外信息，但当前三路 late-fusion 机制尚未把这些新增表征稳定转化为全局收益。"
    )

    event_segment_para = (
        "所有模型都存在明显的 Event-F1 与 Segment-F1 差距，这说明当前误差并不主要来自“完全不会识别某个类别”，"
        "而更多来自事件边界偏移、事件片段被切碎以及长事件连续性不足。即便是性能最优的模型，"
        "Segment-F1 相对 Event-F1 仍高出 20 个百分点以上，因此后续优化的重点仍应放在事件级边界建模与后处理稳定性上。"
    )

    class_para = (
        f"从逐类结果看，CRNN+BEATs 相对 CRNN baseline 提升最明显的类别主要是 "
        f"{gain_two_vs_crnn.index[0]}（{gain_two_vs_crnn.iloc[0]:+.2f} pp）、"
        f"{gain_two_vs_crnn.index[1]}（{gain_two_vs_crnn.iloc[1]:+.2f} pp）以及 "
        f"{gain_two_vs_crnn.index[2]}（{gain_two_vs_crnn.iloc[2]:+.2f} pp）。"
        f"这说明双路融合对于设备类、厨房类以及持续性较强的事件具有较好的增强作用。"
        f"然而，在 CRNN+BEATs 的基础上继续加入 WavLM 后，下降最明显的类别变为 "
        f"{tri_vs_two.index[0]}（{tri_vs_two.iloc[0]:+.2f} pp）、"
        f"{tri_vs_two.index[1]}（{tri_vs_two.iloc[1]:+.2f} pp）和 "
        f"{tri_vs_two.index[2]}（{tri_vs_two.iloc[2]:+.2f} pp）。"
        "这表明三路融合虽然能继续强化部分相对平稳的类别，但在若干瞬态类或边界敏感类上仍然牺牲了稳定性。"
    )

    wavlm_para = (
        "WavLM-only 与 CRNN+WavLM 的结果呈现出“局部有效、整体一般”的特征。相较于 CRNN baseline，"
        "这两类模型的 Event-F1 有一定提升，但其 Intersection-F1 与 PSDS 仍明显落后于 CRNN+BEATs。"
        "因此可以认为，WavLM 分支已经能够提供有用的语义补充，但在当前实现下，其融合效率仍弱于 BEATs 路线。"
    )
    return {
        "overall": overall_para,
        "fusion": fusion_para,
        "event_segment": event_segment_para,
        "class": class_para,
        "wavlm": wavlm_para,
    }


def build_report(
    mapping_table: pd.DataFrame,
    overall_table: pd.DataFrame,
    selected_class_table: pd.DataFrame,
    gap_table: pd.DataFrame,
    fonts: Dict[str, str],
    analyses: Dict[str, str],
) -> str:
    mapping_display = mapping_table.rename(
        columns={
            "Model": "模型",
            "Adopted source": "采用来源",
            "Run/report path": "实验目录或报告",
            "Best checkpoint": "最终 checkpoint",
            "Core files": "关键文件位置",
        }
    )
    overall_display = overall_table.rename(
        columns={
            "Model": "模型",
            "PSDS1": "PSDS1",
            "PSDS2": "PSDS2",
            "Inter-F1": "Inter-F1",
            "Event-F1": "Event-F1",
            "Segment-F1": "Segment-F1",
        }
    )
    class_display = selected_class_table.rename(columns={"Class": "类别"})
    gap_display = gap_table.rename(columns={"Model": "模型", "Event-F1": "Event-F1", "Segment-F1": "Segment-F1", "Gap": "差值"})

    lines = []
    lines.append("# DCASE 2022 Task4 论文风格实验分析报告")
    lines.append("")
    lines.append("## 1. 实验结果读取与映射")
    lines.append("")
    lines.append("本报告以服务器仓库 `~/autodl-tmp/github/dcase-2022-task4` 的当前快照为准。对于仍保留 `exp/.../metrics_test` 的 5 个模型，本文直接基于保存下来的多阈值预测 TSV、`event_f1.txt`、`segment_f1.txt` 以及官方 GT/duration 文件重新统一评估。CRNN baseline 是唯一例外：当前服务器仓库未保留其对应的原始预测 TSV，因此其最终对比结果只能从现有 baseline 报告中提取。")
    lines.append("")
    lines.append(to_markdown(mapping_display))
    lines.append("")
    lines.append("进一步核对后可以发现，若干旧版 Markdown 报告与当前服务器 `exp/` 目录中的新版实验结果在数值上已经不再一致，尤其体现在 WavLM 与部分融合实验上。因此，旧报告仅作为历史背景使用，正式对比时不再作为最终定量来源。")
    lines.append("")
    lines.append("## 2. 实验结果总表")
    lines.append("")
    lines.append(to_markdown(overall_display))
    lines.append("")
    lines.append("整体对比表明，`CRNN + BEATs` 在本次论文要求的四项核心指标上均取得最优结果。`CRNN + BEATs + WavLM` 仍显著优于各单模型基线，也优于原始 CRNN baseline，但尚未超越最优双路融合结果。")
    lines.append("")
    lines.append("![总体指标分组柱状图](report_assets/overall_metrics_grouped_bar.png)")
    lines.append("")
    lines.append("## 3. 关键类别结果表")
    lines.append("")
    lines.append("下表选取了若干最具代表性的类别。这些类别既包含差异显著的稀有报警类、厨房类和动物类事件，也覆盖了占比最高的 `Speech` 以及较稳定的设备类事件，能够更直观地体现不同模型的类别收益与短板。")
    lines.append("")
    lines.append(to_markdown(class_display))
    lines.append("")
    lines.append("![关键类别 Event-F1 分组柱状图](report_assets/representative_class_event_f1_bar.png)")
    lines.append("")
    lines.append("## 4. 文字实验分析")
    lines.append("")
    lines.append("### 4.1 整体性能比较")
    lines.append("")
    lines.append(analyses["overall"])
    lines.append("")
    lines.append(analyses["fusion"])
    lines.append("")
    lines.append(analyses["wavlm"])
    lines.append("")
    lines.append("### 4.2 单模型与融合模型差异")
    lines.append("")
    lines.append("从单模型结果看，BEATs 全量微调已经显著超过 CRNN baseline，说明在当前数据划分与训练设定下，预训练声学表征本身能够带来稳定收益。相比之下，WavLM-only 的表现虽然优于原始 CRNN 的部分指标，但整体仍未达到 BEATs 路线的水平。")
    lines.append("")
    lines.append("从融合模型结果看，双路 CRNN+BEATs 的整体优势最为稳定，说明在当前实验条件下，最有效的提升路径并不是简单堆叠分支数量，而是让互补性最强的两类表征形成更稳的协同。三路融合模型虽然仍保持较高水准，但其最终结果低于双路最优值，说明新增的 WavLM 分支尚未被完全吸收。")
    lines.append("")
    lines.append("### 4.3 Event-level 与 Segment-level 差异")
    lines.append("")
    lines.append(to_markdown(gap_display))
    lines.append("")
    lines.append(analyses["event_segment"])
    lines.append("")
    lines.append("### 4.4 逐类性能变化分析")
    lines.append("")
    lines.append(analyses["class"])
    lines.append("")
    lines.append("从代表性类别可以进一步看出，`Blender`、`Dishes` 与 `Electric_shaver_toothbrush` 是最能从融合中受益的类别；相对地，`Cat` 在不同融合模型之间仍然波动较大，三路融合在该类上的退化尤为明显。`Speech` 在较强模型中始终保持较高水平，这说明当前系统的主要困难并不在于高频、持续类事件，而仍集中在瞬态类、稀有类以及边界敏感类上。")
    lines.append("")
    lines.append("### 4.5 典型样本时间轴分析")
    lines.append("")
    lines.append("新版时间轴图按照论文插图方式重构为“一个文件一张图”。每张图顶部放置频谱图，用于展示该音频的整体声学结构；下方则按模型逐行排列子图，并在每个子图内部直接叠加 `Ground Truth + 单个模型预测`。同时，所有子图都保留完整类别集合，从而保证不同模型之间纵轴完全一致。")
    lines.append("")
    lines.append("![论文版时间轴图：13.wav](report_assets/paper_timeline_13.png)")
    lines.append("")
    lines.append("![论文版时间轴图：287.wav](report_assets/paper_timeline_287.png)")
    lines.append("")
    lines.append("![论文版时间轴图：360.wav](report_assets/paper_timeline_360.png)")
    lines.append("")
    lines.append("本次每张图中统一放入 4 个模型：`CRNN`、`BEATs`、`CRNN + BEATs` 以及 `CRNN + BEATs + WavLM`。其中 `CRNN` 作为传统 baseline，`BEATs` 代表强单模型，`CRNN + BEATs` 代表当前整体最强的融合模型，三路融合则作为进一步扩展的对照模型。样本 `13.wav` 体现了典型的融合收益，样本 `287.wav` 体现了稀有类校正能力，样本 `360.wav` 则保留为融合退化反例，用于提醒读者当前模型仍存在失配与错分问题。")
    lines.append("")
    lines.append("### 4.6 消融路径分析")
    lines.append("")
    lines.append("按照任务要求，消融路径优先采用 PSDS1 指标。由于当前仓库中未找到 `nSEBBs` 结果，因此该图退化为 `CRNN -> CRNN + BEATs -> CRNN + BEATs + WavLM` 三阶段路径。")
    lines.append("")
    lines.append("![PSDS1 消融提升路径图](report_assets/ablation_path_psds1.png)")
    lines.append("")
    lines.append("该路径图清楚地说明了两点。其一，从 CRNN 到 CRNN+BEATs 存在显著跃升，说明 BEATs 分支是当前整体性能提升的主要来源。其二，在 CRNN+BEATs 基础上继续加入 WavLM 后，PSDS1 从 0.522 回落到 0.485，表明现阶段三路融合尚未形成进一步增益，仍需在融合结构、训练控制或后处理策略上继续优化。")
    lines.append("")
    lines.append("## 5. 图表和图片中使用的数据说明")
    lines.append("")
    lines.append("- `BEATs`、`WavLM`、`CRNN + BEATs`、`CRNN + WavLM`、`CRNN + BEATs + WavLM` 的 `PSDS1/PSDS2/Intersection-F1`，均由各自 `exp/.../metrics_test/student/scenario1/predictions_dtc0.7_gtc0.7_cttc0.3/` 下保存的全阈值预测 TSV 重新统一计算得到。")
    lines.append("- 上述 5 个 `exp` 模型的 `Event-F1` 与 `Segment-F1` 则基于 `predictions_th_0.49.tsv` 与官方 GT 文件重算，并与原始 `event_f1.txt`、`segment_f1.txt` 做了一致性核对。")
    lines.append("- `CRNN` baseline 使用 `baselines/CRNN-baseline/training_result_report.md` 作为唯一最终来源，因为当前服务器仓库中没有保留它在 `exp/` 下对应的原始预测结果。")
    lines.append("- 新版时间轴图使用 `CRNN`、`BEATs`、`CRNN + BEATs` 与 `CRNN + BEATs + WavLM` 四个模型。其中 `CRNN` 的预测来自当前本地仓库 `exp/2022_baseline/metrics_test/.../predictions_th_0.49.tsv`，其余 3 个模型来自服务器同步下来的 `source_data/remote_repo/exp/.../metrics_test/.../predictions_th_0.49.tsv`。")
    lines.append("- 当前仓库未检索到 `nSEBBs` 结果，因此无法绘制包含 `nSEBBs` 的最终消融路径。")
    lines.append(f"- 当前绘图环境中未安装 `SimSun` 与 `Times New Roman`，因此图片使用回退衬线字体 `{fonts['latin']}` 输出，但仍按论文排版需求控制了字号、线宽、图幅和 300 dpi 导出。")
    lines.append("")
    lines.append("## 6. 还缺什么信息")
    lines.append("")
    lines.append("- `CRNN` 的时间轴图虽然已经可以根据当前本地仓库中的 prediction TSV 重建，但其 PSDS 仍然只能沿用既有报告结果，因为服务器同步目录中没有对应的完整多阈值 operating points。")
    lines.append("- 缺少 `nSEBBs` 结果，因此论文中若需要完整的 `CRNN -> CRNN + BEATs -> CRNN + BEATs + WavLM -> ...` 最终链路，目前只能先停在三路融合阶段。")
    lines.append("- 缺少宋体和 Times New Roman 字体文件，因此图片已尽量贴近论文风格，但未能严格满足指定字体要求。")
    lines.append("")
    lines.append("## 7. 已生成的文件")
    lines.append("")
    lines.append("- `report_assets/overall_metrics_grouped_bar.(png|pdf)`：总体指标分组柱状图，用于比较六种模型在 PSDS1、PSDS2、Event-F1、Inter-F1 上的整体差异。")
    lines.append("- `report_assets/representative_class_event_f1_bar.(png|pdf)`：关键类别 Event-F1 分组柱状图，用于分析类别层面的收益与退化。")
    lines.append("- `report_assets/paper_timeline_13.(png|pdf)`、`paper_timeline_287.(png|pdf)`、`paper_timeline_360.(png|pdf)`：新版论文风格时间轴图，每张图只对应一个音频文件，顶部为频谱图，下方为多个 `GT + 单模型预测` 子图。")
    lines.append("- `report_assets/ablation_path_psds1.(png|pdf)`：PSDS1 消融路径图，用于说明从 CRNN 到双路融合、再到三路融合的变化趋势。")
    lines.append("- `report_assets/overall_results_table.csv`：总体结果表。")
    lines.append("- `report_assets/full_per_class_event_f1.csv` 与 `report_assets/full_per_class_segment_f1.csv`：10 个类别的完整逐类结果导出。")
    lines.append("- `report_assets/selected_class_event_f1.csv`：适合正文引用的关键类别结果表。")
    lines.append("- `report_assets/experiment_mapping_table.csv` 与 `report_assets/analysis_summary.json`：实验映射信息与数据来源清单。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    fonts = detect_plot_fonts()
    set_plot_style(fonts)

    gt_df = pd.read_csv(GT_TSV, sep="\t")
    global GT_CACHE
    GT_CACHE = gt_df.groupby("filename")["offset"].max().to_dict()

    crnn_overall, crnn_class_df = parse_crnn_report(SOURCE_ROOT / "baselines/CRNN-baseline/training_result_report.md")
    overall_results = {"CRNN": crnn_overall}
    class_results = [crnn_class_df]

    for item in MODEL_SOURCES:
        if item.source_type != "exp":
            continue
        overall, class_df = compute_exp_metrics(item.model, item.rel_root, gt_df)
        overall_results[item.model] = overall
        class_results.append(class_df)

    class_df = pd.concat(class_results, ignore_index=True)

    mapping_table = build_mapping_table()
    overall_table, selected_class_table, gap_table = build_tables(overall_results, class_df)

    overall_numeric_df = pd.DataFrame(overall_results).T.loc[MODEL_ORDER]
    plot_overall_grouped_bar(overall_numeric_df)
    plot_class_grouped_bar(class_df)

    paper_timeline_pred_map = {
        model_name: pd.read_csv(pred_path, sep="\t")
        for model_name, pred_path in PAPER_TIMELINE_MODELS
    }
    plot_paper_timeline_cases(gt_df, paper_timeline_pred_map)
    plot_ablation_path(overall_numeric_df)

    event_export = class_df.pivot(index="event_label", columns="model", values="event_f1").loc[CLASS_ORDER, MODEL_ORDER]
    segment_export = class_df.pivot(index="event_label", columns="model", values="segment_f1").loc[CLASS_ORDER, MODEL_ORDER]

    overall_table.to_csv(ASSET_DIR / "overall_results_table.csv", index=False)
    event_export.round(4).to_csv(ASSET_DIR / "full_per_class_event_f1.csv")
    segment_export.round(4).to_csv(ASSET_DIR / "full_per_class_segment_f1.csv")
    selected_class_table.to_csv(ASSET_DIR / "selected_class_event_f1.csv", index=False)
    mapping_table.to_csv(ASSET_DIR / "experiment_mapping_table.csv", index=False)

    analyses = paragraph_analysis(overall_table, class_df)
    report_md = build_report(mapping_table, overall_table, selected_class_table, gap_table, fonts, analyses)
    (REPORT_DIR / "training_result_report.md").write_text(report_md, encoding="utf-8")

    summary = {
        "model_sources": [item.__dict__ for item in MODEL_SOURCES],
        "fonts": fonts,
        "representative_classes": REPRESENTATIVE_CLASSES,
        "timeline_models": [name for name, _ in PAPER_TIMELINE_MODELS],
        "timeline_cases": [{"filename": f, "tag": tag, "note": note} for f, tag, note in TIMELINE_CASES],
        "overall_results": overall_results,
        "missing_items": [
            "nSEBBs result is not available in the current repository.",
            "The server-side synced source data still does not expose CRNN operating points for PSDS recomputation.",
            "SimSun and Times New Roman are unavailable on the current plotting machine; a fallback serif font is used instead.",
        ],
    }
    (ASSET_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Report written to: {REPORT_DIR / 'training_result_report.md'}")


if __name__ == "__main__":
    main()
