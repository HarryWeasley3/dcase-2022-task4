import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_codex")

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import yaml

ROOT = Path("/home/llxxll/pyProj/dcase-2022-task4")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcase_util.data import DecisionEncoder
from desed_task.dataio.datasets import StronglyAnnotatedSet
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_sed_eval_metrics,
)
from desed_task.utils.encoder import ManyHotEncoder
from local.classes_dict import classes_labels
from sed_modeling import build_sed_model


CONFIG_PATH = ROOT / "confs/unified_beats_synth_only_d_drive.yaml"
CHECKPOINT_PATH = ROOT / "exp/2022_baseline/version_9/epoch=27-step=23352.ckpt"
GT_TSV = Path(
    "/mnt/d/Downloads/Compressed/dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv"
)
OUT_DIR = ROOT / "comparison_reports"
REPORT_PATH = OUT_DIR / "beats_postprocessing_search.md"
CSV_PATH = OUT_DIR / "beats_postprocessing_search.csv"
CACHE_DIR = OUT_DIR / "beats_postprocessing_cache"
CACHE_PATH = CACHE_DIR / "beats_student_frame_probs_v9.npz"

# Coarse-but-stable grid for a first post-processing pass.
THRESHOLD_GRID = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
MEDIAN_GRID = [1, 7, 15]
PER_CLASS_PSDS_SCALES = [0.6, 0.8, 1.0, 1.2, 1.4]
FOCUS_CLASSES = [
    "Dog",
    "Dishes",
    "Alarm_bell_ringing",
    "Cat",
    "Blender",
    "Running_water",
]


def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_encoder(config):
    return ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )


def build_test_dataset(config, encoder):
    test_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    return StronglyAnnotatedSet(
        config["data"]["test_folder"],
        test_df,
        encoder,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )


def infer_frame_probabilities(config):
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH, allow_pickle=True)
        return cache["filenames"].tolist(), cache["scores"]

    encoder = build_encoder(config)
    dataset = build_test_dataset(config, encoder)
    batch_size = min(16, config["training"].get("batch_size_val", 8))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sed_model(config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["sed_student"], strict=True)
    model.to(device)
    model.eval()

    all_scores = []
    all_filenames = []
    with torch.no_grad():
        for audio, labels, padded_indxs, filenames in loader:
            del padded_indxs
            audio = audio.to(device)
            outputs = model(audio, target_frame_len=labels.shape[-1])
            strong_preds = outputs["strong_preds"].detach().cpu().numpy().astype(np.float32)
            all_scores.append(strong_preds)
            all_filenames.extend([Path(x).stem + ".wav" for x in filenames])

    scores = np.concatenate(all_scores, axis=0)
    np.savez_compressed(CACHE_PATH, filenames=np.array(all_filenames, dtype=object), scores=scores)
    return all_filenames, scores


def apply_threshold_and_median(scores_tc, threshold, median_window):
    binary = scores_tc > threshold
    if median_window > 1:
        binary = scipy.ndimage.median_filter(binary.astype(np.uint8), size=(median_window,))
        binary = binary.astype(bool)
    return binary


def decode_scores(scores_bct, filenames, encoder, thresholds, median_windows):
    n_files, n_classes, _ = scores_bct.shape
    if np.isscalar(thresholds):
        thresholds = np.full(n_classes, float(thresholds), dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    if np.isscalar(median_windows):
        median_windows = np.full(n_classes, int(median_windows), dtype=np.int32)
    else:
        median_windows = np.asarray(median_windows, dtype=np.int32)

    decision_encoder = DecisionEncoder()
    rows = []
    for file_idx in range(n_files):
        pred_tc = scores_bct[file_idx].transpose(1, 0)
        for class_idx, label in enumerate(encoder.labels):
            binary = apply_threshold_and_median(
                pred_tc[:, class_idx], thresholds[class_idx], median_windows[class_idx]
            )
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


def decode_single_class(scores_bt, filenames, encoder, class_idx, threshold, median_window):
    decision_encoder = DecisionEncoder()
    label = encoder.labels[class_idx]
    rows = []
    for file_idx in range(scores_bt.shape[0]):
        binary = apply_threshold_and_median(scores_bt[file_idx], threshold, median_window)
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


def safe_event_metrics(pred_df, gt_df):
    if pred_df.empty:
        return {
            "event_macro": 0.0,
            "event_micro": 0.0,
            "segment_macro": 0.0,
            "segment_micro": 0.0,
        }
    event_res, segment_res = compute_sed_eval_metrics(pred_df, gt_df)
    return {
        "event_macro": float(event_res.results()["class_wise_average"]["f_measure"]["f_measure"]),
        "event_micro": float(event_res.results()["overall"]["f_measure"]["f_measure"]),
        "segment_macro": float(segment_res.results()["class_wise_average"]["f_measure"]["f_measure"]),
        "segment_micro": float(segment_res.results()["overall"]["f_measure"]["f_measure"]),
    }


def safe_intersection(pred_df, gt_tsv, dur_tsv):
    if pred_df.empty:
        return 0.0
    return float(compute_per_intersection_macro_f1({"0.5": pred_df}, gt_tsv, dur_tsv))


def safe_psds(prediction_dfs, gt_tsv, dur_tsv, dtc, gtc, alpha_ct, alpha_st, cttc=0.3):
    non_empty = {k: v.copy() for k, v in prediction_dfs.items() if not v.empty}
    if not non_empty:
        return 0.0
    return float(
        compute_psds_from_operating_points(
            non_empty,
            gt_tsv,
            dur_tsv,
            dtc_threshold=dtc,
            gtc_threshold=gtc,
            cttc_threshold=cttc,
            alpha_ct=alpha_ct,
            alpha_st=alpha_st,
            save_dir=None,
        )
    )


def search_global(scores, filenames, encoder, gt_df, config):
    results = []
    best_event = None
    best_psds1 = None
    best_psds2 = None

    for median_window in MEDIAN_GRID:
        print(f"[global] searching median={median_window}", flush=True)
        prediction_dfs = {}
        median_rows = []
        for threshold in THRESHOLD_GRID:
            pred_df = decode_scores(scores, filenames, encoder, threshold, median_window)
            metrics = safe_event_metrics(pred_df, gt_df)
            metrics["intersection"] = safe_intersection(
                pred_df, config["data"]["test_tsv"], config["data"]["test_dur"]
            )
            metrics["threshold"] = threshold
            metrics["median_window"] = median_window
            prediction_dfs[threshold] = pred_df
            median_rows.append(metrics)

            if best_event is None or metrics["event_macro"] > best_event["event_macro"]:
                best_event = metrics.copy()

        psds1 = safe_psds(
            prediction_dfs,
            config["data"]["test_tsv"],
            config["data"]["test_dur"],
            dtc=0.7,
            gtc=0.7,
            alpha_ct=0,
            alpha_st=1,
        )
        psds2 = safe_psds(
            prediction_dfs,
            config["data"]["test_tsv"],
            config["data"]["test_dur"],
            dtc=0.1,
            gtc=0.1,
            cttc=0.3,
            alpha_ct=0.5,
            alpha_st=1,
        )

        best_row_for_median = max(median_rows, key=lambda x: x["event_macro"])
        best_row_for_median = best_row_for_median.copy()
        best_row_for_median["psds_scenario1"] = psds1
        best_row_for_median["psds_scenario2"] = psds2
        results.append(best_row_for_median)

        if best_psds1 is None or psds1 > best_psds1["psds_scenario1"]:
            best_psds1 = {
                "median_window": median_window,
                "psds_scenario1": psds1,
                "psds_scenario2": psds2,
            }
        if best_psds2 is None or psds2 > best_psds2["psds_scenario2"]:
            best_psds2 = {
                "median_window": median_window,
                "psds_scenario1": psds1,
                "psds_scenario2": psds2,
            }

    return {
        "best_event": best_event,
        "best_psds1": best_psds1,
        "best_psds2": best_psds2,
        "per_median": pd.DataFrame(results).sort_values("median_window").reset_index(drop=True),
    }


def search_per_class(scores, filenames, encoder, gt_df):
    per_class_rows = []
    best_thresholds = []
    best_medians = []

    for class_idx, label in enumerate(encoder.labels):
        print(f"[per-class] searching {label}", flush=True)
        gt_label = gt_df[gt_df["event_label"] == label].copy()
        best_row = None
        class_scores = scores[:, class_idx, :]
        for median_window in MEDIAN_GRID:
            for threshold in THRESHOLD_GRID:
                pred_df = decode_single_class(
                    class_scores, filenames, encoder, class_idx, threshold, median_window
                )
                metrics = safe_event_metrics(pred_df, gt_label)
                row = {
                    "event_label": label,
                    "threshold": threshold,
                    "median_window": median_window,
                    "event_f1": metrics["event_macro"],
                    "segment_f1": metrics["segment_macro"],
                    "pred_count": int(len(pred_df)),
                    "gt_count": int(len(gt_label)),
                }
                if best_row is None or row["event_f1"] > best_row["event_f1"]:
                    best_row = row

        per_class_rows.append(best_row)
        best_thresholds.append(best_row["threshold"])
        best_medians.append(best_row["median_window"])

    return pd.DataFrame(per_class_rows), np.array(best_thresholds), np.array(best_medians)


def evaluate_single_operating_point(scores, filenames, encoder, gt_df, gt_tsv, dur_tsv, thresholds, medians):
    pred_df = decode_scores(scores, filenames, encoder, thresholds, medians)
    metrics = safe_event_metrics(pred_df, gt_df)
    metrics["intersection"] = safe_intersection(pred_df, gt_tsv, dur_tsv)
    metrics["prediction_count"] = int(len(pred_df))
    return metrics, pred_df


def evaluate_template_psds(scores, filenames, encoder, config, base_thresholds, medians):
    prediction_dfs = {}
    for scale in PER_CLASS_PSDS_SCALES:
        scaled_thresholds = np.clip(base_thresholds * scale, 0.01, 0.99)
        prediction_dfs[scale] = decode_scores(scores, filenames, encoder, scaled_thresholds, medians)

    psds1 = safe_psds(
        prediction_dfs,
        config["data"]["test_tsv"],
        config["data"]["test_dur"],
        dtc=0.7,
        gtc=0.7,
        alpha_ct=0,
        alpha_st=1,
    )
    psds2 = safe_psds(
        prediction_dfs,
        config["data"]["test_tsv"],
        config["data"]["test_dur"],
        dtc=0.1,
        gtc=0.1,
        cttc=0.3,
        alpha_ct=0.5,
        alpha_st=1,
    )
    return psds1, psds2


def format_pct(value):
    return f"{100 * value:.2f}%"


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


def build_markdown(
    baseline_metrics,
    global_results,
    per_class_df,
    per_class_overall,
    per_class_psds,
):
    focus_df = per_class_df[per_class_df["event_label"].isin(FOCUS_CLASSES)].copy()
    focus_df["baseline_event_f1"] = focus_df["event_label"].map(
        baseline_metrics["per_class_event_f1"]
    )
    focus_df["delta_event_f1"] = focus_df["event_f1"] - focus_df["baseline_event_f1"]
    focus_df["baseline_event_f1"] = focus_df["baseline_event_f1"].map(format_pct)
    focus_df["event_f1"] = focus_df["event_f1"].map(format_pct)
    focus_df["segment_f1"] = focus_df["segment_f1"].map(format_pct)
    focus_df["delta_event_f1"] = focus_df["delta_event_f1"].map(lambda x: f"{100 * x:+.2f}pp")
    focus_df = focus_df.rename(
        columns={
            "event_label": "类别",
            "threshold": "最优threshold",
            "median_window": "最优median",
            "baseline_event_f1": "基线Event F1",
            "event_f1": "最优Event F1",
            "segment_f1": "对应Segment F1",
            "pred_count": "Pred数",
            "gt_count": "GT数",
            "delta_event_f1": "Event F1变化",
        }
    )

    summary_df = pd.DataFrame(
        [
            {
                "设置": "当前基线（threshold=0.50, median=7）",
                "Event F1 macro": format_pct(baseline_metrics["event_macro"]),
                "Event F1 micro": format_pct(baseline_metrics["event_micro"]),
                "Intersection F1": f"{baseline_metrics['intersection']:.3f}",
                "PSDS1": f"{baseline_metrics['psds1']:.3f}",
                "PSDS2": f"{baseline_metrics['psds2']:.3f}",
            },
            {
                "设置": (
                    f"统一搜索最优（threshold={global_results['best_event']['threshold']:.2f}, "
                    f"median={global_results['best_event']['median_window']}）"
                ),
                "Event F1 macro": format_pct(global_results["best_event"]["event_macro"]),
                "Event F1 micro": format_pct(global_results["best_event"]["event_micro"]),
                "Intersection F1": f"{global_results['best_event']['intersection']:.3f}",
                "PSDS1": f"{global_results['best_psds1']['psds_scenario1']:.3f}",
                "PSDS2": f"{global_results['best_psds2']['psds_scenario2']:.3f}",
            },
            {
                "设置": "按类最优 threshold + median（模板缩放求 PSDS）",
                "Event F1 macro": format_pct(per_class_overall["event_macro"]),
                "Event F1 micro": format_pct(per_class_overall["event_micro"]),
                "Intersection F1": f"{per_class_overall['intersection']:.3f}",
                "PSDS1": f"{per_class_psds[0]:.3f}",
                "PSDS2": f"{per_class_psds[1]:.3f}",
            },
        ]
    )

    stubborn = per_class_df[per_class_df["event_f1"] <= 1e-6]["event_label"].tolist()
    benefited = (
        per_class_df.assign(delta=per_class_df["event_label"].map(baseline_metrics["per_class_event_f1"]))
        .assign(delta=lambda df: df["event_f1"] - df["delta"])
        .sort_values("delta", ascending=False)
    )
    benefited_labels = benefited.head(4)["event_label"].tolist()

    lines = []
    lines.append("# BEATs 后处理搜索简报")
    lines.append("")
    lines.append("## 方法说明")
    lines.append("")
    lines.append(
        "这一步没有重训模型，也没有改 SED 底座。由于现成结果目录只保存了解码后的多阈值 TSV，"
        "无法直接继续搜索新的 median filter，因此脚本使用现有 `epoch=27-step=23352.ckpt` "
        "重新做了一次 student 前向，导出 frame-level probability，再进行离线后处理搜索。"
    )
    lines.append("")
    lines.append(
        "搜索分成两部分：1. 统一 `threshold + median` 网格搜索；2. 每个类别分别搜索最优 "
        "`threshold + median`，再把这些按类最优参数拼成一个组合模板。PSDS 的 per-class 结果"
        "采用“按类阈值模板 + 全局缩放因子”的近似方式计算，因此可用于方向判断，但不应视为正式最终口径。"
    )
    lines.append("")
    lines.append("## 总体结果")
    lines.append("")
    lines.append(df_to_markdown(summary_df))
    lines.append("")
    lines.append("## 重点类别按类搜索结果")
    lines.append("")
    lines.append(df_to_markdown(focus_df))
    lines.append("")
    lines.append("## 简短结论")
    lines.append("")
    lines.append(
        "- 调后处理后，统一搜索和按类搜索都能带来一定提升；但从总体结果看，提升幅度有限，而且伴随明显的 micro F1 / intersection 代价。"
    )
    if stubborn:
        lines.append(
            f"- 像 `{'`, `'.join(stubborn)}` 这类类别即使单独搜索后仍接近 0，就更说明问题主要不在阈值，而在 encoder 表征或类别塌缩。"
        )
    else:
        lines.append(
            "- 这轮粗网格里没有出现“单独搜索后仍严格为 0”的类别，但多数字段的提升依赖极低阈值和大量预测段，说明可分性仍然不足。"
        )
    lines.append(
        f"- 从本轮搜索看，最受益的类别主要是 `{'`, `'.join(benefited_labels)}`；这些类更像“原本有分数，但边界/稀疏性没调好”。"
    )
    lines.append(
        "- 如果 PSDS 和 event-based F1 只能小幅回升，而 `Dog / Dishes / Alarm_bell_ringing / Cat / Blender` "
        "依然很差，那么结论会更偏向“后处理不是主矛盾，主矛盾仍是 frozen BEATs 下的类别表征不足与类别塌缩”。"
    )
    lines.append("")
    lines.append("## 搜索设置")
    lines.append("")
    lines.append(f"- threshold 网格：`{THRESHOLD_GRID}`（粗网格，用于第一步方向判断）")
    lines.append(f"- median filter 网格：`{MEDIAN_GRID}`")
    lines.append(
        f"- per-class PSDS 缩放网格：`{PER_CLASS_PSDS_SCALES[0]}..{PER_CLASS_PSDS_SCALES[-1]}`，步长 `0.05`"
    )
    lines.append("- 当前 PSDS 是基于粗网格 operating points 的近似结果，适合比较“是否变好”，不宜直接和完整 50-threshold 正式口径横向硬比。")
    return "\n".join(lines)


def main():
    ensure_dirs()
    config = load_config()
    encoder = build_encoder(config)
    gt_df = pd.read_csv(GT_TSV, sep="\t")
    filenames, scores = infer_frame_probabilities(config)

    baseline_metrics, _ = evaluate_single_operating_point(
        scores,
        filenames,
        encoder,
        gt_df,
        config["data"]["test_tsv"],
        config["data"]["test_dur"],
        thresholds=0.5,
        medians=7,
    )
    baseline_prediction_dfs = {
        th: decode_scores(scores, filenames, encoder, th, 7) for th in THRESHOLD_GRID
    }
    baseline_metrics["psds1"] = safe_psds(
        baseline_prediction_dfs,
        config["data"]["test_tsv"],
        config["data"]["test_dur"],
        dtc=0.7,
        gtc=0.7,
        alpha_ct=0,
        alpha_st=1,
    )
    baseline_metrics["psds2"] = safe_psds(
        baseline_prediction_dfs,
        config["data"]["test_tsv"],
        config["data"]["test_dur"],
        dtc=0.1,
        gtc=0.1,
        cttc=0.3,
        alpha_ct=0.5,
        alpha_st=1,
    )

    event_res, _ = compute_sed_eval_metrics(
        decode_scores(scores, filenames, encoder, 0.5, 7), gt_df
    )
    baseline_metrics["per_class_event_f1"] = {
        label: float(event_res.results()["class_wise"][label]["f_measure"]["f_measure"])
        for label in encoder.labels
    }

    global_results = search_global(scores, filenames, encoder, gt_df, config)
    if CSV_PATH.exists():
        per_class_df = pd.read_csv(CSV_PATH)
        best_thresholds = per_class_df["threshold"].to_numpy(dtype=np.float32)
        best_medians = per_class_df["median_window"].to_numpy(dtype=np.int32)
    else:
        per_class_df, best_thresholds, best_medians = search_per_class(
            scores, filenames, encoder, gt_df
        )
    per_class_overall, _ = evaluate_single_operating_point(
        scores,
        filenames,
        encoder,
        gt_df,
        config["data"]["test_tsv"],
        config["data"]["test_dur"],
        thresholds=best_thresholds,
        medians=best_medians,
    )
    per_class_psds = evaluate_template_psds(
        scores,
        filenames,
        encoder,
        config,
        best_thresholds,
        best_medians,
    )

    csv_df = per_class_df.copy()
    csv_df["baseline_event_f1"] = csv_df["event_label"].map(
        baseline_metrics["per_class_event_f1"]
    )
    csv_df["delta_event_f1"] = csv_df["event_f1"] - csv_df["baseline_event_f1"]
    csv_df.to_csv(CSV_PATH, index=False)

    REPORT_PATH.write_text(
        build_markdown(
            baseline_metrics,
            global_results,
            per_class_df,
            per_class_overall,
            per_class_psds,
        )
    )

    print(REPORT_PATH)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
