import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tensorboard.backend.event_processing import event_accumulator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desed_task.dataio.datasets import StronglyAnnotatedSet
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
)
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.scaler import TorchScaler
from local.classes_dict import classes_labels
from local.utils import batched_decode_preds, compute_sed_eval_metrics
from sed_modeling import build_sed_model


INTERESTING_CLASSES = [
    "Dog",
    "Cat",
    "Dishes",
    "Alarm_bell_ringing",
    "Blender",
    "Running_water",
]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_event_file(version_dir):
    files = sorted(Path(version_dir).glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in {version_dir}")
    return files[-1]


def summarize_curve(version_dir):
    event_file = get_event_file(version_dir)
    ea = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    out = {}
    for tag in [
        "val/obj_metric",
        "val/synth/student/intersection_f1_macro",
        "val/synth/student/event_f1_macro",
        "train/student/loss_strong",
        "val/synth/student/loss_strong",
    ]:
        if tag in tags:
            vals = ea.Scalars(tag)
            out[tag] = [{"step": v.step, "value": float(v.value)} for v in vals]
    return out


def build_encoder(config):
    return ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )


def build_val_loader(config, encoder, batch_size=None):
    gt_df = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
    dataset = StronglyAnnotatedSet(
        config["data"]["synth_val_folder"],
        gt_df,
        encoder,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size or config["training"]["batch_size_val"],
        shuffle=False,
        num_workers=0,
    )
    return gt_df, loader


def load_student_model(config, ckpt_path, device):
    model = build_sed_model(config)
    model.set_input_scaler(
        TorchScaler(
            "instance",
            config["scaler"]["normtype"],
            config["scaler"]["dims"],
        )
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {
        key[len("sed_student.") :]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("sed_student.")
    }
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, ckpt


def tensor_stats(tensors):
    flat = torch.cat([t.reshape(-1) for t in tensors]).float()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def classwise_f1(metric_results):
    return {
        cls: float(values["f_measure"]["f_measure"])
        for cls, values in metric_results["class_wise"].items()
    }


def evaluate_checkpoint(config, ckpt_path, device):
    encoder = build_encoder(config)
    gt_df, loader = build_val_loader(config, encoder)
    model, ckpt = load_student_model(config, ckpt_path, device)

    use_amp = device.type == "cuda" and config["training"]["precision"] == "16-mixed"
    pred_df = pd.DataFrame()
    feature_buffers = {"cnn": [], "beats": [], "fused": [], "merged": []}

    with torch.no_grad():
        for batch_idx, (audio, labels, _, filenames) in enumerate(loader):
            audio = audio.to(device)
            target_len = labels.shape[-1]
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(audio, target_frame_len=target_len, return_intermediates=True)
            strong = outputs["strong_preds"].detach().cpu()
            decoded = batched_decode_preds(
                strong,
                filenames,
                encoder,
                thresholds=[0.5],
                median_filter=config["training"]["median_window"],
            )
            pred_df = pd.concat([pred_df, decoded[0.5]], ignore_index=True)

            if batch_idx < 5:
                feature_buffers["cnn"].append(outputs["cnn_sequence_features"].detach().cpu())
                feature_buffers["beats"].append(outputs["beats_aligned_features"].detach().cpu())
                feature_buffers["fused"].append(outputs["fused_sequence_features"].detach().cpu())
                feature_buffers["merged"].append(
                    outputs["encoder_sequence_features"].detach().cpu()
                )

    event_metric, segment_metric = compute_sed_eval_metrics(pred_df, gt_df)
    event_res = event_metric.results()
    segment_res = segment_metric.results()
    pred_counts = pred_df.groupby("event_label").size().to_dict() if len(pred_df) else {}
    gt_counts = gt_df.groupby("event_label").size().to_dict()
    pred_durations = (
        pred_df["offset"] - pred_df["onset"] if len(pred_df) else pd.Series(dtype=float)
    )
    total_files = gt_df["filename"].nunique()
    predicted_files = pred_df["filename"].nunique() if len(pred_df) else 0

    return {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "intersection": float(
            compute_per_intersection_macro_f1(
                {0.5: pred_df},
                config["data"]["synth_val_tsv"],
                config["data"]["synth_val_dur"],
            )
        ),
        "event_macro": float(event_res["class_wise_average"]["f_measure"]["f_measure"]),
        "event_micro": float(event_res["overall"]["f_measure"]["f_measure"]),
        "segment_macro": float(
            segment_res["class_wise_average"]["f_measure"]["f_measure"]
        ),
        "segment_micro": float(segment_res["overall"]["f_measure"]["f_measure"]),
        "event_classwise": classwise_f1(event_res),
        "segment_classwise": classwise_f1(segment_res),
        "pred_counts": pred_counts,
        "gt_counts": gt_counts,
        "empty_ratio": float((total_files - predicted_files) / total_files),
        "pred_events": int(len(pred_df)),
        "gt_events": int(len(gt_df)),
        "pred_avg_duration": float(pred_durations.mean()) if len(pred_durations) else 0.0,
        "long_ratio_ge_9s": float((pred_durations >= 9.0).mean()) if len(pred_durations) else 0.0,
        "feature_stats": {
            name: tensor_stats(buffers) for name, buffers in feature_buffers.items()
        },
    }


def print_curve_summary(curves):
    print("## Curve Summary")
    for tag, values in curves.items():
        if not values:
            continue
        last = values[-1]["value"]
        max_v = max(v["value"] for v in values)
        min_v = min(v["value"] for v in values)
        print(f"{tag}: n={len(values)} last={last:.6f} max={max_v:.6f} min={min_v:.6f}")
        if tag.startswith("val/"):
            print("  series:", [round(v["value"], 6) for v in values])


def print_checkpoint_summary(name, result):
    print(f"\n## {name}")
    print(f"epoch={result['epoch']} step={result['global_step']}")
    print(
        "intersection={:.6f} event_macro={:.6f} event_micro={:.6f} "
        "segment_macro={:.6f} segment_micro={:.6f}".format(
            result["intersection"],
            result["event_macro"],
            result["event_micro"],
            result["segment_macro"],
            result["segment_micro"],
        )
    )
    print(
        "empty_ratio={:.6f} pred_events={} gt_events={} pred_avg_duration={:.6f} long_ratio_ge_9s={:.6f}".format(
            result["empty_ratio"],
            result["pred_events"],
            result["gt_events"],
            result["pred_avg_duration"],
            result["long_ratio_ge_9s"],
        )
    )
    for name, stats in result["feature_stats"].items():
        print(
            f"{name}: mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )
    print("interesting_classes:")
    for cls in INTERESTING_CLASSES:
        print(
            f"  {cls}: event={result['event_classwise'].get(cls, 0.0):.6f} "
            f"segment={result['segment_classwise'].get(cls, 0.0):.6f} "
            f"pred/gt={result['pred_counts'].get(cls, 0)}/{result['gt_counts'].get(cls, 0)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_file",
        default="confs/crnn_beats_late_fusion_synth_only.yaml",
    )
    parser.add_argument(
        "--version_dir",
        default="exp/2022_baseline/version_19",
    )
    parser.add_argument(
        "--best_ckpt",
        default=None,
    )
    parser.add_argument(
        "--last_ckpt",
        default=None,
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    args = parser.parse_args()

    config = load_config(args.conf_file)
    version_dir = Path(args.version_dir)
    best_ckpt = args.best_ckpt or str(version_dir / "epoch=3-step=2500.ckpt")
    last_ckpt = args.last_ckpt or str(version_dir / "last.ckpt")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    curves = summarize_curve(version_dir)
    print_curve_summary(curves)

    best = evaluate_checkpoint(config, best_ckpt, device)
    last = evaluate_checkpoint(config, last_ckpt, device)
    print_checkpoint_summary("best_epoch3", best)
    print_checkpoint_summary("last", last)


if __name__ == "__main__":
    main()
