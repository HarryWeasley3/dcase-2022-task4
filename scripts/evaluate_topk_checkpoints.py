import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from train_sed import (
    _torch_load_compat,
    build_eval_config_from_checkpoint,
    load_experiment_config,
    resample_data_generate_durations,
    single_run,
)


DEFAULT_RANK_BY = "test/student/psds_score_scenario1"
SUMMARY_COLUMNS = [
    "checkpoint_name",
    "epoch",
    "test/student/psds_score_scenario1",
    "test/student/psds_score_scenario2",
    "test/student/event_f1_macro",
    "test/student/intersection_f1_macro",
    "test/teacher/psds_score_scenario1",
    "test/teacher/psds_score_scenario2",
    "test/teacher/event_f1_macro",
    "test/teacher/intersection_f1_macro",
    "checkpoint_path",
]


def _discover_checkpoints(exp_dir=None, checkpoint_dir=None, checkpoints=None, include_last=False):
    resolved = []
    if checkpoints:
        resolved = [Path(path).expanduser().resolve() for path in checkpoints]
    else:
        search_root = checkpoint_dir or exp_dir
        if search_root is None:
            raise ValueError("Please provide --checkpoints, --checkpoint_dir, or --exp_dir.")
        search_root = Path(search_root).expanduser().resolve()
        if not search_root.exists():
            raise FileNotFoundError(f"Checkpoint search path does not exist: {search_root}")
        resolved = sorted(search_root.rglob("*.ckpt"))

    unique_paths = []
    seen = set()
    for checkpoint_path in resolved:
        if not checkpoint_path.is_file():
            continue
        if checkpoint_path.name == "last.ckpt" and not include_last:
            continue
        checkpoint_str = str(checkpoint_path)
        if checkpoint_str in seen:
            continue
        seen.add(checkpoint_str)
        unique_paths.append(checkpoint_path)

    if not unique_paths:
        raise FileNotFoundError("No checkpoint files were found for evaluation.")

    return unique_paths


def _evaluate_checkpoint(base_config, checkpoint_path, output_root, gpus, fast_dev_run):
    checkpoint = _torch_load_compat(str(checkpoint_path), map_location="cpu")
    config = build_eval_config_from_checkpoint(checkpoint, deepcopy(base_config))
    run_log_dir = output_root / checkpoint_path.stem
    result = single_run(
        config=config,
        log_dir=str(run_log_dir),
        gpus=gpus,
        strong_real=False,
        checkpoint_resume=None,
        checkpoint_init=None,
        test_state_dict=checkpoint["state_dict"],
        fast_dev_run=fast_dev_run,
        evaluation=False,
        synth_only=config["training"].get("synth_only", False),
        return_test_results=True,
    )
    metrics = dict(result.get("test_results", {}))
    metrics["checkpoint_name"] = checkpoint_path.name
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["epoch"] = checkpoint.get("epoch")
    return metrics


def _print_summary(summary_df, rank_by):
    available_columns = [column for column in SUMMARY_COLUMNS if column in summary_df.columns]
    print("\n[top-k rescoring] summary:")
    print(
        summary_df[available_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    best_row = summary_df.iloc[0]
    print(f"\n[top-k rescoring] best checkpoint by {rank_by}: {best_row['checkpoint_path']}")
    print("[top-k rescoring] best metrics:")
    for column in available_columns:
        if column in {"checkpoint_name", "checkpoint_path", "epoch"}:
            continue
        print(f"  - {column}: {best_row[column]:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-score top-k checkpoints with the existing DESED test metrics."
    )
    parser.add_argument("--conf_file", required=True, help="Experiment config used for data/runtime overrides.")
    parser.add_argument("--exp_dir", default=None, help="Experiment directory to recursively scan for checkpoints.")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory containing top-k checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Explicit checkpoint paths. When set, scanning --exp_dir/--checkpoint_dir is skipped.",
    )
    parser.add_argument(
        "--include_last",
        action="store_true",
        default=False,
        help="Also evaluate last.ckpt if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit after checkpoint discovery.",
    )
    parser.add_argument(
        "--rank_by",
        default=DEFAULT_RANK_BY,
        help="Metric column used to rank checkpoints, e.g. "
        "'test/student/psds_score_scenario1' or 'test/teacher/event_f1_macro'.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        default=False,
        help="Sort ascending instead of descending.",
    )
    parser.add_argument(
        "--output_dir",
        default="./exp/topk_rescoring",
        help="Directory where per-checkpoint eval artifacts and the summary TSV are written.",
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="GPU selector following train_sed.py semantics. Use '0' for CPU.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Run each checkpoint in Lightning fast-dev mode for smoke validation only.",
    )
    args = parser.parse_args()

    base_config = load_experiment_config(args.conf_file)
    base_config.setdefault("training", {})
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = _discover_checkpoints(
        exp_dir=args.exp_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoints=args.checkpoints,
        include_last=args.include_last,
    )
    if args.limit is not None:
        checkpoint_paths = checkpoint_paths[: args.limit]

    resample_data_generate_durations(
        base_config["data"],
        test_only=True,
        evaluation=False,
        synth_only=base_config["training"].get("synth_only", False),
    )

    rows = []
    failures = []
    for checkpoint_path in checkpoint_paths:
        print(f"\n[top-k rescoring] evaluating {checkpoint_path}")
        try:
            rows.append(
                _evaluate_checkpoint(
                    base_config=base_config,
                    checkpoint_path=checkpoint_path,
                    output_root=output_root,
                    gpus=args.gpus,
                    fast_dev_run=args.fast_dev_run,
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort batch evaluation
            failures.append((str(checkpoint_path), str(exc)))
            print(f"[top-k rescoring] failed: {checkpoint_path} -> {exc}")

    if not rows:
        raise RuntimeError("All checkpoint evaluations failed.")

    summary_df = pd.DataFrame(rows)
    if args.rank_by not in summary_df.columns:
        raise KeyError(
            f"Unknown rank metric '{args.rank_by}'. Available metrics: {sorted(summary_df.columns)}"
        )

    summary_df = summary_df.sort_values(
        by=args.rank_by,
        ascending=args.ascending,
    ).reset_index(drop=True)

    summary_path = output_root / "topk_rescoring_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    _print_summary(summary_df, args.rank_by)
    print(f"\n[top-k rescoring] summary saved to: {summary_path}")

    if failures:
        failure_path = output_root / "topk_rescoring_failures.tsv"
        pd.DataFrame(failures, columns=["checkpoint_path", "error"]).to_csv(
            failure_path,
            sep="\t",
            index=False,
        )
        print(f"[top-k rescoring] failures saved to: {failure_path}")


if __name__ == "__main__":
    main()
