"""从训练 checkpoint 中提取 BEATs 编码器权重，并合并回 BEATs 官方权重，导出 warm-start 用的 pt 文件。

用法示例：
    python scripts/extract_ckpt.py \
        --src exp/unified_beats_synth_only_a800_finetune/version_5/epoch=55-step=8791.ckpt \
        --base pretrained/beats/BEATs_iter3+_AS2M.pt \
        --out pretrained/beats/BEATs_full_finetune_best_0_78.pt
"""

from argparse import ArgumentParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

PREFIX_CANDIDATES = ("encoder.beats.", "sed_student.encoder.beats.", "beats.")


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_beats_state(checkpoint_obj):
    """在 checkpoint 里找出 BEATs 编码器权重，返回 (state_dict, 容器名, 前缀)。"""
    containers = []
    if isinstance(checkpoint_obj, dict):
        for key in ("sed_student", "state_dict"):
            if isinstance(checkpoint_obj.get(key), dict):
                containers.append((key, checkpoint_obj[key]))
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            containers.append(("checkpoint", checkpoint_obj))

    for container_name, state in containers:
        for prefix in PREFIX_CANDIDATES:
            model = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if model:
                return model, container_name, prefix

    raise RuntimeError(
        "没有在 checkpoint 里找到 BEATs 编码器权重。尝试过的前缀: "
        + ", ".join(PREFIX_CANDIDATES)
    )


def build_parser():
    parser = ArgumentParser(
        description="从训练 checkpoint 中提取 BEATs 编码器权重，并合并进 BEATs 官方权重文件。"
    )
    parser.add_argument(
        "--src",
        default="exp/unified_beats_synth_only_a800_finetune/version_5/epoch=55-step=8791.ckpt",
        help="训练产生的 checkpoint 路径。",
    )
    parser.add_argument(
        "--base",
        default="pretrained/beats/BEATs_iter3+_AS2M.pt",
        help="BEATs 官方权重文件（含 model 键）。",
    )
    parser.add_argument(
        "--out",
        default="pretrained/beats/BEATs_full_finetune_best.pt",
        help="导出的 BEATs 权重路径。",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    src = Path(args.src)
    base = Path(args.base)
    out = Path(args.out)

    for path in (src, base):
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

    checkpoint = torch_load_compat(src, map_location="cpu")
    beats_state, container_name, prefix = extract_beats_state(checkpoint)

    base_ckpt = torch_load_compat(base, map_location="cpu")
    if not isinstance(base_ckpt, dict):
        raise RuntimeError(f"BEATs 官方权重格式异常: {type(base_ckpt)}")

    base_ckpt["model"] = beats_state
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_ckpt, out)

    print("saved:", out)
    print("source_container:", container_name)
    print("matched_prefix:", prefix)
    print("num_params:", len(beats_state))
    print("sample_keys:")
    for key in list(beats_state.keys())[:10]:
        print(" ", key)


if __name__ == "__main__":
    main()
