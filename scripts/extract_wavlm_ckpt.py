from argparse import ArgumentParser
from pathlib import Path

import torch


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_wavlm_state(checkpoint_obj):
    candidate_containers = []

    if isinstance(checkpoint_obj, dict):
        if isinstance(checkpoint_obj.get("sed_student"), dict):
            candidate_containers.append(("sed_student", checkpoint_obj["sed_student"]))
        if isinstance(checkpoint_obj.get("state_dict"), dict):
            candidate_containers.append(("state_dict", checkpoint_obj["state_dict"]))
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            candidate_containers.append(("checkpoint", checkpoint_obj))

    candidate_prefixes = (
        "encoder.wavlm.",
        "wavlm_encoder.wavlm.",
        "sed_student.encoder.wavlm.",
        "sed_student.wavlm_encoder.wavlm.",
        "wavlm.",
    )

    for container_name, state in candidate_containers:
        for prefix in candidate_prefixes:
            model = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if model:
                return model, container_name, prefix

    raise RuntimeError(
        "没有在 checkpoint 里找到 WavLM 编码器权重。"
        "尝试过的前缀包括: "
        + ", ".join(candidate_prefixes)
    )


def build_parser():
    parser = ArgumentParser(
        description="从训练 checkpoint 中提取 WavLM 编码器权重，并导出成 WavLMEncoder 可加载的 pt 文件。"
    )
    parser.add_argument(
        "--src",
        default="exp/WavLM_only/version_0/epoch=49-step=31249.ckpt",
        help="训练产生的 checkpoint 路径。",
    )
    parser.add_argument(
        "--out",
        default="pretrained/wavlm/WavLM_full_finetune_best_0_65.pt",
        help="导出的 WavLM 权重路径。",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {src}")

    checkpoint = torch_load_compat(src, map_location="cpu")
    model, container_name, prefix = extract_wavlm_state(checkpoint)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model}, out)

    print("saved:", out)
    print("source_container:", container_name)
    print("matched_prefix:", prefix)
    print("num_params:", len(model))
    print("sample_keys:")
    for key in list(model.keys())[:10]:
        print(" ", key)


if __name__ == "__main__":
    main()
