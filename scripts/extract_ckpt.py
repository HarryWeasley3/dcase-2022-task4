# python - <<'PY'
import torch
from pathlib import Path

root = Path("/root/autodl-tmp/dcase-2022-task4")
src = root / "exp/unified_beats_synth_only_a800_finetune/version_5/epoch=55-step=8791.ckpt"
base = root / "pretrained/beats/BEATs_iter3+_AS2M.pt"
out = root / "pretrained/beats/BEATs_full_finetune_best_0_78.pt"

ft = torch.load(src, map_location="cpu")
base_ckpt = torch.load(base, map_location="cpu")

model = {}

if "sed_student" in ft:
    state = ft["sed_student"]
    prefix = "encoder.beats."
    model = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

if not model and "state_dict" in ft:
    state = ft["state_dict"]
    prefix = "sed_student.encoder.beats."
    model = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

if not model:
    raise RuntimeError("没有在 checkpoint 里找到 encoder.beats 权重。")

base_ckpt["model"] = model
torch.save(base_ckpt, out)

print("saved:", out)
print("num_params:", len(model))
print("sample_keys:")
for k in list(model.keys())[:10]:
    print(" ", k)