from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from sed_modeling.third_party.beats import BEATs, BEATsConfig


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _disabled_autocast_context(device_type):
    if device_type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=False)
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


class BEATsEncoder(nn.Module):
    """Frozen-or-trainable BEATs feature extractor returning sequence features."""

    def __init__(
        self,
        checkpoint,
        freeze=True,
        feature_layer=None,
        fbank_mean=15.41663,
        fbank_std=6.55582,
        load_branch_weights=False,
        branch_checkpoint="",
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(
                "BEATs encoder requires a valid checkpoint path in model.beats.checkpoint."
            )

        state = _torch_load_compat(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(state["cfg"])
        self.beats = BEATs(cfg)
        self.beats.load_state_dict(state["model"])

        if load_branch_weights:
            branch_checkpoint_path = Path(branch_checkpoint) if branch_checkpoint else None
            if branch_checkpoint_path is None or not branch_checkpoint_path.exists():
                raise FileNotFoundError(
                    "BEATs branch warm-start requires a valid model.beats.branch_checkpoint "
                    "when model.beats.load_branch_weights is true."
                )
            self._load_branch_weights(branch_checkpoint_path)

        self.feature_layer = feature_layer
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std
        self.requires_input_scaler = False
        self.is_frozen = freeze
        self.output_dim = cfg.encoder_embed_dim

        if freeze:
            for param in self.beats.parameters():
                param.requires_grad = False
            self.beats.eval()

    def _load_branch_weights(self, checkpoint_path):
        checkpoint = _torch_load_compat(checkpoint_path, map_location="cpu")
        source_prefix = None

        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
            source_state = checkpoint["model"]
            source_prefix = "model"
        else:
            source_state = checkpoint.get("state_dict", checkpoint)
            for candidate in (
                "beats_encoder.beats.",
                "sed_student.beats_encoder.beats.",
                "encoder.beats.",
                "sed_student.encoder.beats.",
            ):
                if any(key.startswith(candidate) for key in source_state):
                    source_prefix = candidate
                    source_state = {
                        key[len(candidate) :]: value
                        for key, value in source_state.items()
                        if key.startswith(candidate)
                    }
                    break

        target_state = self.beats.state_dict()
        load_state = {}
        missing_source_keys = []
        mismatched_keys = []

        for target_key, target_tensor in target_state.items():
            if target_key not in source_state:
                missing_source_keys.append(target_key)
                continue

            source_tensor = source_state[target_key]
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                mismatched_keys.append(
                    (target_key, tuple(target_tensor.shape), tuple(source_tensor.shape))
                )
                continue

            load_state[target_key] = source_tensor

        incompatible = self.beats.load_state_dict(load_state, strict=False)
        print(
            "[beats branch warm-start] loaded BEATs branch from "
            f"{checkpoint_path} using prefix {source_prefix!r} "
            f"(loaded={len(load_state)}, missing_source={len(missing_source_keys)}, "
            f"shape_mismatch={len(mismatched_keys)})"
        )
        if missing_source_keys:
            print("[beats branch warm-start] missing source keys:", missing_source_keys)
        if mismatched_keys:
            print("[beats branch warm-start] shape mismatch:", mismatched_keys)
        if incompatible.missing_keys:
            print(
                "[beats branch warm-start] target missing keys:",
                incompatible.missing_keys,
            )
        if incompatible.unexpected_keys:
            print(
                "[beats branch warm-start] unexpected keys:",
                incompatible.unexpected_keys,
            )

    def set_input_scaler(self, scaler):
        del scaler

    def prepare_inputs(self, audio):
        return audio

    def _extract_sequence_features(self, audio, padding_mask=None):
        fbank = self.beats.preprocess(
            audio, fbank_mean=self.fbank_mean, fbank_std=self.fbank_std
        )

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(fbank, padding_mask)

        patches = self.beats.patch_embedding(fbank.unsqueeze(1))
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1).transpose(1, 2)
        patches = self.beats.layer_norm(patches)

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(patches, padding_mask)

        if self.beats.post_extract_proj is not None:
            patches = self.beats.post_extract_proj(patches)

        x = self.beats.dropout_input(patches)
        x, _ = self.beats.encoder(
            x,
            padding_mask=padding_mask,
            layer=self.feature_layer,
        )

        return {
            "sequence_features": x,
            "frontend_features": fbank,
            "padding_mask": padding_mask,
        }

    def forward(self, audio, padding_mask=None):
        if self.is_frozen:
            autocast_context = _disabled_autocast_context(audio.device.type)
            with torch.no_grad(), autocast_context:
                return self._extract_sequence_features(audio, padding_mask=padding_mask)
        return self._extract_sequence_features(audio, padding_mask=padding_mask)

    def train(self, mode=True):
        super().train(mode)
        if self.is_frozen:
            self.beats.eval()
        return self
