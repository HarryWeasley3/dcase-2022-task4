from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence

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
    """Frozen-or-trainable BEATs feature extractor returning sequence features.

    Default behavior matches the existing single-layer contract. Callers can
    explicitly request multi-layer hidden states via ``selected_layers`` or
    ``return_all_layers`` when a structured output is needed for experimental
    fusion modules.
    """

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

    @property
    def num_layers(self):
        return len(self.beats.encoder.layers)

    def _validate_requested_layers(
        self, selected_layers: Optional[Sequence[int]]
    ):
        if selected_layers is None:
            return None
        if not isinstance(selected_layers, (list, tuple)):
            raise TypeError(
                "BEATsEncoder selected_layers must be a list or tuple of zero-based "
                f"layer indices, got {type(selected_layers)!r}."
            )
        if len(selected_layers) == 0:
            raise ValueError("BEATsEncoder selected_layers must not be empty.")

        validated = []
        for layer_idx in selected_layers:
            if not isinstance(layer_idx, int):
                raise TypeError(
                    "BEATsEncoder selected_layers must contain integers, "
                    f"got {type(layer_idx)!r}."
                )
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(
                    "BEATsEncoder layer index out of range: "
                    f"{layer_idx}. Valid range is [0, {self.num_layers - 1}]."
                )
            validated.append(layer_idx)
        return validated

    def _validate_feature_layer(self):
        if self.feature_layer is None:
            return None
        feature_layer = int(self.feature_layer)
        if feature_layer < 0 or feature_layer >= self.num_layers:
            raise ValueError(
                "model.beats.feature_layer is out of range: "
                f"{feature_layer}. Valid range is [0, {self.num_layers - 1}]."
            )
        return feature_layer

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

    def _prepare_encoder_inputs(self, audio, padding_mask=None):
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
        return {
            "encoder_inputs": x,
            "frontend_features": fbank,
            "padding_mask": padding_mask,
        }

    @staticmethod
    def _layer_results_to_hidden_states(layer_results, target_layer):
        if target_layer is None:
            raise ValueError("target_layer must be provided when collecting BEATs hidden states.")
        if len(layer_results) == 0:
            raise RuntimeError(
                "BEATs encoder returned no layer_results. The third-party backbone only "
                "collects intermediate states when a concrete target layer is requested."
            )

        hidden_states = [
            layer_output.transpose(0, 1).contiguous()
            for layer_output, _ in layer_results[1:]
        ]
        expected_num_states = int(target_layer) + 1
        if len(hidden_states) != expected_num_states:
            raise RuntimeError(
                "BEATsEncoder expected "
                f"{expected_num_states} hidden states up to target_layer={target_layer}, "
                f"got {len(hidden_states)}."
            )
        return hidden_states

    def _extract_sequence_features(
        self,
        audio,
        padding_mask=None,
        selected_layers=None,
        return_all_layers=False,
        return_layer_dict=False,
    ):
        if return_layer_dict and selected_layers is None and not return_all_layers:
            return_all_layers = True

        prepared = self._prepare_encoder_inputs(audio, padding_mask=padding_mask)
        encoder_inputs = prepared["encoder_inputs"]
        need_layer_outputs = return_all_layers or selected_layers is not None

        if not need_layer_outputs:
            feature_layer = self._validate_feature_layer()
            sequence_features, _ = self.beats.encoder(
                encoder_inputs,
                padding_mask=prepared["padding_mask"],
                layer=feature_layer,
            )
            return {
                "sequence_features": sequence_features,
                "frontend_features": prepared["frontend_features"],
                "padding_mask": prepared["padding_mask"],
            }

        validated_layers = self._validate_requested_layers(selected_layers)
        feature_layer = self._validate_feature_layer()
        candidate_layers = []
        if validated_layers is not None:
            candidate_layers.extend(validated_layers)
        if feature_layer is not None:
            candidate_layers.append(feature_layer)
        if return_all_layers:
            target_layer = self.num_layers - 1
        elif candidate_layers:
            target_layer = max(candidate_layers)
        else:
            raise RuntimeError(
                "BEATsEncoder multi-layer extraction requires at least one selected "
                "layer or an explicit feature_layer."
            )

        sequence_features, layer_results = self.beats.encoder(
            encoder_inputs,
            padding_mask=prepared["padding_mask"],
            layer=target_layer,
        )
        all_hidden_states = self._layer_results_to_hidden_states(
            layer_results,
            target_layer=target_layer,
        )
        if feature_layer is not None:
            sequence_features = all_hidden_states[feature_layer]

        if return_all_layers:
            selected_layer_indices = list(range(target_layer + 1))
        else:
            selected_layer_indices = validated_layers

        selected_hidden_states = (
            [all_hidden_states[layer_idx] for layer_idx in selected_layer_indices]
            if selected_layer_indices is not None
            else None
        )

        outputs = {
            "sequence_features": sequence_features,
            "frontend_features": prepared["frontend_features"],
            "padding_mask": prepared["padding_mask"],
            "selected_layers": selected_layer_indices,
            "selected_hidden_states": selected_hidden_states,
        }
        if return_all_layers:
            outputs["all_hidden_states"] = all_hidden_states
        if return_layer_dict:
            layer_indices = (
                selected_layer_indices
                if selected_layer_indices is not None
                else list(range(self.num_layers))
            )
            layer_hidden_states = (
                selected_hidden_states
                if selected_hidden_states is not None
                else all_hidden_states
            )
            outputs["layer_features"] = {
                layer_idx: hidden_state
                for layer_idx, hidden_state in zip(layer_indices, layer_hidden_states)
            }

        return outputs

    def forward(
        self,
        audio,
        padding_mask=None,
        selected_layers=None,
        return_all_layers=False,
        return_layer_dict=False,
    ):
        if self.is_frozen:
            autocast_context = _disabled_autocast_context(audio.device.type)
            with torch.no_grad(), autocast_context:
                return self._extract_sequence_features(
                    audio,
                    padding_mask=padding_mask,
                    selected_layers=selected_layers,
                    return_all_layers=return_all_layers,
                    return_layer_dict=return_layer_dict,
                )
        return self._extract_sequence_features(
            audio,
            padding_mask=padding_mask,
            selected_layers=selected_layers,
            return_all_layers=return_all_layers,
            return_layer_dict=return_layer_dict,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.is_frozen:
            self.beats.eval()
        return self
