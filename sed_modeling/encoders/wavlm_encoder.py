from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class WavLMEncoder(nn.Module):
    """WavLM feature extractor for the unified SED encoder-decoder framework.

    This wrapper keeps WavLM as an interchangeable encoder: it consumes waveform,
    emits sequence features, and leaves time alignment / decoding to the shared
    project modules. The first version is optimized for a frozen baseline.
    """

    def __init__(
        self,
        checkpoint="",
        bundle_name="WAVLM_BASE_PLUS",
        freeze=True,
        output_layer=None,
        use_bundle_weights=True,
        normalize_waveform=False,
    ):
        super().__init__()
        self.bundle_name = bundle_name
        self.bundle = self._resolve_bundle(bundle_name)
        self.output_layer = output_layer
        self.requires_input_scaler = False
        self.is_frozen = freeze
        self.normalize_waveform = normalize_waveform or getattr(
            self.bundle, "_normalize_waveform", False
        )

        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path is not None and checkpoint_path.exists():
            self.wavlm = torchaudio.models.wavlm_model(**self.bundle._params)
            self._load_checkpoint(self.wavlm, checkpoint_path)
        elif use_bundle_weights:
            self.wavlm = self.bundle.get_model()
        else:
            self.wavlm = torchaudio.models.wavlm_model(**self.bundle._params)

        self.output_dim = self.bundle._params["encoder_embed_dim"]

        if freeze:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm.eval()

    @property
    def num_layers(self):
        return int(self.bundle._params["encoder_num_layers"])

    @staticmethod
    def _resolve_bundle(bundle_name):
        if not hasattr(torchaudio.pipelines, bundle_name):
            raise ValueError(
                f"Unsupported WavLM bundle '{bundle_name}'. Expected one of the torchaudio WavLM bundles."
            )
        bundle = getattr(torchaudio.pipelines, bundle_name)
        model_type = getattr(bundle, "_model_type", "")
        if "wavlm" not in str(model_type).lower():
            raise ValueError(
                f"Bundle '{bundle_name}' is not a WavLM bundle (model_type={model_type})."
            )
        return bundle

    @staticmethod
    def _strip_prefixes(state_dict):
        prefixes = ("model.", "state_dict.", "module.", "wavlm.", "encoder.")
        changed = True
        cleaned = dict(state_dict)
        while changed:
            changed = False
            for prefix in prefixes:
                if cleaned and all(key.startswith(prefix) for key in cleaned.keys()):
                    cleaned = {key[len(prefix):]: value for key, value in cleaned.items()}
                    changed = True
        return cleaned

    def _extract_state_dict(self, checkpoint_obj):
        if isinstance(checkpoint_obj, dict):
            for key in ("model", "state_dict"):
                maybe_state = checkpoint_obj.get(key)
                if isinstance(maybe_state, dict) and maybe_state:
                    return self._strip_prefixes(maybe_state)
            if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
                return self._strip_prefixes(checkpoint_obj)
        raise RuntimeError(
            "Unsupported WavLM checkpoint format. "
            "Please provide a torchaudio-compatible WavLM state dict, or leave "
            "model.wavlm.use_bundle_weights=true to use the torchaudio bundle."
        )

    def _load_checkpoint(self, model, checkpoint_path):
        state = _torch_load_compat(checkpoint_path, map_location="cpu")
        state_dict = self._extract_state_dict(state)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load the provided WavLM checkpoint into the torchaudio WavLM model. "
                "This wrapper currently expects torchaudio-compatible WavLM weights."
            ) from exc

    def set_input_scaler(self, scaler):
        del scaler

    def prepare_inputs(self, audio):
        return audio

    def _normalize_audio(self, audio):
        if not self.normalize_waveform:
            return audio
        return F.layer_norm(audio, audio.shape[-1:])

    def _padding_mask_to_lengths(self, audio, padding_mask):
        if padding_mask is None:
            return None
        if padding_mask.ndim != 2:
            raise ValueError(
                f"WavLMEncoder expects padding_mask with shape [batch, time], got {tuple(padding_mask.shape)}"
            )
        lengths = (~padding_mask).sum(dim=-1)
        return lengths.to(device=audio.device, dtype=torch.long)

    def _validate_requested_layers(
        self, selected_layers: Optional[Sequence[int]]
    ):
        if selected_layers is None:
            return None
        if not isinstance(selected_layers, (list, tuple)):
            raise TypeError(
                "WavLMEncoder selected_layers must be a list or tuple of zero-based "
                f"layer indices, got {type(selected_layers)!r}."
            )
        if len(selected_layers) == 0:
            raise ValueError("WavLMEncoder selected_layers must not be empty.")

        validated = []
        for layer_idx in selected_layers:
            if not isinstance(layer_idx, int):
                raise TypeError(
                    "WavLMEncoder selected_layers must contain integers, "
                    f"got {type(layer_idx)!r}."
                )
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(
                    "WavLMEncoder layer index out of range: "
                    f"{layer_idx}. Valid range is [0, {self.num_layers - 1}]."
                )
            validated.append(layer_idx)
        return validated

    def _validate_output_layer(self):
        if self.output_layer is None:
            return None
        output_layer = int(self.output_layer)
        if output_layer <= 0 or output_layer > self.num_layers:
            raise ValueError(
                "model.wavlm.output_layer is out of range: "
                f"{output_layer}. Valid range is [1, {self.num_layers}]."
            )
        return output_layer

    def _extract_sequence_features(
        self,
        audio,
        padding_mask=None,
        selected_layers=None,
        return_all_layers=False,
        return_layer_dict=False,
    ):
        audio = self._normalize_audio(audio)
        lengths = self._padding_mask_to_lengths(audio, padding_mask)

        if return_layer_dict and selected_layers is None and not return_all_layers:
            return_all_layers = True

        need_layer_outputs = return_all_layers or selected_layers is not None
        validated_output_layer = self._validate_output_layer()

        if not need_layer_outputs:
            if validated_output_layer is not None:
                features, lengths = self.wavlm.extract_features(
                    audio,
                    lengths=lengths,
                    num_layers=validated_output_layer,
                )
                sequence_features = features[-1]
            else:
                sequence_features, lengths = self.wavlm(audio, lengths=lengths)
            return {
                "sequence_features": sequence_features,
                "frontend_features": audio,
                "lengths": lengths,
            }

        validated_layers = self._validate_requested_layers(selected_layers)

        if return_all_layers or validated_output_layer is None:
            layer_outputs, lengths = self.wavlm.extract_features(
                audio,
                lengths=lengths,
                num_layers=None,
            )
        else:
            max_requested_layer = max(validated_layers) + 1
            layer_outputs, lengths = self.wavlm.extract_features(
                audio,
                lengths=lengths,
                num_layers=max(max_requested_layer, validated_output_layer),
            )

        all_hidden_states = [hidden_state.contiguous() for hidden_state in layer_outputs]
        if not all_hidden_states:
            raise RuntimeError("WavLMEncoder did not return any hidden states.")

        if validated_output_layer is not None:
            sequence_features = all_hidden_states[validated_output_layer - 1]
        else:
            sequence_features = all_hidden_states[-1]

        if return_all_layers:
            selected_layer_indices = list(range(len(all_hidden_states)))
        else:
            selected_layer_indices = validated_layers

        selected_hidden_states = (
            [all_hidden_states[layer_idx] for layer_idx in selected_layer_indices]
            if selected_layer_indices is not None
            else None
        )

        outputs = {
            "sequence_features": sequence_features,
            "frontend_features": audio,
            "lengths": lengths,
            "selected_layers": selected_layer_indices,
            "selected_hidden_states": selected_hidden_states,
        }
        if return_all_layers:
            outputs["all_hidden_states"] = all_hidden_states
        if return_layer_dict:
            layer_indices = (
                selected_layer_indices
                if selected_layer_indices is not None
                else list(range(len(all_hidden_states)))
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
            autocast_context = (
                torch.amp.autocast("cuda", enabled=False)
                if audio.device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
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
            self.wavlm.eval()
        return self
