from pathlib import Path

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

    def _extract_sequence_features(self, audio, padding_mask=None):
        audio = self._normalize_audio(audio)
        lengths = self._padding_mask_to_lengths(audio, padding_mask)

        if self.output_layer is not None:
            features, lengths = self.wavlm.extract_features(
                audio,
                lengths=lengths,
                num_layers=int(self.output_layer),
            )
            sequence_features = features[-1]
        else:
            sequence_features, lengths = self.wavlm(audio, lengths=lengths)

        return {
            "sequence_features": sequence_features,
            "frontend_features": audio,
            "lengths": lengths,
        }

    def forward(self, audio, padding_mask=None):
        if self.is_frozen:
            autocast_context = (
                torch.amp.autocast("cuda", enabled=False)
                if audio.device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with torch.no_grad(), autocast_context:
                return self._extract_sequence_features(audio, padding_mask=padding_mask)
        return self._extract_sequence_features(audio, padding_mask=padding_mask)

    def train(self, mode=True):
        super().train(mode)
        if self.is_frozen:
            self.wavlm.eval()
        return self
