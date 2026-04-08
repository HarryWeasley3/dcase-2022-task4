from pathlib import Path

import torch
import torch.nn as nn

from sed_modeling.third_party.beats import BEATs, BEATsConfig


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class BEATsEncoder(nn.Module):
    """Frozen-or-trainable BEATs feature extractor returning sequence features."""

    def __init__(
        self,
        checkpoint,
        freeze=True,
        feature_layer=None,
        fbank_mean=15.41663,
        fbank_std=6.55582,
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
            self.beats.eval()
        return self
