from copy import deepcopy

import torch
import torch.nn as nn


class CRNNBEATsWavLMLateFusionModel(nn.Module):
    """Three-branch late fusion with CRNN, BEATs, and WavLM."""

    def __init__(
        self,
        crnn_encoder,
        beats_encoder,
        wavlm_encoder,
        fusion_aligner,
        merge_mlp,
        decoder,
        label_aligner,
        use_branch_layernorm=False,
        build_config=None,
    ):
        super().__init__()
        self.crnn_encoder = crnn_encoder
        self.beats_encoder = beats_encoder
        self.wavlm_encoder = wavlm_encoder
        self.fusion_aligner = fusion_aligner
        self.merge_mlp = merge_mlp
        self.decoder = decoder
        self.label_aligner = label_aligner
        self.encoder_type = "crnn_beats_wavlm_late_fusion"
        self.use_branch_layernorm = use_branch_layernorm
        self.build_config = deepcopy(build_config) if build_config is not None else None

        if use_branch_layernorm:
            self.cnn_pre_fusion_norm = nn.LayerNorm(crnn_encoder.output_dim)
            self.beats_pre_fusion_norm = nn.LayerNorm(beats_encoder.output_dim)
            self.wavlm_pre_fusion_norm = nn.LayerNorm(wavlm_encoder.output_dim)
        else:
            self.cnn_pre_fusion_norm = nn.Identity()
            self.beats_pre_fusion_norm = nn.Identity()
            self.wavlm_pre_fusion_norm = nn.Identity()

    @property
    def needs_input_scaler(self):
        return getattr(self.crnn_encoder, "requires_input_scaler", False)

    @property
    def is_frozen_encoder(self):
        return False

    def set_input_scaler(self, scaler):
        self.crnn_encoder.set_input_scaler(scaler)

    def prepare_inputs(self, audio):
        return self.crnn_encoder.prepare_inputs(audio)

    def make_teacher_copy(self, share_encoder_if_frozen=True):
        share_frozen_beats = share_encoder_if_frozen
        share_frozen_wavlm = share_encoder_if_frozen
        if self.build_config is not None:
            teacher_cfg = self.build_config.get("model", {}).get("teacher", {})
            share_frozen_beats = teacher_cfg.get(
                "share_frozen_beats",
                teacher_cfg.get("share_frozen_encoder", share_encoder_if_frozen),
            )
            share_frozen_wavlm = teacher_cfg.get(
                "share_frozen_wavlm",
                teacher_cfg.get("share_frozen_encoder", share_encoder_if_frozen),
            )

        if self.build_config is None:
            teacher = deepcopy(self)
        else:
            from .sed_model import build_sed_model

            teacher = build_sed_model(self.build_config)
            teacher.load_state_dict(self.state_dict(), strict=True)

        if share_frozen_beats and getattr(self.beats_encoder, "is_frozen", False):
            teacher.beats_encoder = self.beats_encoder
        if share_frozen_wavlm and getattr(self.wavlm_encoder, "is_frozen", False):
            teacher.wavlm_encoder = self.wavlm_encoder

        return teacher

    def _fuse(self, cnn_features, beats_features, wavlm_features):
        beats_aligned = self.fusion_aligner(beats_features, cnn_features)
        wavlm_aligned = self.fusion_aligner(wavlm_features, cnn_features)

        fused = torch.cat(
            [
                self.cnn_pre_fusion_norm(cnn_features),
                self.beats_pre_fusion_norm(beats_aligned),
                self.wavlm_pre_fusion_norm(wavlm_aligned),
            ],
            dim=-1,
        )
        merged = self.merge_mlp(fused)
        return beats_aligned, wavlm_aligned, fused, merged

    def forward(
        self,
        audio,
        target_frame_len=None,
        padding_mask=None,
        return_intermediates=False,
    ):
        cnn_outputs = self.crnn_encoder(audio, padding_mask=padding_mask)
        beats_outputs = self.beats_encoder(audio, padding_mask=padding_mask)
        wavlm_outputs = self.wavlm_encoder(audio, padding_mask=padding_mask)

        cnn_features = cnn_outputs["sequence_features"]
        beats_features = beats_outputs["sequence_features"]
        wavlm_features = wavlm_outputs["sequence_features"]
        (
            beats_aligned,
            wavlm_aligned,
            fused_features,
            merged_features,
        ) = self._fuse(cnn_features, beats_features, wavlm_features)

        decoder_inputs = self.label_aligner(merged_features, target_frame_len)
        decoder_outputs = self.decoder(decoder_inputs)

        outputs = {
            "strong_preds": decoder_outputs["strong_preds"],
            "weak_preds": decoder_outputs["weak_preds"],
            "strong_logits": decoder_outputs["strong_logits"],
            "encoder_sequence_features": merged_features,
            "aligned_sequence_features": decoder_inputs,
            "cnn_sequence_features": cnn_features,
            "beats_sequence_features": beats_features,
            "wavlm_sequence_features": wavlm_features,
            "beats_aligned_features": beats_aligned,
            "wavlm_aligned_features": wavlm_aligned,
            "fused_sequence_features": fused_features,
        }

        if return_intermediates:
            outputs["encoder_frontend_features"] = cnn_outputs.get("frontend_features")
            outputs["beats_frontend_features"] = beats_outputs.get("frontend_features")
            outputs["wavlm_frontend_features"] = wavlm_outputs.get("frontend_features")
            outputs["decoder_inputs"] = decoder_outputs.get("decoder_inputs")
            outputs["decoder_frame_features"] = decoder_outputs.get("frame_features")
            outputs["attention_weights"] = decoder_outputs.get("attention_weights")

        return outputs
