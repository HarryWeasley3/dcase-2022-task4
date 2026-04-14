from copy import deepcopy

import torch
import torch.nn as nn


class CRNNWavLMLateFusionModel(nn.Module):
    """Paper-style late fusion:

    waveform -> CRNN CNN branch
            -> WavLM branch -> align to CNN time
            -> concat/add -> Merge MLP -> shared BiGRU/classifier
    """

    def __init__(
        self,
        crnn_encoder,
        wavlm_encoder,
        fusion_aligner,
        merge_mlp,
        decoder,
        label_aligner,
        fusion_type="concat",
        use_branch_layernorm=False,
        build_config=None,
    ):
        super().__init__()
        self.crnn_encoder = crnn_encoder
        self.wavlm_encoder = wavlm_encoder
        self.fusion_aligner = fusion_aligner
        self.merge_mlp = merge_mlp
        self.decoder = decoder
        self.label_aligner = label_aligner
        self.fusion_type = fusion_type
        self.encoder_type = "crnn_wavlm_late_fusion"
        self.use_branch_layernorm = use_branch_layernorm
        self.build_config = deepcopy(build_config) if build_config is not None else None

        if use_branch_layernorm:
            self.cnn_pre_fusion_norm = nn.LayerNorm(crnn_encoder.output_dim)
            self.wavlm_pre_fusion_norm = nn.LayerNorm(wavlm_encoder.output_dim)
        else:
            self.cnn_pre_fusion_norm = nn.Identity()
            self.wavlm_pre_fusion_norm = nn.Identity()

        if fusion_type == "add":
            wavlm_dim = wavlm_encoder.output_dim
            cnn_dim = crnn_encoder.output_dim
            self.add_proj = (
                nn.Identity() if wavlm_dim == cnn_dim else nn.Linear(wavlm_dim, cnn_dim)
            )
        else:
            self.add_proj = None

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
        share_frozen_wavlm = share_encoder_if_frozen
        if self.build_config is not None:
            teacher_cfg = self.build_config.get("model", {}).get("teacher", {})
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

        if share_frozen_wavlm and getattr(self.wavlm_encoder, "is_frozen", False):
            teacher.wavlm_encoder = self.wavlm_encoder

        return teacher

    def _fuse(self, cnn_features, wavlm_features):
        wavlm_aligned = self.fusion_aligner(wavlm_features, cnn_features)
        cnn_for_fusion = self.cnn_pre_fusion_norm(cnn_features)
        wavlm_for_fusion = self.wavlm_pre_fusion_norm(wavlm_aligned)
        if self.fusion_type == "concat":
            fused = torch.cat([cnn_for_fusion, wavlm_for_fusion], dim=-1)
        elif self.fusion_type == "add":
            fused = cnn_for_fusion + self.add_proj(wavlm_for_fusion)
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
        merged = self.merge_mlp(fused)
        return wavlm_aligned, fused, merged

    def forward(
        self,
        audio,
        target_frame_len=None,
        padding_mask=None,
        return_intermediates=False,
    ):
        cnn_outputs = self.crnn_encoder(audio, padding_mask=padding_mask)
        wavlm_outputs = self.wavlm_encoder(audio, padding_mask=padding_mask)

        cnn_features = cnn_outputs["sequence_features"]
        wavlm_features = wavlm_outputs["sequence_features"]
        wavlm_aligned, fused_features, merged_features = self._fuse(
            cnn_features, wavlm_features
        )

        decoder_inputs = self.label_aligner(merged_features, target_frame_len)
        decoder_outputs = self.decoder(decoder_inputs)

        outputs = {
            "strong_preds": decoder_outputs["strong_preds"],
            "weak_preds": decoder_outputs["weak_preds"],
            "strong_logits": decoder_outputs["strong_logits"],
            "encoder_sequence_features": merged_features,
            "aligned_sequence_features": decoder_inputs,
            "cnn_sequence_features": cnn_features,
            "wavlm_sequence_features": wavlm_features,
            "wavlm_aligned_features": wavlm_aligned,
            "fused_sequence_features": fused_features,
        }

        if return_intermediates:
            outputs["encoder_frontend_features"] = cnn_outputs.get("frontend_features")
            outputs["decoder_inputs"] = decoder_outputs.get("decoder_inputs")
            outputs["decoder_frame_features"] = decoder_outputs.get("frame_features")
            outputs["attention_weights"] = decoder_outputs.get("attention_weights")

        return outputs
