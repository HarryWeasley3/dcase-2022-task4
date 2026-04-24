from copy import deepcopy

import torch.nn as nn


class CRNNBEATsWavLMResidualGatedFusionModel(nn.Module):
    """Three-way residual gated fusion with BEATs as the main path.

    Legacy mode aligns BEATs/WavLM to the CRNN timeline first.
    BEATs time-anchor mode keeps BEATs on its native timeline, aligns WavLM and
    CRNN to BEATs, and performs SSL fusion before the final CRNN residual merge.
    """

    def __init__(
        self,
        crnn_encoder,
        beats_encoder,
        wavlm_encoder,
        fusion_aligner,
        fusion_block,
        decoder,
        label_aligner,
        build_config=None,
    ):
        super().__init__()
        self.crnn_encoder = crnn_encoder
        self.beats_encoder = beats_encoder
        self.wavlm_encoder = wavlm_encoder
        self.fusion_aligner = fusion_aligner
        self.fusion_block = fusion_block
        self.decoder = decoder
        self.label_aligner = label_aligner
        self.encoder_type = "crnn_beats_wavlm_residual_gated_fusion"
        self.build_config = deepcopy(build_config) if build_config is not None else None
        self.use_beats_time_anchor = getattr(
            fusion_block, "use_beats_time_anchor", False
        )

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

        if self.use_beats_time_anchor:
            beats_aligned = beats_features
            wavlm_aligned = self.fusion_aligner(wavlm_features, beats_features)
            crnn_aligned = self.fusion_aligner(cnn_features, beats_features)
        else:
            beats_aligned = self.fusion_aligner(beats_features, cnn_features)
            wavlm_aligned = self.fusion_aligner(wavlm_features, cnn_features)
            crnn_aligned = cnn_features

        fusion_outputs = self.fusion_block(
            crnn_aligned,
            beats_aligned,
            wavlm_aligned,
        )

        decoder_inputs = self.label_aligner(
            fusion_outputs["fused_output"], target_frame_len
        )
        decoder_outputs = self.decoder(decoder_inputs)

        outputs = {
            "strong_preds": decoder_outputs["strong_preds"],
            "weak_preds": decoder_outputs["weak_preds"],
            "strong_logits": decoder_outputs["strong_logits"],
            "encoder_sequence_features": fusion_outputs["fused_output"],
            "aligned_sequence_features": decoder_inputs,
            "cnn_sequence_features": cnn_features,
            "beats_sequence_features": beats_features,
            "wavlm_sequence_features": wavlm_features,
            "use_beats_time_anchor": self.use_beats_time_anchor,
            "beats_aligned_features": beats_aligned,
            "wavlm_aligned_features": wavlm_aligned,
            "beats_anchor_sequence_features": beats_features,
            "wavlm_aligned_to_beats_features": (
                wavlm_aligned if self.use_beats_time_anchor else None
            ),
            "crnn_aligned_to_beats_features": (
                crnn_aligned if self.use_beats_time_anchor else None
            ),
            "crnn_projected_features": fusion_outputs["crnn_projected"],
            "beats_projected_features": fusion_outputs["beats_projected"],
            "wavlm_projected_features": fusion_outputs["wavlm_projected"],
            "crnn_normalized_features": fusion_outputs["crnn_normalized"],
            "beats_normalized_features": fusion_outputs["beats_normalized"],
            "wavlm_normalized_features": fusion_outputs["wavlm_normalized"],
            "ssl_main_normalized_features": fusion_outputs["ssl_main_normalized"],
            "fusion_gate_crnn": fusion_outputs["gate_crnn"],
            "fusion_gate_wavlm": fusion_outputs["gate_wavlm"],
            "fusion_alpha_crnn": fusion_outputs["alpha_crnn"],
            "fusion_alpha_wavlm": fusion_outputs["alpha_wavlm"],
            "fused_ssl_sequence_features": fusion_outputs["fused_ssl"],
            "fused_sequence_features": fusion_outputs["fused"],
            "final_fused_sequence_features": fusion_outputs["fused_output"],
        }

        if return_intermediates:
            outputs["encoder_frontend_features"] = cnn_outputs.get("frontend_features")
            outputs["beats_frontend_features"] = beats_outputs.get("frontend_features")
            outputs["wavlm_frontend_features"] = wavlm_outputs.get("frontend_features")
            outputs["decoder_inputs"] = decoder_outputs.get("decoder_inputs")
            outputs["decoder_frame_features"] = decoder_outputs.get("frame_features")
            outputs["attention_weights"] = decoder_outputs.get("attention_weights")
            outputs["gate_crnn_inputs"] = fusion_outputs["gate_crnn_inputs"]
            outputs["gate_wavlm_inputs"] = fusion_outputs["gate_wavlm_inputs"]

        return outputs
