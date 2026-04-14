from copy import deepcopy

import torch.nn as nn


class CRNNBEATsResidualGatedFusionModel(nn.Module):
    """Residual gated fusion with a CRNN main path and BEATs residual branch.

    Data flow:
        waveform -> CRNN CNN branch
                -> frozen BEATs branch -> align to CNN time
                -> projection + per-branch LayerNorm
                -> gate([cnn, beats]) and residual fusion
                -> shared BiGRU / strong-weak heads
    """

    def __init__(
        self,
        crnn_encoder,
        beats_encoder,
        fusion_aligner,
        fusion_block,
        decoder,
        label_aligner,
        build_config=None,
    ):
        super().__init__()
        self.crnn_encoder = crnn_encoder
        self.beats_encoder = beats_encoder
        self.fusion_aligner = fusion_aligner
        self.fusion_block = fusion_block
        self.decoder = decoder
        self.label_aligner = label_aligner
        self.encoder_type = "crnn_beats_residual_gated_fusion"
        self.build_config = deepcopy(build_config) if build_config is not None else None

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
        if self.build_config is not None:
            teacher_cfg = self.build_config.get("model", {}).get("teacher", {})
            share_frozen_beats = teacher_cfg.get(
                "share_frozen_beats",
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

        cnn_features = cnn_outputs["sequence_features"]
        beats_features = beats_outputs["sequence_features"]
        beats_aligned = self.fusion_aligner(beats_features, cnn_features)
        fusion_outputs = self.fusion_block(cnn_features, beats_aligned)

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
            "beats_aligned_features": beats_aligned,
            "cnn_projected_features": fusion_outputs["cnn_projected"],
            "beats_projected_features": fusion_outputs["beats_projected"],
            "cnn_normalized_features": fusion_outputs["cnn_normalized"],
            "beats_normalized_features": fusion_outputs["beats_normalized"],
            "fusion_gate": fusion_outputs["gate"],
            "fused_sequence_features": fusion_outputs["fused"],
        }

        if return_intermediates:
            outputs["encoder_frontend_features"] = cnn_outputs.get("frontend_features")
            outputs["decoder_inputs"] = decoder_outputs.get("decoder_inputs")
            outputs["decoder_frame_features"] = decoder_outputs.get("frame_features")
            outputs["attention_weights"] = decoder_outputs.get("attention_weights")

        return outputs
