from copy import deepcopy

import torch.nn as nn


class CRNNWavLMResidualGatedFusionModel(nn.Module):
    """Residual gated fusion with a CRNN trunk and WavLM residual branch.

    Data flow:
        waveform -> CRNN CNN branch
                -> frozen WavLM branch -> align to CNN time
                -> projection + per-branch LayerNorm
                -> gate([cnn, wavlm]) and residual fusion
                -> shared BiGRU / strong-weak heads
    """

    def __init__(
        self,
        crnn_encoder,
        wavlm_encoder,
        fusion_aligner,
        fusion_block,
        decoder,
        label_aligner,
        build_config=None,
    ):
        super().__init__()
        self.crnn_encoder = crnn_encoder
        self.wavlm_encoder = wavlm_encoder
        self.fusion_aligner = fusion_aligner
        self.fusion_block = fusion_block
        self.decoder = decoder
        self.label_aligner = label_aligner
        self.encoder_type = "crnn_wavlm_residual_gated_fusion"
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
        wavlm_aligned = self.fusion_aligner(wavlm_features, cnn_features)
        fusion_outputs = self.fusion_block(cnn_features, wavlm_aligned)

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
            "wavlm_sequence_features": wavlm_features,
            "wavlm_aligned_features": wavlm_aligned,
            "cnn_projected_features": fusion_outputs["cnn_projected"],
            "wavlm_projected_features": fusion_outputs["auxiliary_projected"],
            "cnn_normalized_features": fusion_outputs["cnn_normalized"],
            "wavlm_normalized_features": fusion_outputs["auxiliary_normalized"],
            "fusion_gate": fusion_outputs["gate"],
            "fused_sequence_features": fusion_outputs["fused"],
        }

        if return_intermediates:
            outputs["encoder_frontend_features"] = cnn_outputs.get("frontend_features")
            outputs["decoder_inputs"] = decoder_outputs.get("decoder_inputs")
            outputs["decoder_frame_features"] = decoder_outputs.get("frame_features")
            outputs["attention_weights"] = decoder_outputs.get("attention_weights")

        return outputs
