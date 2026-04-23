from copy import deepcopy

import torch
import torch.nn as nn


class CRNNBEATsWavLMHierarchicalSSLFusionModel(nn.Module):
    """Stage-A model: CRNN + [BEATs-main hierarchical WavLM fusion].

    Data flow:
        waveform -> CRNN (unchanged)
                -> BEATs selected SSL layers
                -> WavLM selected SSL layers
                -> hierarchical SSL fusion with BEATs as the main path
                -> align fused SSL features to the CRNN timeline
                -> concat(CRNN, fused SSL) -> Merge MLP -> shared decoder
    """

    def __init__(
        self,
        crnn_encoder,
        beats_encoder,
        wavlm_encoder,
        ssl_fusion_block,
        final_fusion_aligner,
        merge_mlp,
        decoder,
        label_aligner,
        beats_layers,
        wavlm_layers,
        use_branch_layernorm=False,
        build_config=None,
    ):
        super().__init__()
        self.crnn_encoder = crnn_encoder
        self.beats_encoder = beats_encoder
        self.wavlm_encoder = wavlm_encoder
        self.ssl_fusion_block = ssl_fusion_block
        self.final_fusion_aligner = final_fusion_aligner
        self.merge_mlp = merge_mlp
        self.decoder = decoder
        self.label_aligner = label_aligner
        self.beats_layers = self._validate_stage_layers("beats_layers", beats_layers)
        self.wavlm_layers = self._validate_stage_layers("wavlm_layers", wavlm_layers)
        if len(self.beats_layers) != len(self.wavlm_layers):
            raise ValueError(
                "Hierarchical SSL fusion requires the same number of BEATs and WavLM "
                f"stage layers, got beats={self.beats_layers} wavlm={self.wavlm_layers}."
            )
        expected_stages = getattr(self.ssl_fusion_block, "num_stages", None)
        if expected_stages is not None and len(self.beats_layers) != expected_stages:
            raise ValueError(
                "Hierarchical SSL fusion layer config must match the configured "
                f"num_stages={expected_stages}, got beats_layers={self.beats_layers} "
                f"and wavlm_layers={self.wavlm_layers}."
            )
        self.encoder_type = "crnn_beats_wavlm_hierarchical_ssl_fusion"
        self.use_branch_layernorm = use_branch_layernorm
        self.build_config = deepcopy(build_config) if build_config is not None else None

        if use_branch_layernorm:
            self.cnn_pre_fusion_norm = nn.LayerNorm(crnn_encoder.output_dim)
            self.ssl_pre_fusion_norm = nn.LayerNorm(ssl_fusion_block.output_dim)
        else:
            self.cnn_pre_fusion_norm = nn.Identity()
            self.ssl_pre_fusion_norm = nn.Identity()

    @staticmethod
    def _validate_stage_layers(name, layer_indices):
        if not isinstance(layer_indices, (list, tuple)):
            raise TypeError(f"{name} must be a list/tuple of zero-based layer indices.")
        if len(layer_indices) == 0:
            raise ValueError(f"{name} must contain at least one layer index.")
        validated = []
        for layer_idx in layer_indices:
            if not isinstance(layer_idx, int):
                raise TypeError(f"{name} must contain integers, got {type(layer_idx)!r}.")
            if layer_idx < 0:
                raise ValueError(f"{name} does not support negative indices, got {layer_idx}.")
            validated.append(layer_idx)
        return validated

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

    def _extract_ssl_branch_outputs(self, audio, padding_mask=None):
        beats_outputs = self.beats_encoder(
            audio,
            padding_mask=padding_mask,
            selected_layers=self.beats_layers,
            return_layer_dict=True,
        )
        wavlm_outputs = self.wavlm_encoder(
            audio,
            padding_mask=padding_mask,
            selected_layers=self.wavlm_layers,
            return_layer_dict=True,
        )
        beats_selected_features = beats_outputs.get("selected_hidden_states")
        wavlm_selected_features = wavlm_outputs.get("selected_hidden_states")
        if beats_selected_features is None:
            raise RuntimeError(
                "BEATsEncoder did not return selected_hidden_states when hierarchical "
                "SSL fusion requested them."
            )
        if wavlm_selected_features is None:
            raise RuntimeError(
                "WavLMEncoder did not return selected_hidden_states when hierarchical "
                "SSL fusion requested them."
            )
        return (
            beats_outputs,
            wavlm_outputs,
            beats_selected_features,
            wavlm_selected_features,
        )

    def forward(
        self,
        audio,
        target_frame_len=None,
        padding_mask=None,
        return_intermediates=False,
    ):
        cnn_outputs = self.crnn_encoder(audio, padding_mask=padding_mask)
        (
            beats_outputs,
            wavlm_outputs,
            beats_selected_features,
            wavlm_selected_features,
        ) = self._extract_ssl_branch_outputs(audio, padding_mask=padding_mask)

        cnn_features = cnn_outputs["sequence_features"]
        ssl_outputs = self.ssl_fusion_block(
            beats_selected_features=beats_selected_features,
            wavlm_selected_features=wavlm_selected_features,
        )
        fused_ssl_features = ssl_outputs["fused_ssl_output"]
        fused_ssl_aligned = self.final_fusion_aligner(fused_ssl_features, cnn_features)

        fused_features = torch.cat(
            [
                self.cnn_pre_fusion_norm(cnn_features),
                self.ssl_pre_fusion_norm(fused_ssl_aligned),
            ],
            dim=-1,
        )
        merged_features = self.merge_mlp(fused_features)

        decoder_inputs = self.label_aligner(merged_features, target_frame_len)
        decoder_outputs = self.decoder(decoder_inputs)

        outputs = {
            "strong_preds": decoder_outputs["strong_preds"],
            "weak_preds": decoder_outputs["weak_preds"],
            "strong_logits": decoder_outputs["strong_logits"],
            "encoder_sequence_features": merged_features,
            "aligned_sequence_features": decoder_inputs,
            "crnn_sequence_features": cnn_features,
            "beats_sequence_features": beats_outputs["sequence_features"],
            "wavlm_sequence_features": wavlm_outputs["sequence_features"],
            "beats_selected_layers": beats_outputs.get("selected_layers"),
            "wavlm_selected_layers": wavlm_outputs.get("selected_layers"),
            "fused_ssl_sequence_features": fused_ssl_features,
            "fused_ssl_aligned_features": fused_ssl_aligned,
            "fused_sequence_features": fused_features,
        }

        if return_intermediates:
            outputs["encoder_frontend_features"] = cnn_outputs.get("frontend_features")
            outputs["beats_frontend_features"] = beats_outputs.get("frontend_features")
            outputs["wavlm_frontend_features"] = wavlm_outputs.get("frontend_features")
            outputs["decoder_inputs"] = decoder_outputs.get("decoder_inputs")
            outputs["decoder_frame_features"] = decoder_outputs.get("frame_features")
            outputs["attention_weights"] = decoder_outputs.get("attention_weights")
            outputs["beats_selected_features"] = beats_selected_features
            outputs["wavlm_selected_features"] = wavlm_selected_features
            for key, value in ssl_outputs.items():
                outputs[f"ssl_{key}"] = value

        return outputs
