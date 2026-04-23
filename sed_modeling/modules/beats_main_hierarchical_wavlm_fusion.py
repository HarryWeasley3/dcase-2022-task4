import torch
import torch.nn as nn

from .beats_main_residual_gated_fusion import BEATsMainResidualGatedFusion
from .residual_gated_fusion import ResidualGatedFusion
from .time_aligner import FusionTimeAligner


class BEATsMainHierarchicalWavLMFusion(nn.Module):
    """Hierarchical SSL fusion with BEATs as the main branch.

    Stage 1:
        BEATs[layer_1] + gated WavLM[layer_1]

    Stage 2 (optional):
        BEATs[layer_2] + gated Stage1 + gated WavLM[layer_2]

    The module is intentionally lightweight so it can be slotted into the
    existing encoder -> decoder pipeline without changing the trainer or the
    shared decoder.
    """

    def __init__(
        self,
        beats_input_dim,
        wavlm_input_dim,
        fuse_dim,
        num_stages=1,
        align_method="adaptive_avg",
        interpolate_mode="linear",
        gate_mode="channel",
        gate_hidden_dim=None,
        gate_activation="gelu",
        gate_dropout=0.0,
        use_alpha_scale=False,
        alpha_init=1.0,
    ):
        super().__init__()
        if int(num_stages) not in {1, 2}:
            raise ValueError(
                "BEATsMainHierarchicalWavLMFusion only supports num_stages=1 or 2, "
                f"got {num_stages}."
            )

        self.num_stages = int(num_stages)
        self.output_dim = fuse_dim
        self.aligner = FusionTimeAligner(
            method=align_method,
            interpolate_mode=interpolate_mode,
        )

        self.stage1_fusion = ResidualGatedFusion(
            cnn_input_dim=wavlm_input_dim,
            beats_input_dim=beats_input_dim,
            fuse_dim=fuse_dim,
            gate_mode=gate_mode,
            gate_hidden_dim=gate_hidden_dim,
            gate_activation=gate_activation,
            gate_dropout=gate_dropout,
            use_post_fusion_proj=False,
            use_alpha_scale=use_alpha_scale,
            alpha_init=alpha_init,
        )

        if self.num_stages == 2:
            self.stage2_fusion = BEATsMainResidualGatedFusion(
                beats_input_dim=beats_input_dim,
                crnn_input_dim=fuse_dim,
                wavlm_input_dim=wavlm_input_dim,
                fuse_dim=fuse_dim,
                gate_mode=gate_mode,
                gate_hidden_dim=gate_hidden_dim,
                gate_activation=gate_activation,
                gate_dropout=gate_dropout,
                use_post_fusion_proj=False,
                use_alpha_scale=use_alpha_scale,
                alpha_init=alpha_init,
            )
        else:
            self.stage2_fusion = None

    def _validate_stage_inputs(self, branch_name, features):
        if not isinstance(features, (list, tuple)):
            raise TypeError(
                f"{branch_name} features must be a list/tuple of [batch, time, dim] tensors."
            )
        if len(features) < self.num_stages:
            raise ValueError(
                f"{branch_name} features must provide at least {self.num_stages} stages, "
                f"got {len(features)}."
            )
        validated = []
        for stage_idx, feature in enumerate(features[: self.num_stages], start=1):
            if not torch.is_tensor(feature) or feature.ndim != 3:
                raise ValueError(
                    f"{branch_name} stage {stage_idx} must have shape [batch, time, dim], "
                    f"got {type(feature)!r} / "
                    f"{tuple(feature.shape) if torch.is_tensor(feature) else 'n/a'}."
                )
            validated.append(feature)
        return validated

    def forward(self, beats_selected_features, wavlm_selected_features):
        beats_features = self._validate_stage_inputs(
            "BEATs selected", beats_selected_features
        )
        wavlm_features = self._validate_stage_inputs(
            "WavLM selected", wavlm_selected_features
        )

        stage1_beats = beats_features[0]
        stage1_wavlm = self.aligner(wavlm_features[0], stage1_beats)
        stage1_outputs = self.stage1_fusion(stage1_wavlm, stage1_beats)
        stage1_output = stage1_outputs["fused_output"]

        outputs = {
            "beats_selected_features": list(beats_features),
            "wavlm_selected_features": list(wavlm_features),
            "stage1_main_features": stage1_beats,
            "stage1_aux_features": stage1_wavlm,
            "stage1_output": stage1_output,
            "stage1_gate": stage1_outputs["gate"],
            "stage1_main_projected": stage1_outputs["beats_projected"],
            "stage1_aux_projected": stage1_outputs["cnn_projected"],
            "stage1_main_normalized": stage1_outputs["beats_normalized"],
            "stage1_aux_normalized": stage1_outputs["cnn_normalized"],
            "fused_ssl_output": stage1_output,
        }

        if self.num_stages == 1:
            return outputs

        stage2_beats = beats_features[1]
        stage1_aligned = self.aligner(stage1_output, stage2_beats)
        stage2_wavlm = self.aligner(wavlm_features[1], stage2_beats)
        stage2_outputs = self.stage2_fusion(
            stage1_aligned,
            stage2_beats,
            stage2_wavlm,
        )
        outputs.update(
            {
                "stage2_main_features": stage2_beats,
                "stage2_previous_stage_features": stage1_aligned,
                "stage2_aux_features": stage2_wavlm,
                "stage2_output": stage2_outputs["fused_output"],
                "stage2_previous_stage_gate": stage2_outputs["gate_crnn"],
                "stage2_aux_gate": stage2_outputs["gate_wavlm"],
                "stage2_main_projected": stage2_outputs["beats_projected"],
                "stage2_previous_stage_projected": stage2_outputs["crnn_projected"],
                "stage2_aux_projected": stage2_outputs["wavlm_projected"],
                "stage2_main_normalized": stage2_outputs["beats_normalized"],
                "stage2_previous_stage_normalized": stage2_outputs["crnn_normalized"],
                "stage2_aux_normalized": stage2_outputs["wavlm_normalized"],
                "fused_ssl_output": stage2_outputs["fused_output"],
            }
        )
        return outputs
