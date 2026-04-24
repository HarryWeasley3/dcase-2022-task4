import torch
import torch.nn as nn


def _build_activation(name):
    activation_name = name.lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported gate activation: {name}")


class BEATsMainResidualGatedFusion(nn.Module):
    """Three-way residual gated fusion with a BEATs main path.

    Two forward modes are supported:

    1. Legacy mode: align every branch to the CRNN timeline and fuse BEATs, CRNN,
       and WavLM in one step.
    2. BEATs time-anchor mode: keep BEATs on its native time grid, first fuse
       BEATs(main) + WavLM(aux) on the BEATs grid, then inject aligned CRNN
       features as a second residual correction.

    ``stopgrad_gate_inputs`` only detaches the controller inputs seen by the gate
    MLPs. The residual fusion path itself keeps normal gradients.
    """

    def __init__(
        self,
        beats_input_dim,
        crnn_input_dim,
        wavlm_input_dim,
        fuse_dim,
        gate_mode="channel",
        gate_hidden_dim=None,
        gate_activation="gelu",
        gate_dropout=0.0,
        use_post_fusion_proj=False,
        post_fusion_dim=None,
        post_fusion_dropout=0.0,
        use_alpha_scale=False,
        alpha_init=1.0,
        gate_bias_init=-1.5,
        stopgrad_gate_inputs=False,
        use_beats_time_anchor=False,
    ):
        super().__init__()
        self.gate_mode = gate_mode.lower()
        if self.gate_mode not in {"channel", "frame"}:
            raise ValueError(
                f"Unsupported gate_mode: {gate_mode}. Expected 'channel' or 'frame'."
            )
        self.stopgrad_gate_inputs = bool(stopgrad_gate_inputs)
        self.use_beats_time_anchor = bool(use_beats_time_anchor)

        self.beats_proj = (
            nn.Identity()
            if beats_input_dim == fuse_dim
            else nn.Linear(beats_input_dim, fuse_dim)
        )
        self.crnn_proj = (
            nn.Identity()
            if crnn_input_dim == fuse_dim
            else nn.Linear(crnn_input_dim, fuse_dim)
        )
        self.wavlm_proj = (
            nn.Identity()
            if wavlm_input_dim == fuse_dim
            else nn.Linear(wavlm_input_dim, fuse_dim)
        )

        self.beats_norm = nn.LayerNorm(fuse_dim)
        self.crnn_norm = nn.LayerNorm(fuse_dim)
        self.wavlm_norm = nn.LayerNorm(fuse_dim)
        self.ssl_main_norm = nn.LayerNorm(fuse_dim)

        gate_hidden_dim = gate_hidden_dim or fuse_dim
        gate_output_dim = fuse_dim if self.gate_mode == "channel" else 1

        self.crnn_gate_net = self._build_gate_net(
            fuse_dim,
            gate_hidden_dim,
            gate_output_dim,
            gate_activation,
            gate_dropout,
            gate_bias_init,
        )
        self.wavlm_gate_net = self._build_gate_net(
            fuse_dim,
            gate_hidden_dim,
            gate_output_dim,
            gate_activation,
            gate_dropout,
            gate_bias_init,
        )

        self.use_alpha_scale = use_alpha_scale
        if use_alpha_scale:
            self.alpha_crnn = nn.Parameter(torch.tensor(float(alpha_init)))
            self.alpha_wavlm = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer(
                "alpha_crnn",
                torch.tensor(float(alpha_init)),
                persistent=False,
            )
            self.register_buffer(
                "alpha_wavlm",
                torch.tensor(float(alpha_init)),
                persistent=False,
            )

        output_dim = post_fusion_dim or fuse_dim
        self.use_post_fusion_proj = use_post_fusion_proj
        if use_post_fusion_proj:
            self.post_fusion_proj = nn.Sequential(
                nn.Linear(fuse_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(post_fusion_dropout),
            )
        else:
            self.post_fusion_proj = nn.Identity()
        self.output_dim = output_dim if use_post_fusion_proj else fuse_dim

    def _maybe_detach_gate_input(self, tensor):
        if self.stopgrad_gate_inputs:
            return tensor.detach()
        return tensor

    def _compute_gate(self, gate_net, main_features, aux_features):
        gate_inputs = torch.cat(
            [
                self._maybe_detach_gate_input(main_features),
                self._maybe_detach_gate_input(aux_features),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(gate_net(gate_inputs))
        return gate, gate_inputs

    @staticmethod
    def _build_gate_net(
        fuse_dim,
        gate_hidden_dim,
        gate_output_dim,
        gate_activation,
        gate_dropout,
        gate_bias_init,
    ):
        gate_net = nn.Sequential(
            nn.Linear(fuse_dim * 2, gate_hidden_dim),
            _build_activation(gate_activation),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, gate_output_dim),
        )
        final_gate_layer = gate_net[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.constant_(final_gate_layer.bias, gate_bias_init)
        return gate_net

    def forward(self, crnn_features, beats_features, wavlm_features):
        beats_projected = self.beats_proj(beats_features)
        crnn_projected = self.crnn_proj(crnn_features)
        wavlm_projected = self.wavlm_proj(wavlm_features)

        beats_normalized = self.beats_norm(beats_projected)
        crnn_normalized = self.crnn_norm(crnn_projected)
        wavlm_normalized = self.wavlm_norm(wavlm_projected)

        gate_wavlm, gate_wavlm_inputs = self._compute_gate(
            self.wavlm_gate_net,
            beats_normalized,
            wavlm_normalized,
        )

        fused_ssl = beats_normalized + self.alpha_wavlm * gate_wavlm * wavlm_normalized

        if self.use_beats_time_anchor:
            crnn_gate_main = self.ssl_main_norm(fused_ssl)
        else:
            crnn_gate_main = beats_normalized

        gate_crnn, gate_crnn_inputs = self._compute_gate(
            self.crnn_gate_net,
            crnn_gate_main,
            crnn_normalized,
        )

        fused = fused_ssl + self.alpha_crnn * gate_crnn * crnn_normalized
        fused_output = self.post_fusion_proj(fused)

        return {
            "beats_projected": beats_projected,
            "crnn_projected": crnn_projected,
            "wavlm_projected": wavlm_projected,
            "beats_normalized": beats_normalized,
            "crnn_normalized": crnn_normalized,
            "wavlm_normalized": wavlm_normalized,
            "ssl_main_normalized": crnn_gate_main,
            "gate_crnn_inputs": gate_crnn_inputs,
            "gate_wavlm_inputs": gate_wavlm_inputs,
            "gate_crnn": gate_crnn,
            "gate_wavlm": gate_wavlm,
            "alpha_crnn": self.alpha_crnn,
            "alpha_wavlm": self.alpha_wavlm,
            "fused_ssl": fused_ssl,
            "fused": fused,
            "fused_output": fused_output,
        }
