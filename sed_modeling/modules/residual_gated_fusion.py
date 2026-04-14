import torch
import torch.nn as nn


def _build_activation(name):
    activation_name = name.lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported gate activation: {name}")


class ResidualGatedFusion(nn.Module):
    """Residual gated fusion with a BEATs main path and CRNN correction branch.

    This block keeps the stronger BEATs stream as the main path and applies the
    CRNN stream only as a gated residual correction:

        fused = beats_norm + alpha * gate * cnn_norm

    The explicit projections + per-branch LayerNorm keep the two streams on a
    comparable scale, while the gate and alpha keep the CRNN residual
    conservative at initialization so the model starts close to the strong
    BEATs-only solution.
    """

    def __init__(
        self,
        cnn_input_dim,
        beats_input_dim,
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
    ):
        super().__init__()
        self.gate_mode = gate_mode.lower()
        if self.gate_mode not in {"channel", "frame"}:
            raise ValueError(
                f"Unsupported gate_mode: {gate_mode}. Expected 'channel' or 'frame'."
            )

        self.cnn_proj = (
            nn.Identity()
            if cnn_input_dim == fuse_dim
            else nn.Linear(cnn_input_dim, fuse_dim)
        )
        self.beats_proj = (
            nn.Identity()
            if beats_input_dim == fuse_dim
            else nn.Linear(beats_input_dim, fuse_dim)
        )
        self.cnn_norm = nn.LayerNorm(fuse_dim)
        self.beats_norm = nn.LayerNorm(fuse_dim)

        gate_hidden_dim = gate_hidden_dim or fuse_dim
        gate_output_dim = fuse_dim if self.gate_mode == "channel" else 1
        gate_activation_layer = _build_activation(gate_activation)
        self.gate_net = nn.Sequential(
            nn.Linear(fuse_dim * 2, gate_hidden_dim),
            gate_activation_layer,
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, gate_output_dim),
        )
        final_gate_layer = self.gate_net[-1]
        if isinstance(final_gate_layer, nn.Linear):
            # Start with a small gate so the CRNN branch only makes a weak
            # residual correction until training proves it is helpful.
            nn.init.constant_(final_gate_layer.bias, gate_bias_init)

        self.use_alpha_scale = use_alpha_scale
        if use_alpha_scale:
            # A learnable alpha lets us stay close to the BEATs-only baseline at
            # initialization and grow the CRNN residual only when it improves.
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer(
                "alpha",
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

    def forward(self, cnn_features, beats_features):
        cnn_projected = self.cnn_proj(cnn_features)
        beats_projected = self.beats_proj(beats_features)

        cnn_normalized = self.cnn_norm(cnn_projected)
        beats_normalized = self.beats_norm(beats_projected)

        gate_inputs = torch.cat([cnn_normalized, beats_normalized], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_inputs))

        fused = beats_normalized + self.alpha * gate * cnn_normalized
        fused_output = self.post_fusion_proj(fused)

        return {
            "cnn_projected": cnn_projected,
            "beats_projected": beats_projected,
            "auxiliary_projected": beats_projected,
            "cnn_normalized": cnn_normalized,
            "beats_normalized": beats_normalized,
            "auxiliary_normalized": beats_normalized,
            "gate": gate,
            "fused": fused,
            "fused_output": fused_output,
        }
