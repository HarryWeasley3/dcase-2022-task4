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
    """Residual gated fusion for a strong CRNN trunk + weaker BEATs branch.

    This block keeps the CRNN stream as the main path and lets the frozen BEATs
    stream contribute only as a gated residual:

        fused = cnn_norm + gate * beats_norm

    The explicit projections + per-branch LayerNorm help mitigate the feature
    scale mismatch we observed in diagnostics, while the gate gives the model a
    direct way to decide when BEATs should complement the CRNN features.
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

        self.use_alpha_scale = use_alpha_scale
        if use_alpha_scale:
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

        fused = cnn_normalized + self.alpha * gate * beats_normalized
        fused_output = self.post_fusion_proj(fused)

        return {
            "cnn_projected": cnn_projected,
            "beats_projected": beats_projected,
            "cnn_normalized": cnn_normalized,
            "beats_normalized": beats_normalized,
            "gate": gate,
            "fused": fused,
            "fused_output": fused_output,
        }
