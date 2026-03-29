import torch.nn as nn


class MergeMLP(nn.Module):
    """Projection block used after late-fusion feature concatenation."""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation="gelu",
        dropout=0.5,
        use_layernorm=False,
    ):
        super().__init__()
        activation_name = activation.lower()
        if activation_name == "relu":
            activation_layer = nn.ReLU()
        elif activation_name == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError(f"Unsupported merge activation: {activation}")

        layers = [nn.Linear(input_dim, output_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(output_dim))
        layers.extend([activation_layer, nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, sequence_features):
        return self.net(sequence_features)
