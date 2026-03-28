import torch
import torch.nn as nn

from desed_task.nnet.RNN import BidirectionalGRU


class SEDDecoder(nn.Module):
    """Shared SED decoder for all encoders.

    The decoder consumes time-aligned sequence features and applies an optional
    projection, an optional temporal module, and shared strong/weak prediction heads.
    """

    def __init__(
        self,
        input_dim,
        n_classes,
        input_proj_dim=None,
        use_gru=True,
        hidden_dim=128,
        rnn_layers=2,
        dropout=0.5,
        dropout_recurrent=0.0,
        attention=True,
    ):
        super().__init__()
        self.attention = attention
        self.use_gru = use_gru

        if input_proj_dim is not None and input_proj_dim != input_dim:
            self.input_proj = nn.Linear(input_dim, input_proj_dim)
            temporal_in_dim = input_proj_dim
        else:
            self.input_proj = nn.Identity()
            temporal_in_dim = input_dim

        if use_gru:
            self.temporal = BidirectionalGRU(
                n_in=temporal_in_dim,
                n_hidden=hidden_dim,
                dropout=dropout_recurrent,
                num_layers=rnn_layers,
            )
            head_dim = hidden_dim * 2
        else:
            self.temporal = nn.Identity()
            head_dim = temporal_in_dim

        self.dropout = nn.Dropout(dropout)
        self.strong_head = nn.Linear(head_dim, n_classes)
        if attention:
            self.attention_head = nn.Linear(head_dim, n_classes)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence_features, pad_mask=None):
        decoder_inputs = self.input_proj(sequence_features)
        temporal_features = self.temporal(decoder_inputs)
        temporal_features = self.dropout(temporal_features)

        strong_logits_bt = self.strong_head(temporal_features)
        strong_sequence = torch.sigmoid(strong_logits_bt)

        if self.attention:
            attention_logits = self.attention_head(temporal_features)
            if pad_mask is not None:
                attention_logits = attention_logits.masked_fill(
                    pad_mask.unsqueeze(-1), -1e30
                )
            attention_weights = self.softmax(attention_logits)
            attention_weights = torch.clamp(attention_weights, min=1e-7, max=1.0)
            weak_preds = (strong_sequence * attention_weights).sum(1) / attention_weights.sum(1)
        else:
            attention_weights = None
            weak_preds = strong_sequence.mean(1)

        return {
            "strong_logits": strong_logits_bt.transpose(1, 2).contiguous(),
            "strong_preds": strong_sequence.transpose(1, 2).contiguous(),
            "weak_preds": weak_preds,
            "decoder_inputs": decoder_inputs,
            "frame_features": temporal_features,
            "attention_weights": attention_weights,
        }
