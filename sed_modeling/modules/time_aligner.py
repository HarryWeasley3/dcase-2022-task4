import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAligner(nn.Module):
    """Align encoder time steps to the label frame grid seen by the decoder.

    Different encoders naturally emit different sequence lengths. Traditional SED
    losses and post-processing in this project operate on a fixed frame grid, so we
    centralize the time alignment here instead of scattering resize logic across the
    training loop.
    """

    def __init__(self, method="interpolate", interpolate_mode="linear"):
        super().__init__()
        self.method = method
        self.interpolate_mode = interpolate_mode

    def forward(self, sequence, target_length):
        if target_length is None:
            return sequence
        if sequence.ndim != 3:
            raise ValueError(
                f"TimeAligner expects [batch, time, dim] inputs, got {tuple(sequence.shape)}"
            )

        current_length = sequence.shape[1]
        if current_length == target_length:
            return sequence

        sequence_t = sequence.transpose(1, 2)

        if self.method == "interpolate":
            align_corners = False if self.interpolate_mode in {"linear"} else None
            aligned = F.interpolate(
                sequence_t,
                size=target_length,
                mode=self.interpolate_mode,
                align_corners=align_corners,
            )
        elif self.method == "nearest":
            aligned = F.interpolate(sequence_t, size=target_length, mode="nearest")
        elif self.method == "adaptive_avg":
            aligned = F.adaptive_avg_pool1d(sequence_t, output_size=target_length)
        elif self.method == "adaptive_max":
            aligned = F.adaptive_max_pool1d(sequence_t, output_size=target_length)
        else:
            raise ValueError(f"Unsupported alignment method: {self.method}")

        return aligned.transpose(1, 2).contiguous()
