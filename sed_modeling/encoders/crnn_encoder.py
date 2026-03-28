import math
import warnings

import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from desed_task.nnet.CNN import CNN


class CRNNEncoder(nn.Module):
    """Waveform-to-sequence encoder derived from the original CRNN frontend."""

    def __init__(self, feat_params, encoder_cfg):
        super().__init__()
        self.n_in_channel = encoder_cfg.get("n_in_channel", 1)
        self.cnn_integration = encoder_cfg.get("cnn_integration", False)
        self.train_cnn = encoder_cfg.get("train_cnn", True)
        self.freeze_bn = encoder_cfg.get("freeze_bn", False)
        self.requires_input_scaler = True
        self.is_frozen = not self.train_cnn

        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        self.amplitude_to_db = AmplitudeToDB(stype="amplitude")
        self.amplitude_to_db.amin = 1e-5
        self.input_scaler = nn.Identity()

        n_in_cnn = self.n_in_channel if not self.cnn_integration else 1
        self.cnn = CNN(
            n_in_channel=n_in_cnn,
            activation=encoder_cfg.get("activation", "glu"),
            conv_dropout=encoder_cfg.get("dropout", 0.5),
            kernel_size=encoder_cfg.get("kernel_size", [3, 3, 3]),
            padding=encoder_cfg.get("padding", [1, 1, 1]),
            stride=encoder_cfg.get("stride", [1, 1, 1]),
            nb_filters=encoder_cfg.get("nb_filters", [64, 64, 64]),
            pooling=encoder_cfg.get("pooling", [(1, 4), (1, 4), (1, 4)]),
        )

        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.output_dim = self._infer_output_dim(
            feat_params["n_mels"],
            self.cnn.nb_filters[-1],
            encoder_cfg.get("pooling", [(1, 4), (1, 4), (1, 4)]),
            self.n_in_channel if self.cnn_integration else 1,
        )

    def set_input_scaler(self, scaler):
        self.input_scaler = scaler if scaler is not None else nn.Identity()

    def prepare_inputs(self, audio):
        mels = self.mel_spec(audio)
        return self.amplitude_to_db(mels).clamp(min=-50, max=80)

    def _infer_output_dim(self, n_mels, final_channels, pooling, n_input_channels):
        freq = n_mels
        for _, pool_freq in pooling:
            freq = math.floor(freq / pool_freq)
        freq = max(freq, 1)
        channels = final_channels * (n_input_channels if self.cnn_integration else 1)
        return channels * freq

    def forward(self, audio, padding_mask=None):
        del padding_mask

        frontend_features = self.input_scaler(self.prepare_inputs(audio))
        x = frontend_features.transpose(1, 2).unsqueeze(1)

        if self.cnn_integration:
            batch_size_in, n_channels_in = x.size(0), x.size(1)
            x = x.view(batch_size_in * n_channels_in, 1, *x.shape[2:])
        else:
            batch_size_in, n_channels_in = x.size(0), x.size(1)

        x = self.cnn(x)
        batch_size, channels, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(batch_size_in, channels * n_channels_in, frames, freq)
            batch_size = batch_size_in

        if freq != 1:
            warnings.warn(
                f"CRNN encoder keeps {freq} frequency bins after pooling; flattening them into the feature dimension."
            )
            x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, frames, channels * freq)
        else:
            x = x.squeeze(-1).permute(0, 2, 1).contiguous()

        return {
            "sequence_features": x,
            "frontend_features": frontend_features,
        }

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
        return self
