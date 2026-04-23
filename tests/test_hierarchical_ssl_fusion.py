import torch
import torch.nn as nn

from sed_modeling.encoders.beats_encoder import BEATsEncoder
from sed_modeling.encoders.wavlm_encoder import WavLMEncoder
from sed_modeling.models.sed_model import build_sed_model, resolve_model_config


class _FakeBeatsBackbone:
    def __init__(self, num_layers=4):
        self.layers = [object() for _ in range(num_layers)]

    def __call__(self, x, padding_mask=None, layer=None):
        del padding_mask
        current = x
        layer_results = []
        for layer_idx in range(len(self.layers)):
            current = current + float(layer_idx + 1)
            layer_results.append((current.transpose(0, 1).contiguous(), None))
            if layer is not None and layer_idx == layer:
                return current, layer_results
        return current, layer_results


class _FakeBeatsModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.encoder = _FakeBeatsBackbone(num_layers=num_layers)
        self.layer_norm = nn.Identity()
        self.post_extract_proj = None
        self.dropout_input = nn.Identity()

    def preprocess(self, audio, fbank_mean=None, fbank_std=None):
        del fbank_mean, fbank_std
        return audio.unsqueeze(-1).repeat(1, 1, 2)

    def forward_padding_mask(self, features, padding_mask):
        del features
        return padding_mask

    def patch_embedding(self, features):
        # [B, 1, T, F] -> [B, 2, T, 1]
        return features.mean(dim=-1, keepdim=True).repeat(1, 2, 1, 1)


class _FakeWavLMModel(nn.Module):
    def __init__(self, num_layers=4, feature_dim=6):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim

    def forward(self, audio, lengths=None):
        batch_size = audio.shape[0]
        sequence_features = torch.arange(
            batch_size * 5 * self.feature_dim,
            dtype=audio.dtype,
            device=audio.device,
        ).reshape(batch_size, 5, self.feature_dim)
        return sequence_features, lengths

    def extract_features(self, audio, lengths=None, num_layers=None):
        batch_size = audio.shape[0]
        requested_layers = self.num_layers if num_layers is None else int(num_layers)
        features = []
        for layer_idx in range(requested_layers):
            feature = torch.full(
                (batch_size, 4, self.feature_dim),
                float(layer_idx + 1),
                dtype=audio.dtype,
                device=audio.device,
            )
            features.append(feature)
        return features, lengths


def _make_fake_beats_encoder():
    encoder = BEATsEncoder.__new__(BEATsEncoder)
    nn.Module.__init__(encoder)
    encoder.beats = _FakeBeatsModel(num_layers=4)
    encoder.feature_layer = None
    encoder.fbank_mean = 0.0
    encoder.fbank_std = 1.0
    encoder.requires_input_scaler = False
    encoder.is_frozen = False
    encoder.output_dim = 2
    return encoder


def _make_fake_wavlm_encoder():
    encoder = WavLMEncoder.__new__(WavLMEncoder)
    nn.Module.__init__(encoder)
    encoder.bundle = type(
        "_Bundle",
        (),
        {"_params": {"encoder_num_layers": 4, "encoder_embed_dim": 6}},
    )()
    encoder.output_layer = None
    encoder.requires_input_scaler = False
    encoder.is_frozen = False
    encoder.normalize_waveform = False
    encoder.wavlm = _FakeWavLMModel(num_layers=4, feature_dim=6)
    encoder.output_dim = 6
    return encoder


def test_beats_encoder_default_and_multilayer_contracts():
    encoder = _make_fake_beats_encoder()
    audio = torch.randn(2, 12)

    default_outputs = encoder(audio)
    assert set(default_outputs.keys()) == {
        "sequence_features",
        "frontend_features",
        "padding_mask",
    }
    assert default_outputs["sequence_features"].shape[:2] == (2, 12)

    multilayer_outputs = encoder(
        audio,
        selected_layers=[1, 3],
        return_layer_dict=True,
    )
    assert multilayer_outputs["selected_layers"] == [1, 3]
    assert len(multilayer_outputs["selected_hidden_states"]) == 2
    assert set(multilayer_outputs["layer_features"].keys()) == {1, 3}


def test_wavlm_encoder_default_and_multilayer_contracts():
    encoder = _make_fake_wavlm_encoder()
    audio = torch.randn(2, 12)

    default_outputs = encoder(audio)
    assert set(default_outputs.keys()) == {
        "sequence_features",
        "frontend_features",
        "lengths",
    }
    assert default_outputs["sequence_features"].shape == (2, 5, 6)

    multilayer_outputs = encoder(
        audio,
        selected_layers=[0, 2],
        return_layer_dict=True,
    )
    assert multilayer_outputs["selected_layers"] == [0, 2]
    assert len(multilayer_outputs["selected_hidden_states"]) == 2
    assert set(multilayer_outputs["layer_features"].keys()) == {0, 2}


class _FakeCRNNEncoder(nn.Module):
    def __init__(self, feat_params, encoder_cfg):
        super().__init__()
        del feat_params, encoder_cfg
        self.output_dim = 16
        self.requires_input_scaler = True
        self.is_frozen = False
        self.input_scaler = nn.Identity()

    def set_input_scaler(self, scaler):
        self.input_scaler = scaler if scaler is not None else nn.Identity()

    def prepare_inputs(self, audio):
        return audio

    def forward(self, audio, padding_mask=None):
        del padding_mask
        batch_size = audio.shape[0]
        sequence_features = torch.arange(
            batch_size * 14 * self.output_dim,
            dtype=audio.dtype,
            device=audio.device,
        ).reshape(batch_size, 14, self.output_dim)
        return {
            "sequence_features": sequence_features,
            "frontend_features": audio.unsqueeze(1),
        }


class _FakeBranchEncoder(nn.Module):
    def __init__(self, output_dim, base_length, num_layers, freeze=True, **kwargs):
        super().__init__()
        del kwargs
        self.output_dim = output_dim
        self.base_length = base_length
        self.num_layers = num_layers
        self.is_frozen = freeze
        self.requires_input_scaler = False

    def set_input_scaler(self, scaler):
        del scaler

    def prepare_inputs(self, audio):
        return audio

    def _make_feature(self, audio, layer_idx, time_length):
        batch_size = audio.shape[0]
        feature = torch.full(
            (batch_size, time_length, self.output_dim),
            float(layer_idx + 1),
            dtype=audio.dtype,
            device=audio.device,
        )
        return feature

    def forward(
        self,
        audio,
        padding_mask=None,
        selected_layers=None,
        return_all_layers=False,
        return_layer_dict=False,
    ):
        del padding_mask
        sequence_features = self._make_feature(audio, self.num_layers - 1, self.base_length)
        outputs = {
            "sequence_features": sequence_features,
            "frontend_features": audio,
        }
        if return_layer_dict and selected_layers is None and not return_all_layers:
            return_all_layers = True
        if return_all_layers:
            selected_layers = list(range(self.num_layers))
        if selected_layers is not None:
            selected_hidden_states = [
                self._make_feature(audio, layer_idx, self.base_length + layer_idx + 1)
                for layer_idx in selected_layers
            ]
            outputs["selected_layers"] = list(selected_layers)
            outputs["selected_hidden_states"] = selected_hidden_states
            if return_layer_dict:
                outputs["layer_features"] = dict(
                    zip(selected_layers, selected_hidden_states)
                )
        return outputs


class _FakeBEATsEncoder(_FakeBranchEncoder):
    def __init__(
        self,
        checkpoint="",
        freeze=True,
        feature_layer=None,
        fbank_mean=0.0,
        fbank_std=1.0,
        load_branch_weights=False,
        branch_checkpoint="",
    ):
        del checkpoint, feature_layer, fbank_mean, fbank_std, load_branch_weights, branch_checkpoint
        super().__init__(output_dim=24, base_length=9, num_layers=12, freeze=freeze)


class _FakeWavLMEncoder(_FakeBranchEncoder):
    def __init__(
        self,
        checkpoint="",
        bundle_name="WAVLM_BASE_PLUS",
        freeze=True,
        output_layer=None,
        use_bundle_weights=True,
        normalize_waveform=False,
    ):
        del checkpoint, bundle_name, output_layer, use_bundle_weights, normalize_waveform
        super().__init__(output_dim=20, base_length=7, num_layers=12, freeze=freeze)


def _base_config():
    return {
        "feats": {
            "sample_rate": 16000,
            "n_window": 400,
            "hop_length": 160,
            "f_min": 0,
            "f_max": 8000,
            "n_mels": 64,
        },
        "net": {
            "nclass": 10,
            "dropout": 0.0,
            "rnn_type": "GRU",
            "n_RNN_cell": 16,
            "rnn_layers": 1,
            "attention": True,
        },
        "model": {
            "align": {
                "method": "interpolate",
                "interpolate_mode": "linear",
            },
            "branches": {
                "crnn": {"enabled": True, "train": True},
                "beats": {
                    "enabled": True,
                    "pretrained_checkpoint": "unused_beats.pt",
                    "freeze": True,
                },
                "wavlm": {
                    "enabled": True,
                    "pretrained_checkpoint": "unused_wavlm.pt",
                    "freeze": True,
                    "bundle_name": "WAVLM_BASE_PLUS",
                    "use_bundle_weights": False,
                },
            },
            "fusion": {
                "strategy": "hierarchical_ssl",
                "fusion_type": "concat",
                "main_branch": "beats",
                "align_method": "adaptive_avg",
                "interpolate_mode": "linear",
                "merge_mlp_dim": 32,
                "merge_activation": "gelu",
                "merge_dropout": 0.0,
                "use_layernorm": True,
            },
            "ssl_hierarchical_fusion": {
                "enable": True,
                "beats_main": True,
                "num_stages": 2,
                "beats_layers": [3, 8],
                "wavlm_layers": [2, 7],
                "fuse_dim": 24,
                "align_method": "adaptive_avg",
                "interpolate_mode": "linear",
                "gate_mode": "channel",
                "gate_hidden_dim": 16,
                "gate_dropout": 0.0,
                "use_alpha_scale": True,
                "alpha_init": 0.2,
            },
            "decoder": {
                "input_proj_dim": None,
                "use_gru": False,
                "hidden_dim": 16,
                "rnn_layers": 1,
                "dropout": 0.0,
                "dropout_recurrent": 0.0,
                "attention": True,
            },
            "decoder_warmstart": {"enable": False, "checkpoint": ""},
            "crnn_warmstart": {"enable": False, "checkpoint": ""},
            "wavlm_warmstart": {"enable": False, "checkpoint": ""},
            "teacher": {
                "share_frozen_encoder": False,
                "share_frozen_beats": False,
                "share_frozen_wavlm": False,
            },
        },
    }


def test_resolve_model_config_maps_hierarchical_ssl_route():
    model_cfg = resolve_model_config(_base_config())
    assert model_cfg["model_type"] == "crnn_beats_wavlm_hierarchical_ssl_fusion"
    assert model_cfg["fusion"]["strategy"] == "hierarchical_ssl"


def test_build_and_forward_stage_a_and_existing_three_branch_routes(monkeypatch):
    monkeypatch.setattr(
        "sed_modeling.models.sed_model.CRNNEncoder",
        _FakeCRNNEncoder,
    )
    monkeypatch.setattr(
        "sed_modeling.models.sed_model.BEATsEncoder",
        _FakeBEATsEncoder,
    )
    monkeypatch.setattr(
        "sed_modeling.models.sed_model.WavLMEncoder",
        _FakeWavLMEncoder,
    )

    audio = torch.randn(2, 160)

    hierarchical_config = _base_config()
    stage_a_model = build_sed_model(hierarchical_config)
    stage_a_outputs = stage_a_model(
        audio,
        target_frame_len=25,
        return_intermediates=True,
    )
    assert stage_a_outputs["strong_preds"].shape == (2, 10, 25)
    assert stage_a_outputs["weak_preds"].shape == (2, 10)
    assert "ssl_stage1_output" in stage_a_outputs
    assert "ssl_stage2_output" in stage_a_outputs

    late_config = _base_config()
    late_config["model"]["fusion"] = {
        "strategy": "late",
        "fusion_type": "concat",
        "align_method": "adaptive_avg",
        "interpolate_mode": "linear",
        "merge_mlp_dim": 32,
        "merge_activation": "gelu",
        "merge_dropout": 0.0,
        "use_layernorm": True,
    }
    late_model = build_sed_model(late_config)
    late_outputs = late_model(audio, target_frame_len=25)
    assert late_outputs["strong_preds"].shape == (2, 10, 25)

    residual_config = _base_config()
    residual_config["model"]["fusion"] = {
        "strategy": "residual_gated",
        "main_branch": "beats",
        "align_method": "adaptive_avg",
        "interpolate_mode": "linear",
        "fuse_dim": 24,
        "norm_type": "layernorm",
        "gate_mode": "channel",
        "gate_hidden_dim": 16,
        "gate_activation": "gelu",
        "gate_dropout": 0.0,
        "use_post_fusion_proj": True,
        "post_fusion_dim": 24,
        "post_fusion_dropout": 0.0,
        "use_alpha_scale": True,
        "alpha_init": 0.2,
        "merge_mlp_dim": 32,
    }
    residual_model = build_sed_model(residual_config)
    residual_outputs = residual_model(audio, target_frame_len=25)
    assert residual_outputs["strong_preds"].shape == (2, 10, 25)
