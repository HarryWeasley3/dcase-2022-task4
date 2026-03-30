from copy import deepcopy

import torch.nn as nn

from sed_modeling.decoders import SEDDecoder
from sed_modeling.encoders import BEATsEncoder, CRNNEncoder
from sed_modeling.modules import (
    FusionTimeAligner,
    MergeMLP,
    ResidualGatedFusion,
    TimeAligner,
)

from .crnn_beats_late_fusion import CRNNBEATsLateFusionModel
from .crnn_beats_residual_gated_fusion import CRNNBEATsResidualGatedFusionModel


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_model_config(config):
    net_cfg = config.get("net", {})
    model_cfg = {
        "model_type": "single_encoder",
        "encoder_type": "crnn",
        "align": {
            "method": "interpolate",
            "interpolate_mode": "linear",
        },
        "crnn_encoder": {
            "n_in_channel": net_cfg.get("n_in_channel", 1),
            "dropout": net_cfg.get("dropout", 0.5),
            "train_cnn": net_cfg.get("train_cnn", True),
            "activation": net_cfg.get("activation", "glu"),
            "kernel_size": net_cfg.get("kernel_size", [3, 3, 3]),
            "padding": net_cfg.get("padding", [1, 1, 1]),
            "stride": net_cfg.get("stride", [1, 1, 1]),
            "nb_filters": net_cfg.get("nb_filters", [64, 64, 64]),
            "pooling": net_cfg.get("pooling", [(1, 4), (1, 4), (1, 4)]),
            "cnn_integration": net_cfg.get("cnn_integration", False),
            "freeze_bn": net_cfg.get("freeze_bn", False),
        },
        "beats": {
            "checkpoint": "",
            "freeze": True,
            "feature_layer": None,
            "fbank_mean": 15.41663,
            "fbank_std": 6.55582,
        },
        "decoder": {
            "input_proj_dim": None,
            "use_gru": net_cfg.get("rnn_type", "BGRU") == "BGRU",
            "hidden_dim": net_cfg.get("n_RNN_cell", 128),
            "rnn_layers": net_cfg.get(
                "rnn_layers", net_cfg.get("n_layers_RNN", 2)
            ),
            "dropout": net_cfg.get("dropout", 0.5),
            "dropout_recurrent": net_cfg.get("dropout_recurrent", 0.0),
            "attention": net_cfg.get("attention", True),
        },
        "fusion": {
            "enabled": False,
            "fusion_type": "concat",
            "align_method": "adaptive_avg",
            "interpolate_mode": "linear",
            "merge_mlp_dim": 256,
            "merge_activation": "gelu",
            "merge_dropout": net_cfg.get("dropout", 0.5),
            "use_layernorm": False,
            "fuse_dim": 256,
            "norm_type": "layernorm",
            "gate_mode": "channel",
            "gate_hidden_dim": None,
            "gate_activation": "gelu",
            "gate_dropout": net_cfg.get("dropout", 0.5),
            "use_post_fusion_proj": True,
            "post_fusion_dim": 256,
            "post_fusion_dropout": net_cfg.get("dropout", 0.5),
            "use_alpha_scale": False,
            "alpha_init": 1.0,
        },
        "teacher": {
            "share_frozen_encoder": True,
            "share_frozen_beats": True,
        },
    }

    if "model" in config:
        _deep_update(model_cfg, config["model"])

    return model_cfg


class SEDModel(nn.Module):
    """Unified waveform -> encoder -> aligner -> decoder SED model."""

    def __init__(self, encoder, decoder, aligner, encoder_type, build_config=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.aligner = aligner
        self.encoder_type = encoder_type
        self.build_config = deepcopy(build_config) if build_config is not None else None

    @property
    def needs_input_scaler(self):
        return getattr(self.encoder, "requires_input_scaler", False)

    @property
    def is_frozen_encoder(self):
        return getattr(self.encoder, "is_frozen", False)

    def set_input_scaler(self, scaler):
        if hasattr(self.encoder, "set_input_scaler"):
            self.encoder.set_input_scaler(scaler)

    def prepare_inputs(self, audio):
        if hasattr(self.encoder, "prepare_inputs"):
            return self.encoder.prepare_inputs(audio)
        return audio

    def make_teacher_copy(self, share_encoder_if_frozen=True):
        if self.build_config is None:
            teacher = deepcopy(self)
            if share_encoder_if_frozen and self.is_frozen_encoder:
                teacher.encoder = self.encoder
            return teacher

        teacher = build_sed_model(self.build_config)
        teacher.load_state_dict(self.state_dict(), strict=True)
        if share_encoder_if_frozen and self.is_frozen_encoder:
            teacher.encoder = self.encoder
        return teacher

    def forward(
        self,
        audio,
        target_frame_len=None,
        padding_mask=None,
        return_intermediates=False,
    ):
        encoder_outputs = self.encoder(audio, padding_mask=padding_mask)
        aligned_features = self.aligner(
            encoder_outputs["sequence_features"], target_frame_len
        )
        decoder_outputs = self.decoder(aligned_features)

        outputs = {
            "encoder_sequence_features": encoder_outputs["sequence_features"],
            "aligned_sequence_features": aligned_features,
            "strong_preds": decoder_outputs["strong_preds"],
            "weak_preds": decoder_outputs["weak_preds"],
            "strong_logits": decoder_outputs["strong_logits"],
        }

        if return_intermediates:
            outputs["encoder_frontend_features"] = encoder_outputs.get(
                "frontend_features"
            )
            outputs["decoder_inputs"] = decoder_outputs.get("decoder_inputs")
            outputs["decoder_frame_features"] = decoder_outputs.get("frame_features")
            outputs["attention_weights"] = decoder_outputs.get("attention_weights")

        return outputs


def build_sed_model(config):
    model_cfg = resolve_model_config(config)
    model_type = model_cfg.get("model_type", "single_encoder").lower()
    encoder_type = model_cfg["encoder_type"].lower()
    num_classes = config.get("net", {}).get("nclass")
    if num_classes is None:
        raise ValueError("config['net']['nclass'] must be defined for the shared decoder.")

    if model_type == "crnn_beats_late_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        beats_encoder = BEATsEncoder(**model_cfg["beats"])
        fusion_cfg = model_cfg["fusion"]
        fusion_aligner = FusionTimeAligner(
            method=fusion_cfg.get("align_method", "adaptive_avg"),
            interpolate_mode=fusion_cfg.get("interpolate_mode", "linear"),
        )
        fusion_type = fusion_cfg.get("fusion_type", "concat").lower()
        if fusion_type == "concat":
            fusion_input_dim = crnn_encoder.output_dim + beats_encoder.output_dim
        elif fusion_type == "add":
            fusion_input_dim = crnn_encoder.output_dim
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        merge_mlp = MergeMLP(
            input_dim=fusion_input_dim,
            output_dim=fusion_cfg.get("merge_mlp_dim", 256),
            activation=fusion_cfg.get("merge_activation", "gelu"),
            dropout=fusion_cfg.get("merge_dropout", 0.5),
            use_layernorm=fusion_cfg.get("use_layernorm", False),
        )
        label_aligner = TimeAligner(**model_cfg["align"])
        decoder = SEDDecoder(
            input_dim=fusion_cfg.get("merge_mlp_dim", 256),
            n_classes=num_classes,
            **model_cfg["decoder"],
        )
        return CRNNBEATsLateFusionModel(
            crnn_encoder=crnn_encoder,
            beats_encoder=beats_encoder,
            fusion_aligner=fusion_aligner,
            merge_mlp=merge_mlp,
            decoder=decoder,
            label_aligner=label_aligner,
            fusion_type=fusion_type,
            build_config=config,
        )

    if model_type == "crnn_beats_residual_gated_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        beats_encoder = BEATsEncoder(**model_cfg["beats"])
        fusion_cfg = model_cfg["fusion"]
        fusion_aligner = FusionTimeAligner(
            method=fusion_cfg.get("align_method", "adaptive_avg"),
            interpolate_mode=fusion_cfg.get("interpolate_mode", "linear"),
        )
        norm_type = fusion_cfg.get("norm_type", "layernorm").lower()
        if norm_type != "layernorm":
            raise ValueError(
                "Residual gated fusion currently supports norm_type='layernorm' only."
            )

        fuse_dim = fusion_cfg.get("fuse_dim", fusion_cfg.get("merge_mlp_dim", 256))
        fusion_block = ResidualGatedFusion(
            cnn_input_dim=crnn_encoder.output_dim,
            beats_input_dim=beats_encoder.output_dim,
            fuse_dim=fuse_dim,
            gate_mode=fusion_cfg.get("gate_mode", "channel"),
            gate_hidden_dim=fusion_cfg.get("gate_hidden_dim"),
            gate_activation=fusion_cfg.get("gate_activation", "gelu"),
            gate_dropout=fusion_cfg.get("gate_dropout", 0.5),
            use_post_fusion_proj=fusion_cfg.get("use_post_fusion_proj", True),
            post_fusion_dim=fusion_cfg.get("post_fusion_dim", fuse_dim),
            post_fusion_dropout=fusion_cfg.get("post_fusion_dropout", 0.5),
            use_alpha_scale=fusion_cfg.get("use_alpha_scale", False),
            alpha_init=fusion_cfg.get("alpha_init", 1.0),
        )
        label_aligner = TimeAligner(**model_cfg["align"])
        decoder = SEDDecoder(
            input_dim=fusion_block.output_dim,
            n_classes=num_classes,
            **model_cfg["decoder"],
        )
        return CRNNBEATsResidualGatedFusionModel(
            crnn_encoder=crnn_encoder,
            beats_encoder=beats_encoder,
            fusion_aligner=fusion_aligner,
            fusion_block=fusion_block,
            decoder=decoder,
            label_aligner=label_aligner,
            build_config=config,
        )

    if encoder_type == "crnn":
        encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
    elif encoder_type == "beats":
        encoder = BEATsEncoder(**model_cfg["beats"])
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    aligner = TimeAligner(**model_cfg["align"])
    decoder = SEDDecoder(
        input_dim=encoder.output_dim,
        n_classes=num_classes,
        **model_cfg["decoder"],
    )
    return SEDModel(
        encoder,
        decoder,
        aligner,
        encoder_type=encoder_type,
        build_config=config,
    )
