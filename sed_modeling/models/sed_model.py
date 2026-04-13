from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from sed_modeling.decoders import SEDDecoder
from sed_modeling.encoders import BEATsEncoder, CRNNEncoder, WavLMEncoder
from sed_modeling.modules import (
    BEATsMainResidualGatedFusion,
    FusionTimeAligner,
    MergeMLP,
    ResidualGatedFusion,
    TimeAligner,
)

from .crnn_beats_late_fusion import CRNNBEATsLateFusionModel
from .crnn_beats_residual_gated_fusion import CRNNBEATsResidualGatedFusionModel
from .crnn_beats_wavlm_late_fusion import CRNNBEATsWavLMLateFusionModel
from .crnn_beats_wavlm_residual_gated_fusion import (
    CRNNBEATsWavLMResidualGatedFusionModel,
)
from .crnn_wavlm_late_fusion import CRNNWavLMLateFusionModel
from .crnn_wavlm_residual_gated_fusion import CRNNWavLMResidualGatedFusionModel


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_strategy_name(strategy):
    if strategy is None:
        return None
    strategy_name = str(strategy).lower()
    if strategy_name in {"late", "late_fusion", "concat"}:
        return "late"
    if strategy_name in {"residual_gated", "gated", "gate", "residual"}:
        return "residual_gated"
    return strategy_name


def _normalize_unified_fusion_cfg(fusion_cfg):
    if not isinstance(fusion_cfg, dict):
        return fusion_cfg

    normalized = deepcopy(fusion_cfg)
    strategy = _normalize_strategy_name(
        normalized.get("strategy", normalized.get("mode"))
    )
    selected_cfg = None
    if strategy == "late" and isinstance(normalized.get("late"), dict):
        selected_cfg = deepcopy(normalized["late"])
    elif strategy == "residual_gated" and isinstance(
        normalized.get("residual_gated"), dict
    ):
        selected_cfg = deepcopy(normalized["residual_gated"])

    if selected_cfg is not None:
        combine = selected_cfg.pop("combine", None)
        if combine is not None and "fusion_type" not in selected_cfg:
            selected_cfg["fusion_type"] = combine
        normalized.pop("late", None)
        normalized.pop("residual_gated", None)
        _deep_update(normalized, selected_cfg)

    combine = normalized.pop("combine", None)
    if combine is not None and "fusion_type" not in normalized:
        normalized["fusion_type"] = combine
    if strategy is not None:
        normalized["strategy"] = strategy

    return normalized


def _normalize_decoder_warmstart(model_cfg):
    decoder_warmstart = model_cfg.get("decoder_warmstart", {})
    if not isinstance(decoder_warmstart, dict):
        decoder_warmstart = {}

    fusion_cfg = model_cfg.get("fusion", {})
    if fusion_cfg.get("load_beats_decoder_head", False):
        decoder_warmstart.setdefault("enable", True)
        decoder_warmstart.setdefault(
            "checkpoint",
            fusion_cfg.get("beats_decoder_head_ckpt", ""),
        )
    elif fusion_cfg.get("beats_decoder_head_ckpt") and "checkpoint" not in decoder_warmstart:
        decoder_warmstart["checkpoint"] = fusion_cfg.get("beats_decoder_head_ckpt", "")

    decoder_warmstart.setdefault("enable", False)
    decoder_warmstart.setdefault("checkpoint", "")
    model_cfg["decoder_warmstart"] = decoder_warmstart
    return model_cfg


def _normalize_beats_branch_cfg(beats_branch_cfg):
    if not isinstance(beats_branch_cfg, dict):
        return beats_branch_cfg

    normalized = deepcopy(beats_branch_cfg)
    pretrained_checkpoint = normalized.pop("pretrained_checkpoint", None)
    if pretrained_checkpoint is not None:
        normalized["checkpoint"] = pretrained_checkpoint

    warmstart_cfg = normalized.pop("warmstart", None)
    if isinstance(warmstart_cfg, dict):
        warmstart_enable = warmstart_cfg.get("enable")
        warmstart_checkpoint = warmstart_cfg.get("checkpoint")
        if warmstart_enable is not None:
            normalized["load_branch_weights"] = warmstart_enable
        if warmstart_checkpoint is not None:
            normalized["branch_checkpoint"] = warmstart_checkpoint

    return normalized


def _normalize_wavlm_branch_cfg(wavlm_branch_cfg):
    if not isinstance(wavlm_branch_cfg, dict):
        return wavlm_branch_cfg

    normalized = deepcopy(wavlm_branch_cfg)
    pretrained_checkpoint = normalized.pop("pretrained_checkpoint", None)
    if pretrained_checkpoint is not None:
        normalized["checkpoint"] = pretrained_checkpoint
    return normalized


def _validate_main_branch(main_branch, allowed, enabled_branches):
    if main_branch is None:
        return
    main_branch_name = str(main_branch).lower()
    if main_branch_name not in allowed:
        raise ValueError(
            "Unsupported fusion.main_branch="
            f"{main_branch!r} for enabled branches {enabled_branches}. "
            f"Allowed values: {sorted(allowed)}"
        )


def _resolve_unified_model_axes(model_cfg):
    branches_cfg = model_cfg.get("branches")
    if not isinstance(branches_cfg, dict):
        return _normalize_decoder_warmstart(model_cfg)

    crnn_branch_cfg = deepcopy(branches_cfg.get("crnn", {}))
    beats_branch_cfg = deepcopy(branches_cfg.get("beats", {}))
    wavlm_branch_cfg = deepcopy(branches_cfg.get("wavlm", {}))

    if crnn_branch_cfg:
        train_cnn = crnn_branch_cfg.get("train", crnn_branch_cfg.get("train_cnn"))
        if train_cnn is not None:
            model_cfg["crnn_encoder"]["train_cnn"] = train_cnn

        warmstart_cfg = crnn_branch_cfg.get("warmstart")
        if isinstance(warmstart_cfg, dict):
            _deep_update(model_cfg["crnn_warmstart"], warmstart_cfg)

    if beats_branch_cfg:
        beats_branch_cfg = _normalize_beats_branch_cfg(beats_branch_cfg)
        beats_branch_cfg.pop("enabled", None)
        model_cfg["beats"] = _deep_update(model_cfg["beats"], beats_branch_cfg)

    if wavlm_branch_cfg:
        wavlm_branch_cfg = _normalize_wavlm_branch_cfg(wavlm_branch_cfg)
        wavlm_branch_cfg.pop("enabled", None)
        model_cfg["wavlm"] = _deep_update(model_cfg["wavlm"], wavlm_branch_cfg)

    fusion_cfg = _normalize_unified_fusion_cfg(model_cfg.get("fusion", {}))
    model_cfg["fusion"] = fusion_cfg

    enabled_branches = [
        branch_name
        for branch_name in ("crnn", "beats", "wavlm")
        if branches_cfg.get(branch_name, {}).get("enabled", False)
    ]
    if not enabled_branches:
        raise ValueError("model.branches must enable at least one branch.")

    if len(enabled_branches) == 1:
        model_cfg["model_type"] = "single_encoder"
        model_cfg["encoder_type"] = enabled_branches[0]
        model_cfg["fusion"]["enabled"] = False
        return _normalize_decoder_warmstart(model_cfg)

    fusion_cfg["enabled"] = True
    strategy = _normalize_strategy_name(
        fusion_cfg.get("strategy", fusion_cfg.get("mode"))
    )
    if strategy is None:
        fusion_type = str(fusion_cfg.get("fusion_type", "")).lower()
        if fusion_type in {"residual_gated", "beats_main_residual_gated"}:
            strategy = "residual_gated"
        else:
            strategy = "late"
    fusion_cfg["strategy"] = strategy

    enabled_tuple = tuple(enabled_branches)
    main_branch = fusion_cfg.get("main_branch", "auto")

    if strategy == "late":
        fusion_cfg.setdefault("fusion_type", "concat")
        mapping = {
            ("crnn", "beats"): "crnn_beats_late_fusion",
            ("crnn", "wavlm"): "crnn_wavlm_late_fusion",
            ("crnn", "beats", "wavlm"): "crnn_beats_wavlm_late_fusion",
        }
    elif strategy == "residual_gated":
        mapping = {
            ("crnn", "beats"): "crnn_beats_residual_gated_fusion",
            ("crnn", "wavlm"): "crnn_wavlm_residual_gated_fusion",
            ("crnn", "beats", "wavlm"): "crnn_beats_wavlm_residual_gated_fusion",
        }
        if enabled_tuple == ("crnn", "beats"):
            _validate_main_branch(main_branch, {"auto", "beats"}, enabled_branches)
            fusion_cfg.setdefault("fusion_type", "residual_gated")
            fusion_cfg["main_branch"] = "beats"
        elif enabled_tuple == ("crnn", "wavlm"):
            _validate_main_branch(main_branch, {"auto", "wavlm"}, enabled_branches)
            fusion_cfg.setdefault("fusion_type", "residual_gated")
            fusion_cfg["main_branch"] = "wavlm"
        elif enabled_tuple == ("crnn", "beats", "wavlm"):
            _validate_main_branch(main_branch, {"auto", "beats"}, enabled_branches)
            fusion_cfg.setdefault("fusion_type", "beats_main_residual_gated")
            fusion_cfg["main_branch"] = "beats"
    else:
        raise ValueError(f"Unsupported fusion strategy: {strategy}")

    if enabled_tuple not in mapping:
        raise ValueError(
            "Unsupported enabled branch combination for unified config: "
            f"{enabled_branches}. Supported multi-branch routes are "
            "CRNN+BEATs, CRNN+WavLM, and CRNN+BEATs+WavLM."
        )

    model_cfg["model_type"] = mapping[enabled_tuple]
    model_cfg["encoder_type"] = model_cfg["model_type"]
    return _normalize_decoder_warmstart(model_cfg)


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
            "load_branch_weights": False,
            "branch_checkpoint": "",
        },
        "wavlm": {
            "checkpoint": "",
            "bundle_name": "WAVLM_BASE_PLUS",
            "freeze": True,
            "output_layer": None,
            "use_bundle_weights": True,
            "normalize_waveform": False,
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
        "decoder_warmstart": {
            "enable": False,
            "checkpoint": "",
        },
        "fusion": {
            "enabled": False,
            "strategy": None,
            "main_branch": "auto",
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
            "load_beats_decoder_head": False,
            "beats_decoder_head_ckpt": "",
        },
        "teacher": {
            "share_frozen_encoder": True,
            "share_frozen_beats": True,
            "share_frozen_wavlm": True,
        },
        "crnn_warmstart": {
            "enable": False,
            "checkpoint": "",
        },
    }

    if "model" in config:
        _deep_update(model_cfg, deepcopy(config["model"]))

    return _resolve_unified_model_axes(model_cfg)


def _load_decoder_head_warmstart(model, model_cfg):
    decoder_warmstart = model_cfg.get("decoder_warmstart", {})
    if not decoder_warmstart.get("enable", False):
        return

    ckpt_path = decoder_warmstart.get("checkpoint", "")
    if not ckpt_path:
        print(
            "[decoder warm-start] enabled, but model.decoder_warmstart.checkpoint "
            "is empty; skipping."
        )
        return

    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.is_file():
        print(f"[decoder warm-start] checkpoint not found: {ckpt_path}; skipping.")
        return

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    source_prefix = None
    for candidate in ("sed_student.decoder.", "decoder."):
        if any(k.startswith(candidate) for k in source_state):
            source_prefix = candidate
            break

    if source_prefix is None:
        print(
            "[decoder warm-start] no compatible decoder prefix found in "
            f"{ckpt_path}; checked sed_student.decoder. and decoder.; skipping."
        )
        return

    load_state = {}
    loaded_keys = []
    missing_source_keys = []
    mismatched_keys = []
    target_prefixes = (
        "decoder.temporal.",
        "decoder.strong_head.",
        "decoder.attention_head.",
    )
    for target_key, target_tensor in target_state.items():
        if not any(target_key.startswith(prefix) for prefix in target_prefixes):
            continue

        source_key = f"{source_prefix}{target_key[len('decoder.'):]}"
        if source_key not in source_state:
            missing_source_keys.append((target_key, source_key))
            continue

        source_tensor = source_state[source_key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatched_keys.append(
                (
                    target_key,
                    source_key,
                    tuple(target_tensor.shape),
                    tuple(source_tensor.shape),
                )
            )
            continue

        load_state[target_key] = source_tensor
        loaded_keys.append((target_key, source_key))

    skipped_target_keys = [
        key for key in target_state if key.startswith("decoder.input_proj.")
    ]

    missing_keys, unexpected_keys = model.load_state_dict(load_state, strict=False)
    relevant_missing_keys = [
        key
        for key in missing_keys
        if any(key.startswith(prefix) for prefix in target_prefixes)
    ]

    print(
        "[decoder warm-start] loaded decoder/head from "
        f"{ckpt_path} using prefix {source_prefix}"
    )
    print(
        "[decoder warm-start] loaded "
        f"{len(loaded_keys)} keys; missing source={len(missing_source_keys)}; "
        f"shape mismatch={len(mismatched_keys)}"
    )
    for target_key, source_key in loaded_keys:
        print(f"  loaded: {source_key} -> {target_key}")
    for target_key in skipped_target_keys:
        print(
            f"  skipped target-only module: {target_key} "
            "(decoder.input_proj remains on the current experiment defaults)"
        )
    for target_key, source_key in missing_source_keys:
        print(f"  missing source: {source_key} for target {target_key}")
    for target_key, source_key, target_shape, source_shape in mismatched_keys:
        print(
            "  shape mismatch: "
            f"{source_key} {source_shape} -> {target_key} {target_shape}"
        )
    if relevant_missing_keys:
        print(
            "[decoder warm-start] load_state_dict missing keys "
            f"(within decoder/head scope): {relevant_missing_keys}"
        )
    if unexpected_keys:
        print(
            "[decoder warm-start] load_state_dict unexpected keys: "
            f"{unexpected_keys}"
        )


def _load_crnn_encoder_warmstart(model, model_cfg):
    crnn_cfg = model_cfg.get("crnn_warmstart", {})
    if not crnn_cfg.get("enable", False):
        return

    ckpt_path = crnn_cfg.get("checkpoint", "")
    if not ckpt_path:
        print(
            "[crnn warm-start] enabled, but model.crnn_warmstart.checkpoint is empty; "
            "skipping."
        )
        return

    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.is_file():
        print(f"[crnn warm-start] checkpoint not found: {ckpt_path}; skipping.")
        return

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    source_prefix = None
    source_target_prefix = None
    for candidate, target_prefix in (
        ("crnn_encoder.cnn.", "crnn_encoder.cnn."),
        ("sed_student.cnn.", "crnn_encoder.cnn."),
        ("cnn.", "crnn_encoder.cnn."),
    ):
        if any(k.startswith(candidate) for k in source_state):
            source_prefix = candidate
            source_target_prefix = target_prefix
            break

    if source_prefix is None:
        print(
            "[crnn warm-start] no compatible CRNN prefix found in "
            f"{ckpt_path}; checked crnn_encoder.cnn., sed_student.cnn., and cnn.; skipping."
        )
        return

    load_state = {}
    loaded_keys = []
    missing_source_keys = []
    mismatched_keys = []

    for target_key, target_tensor in target_state.items():
        if not target_key.startswith(source_target_prefix):
            continue

        source_key = f"{source_prefix}{target_key[len(source_target_prefix):]}"
        if source_key not in source_state:
            missing_source_keys.append((target_key, source_key))
            continue

        source_tensor = source_state[source_key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatched_keys.append(
                (
                    target_key,
                    source_key,
                    tuple(target_tensor.shape),
                    tuple(source_tensor.shape),
                )
            )
            continue

        load_state[target_key] = source_tensor
        loaded_keys.append((target_key, source_key))

    skipped_target_keys = [
        key for key in target_state if key.startswith("crnn_encoder.") and key not in load_state
    ]

    missing_keys, unexpected_keys = model.load_state_dict(load_state, strict=False)
    relevant_missing_keys = [
        key for key in missing_keys if key.startswith(source_target_prefix)
    ]

    print(
        "[crnn warm-start] loaded CRNN encoder from "
        f"{ckpt_path} using prefix {source_prefix}"
    )
    print(
        "[crnn warm-start] loaded "
        f"{len(loaded_keys)} keys; missing source={len(missing_source_keys)}; "
        f"shape mismatch={len(mismatched_keys)}"
    )
    for target_key, source_key in loaded_keys:
        print(f"  loaded: {source_key} -> {target_key}")
    for target_key in skipped_target_keys:
        print(
            f"  skipped target-only module: {target_key} "
            "(frontend buffers / non-CRNN-CNN modules stay on current model defaults)"
        )
    for target_key, source_key in missing_source_keys:
        print(f"  missing source: {source_key} for target {target_key}")
    for target_key, source_key, target_shape, source_shape in mismatched_keys:
        print(
            "  shape mismatch: "
            f"{source_key} {source_shape} -> {target_key} {target_shape}"
        )
    if relevant_missing_keys:
        print(
            "[crnn warm-start] load_state_dict missing keys "
            f"(within CRNN encoder scope): {relevant_missing_keys}"
        )
    if unexpected_keys:
        print(
            "[crnn warm-start] load_state_dict unexpected keys: "
            f"{unexpected_keys}"
        )


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


def _build_time_aligner(model_cfg):
    return TimeAligner(**model_cfg["align"])


def _build_fusion_aligner(fusion_cfg):
    return FusionTimeAligner(
        method=fusion_cfg.get("align_method", "adaptive_avg"),
        interpolate_mode=fusion_cfg.get("interpolate_mode", "linear"),
    )


def _build_shared_decoder(input_dim, config, model_cfg):
    num_classes = config.get("net", {}).get("nclass")
    if num_classes is None:
        raise ValueError("config['net']['nclass'] must be defined for the shared decoder.")
    return SEDDecoder(
        input_dim=input_dim,
        n_classes=num_classes,
        **model_cfg["decoder"],
    )


def build_sed_model(config):
    model_cfg = resolve_model_config(config)
    model_type = model_cfg.get("model_type", "single_encoder").lower()
    encoder_type = model_cfg["encoder_type"].lower()

    if model_type == "crnn_beats_late_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        beats_encoder = BEATsEncoder(**model_cfg["beats"])
        fusion_cfg = model_cfg["fusion"]
        fusion_aligner = _build_fusion_aligner(fusion_cfg)
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
        decoder = _build_shared_decoder(
            fusion_cfg.get("merge_mlp_dim", 256), config, model_cfg
        )
        model = CRNNBEATsLateFusionModel(
            crnn_encoder=crnn_encoder,
            beats_encoder=beats_encoder,
            fusion_aligner=fusion_aligner,
            merge_mlp=merge_mlp,
            decoder=decoder,
            label_aligner=_build_time_aligner(model_cfg),
            fusion_type=fusion_type,
            use_branch_layernorm=fusion_cfg.get("use_layernorm", False),
            build_config=config,
        )
        _load_crnn_encoder_warmstart(model, model_cfg)
        _load_decoder_head_warmstart(model, model_cfg)
        return model

    if model_type == "crnn_wavlm_late_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        wavlm_encoder = WavLMEncoder(**model_cfg["wavlm"])
        fusion_cfg = model_cfg["fusion"]
        fusion_aligner = _build_fusion_aligner(fusion_cfg)
        fusion_type = fusion_cfg.get("fusion_type", "concat").lower()
        if fusion_type == "concat":
            fusion_input_dim = crnn_encoder.output_dim + wavlm_encoder.output_dim
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
        decoder = _build_shared_decoder(
            fusion_cfg.get("merge_mlp_dim", 256), config, model_cfg
        )
        model = CRNNWavLMLateFusionModel(
            crnn_encoder=crnn_encoder,
            wavlm_encoder=wavlm_encoder,
            fusion_aligner=fusion_aligner,
            merge_mlp=merge_mlp,
            decoder=decoder,
            label_aligner=_build_time_aligner(model_cfg),
            fusion_type=fusion_type,
            use_branch_layernorm=fusion_cfg.get("use_layernorm", False),
            build_config=config,
        )
        _load_crnn_encoder_warmstart(model, model_cfg)
        _load_decoder_head_warmstart(model, model_cfg)
        return model

    if model_type == "crnn_beats_wavlm_late_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        beats_encoder = BEATsEncoder(**model_cfg["beats"])
        wavlm_encoder = WavLMEncoder(**model_cfg["wavlm"])
        fusion_cfg = model_cfg["fusion"]
        fusion_type = fusion_cfg.get("fusion_type", "concat").lower()
        if fusion_type != "concat":
            raise ValueError(
                "CRNN + BEATs + WavLM late fusion currently supports "
                "fusion_type='concat' only."
            )

        merge_mlp = MergeMLP(
            input_dim=(
                crnn_encoder.output_dim
                + beats_encoder.output_dim
                + wavlm_encoder.output_dim
            ),
            output_dim=fusion_cfg.get("merge_mlp_dim", 256),
            activation=fusion_cfg.get("merge_activation", "gelu"),
            dropout=fusion_cfg.get("merge_dropout", 0.5),
            use_layernorm=fusion_cfg.get("use_layernorm", False),
        )
        decoder = _build_shared_decoder(
            fusion_cfg.get("merge_mlp_dim", 256), config, model_cfg
        )
        model = CRNNBEATsWavLMLateFusionModel(
            crnn_encoder=crnn_encoder,
            beats_encoder=beats_encoder,
            wavlm_encoder=wavlm_encoder,
            fusion_aligner=_build_fusion_aligner(fusion_cfg),
            merge_mlp=merge_mlp,
            decoder=decoder,
            label_aligner=_build_time_aligner(model_cfg),
            use_branch_layernorm=fusion_cfg.get("use_layernorm", False),
            build_config=config,
        )
        _load_crnn_encoder_warmstart(model, model_cfg)
        _load_decoder_head_warmstart(model, model_cfg)
        return model

    if model_type == "crnn_beats_residual_gated_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        beats_encoder = BEATsEncoder(**model_cfg["beats"])
        fusion_cfg = model_cfg["fusion"]
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
        model = CRNNBEATsResidualGatedFusionModel(
            crnn_encoder=crnn_encoder,
            beats_encoder=beats_encoder,
            fusion_aligner=_build_fusion_aligner(fusion_cfg),
            fusion_block=fusion_block,
            decoder=_build_shared_decoder(fusion_block.output_dim, config, model_cfg),
            label_aligner=_build_time_aligner(model_cfg),
            build_config=config,
        )
        _load_crnn_encoder_warmstart(model, model_cfg)
        _load_decoder_head_warmstart(model, model_cfg)
        return model

    if model_type == "crnn_wavlm_residual_gated_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        wavlm_encoder = WavLMEncoder(**model_cfg["wavlm"])
        fusion_cfg = model_cfg["fusion"]
        norm_type = fusion_cfg.get("norm_type", "layernorm").lower()
        if norm_type != "layernorm":
            raise ValueError(
                "Residual gated fusion currently supports norm_type='layernorm' only."
            )

        fuse_dim = fusion_cfg.get("fuse_dim", fusion_cfg.get("merge_mlp_dim", 256))
        fusion_block = ResidualGatedFusion(
            cnn_input_dim=crnn_encoder.output_dim,
            beats_input_dim=wavlm_encoder.output_dim,
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
        model = CRNNWavLMResidualGatedFusionModel(
            crnn_encoder=crnn_encoder,
            wavlm_encoder=wavlm_encoder,
            fusion_aligner=_build_fusion_aligner(fusion_cfg),
            fusion_block=fusion_block,
            decoder=_build_shared_decoder(fusion_block.output_dim, config, model_cfg),
            label_aligner=_build_time_aligner(model_cfg),
            build_config=config,
        )
        _load_crnn_encoder_warmstart(model, model_cfg)
        _load_decoder_head_warmstart(model, model_cfg)
        return model

    if model_type == "crnn_beats_wavlm_residual_gated_fusion":
        crnn_encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
        beats_encoder = BEATsEncoder(**model_cfg["beats"])
        wavlm_encoder = WavLMEncoder(**model_cfg["wavlm"])
        fusion_cfg = model_cfg["fusion"]
        fusion_type = fusion_cfg.get("fusion_type", "beats_main_residual_gated").lower()
        if fusion_type != "beats_main_residual_gated":
            raise ValueError(
                "CRNN + BEATs + WavLM residual gated fusion currently supports "
                "fusion_type='beats_main_residual_gated' only."
            )

        norm_type = fusion_cfg.get("norm_type", "layernorm").lower()
        if norm_type != "layernorm":
            raise ValueError(
                "Three-way residual gated fusion currently supports norm_type='layernorm' only."
            )

        fuse_dim = fusion_cfg.get("fuse_dim", beats_encoder.output_dim)
        fusion_block = BEATsMainResidualGatedFusion(
            beats_input_dim=beats_encoder.output_dim,
            crnn_input_dim=crnn_encoder.output_dim,
            wavlm_input_dim=wavlm_encoder.output_dim,
            fuse_dim=fuse_dim,
            gate_mode=fusion_cfg.get("gate_mode", "channel"),
            gate_hidden_dim=fusion_cfg.get("gate_hidden_dim"),
            gate_activation=fusion_cfg.get("gate_activation", "gelu"),
            gate_dropout=fusion_cfg.get("gate_dropout", 0.5),
            use_post_fusion_proj=fusion_cfg.get("use_post_fusion_proj", True),
            post_fusion_dim=fusion_cfg.get(
                "post_fusion_dim",
                fusion_cfg.get("merge_mlp_dim", fuse_dim),
            ),
            post_fusion_dropout=fusion_cfg.get("post_fusion_dropout", 0.5),
            use_alpha_scale=fusion_cfg.get("use_alpha_scale", False),
            alpha_init=fusion_cfg.get("alpha_init", 1.0),
        )
        model = CRNNBEATsWavLMResidualGatedFusionModel(
            crnn_encoder=crnn_encoder,
            beats_encoder=beats_encoder,
            wavlm_encoder=wavlm_encoder,
            fusion_aligner=_build_fusion_aligner(fusion_cfg),
            fusion_block=fusion_block,
            decoder=_build_shared_decoder(fusion_block.output_dim, config, model_cfg),
            label_aligner=_build_time_aligner(model_cfg),
            build_config=config,
        )
        _load_crnn_encoder_warmstart(model, model_cfg)
        _load_decoder_head_warmstart(model, model_cfg)
        return model

    if encoder_type == "crnn":
        encoder = CRNNEncoder(config["feats"], model_cfg["crnn_encoder"])
    elif encoder_type == "beats":
        encoder = BEATsEncoder(**model_cfg["beats"])
    elif encoder_type == "wavlm":
        encoder = WavLMEncoder(**model_cfg["wavlm"])
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    model = SEDModel(
        encoder=encoder,
        decoder=_build_shared_decoder(encoder.output_dim, config, model_cfg),
        aligner=_build_time_aligner(model_cfg),
        encoder_type=encoder_type,
        build_config=config,
    )
    _load_decoder_head_warmstart(model, model_cfg)
    return model
