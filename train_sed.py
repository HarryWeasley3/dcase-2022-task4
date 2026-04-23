import argparse
import inspect
import os
import random
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import (StronglyAnnotatedSet, UnlabeledSet,
                                        WeakSet)
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup, WarmupCosineScheduler
from local.classes_dict import classes_labels
from local.resample_folder import resample_folder
from local.sed_trainer import SEDTask4
from local.utils import generate_tsv_wav_durations
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sed_modeling import build_sed_model


def _torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_profiles(config):
    resolved = deepcopy(config)
    top_level_profiles = resolved.pop("profiles", None)

    experiment_cfg = resolved.get("experiment", {})
    experiment_profiles = None
    selected_profile = None
    if isinstance(experiment_cfg, dict):
        experiment_profiles = experiment_cfg.pop("profiles", None)
        selected_profile = experiment_cfg.get("profile")

    if selected_profile is None:
        selected_profile = resolved.pop("profile", None)

    profiles = experiment_profiles or top_level_profiles or {}
    if selected_profile is None:
        return resolved

    if selected_profile not in profiles:
        raise KeyError(
            f"Unknown config profile '{selected_profile}'. "
            f"Available profiles: {sorted(profiles.keys())}"
        )

    _deep_update(resolved, deepcopy(profiles[selected_profile]))
    resolved.setdefault("experiment", {})
    resolved["experiment"]["profile"] = selected_profile
    return resolved


def load_experiment_config(conf_file):
    with open(conf_file, "r") as f:
        configs = yaml.safe_load(f) or {}
    return _resolve_profiles(configs)


class _NoOpScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0

    def _get_scaling_factor(self):
        return 1.0

    def step(self):
        self.step_num += 1


def _build_trainer_kwargs(
    config, gpus, n_epochs, callbacks, logger, checkpoint_resume, flush_logs_every_n_steps
):
    trainer_signature = inspect.signature(pl.Trainer)
    trainer_kwargs = {
        "precision": config["training"]["precision"],
        "max_epochs": n_epochs,
        "callbacks": callbacks,
        "accumulate_grad_batches": config["training"]["accumulate_batches"],
        "logger": logger,
        "gradient_clip_val": config["training"]["gradient_clip"],
        "check_val_every_n_epoch": config["training"]["validation_interval"],
        "num_sanity_val_steps": 0,
    }

    backend = config["training"].get("backend")
    if "flush_logs_every_n_steps" in trainer_signature.parameters:
        trainer_kwargs["flush_logs_every_n_steps"] = flush_logs_every_n_steps

    if "gpus" in trainer_signature.parameters:
        trainer_kwargs["gpus"] = gpus
        trainer_kwargs["strategy"] = backend
        trainer_kwargs["resume_from_checkpoint"] = checkpoint_resume
    else:
        if gpus == "0":
            trainer_kwargs["accelerator"] = "cpu"
            trainer_kwargs["devices"] = 1
        else:
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = 1

        # `dp` is not supported in Lightning 2.x and is unnecessary for single-device runs.
        if backend not in (None, "dp"):
            trainer_kwargs["strategy"] = backend

    return trainer_kwargs


def _load_init_checkpoint_weights(model, checkpoint_path):
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = _torch_load_compat(checkpoint_path, map_location=map_location)
    source_state_dict = checkpoint["state_dict"]
    target_state_dict = model.state_dict()

    candidate_states = []
    if any(key.startswith("sed_student.") for key in source_state_dict):
        candidate_states.append(
            (
                "sed_student.",
                {
                    key[len("sed_student.") :]: value
                    for key, value in source_state_dict.items()
                    if key.startswith("sed_student.")
                },
            )
        )
    candidate_states.append(("", source_state_dict))

    selected_prefix = None
    selected_state_dict = None
    load_state = {}
    unexpected_keys = []
    shape_mismatch = []

    for prefix, candidate_state in candidate_states:
        current_load_state = {}
        current_unexpected = []
        current_shape_mismatch = []
        for key, value in candidate_state.items():
            target_value = target_state_dict.get(key)
            if target_value is None:
                current_unexpected.append(key)
            elif target_value.shape != value.shape:
                current_shape_mismatch.append(
                    (key, tuple(value.shape), tuple(target_value.shape))
                )
            else:
                current_load_state[key] = value
        if len(current_load_state) > len(load_state):
            selected_prefix = prefix
            selected_state_dict = candidate_state
            load_state = current_load_state
            unexpected_keys = current_unexpected
            shape_mismatch = current_shape_mismatch

    incompatible = model.load_state_dict(load_state, strict=False)
    print(
        f"[init_from_checkpoint] loaded weights from {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch')}, source_prefix={selected_prefix!r}, "
        f"loaded_keys={len(load_state)}, missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(unexpected_keys)}, shape_mismatch={len(shape_mismatch)})"
    )
    if incompatible.missing_keys:
        print(
            "[init_from_checkpoint] missing keys:",
            incompatible.missing_keys,
        )
    if unexpected_keys:
        print(
            "[init_from_checkpoint] unexpected keys:",
            unexpected_keys,
        )
    if shape_mismatch:
        print(
            "[init_from_checkpoint] shape mismatch:",
            shape_mismatch,
        )
    if len(load_state) == 0:
        raise RuntimeError(
            f"[init_from_checkpoint] no compatible model weights were loaded from {checkpoint_path}."
        )


def resample_data_generate_durations(
    config_data, test_only=False, evaluation=False, synth_only=False
):
    computed = False
    if evaluation:
        dsets = ["eval_folder"]
    elif synth_only:
        dsets = ["synth_folder", "synth_val_folder"]
        if test_only or config_data["test_folder"] != config_data["synth_val_folder"]:
            dsets.append("test_folder")
    elif not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    else:
        dsets = ["test_folder"]

    for dset in dsets:
        if not config_data.get(dset + "_44k") or not config_data.get(dset):
            continue
        computed = (
            resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
            )
            or computed
        )

    if not evaluation:
        duration_sets = ["synth_val", "test"]
        if synth_only and config_data["test_dur"] == config_data["synth_val_dur"]:
            duration_sets = ["synth_val"]
        for base_set in duration_sets:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )


def single_run(
    config,
    log_dir,
    gpus,
    strong_real=False,
    checkpoint_resume=None,
    checkpoint_init=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    synth_only=False,
):
    """
    Running sound event detection baselin

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        checkpoint_init (str, optional): path to checkpoint whose model weights should be used
            to initialize a new training run without restoring optimizer/scheduler state.
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )
    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"], encoder, pad_to=None, return_filename=True
        )

    test_dataset = devtest_dataset

    ##### model definition  ############
    sed_student = build_sed_model(config)
    if checkpoint_init is not None:
        _load_init_checkpoint_weights(sed_student, checkpoint_init)

    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )

        if strong_real and not synth_only:
            strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
            strong_set = StronglyAnnotatedSet(
                config["data"]["strong_folder"],
                strong_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )

        synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        synth_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            synth_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )

        if synth_only:
            train_dataset = synth_set
            valid_dataset = synth_val
            batch_sampler = None
            batch_size = config["training"]["batch_size"]
            if isinstance(batch_size, (list, tuple)):
                batch_size = batch_size[0]
            epoch_len = max(
                1,
                len(train_dataset)
                // (batch_size * config["training"]["accumulate_batches"]),
            )
        else:
            weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
            train_weak_df = weak_df.sample(
                frac=config["training"]["weak_split"],
                random_state=config["training"]["seed"],
            )
            valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
            train_weak_df = train_weak_df.reset_index(drop=True)
            weak_set = WeakSet(
                config["data"]["weak_folder"],
                train_weak_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )

            unlabeled_set = UnlabeledSet(
                config["data"]["unlabeled_folder"],
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )

            weak_val = WeakSet(
                config["data"]["weak_folder"],
                valid_weak_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
                return_filename=True,
            )

            if strong_real:
                strong_full_set = torch.utils.data.ConcatDataset([strong_set, synth_set])
                tot_train_data = [strong_full_set, weak_set, unlabeled_set]
            else:
                tot_train_data = [synth_set, weak_set, unlabeled_set]
            train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

            batch_sizes = config["training"]["batch_size"]
            samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
            batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

            valid_dataset = torch.utils.data.ConcatDataset([synth_val, weak_val])

            ##### training params and optimizers ############
            epoch_len = max(
                1,
                min(
                    [
                        len(tot_train_data[indx])
                        // (
                            config["training"]["batch_size"][indx]
                            * config["training"]["accumulate_batches"]
                        )
                        for indx in range(len(tot_train_data))
                    ]
                ),
            )

        trainable_parameters = [
            parameter
            for parameter in sed_student.parameters()
            if parameter.requires_grad
        ]
        if len(trainable_parameters) == 0:
            raise RuntimeError("No trainable parameters were found for the selected encoder/decoder setup.")
        opt = torch.optim.Adam(
            trainable_parameters, config["opt"]["lr"], betas=(0.9, 0.999)
        )
        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        scheduler_name = config["opt"].get("scheduler", "exponential_warmup")
        if scheduler_name in (None, "exponential_warmup", "exp_warmup"):
            scheduler_impl = ExponentialWarmup(opt, config["opt"]["lr"], exp_steps)
        elif scheduler_name in ("warmup_cosine", "cosine"):
            total_steps = config["training"]["n_epochs"] * epoch_len
            scheduler_impl = WarmupCosineScheduler(
                opt,
                config["opt"]["lr"],
                exp_steps,
                total_steps,
                min_lr=config["opt"].get("min_lr", 1e-6),
                decay_power=config["opt"].get("decay_power", 1.0),
            )
        elif scheduler_name in ("none", "disabled", False):
            scheduler_impl = _NoOpScheduler(opt)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        exp_scheduler = {
            "scheduler": scheduler_impl,
            "interval": "step",
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]),
            config["log_dir"].split("/")[-1],
        )
        print(f"experiment dir: {logger.log_dir}")

        callbacks = [
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max",
            ),
            ModelCheckpoint(
                logger.log_dir,
                monitor="val/obj_metric",
                save_top_k=1,
                mode="max",
                save_last=True,
            ),
        ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    desed_training = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    if len(gpus.split(",")) > 1:
        raise NotImplementedError("Multiple GPUs are currently not supported")

    trainer_kwargs = _build_trainer_kwargs(
        config,
        gpus,
        n_epochs,
        callbacks,
        logger,
        checkpoint_resume,
        flush_logs_every_n_steps,
    )
    trainer_kwargs.update(
        {
            "log_every_n_steps": log_every_n_steps,
            "limit_train_batches": limit_train_batches,
            "limit_val_batches": limit_val_batches,
            "limit_test_batches": limit_test_batches,
        }
    )
    trainer = pl.Trainer(**trainer_kwargs)

    if test_state_dict is None:
        # start tracking energy consumption
        fit_kwargs = {}
        if checkpoint_resume is not None and "gpus" not in inspect.signature(pl.Trainer).parameters:
            fit_kwargs["ckpt_path"] = checkpoint_resume
        trainer.fit(desed_training, **fit_kwargs)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = _torch_load_compat(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)
    trainer.test(desed_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/2022_baseline",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )

    parser.add_argument(
        "--strong_real",
        action="store_true",
        default=False,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        default=None,
        help="Initialize model weights from a checkpoint without restoring optimizer/scheduler state.",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='0', "
        "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )

    parser.add_argument(
        "--eval_from_checkpoint", default=None, help="Evaluate the model specified"
    )
    parser.add_argument(
        "--synth_only",
        action="store_true",
        default=False,
        help="Train with synthetic train/validation data only, without weak, unlabeled or strong real data.",
    )

    args = parser.parse_args()

    configs = load_experiment_config(args.conf_file)

    configs.setdefault("training", {})
    configs["training"]["synth_only"] = args.synth_only or configs["training"].get(
        "synth_only", False
    )

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    if args.resume_from_checkpoint is not None and args.init_from_checkpoint is not None:
        raise ValueError(
            "--resume_from_checkpoint and --init_from_checkpoint are mutually exclusive."
        )

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        checkpoint = _torch_load_compat(test_from_checkpoint, map_location=map_location)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        configs = configs_ckpt
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)

    test_only = test_from_checkpoint is not None
    synth_only = configs["training"]["synth_only"]
    if synth_only and args.strong_real:
        warnings.warn("--strong_real is ignored when --synth_only is enabled.")
        args.strong_real = False
    resample_data_generate_durations(
        configs["data"], test_only, evaluation, synth_only=synth_only
    )
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.strong_real,
        args.resume_from_checkpoint,
        args.init_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
        synth_only,
    )
