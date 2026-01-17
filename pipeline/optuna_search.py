from __future__ import annotations

import logging
import os
from typing import Dict

import optuna

from .train_arcface import ArcFaceConfig, ArcFaceTrainer, create_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_trial_configs(trial: optuna.Trial) -> Dict:
    mix_method = trial.suggest_categorical("mix_method", ["mixup", "cutmix", "tokenmix", "mixtoken"])
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    smoothing = trial.suggest_float("smoothing", 0.05, 0.2)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    ema_decay = trial.suggest_float("ema_decay", 0.995, 0.9999)
    use_curricularface = trial.suggest_categorical("use_curricularface", [True, False])
    use_manifold_mixup = trial.suggest_categorical("use_manifold_mixup", [True, False])

    lora_rank = trial.suggest_categorical("lora_rank", [0, 4, 8])
    token_merging_ratio = trial.suggest_categorical("token_merging_ratio", [0.0, 0.3, 0.5])
    token_learner_tokens = trial.suggest_categorical("token_learner_tokens", [0, 6, 8])
    mixstyle = trial.suggest_categorical("mixstyle", [True, False])

    backbone_cfg = {
        "cnn_model": trial.suggest_categorical(
            "cnn_model", ["convnextv2_base", "convnextv2_base.fcmae_ft_in22k_in1k"]
        ),
        "vit_model": trial.suggest_categorical(
            "vit_model", ["swinv2_base_window12_192_22k", "dinov2_base14", "eva02_base_patch14_224"]
        ),
        "token_merging_ratio": None if token_merging_ratio == 0.0 else token_merging_ratio,
        "token_learner_tokens": None if token_learner_tokens == 0 else token_learner_tokens,
        "lora_rank": None if lora_rank == 0 else lora_rank,
        "lora_alpha": 16,
        "mixstyle": mixstyle,
        "mixstyle_p": 0.4,
        "mixstyle_alpha": 0.1,
        "use_ia3": trial.suggest_categorical("use_ia3", [True, False]),
        "cnn_drop_path_rate": trial.suggest_float("cnn_drop_path_rate", 0.0, 0.15),
        "vit_drop_path_rate": trial.suggest_float("vit_drop_path_rate", 0.1, 0.4),
    }

    augmentations = {
        "random_resized_crop": True,
        "rrc_scale": [0.5, 1.0],
        "randaugment": {
            "num_ops": trial.suggest_int("randaugment_num_ops", 1, 3),
            "magnitude": trial.suggest_int("randaugment_magnitude", 5, 12),
        },
        "trivialaugment": trial.suggest_categorical("trivialaugment", [True, False]),
        "color_jitter": {
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.2,
            "hue": 0.1,
        },
        "random_erasing": {"p": 0.2},
        "normalize": True,
    }

    return {
        "mix_method": mix_method,
        "gamma": gamma,
        "smoothing": smoothing,
        "lr": lr,
        "ema_decay": ema_decay,
        "use_curricularface": use_curricularface,
        "use_manifold_mixup": use_manifold_mixup,
        "backbone": backbone_cfg,
        "augmentations": augmentations,
    }


def objective(trial: optuna.Trial) -> float:
    trial_cfg = build_trial_configs(trial)

    dataloader = create_dataloader(
        batch_size=32,
        augment=True,
        image_size=224,
        augmentations=trial_cfg["augmentations"],
        root="./data",
    )

    config = ArcFaceConfig(
        num_classes=100,
        lr=trial_cfg["lr"],
        gamma=trial_cfg["gamma"],
        smoothing=trial_cfg["smoothing"],
        epochs=8,
        mix_method=trial_cfg["mix_method"],
        use_curricularface=trial_cfg["use_curricularface"],
        use_evidential=False,
        ema_decay=trial_cfg["ema_decay"],
        use_manifold_mixup=trial_cfg["use_manifold_mixup"],
        backbone=trial_cfg["backbone"],
        augmentations=trial_cfg["augmentations"],
    )

    snapshot_dir = "./optuna_snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    log_csv = f"./optuna_logs/arcface_trial_{trial.number}.csv"
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)

    trainer = ArcFaceTrainer(
        dataloader=dataloader,
        config=config,
    )
    trainer.train()

    try:
        with open(log_csv, "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]
        _, _, acc, f1 = last_line.strip().split(",")
        return float(f1)
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to parse metrics for trial %s: %s", trial.number, exc)
        return 0.0


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    logging.info("Best Trial: %s", study.best_trial.params)
    with open("best_arcface_params.txt", "w", encoding="utf-8") as f:
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")

