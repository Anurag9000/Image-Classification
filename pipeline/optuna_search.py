from __future__ import annotations

import logging
import os
from typing import Dict

import optuna

from .train_arcface import ArcFaceConfig, ArcFaceTrainer, create_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


import argparse

def build_trial_configs(trial: optuna.Trial) -> Dict:
    # (Same as before but keeping it clean)
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
        "cnn_model": "convnextv2_base", # trial.suggest_categorical("cnn_model", ["convnextv2_base", ...])
        "vit_model": "swinv2_base_window12_192_22k", # trial.suggest_categorical("vit_model", ["swinv2_base_window12_192_22k", ...])
        "token_merging_ratio": None if token_merging_ratio == 0.0 else token_merging_ratio,
        "token_learner_tokens": None if token_learner_tokens == 0 else token_learner_tokens,
        "cnn_lora_rank": None if lora_rank == 0 else lora_rank,
        "vit_lora_rank": None if lora_rank == 0 else lora_rank,
        "lora_alpha": 16,
        "mixstyle": mixstyle,
        "mixstyle_p": 0.4,
        "mixstyle_alpha": 0.1,
        "cnn_ia3": trial.suggest_categorical("use_ia3_cnn", [True, False]),
        "vit_ia3": trial.suggest_categorical("use_ia3_vit", [True, False]),
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


def objective(trial: optuna.Trial, root_dirs: list[str], num_classes: int) -> float:
    trial_cfg = build_trial_configs(trial)

    from .files_dataset import create_garbage_loader
    
    dataloader, val_loader, _ = create_garbage_loader(
        root_dirs=root_dirs,
        batch_size=32,
        num_workers=4,
        val_split=0.2,
    )

    config = ArcFaceConfig(
        num_classes=num_classes,
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
    
    config.snapshot_dir = snapshot_dir
    config.log_csv = log_csv

    trainer = ArcFaceTrainer(
        train_loader=dataloader,
        val_loader=val_loader,
        config=config,
    )
    trainer.train()

    try:
        with open(log_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return 0.0
            last_line = lines[-1]
            
        parts = last_line.strip().split(",")
        # New format: epoch, train_loss, val_loss, val_acc, val_f1
        if len(parts) >= 2:
             # Find f1 column by header
             header = lines[0].strip().split(",")
             if "val_f1" in header:
                 idx = header.index("val_f1")
                 return float(parts[idx])
             else:
                 # Fallback to last column if val_f1 not found
                 return float(parts[-1])
        else:
             logging.warning("CSV format mismatch or no data: %s", last_line)
             return 0.0
             
    except Exception as exc:
        logging.error("Failed to parse metrics for trial %s: %s", trial.number, exc)
        return 0.0
    except FileNotFoundError:
        logging.warning("Log CSV not found for trial %s (Training likely failed early).", trial.number)
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter sweep.")
    parser.add_argument("--data", nargs="+", default=["./data/Dataset_Final"], help="Path(s) to dataset root(s).")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes in dataset.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, args.data, args.num_classes), n_trials=args.trials)

    logging.info("Best Trial: %s", study.best_trial.params)
    with open("best_arcface_params.txt", "w", encoding="utf-8") as f:
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()

