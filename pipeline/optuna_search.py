import optuna
import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train_arcface import ArcFaceTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)


def objective(trial):
    mix_method = trial.suggest_categorical("mix_method", ["mixup", "cutmix", "tokenmix", "mixtoken"])
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    smoothing = trial.suggest_float("smoothing", 0.05, 0.2)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    use_curricularface = trial.suggest_categorical("use_curricularface", [True, False])

    dataloader = get_dataloader()
    trainer = ArcFaceTrainer(
        dataloader=dataloader,
        num_classes=100,
        mix_method=mix_method,
        use_curricularface=use_curricularface,
        use_evidential=False,
        lr=lr,
        gamma=gamma,
        smoothing=smoothing,
        snapshot_dir="./optuna_snapshots",
        log_csv=f"./optuna_logs/arcface_trial_{trial.number}.csv"
    )
    trainer.train(epochs=10)

    # Load final epoch metrics from CSV
    try:
        with open(f"./optuna_logs/arcface_trial_{trial.number}.csv") as f:
            last_line = f.readlines()[-1]
        _, _, acc, f1 = last_line.strip().split(",")
        return float(f1)  # Use F1 as optimization target
    except:
        return 0.0


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    logging.info(f"Best Trial:\n{study.best_trial.params}")
    with open("best_arcface_params.txt", "w") as f:
        for k, v in study.best_trial.params.items():
            f.write(f"{k}: {v}\n")
