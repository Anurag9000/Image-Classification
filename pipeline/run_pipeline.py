# run_pipeline.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from train_arcface import ArcFaceTrainer
from fine_tune_distill import FineTuneDistillTrainer
from evaluate import Evaluator, calibrate_temperature
from utils import setup_logger, apply_swa, save_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_phase():
    train_loader, _ = get_dataloader()
    trainer = ArcFaceTrainer(train_loader, num_classes=100, mix_method='mixup', use_curricularface=True)
    trainer.train(epochs=30)


def fine_tune_phase():
    train_loader, _ = get_dataloader()
    distiller = FineTuneDistillTrainer(train_loader, num_classes=100)
    distiller.train(epochs=10)


def evaluate_phase():
    _, test_loader = get_dataloader()
    evaluator = Evaluator(test_loader, num_classes=100)

    # Load trained model
    model, head = evaluator.load_model("best_model.pth")

    # Evaluate with TTA, OOD detection, Calibration
    y_true, y_pred, y_conf = evaluator.evaluate(model, head, use_tta=True, ood_detection=True)

    # Optional: Temperature calibration on logits
    # (You can capture logits during evaluate if needed)
    # logits = capture_logits_during_evaluation()
    # temperature = calibrate_temperature(logits, torch.tensor(y_true).to(logits.device))


def swa_and_snapshot_phase():
    # Example pseudo-usage (replace with your models)
    dummy_model = torch.nn.Linear(10, 2).cuda()
    swa_model = torch.nn.Linear(10, 2).cuda()
    for step in range(100):
        # training_step(dummy_model)
        apply_swa(dummy_model, swa_model, swa_start=80, step=step)
        if step % 20 == 0:
            save_snapshot(dummy_model, epoch=step)


if __name__ == "__main__":
    setup_logger("./logs/full_pipeline.log")

    logging.info("==== TRAIN PHASE ====")
    train_phase()

    logging.info("==== FINE-TUNE + DISTILL PHASE ====")
    fine_tune_phase()

    logging.info("==== EVALUATE PHASE ====")
    evaluate_phase()

    logging.info("==== SWA + SNAPSHOT PHASE ====")
    swa_and_snapshot_phase()
