import torch
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_arcface import ArcFaceTrainer
from train_supcon import SupConPretrainer
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


def train_supcon_phase():
    train_loader, _ = get_dataloader(batch_size=16)
    logging.info("===> Starting SupCon Pretraining Phase")
    trainer = SupConPretrainer(train_loader, num_classes=100, num_views=4, temperature=0.07)
    trainer.train(steps=100)


def train_arcface_phase():
    train_loader, _ = get_dataloader(batch_size=32)
    logging.info("===> Starting ArcFace Training Phase")
    trainer = ArcFaceTrainer(
        train_loader,
        num_classes=100,
        mix_method='mixup',
        use_curricularface=True,
        use_evidential=False
    )
    trainer.train(epochs=30)


def fine_tune_distill_phase():
    train_loader, _ = get_dataloader(batch_size=32)
    logging.info("===> Starting Fine-Tune + Distillation Phase")
    distiller = FineTuneDistillTrainer(
        train_loader,
        num_classes=100,
        distill_weight=0.3,
        mix_method='mixup',
        use_evidential=False
    )
    distiller.train(epochs=10)


def evaluate_phase():
    _, test_loader = get_dataloader()
    logging.info("===> Starting Evaluation Phase")

    evaluator = Evaluator(test_loader, num_classes=100)
    model, head = evaluator.load_model("best_model.pth")
    y_true, y_pred, y_conf = evaluator.evaluate(model, head, use_tta=True, ood_detection=True)

    # Optional: temperature scaling if logits are saved
    # temperature = calibrate_temperature(logits, torch.tensor(y_true).to(logits.device))


def swa_and_snapshot_phase():
    dummy_model = torch.nn.Linear(10, 2).cuda()
    swa_model = torch.nn.Linear(10, 2).cuda()
    for step in range(100):
        apply_swa(dummy_model, swa_model, swa_start=80, step=step)
        if step % 20 == 0:
            save_snapshot(dummy_model, epoch=step, folder="./snapshots_dummy")


if __name__ == "__main__":
    setup_logger("./logs/full_pipeline.log")

    logging.info("==== SUPCON PHASE ====")
    train_supcon_phase()

    logging.info("==== ARC FACE PHASE ====")
    train_arcface_phase()

    logging.info("==== FINE-TUNE + DISTILLATION PHASE ====")
    fine_tune_distill_phase()

    logging.info("==== EVALUATION PHASE ====")
    evaluate_phase()

    logging.info("==== SWA / SNAPSHOT PHASE ====")
    swa_and_snapshot_phase()
