# utils.py

import torch
import logging
import os
from pathlib import Path
import torch.nn.functional as F

# ------------------------------
# Logger Setup
# ------------------------------
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# ------------------------------
# Checkpoint Utilities
# ------------------------------
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    logging.info(f"✅ Checkpoint saved at {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"🔄 Checkpoint loaded from {filename}")
    else:
        logging.warning(f"⚠️ No checkpoint found at {filename}")


# ------------------------------
# AMP (Automatic Mixed Precision) Wrapper
# ------------------------------
class AMPWrapper:
    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()

    def backward(self, loss, optimizer):
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()


# ------------------------------
# Stochastic Weight Averaging (SWA) Utility
# ------------------------------
def apply_swa(model, swa_model, swa_start, step):
    if step >= swa_start:
        for swa_param, param in zip(swa_model.parameters(), model.parameters()):
            swa_param.data.mul_(0.99).add_(0.01 * param.data)


# ------------------------------
# Snapshot Ensembling Utility
# ------------------------------
def save_snapshot(model, epoch, folder="./snapshots"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    snapshot_file = os.path.join(folder, f"snapshot_epoch_{epoch}.pth")
    torch.save(model.state_dict(), snapshot_file)
    logging.info(f"📸 Snapshot saved at {snapshot_file}")


# ------------------------------
# Apply Gradient Centralization (GC)
# ------------------------------
def apply_gradient_centralization(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None and len(param.shape) > 1:
                param.grad.data.add_(-param.grad.data.mean(dim=tuple(range(1, len(param.shape))), keepdim=True))
    return optimizer


# ------------------------------
# Apply Token Merging (ToMe) Utility
# ------------------------------
def apply_token_merging(vit_model, ratio=0.5):
    try:
        import tome
    except ImportError:
        logging.error("ToMe library not found. Install: pip install git+https://github.com/GeorgeCazenavette/tome.git")
        return vit_model

    vit_model = tome.patch_vit(vit_model)
    vit_model.r = ratio
    logging.info(f"ToMe activated with ratio {ratio}")
    return vit_model


if __name__ == "__main__":
    Path("./logs").mkdir(parents=True, exist_ok=True)
    setup_logger("./logs/training.log")

    dummy_model = torch.nn.Linear(10, 2).cuda()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())

    save_checkpoint({
        'state_dict': dummy_model.state_dict(),
        'optimizer': dummy_optimizer.state_dict()
    }, filename="test_checkpoint.pth")

    load_checkpoint(dummy_model, dummy_optimizer, filename="test_checkpoint.pth")
