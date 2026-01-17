import os
import torch
import torch.nn as nn
from utils.utils import save_checkpoint, load_checkpoint, apply_swa, save_snapshot

def test_checkpoint_save_load():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Save
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filename = "test_utils_checkpoint.pth"
    save_checkpoint(state, filename=filename)
    assert os.path.exists(filename)
    
    initial_weight = model.weight.clone()
    
    # Modify model
    with torch.no_grad():
        model.weight.fill_(5.0)
    
    # Load
    load_checkpoint(model, optimizer, filename=filename)
    assert torch.allclose(model.weight, initial_weight)
    
    if os.path.exists(filename):
        os.remove(filename)

def test_swa():
    model = nn.Linear(2, 2)
    swa_model = nn.Linear(2, 2)
    
    with torch.no_grad():
        model.weight.fill_(1.0)
        swa_model.weight.fill_(0.0)
    
    apply_swa(model, swa_model, swa_start=0, step=1, alpha=0.5)
    assert torch.allclose(swa_model.weight, torch.ones_like(swa_model.weight) * 0.5)

def test_save_snapshot():
    model = nn.Linear(2, 2)
    folder = "./test_snapshots"
    save_snapshot(model, epoch=1, folder=folder)
    assert os.path.exists(os.path.join(folder, "snapshot_epoch_1.pth"))
    
    import shutil
    if os.path.exists(folder):
        shutil.rmtree(folder)
