import torch
import torch.nn as nn
from pipeline.optimizers import SAM, Lookahead, ModelEMA, apply_gradient_centralization

def test_sam_optimizer():
    model = nn.Linear(10, 2)
    optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.1)
    
    # First step
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.first_step(zero_grad=True)
    
    # Second step
    loss = model(x).sum()
    loss.backward()
    optimizer.second_step(zero_grad=True)

def test_lookahead_optimizer():
    model = nn.Linear(10, 2)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    
    x = torch.randn(2, 10)
    for _ in range(6):
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
    
    # Check if update_slow was called implicitly/explicitly
    optimizer.update_slow()

def test_model_ema():
    model = nn.Linear(10, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        
    ema = ModelEMA(model, decay=0.9)
    
    # Update model
    with torch.no_grad():
        model.weight.fill_(1.0)
    
    ema.update(model)
    # Check if ema weight moved: 0.0 * 0.9 + 1.0 * 0.1 = 0.1
    expected = torch.ones_like(model.weight) * 0.1
    assert torch.allclose(ema.ema_model.weight, expected)

def test_gradient_centralization():
    model = nn.Conv2d(3, 16, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    apply_gradient_centralization(optimizer)
    
    # No easy way to verify without checking hooks, but we ensure it doesn't crash
    x = torch.randn(1, 3, 8, 8)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
