import torch
import torch.nn as nn
from pipeline.adapters import LoRAConfig, IA3Config, inject_lora, inject_ia3, LoRAInjectedLinear, IA3Linear

def test_lora_injection():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.others = nn.LayerNorm(10)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x
            
    model = SimpleModel()
    config = LoRAConfig(rank=4, target_modules=["fc1"])
    
    inject_lora(model, config)
    
    assert isinstance(model.fc1, LoRAInjectedLinear)
    assert not isinstance(model.fc2, LoRAInjectedLinear)
    assert model.fc1.rank == 4
    
    # Test forward
    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2, 10)

def test_ia3_injection():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
        
        def forward(self, x):
            return self.fc1(x)
            
    model = SimpleModel()
    config = IA3Config(target_modules=["fc1"])
    
    inject_ia3(model, config)
    
    assert isinstance(model.fc1, IA3Linear)
    
    # Test forward
    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2, 20)
