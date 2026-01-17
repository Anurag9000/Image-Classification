import torch
from pipeline.augmentations import mixup, cutmix, tokenmix, mixtoken, build_train_transform, build_eval_transform

def test_mixup():
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    mixed_x, y_a, y_b, lam = mixup(x, y, alpha=1.0)
    
    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0 <= lam <= 1

def test_cutmix():
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    mixed_x, y_a, y_b, lam = cutmix(x, y, alpha=1.0)
    
    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0 <= lam <= 1

def test_tokenmix():
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    mixed_x, y_a, y_b, lam = tokenmix(x, y, token_size=8)
    
    assert mixed_x.shape == x.shape
    assert 0 <= lam <= 1

def test_builders():
    train_t = build_train_transform(image_size=32)
    eval_t = build_eval_transform(image_size=32)
    
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
    
    out_train = train_t(dummy_img)
    out_eval = eval_t(dummy_img)
    
    assert out_train.shape == (3, 32, 32)
    assert out_eval.shape == (3, 32, 32)
