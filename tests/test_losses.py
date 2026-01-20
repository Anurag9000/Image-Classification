import torch
from pipeline.losses import AdaFace, CurricularFace, FocalLoss, SupConLoss, EvidentialLoss

def test_adaface():
    loss_fn = AdaFace(embedding_size=128, num_classes=10)
    feats = torch.randn(4, 128)
    labels = torch.randint(0, 10, (4,))
    output = loss_fn(feats, labels)
    assert output.shape == (4, 10)

def test_curricularface():
    loss_fn = CurricularFace(embedding_size=128, num_classes=10)
    feats = torch.randn(4, 128)
    labels = torch.randint(0, 10, (4,))
    output = loss_fn(feats, labels)
    assert output.shape == (4, 10)

def test_focalloss():
    loss_fn = FocalLoss()
    logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, labels)
    assert loss.dim() == 0

def test_supconloss():
    loss_fn = SupConLoss()
    feats = torch.randn(8, 128)
    labels = torch.randint(0, 10, (8,))
    loss = loss_fn(feats, labels)
    assert loss.dim() == 0

def test_evidentialloss():
    loss_fn = EvidentialLoss(num_classes=10)
    logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, labels)
    assert loss.dim() == 0
