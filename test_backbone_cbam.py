
import torch
import logging
from pipeline.backbone import HybridBackbone, BackboneConfig

logging.basicConfig(level=logging.INFO)

def test_cbam_injection():
    print("Testing Interleaved CBAM Injection on MobileNetV3...")
    cfg = BackboneConfig(
        cnn_model="mobilenetv3_large_100", 
        vit_model=None, # Pure CNN for clarity
        use_cbam=True,
        pretrained=False # Speed up
    )
    model = HybridBackbone(cfg)
    
    print("\n--- Inspecting MobileNetV3 Attributes ---")
    print(dir(model.cnn_backbone))
    
    # Check structure
    print("\n--- Checking Model Structure ---")
    blocks = model.cnn_backbone.blocks
    print(f"Total entries in blocks: {len(blocks)}")
    
    cbam_count = 0
    for name, module in blocks.named_children():
        print(f"Module: {name} -> {type(module).__name__}")
        if "cbam" in name or "CBAM" in type(module).__name__:
            cbam_count += 1
            
    print(f"\nTotal CBAM modules found: {cbam_count}")
    if cbam_count > 1:
        print("SUCCESS: Multiple CBAM modules injected.")
    else:
        print("FAILURE: CBAM modules not found or only one found.")

    # Check forward pass
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"\nForward Pass Output Shape: {out.shape}")
    print("Forward Pass Successful.")

if __name__ == "__main__":
    test_cbam_injection()
