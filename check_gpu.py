import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA Available: YES")
    print(f"CUDA Version (Torch build): {torch.version.cuda}")
    if torch.backends.cudnn.is_available():
        print(f"CuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("CuDNN Available: NO")
    
    device_count = torch.cuda.device_count()
    print(f"GPU Device Count: {device_count}")
    
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        # Check basic tensor op
        try:
            x = torch.tensor([1.0]).cuda(i)
            print(f"  Test Tensor: Allocated successfully on Device {i}")
        except Exception as e:
            print(f"  Test Tensor: FAILED on Device {i} ({e})")

else:
    print("CUDA Available: NO (Running on CPU)")
