# Installation Guide for NVIDIA RTX 3050

## System Requirements

- **GPU:** NVIDIA RTX 3050 (CUDA Compute Capability 8.6)
- **CUDA:** 11.8 or newer
- **Python:** 3.8 or newer
- **OS:** Ubuntu Linux

## Step-by-Step Installation

### 1. Verify NVIDIA Driver

Check that your NVIDIA driver is installed:

```bash
nvidia-smi
```

You should see your RTX 3050 listed. If not, install NVIDIA drivers first.

### 2. Install PyTorch with CUDA 11.8 Support

**IMPORTANT:** Create a virtual environment to avoid PEP 668 errors:

```bash
# Install venv (if missing)
sudo apt install python3.12-venv

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate
```

Then install PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This installs:
- PyTorch 2.x with CUDA 11.8 support
- TorchVision with CUDA kernels
- TorchAudio

### 3. Verify CUDA Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
PyTorch: 2.x.x+cu118
CUDA Available: True
CUDA Version: 11.8
GPU: NVIDIA GeForce RTX 3050
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

This will install all other required libraries.

### 5. Verify Installation

Run the GPU check script:

```bash
python3 check_gpu.py
# Or use the script: ./scripts/run_tiny_gpu.sh
```

## Alternative: Install Everything at Once

If you prefer to install everything in one command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip3 install -r requirements.txt
```

## Troubleshooting

### Issue: "CUDA out of memory"

Your RTX 3050 has 4GB VRAM. If you encounter OOM errors:

1. Reduce batch size in `configs/config_tiny.yaml`:
   ```yaml
   dataset:
     batch_size: 32  # or even 16
   
   supcon:
     batch_size: 32  # or even 16
   ```

2. Enable gradient checkpointing (if needed)

### Issue: "No CUDA-capable device detected"

1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA: 
   ```bash
   pip3 uninstall torch torchvision torchaudio
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Slow training

Verify GPU is being used:
```bash
watch -n 1 nvidia-smi
```

Run training in another terminal and you should see GPU utilization.

## Memory Optimization for RTX 3050 (4GB VRAM)

Your RTX 3050 has limited VRAM. Here are recommended settings:

**In `configs/config_tiny.yaml`:**
```yaml
dataset:
  batch_size: 32        # Reduced from 64
  
supcon:
  batch_size: 32        # Reduced from 64
  use_amp: true         # Keep enabled for memory efficiency
  
arcface:
  use_amp: true         # Keep enabled
```

## Performance Tips

1. **Use AMP (Automatic Mixed Precision):** Already enabled in config
2. **Monitor GPU usage:** `watch -n 1 nvidia-smi`
3. **Close other GPU applications** while training
4. **Use 4 workers** for data loading (already configured)

## Verification Checklist

- [ ] NVIDIA driver installed (`nvidia-smi` works)
- [ ] PyTorch with CUDA 11.8 installed
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU name shows "RTX 3050"
- [ ] All dependencies installed from requirements.txt
- [ ] `python3 check_gpu.py` runs successfully

Once all checks pass, you're ready to train!
