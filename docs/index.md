# Pipeline Documentation

This document offers a deeper dive into the core modules and how to extend them.

## Training Phases

### 1. `pipeline/train_supcon.py`
- **Goal**: Contrastive warm start on the target dataset.
- **Model**: `HybridBackbone` (ConvNeXt-V2 + SwinV2) with optional EMA.
- **Loss**: `SupConLoss` with temperature `τ`.
- **Optimiser**: SAM(AdamW) + Lookahead + gradient centralisation, optional AMP.
- **Key hooks**:
  - `SupConConfig` dataclass controls `num_views`, `lr`, `steps`, etc.
  - EMA snapshots saved alongside standard checkpoints.

### 2. `pipeline/train_arcface.py`
- **Goal**: Metric-learning fine-tune with ArcFace/AdaFace style heads.
- **Config**: `ArcFaceConfig` toggles CurricularFace, Evidential loss, mix strategy, EMA, `torch.compile`.
- **Mixing**: `mixup_cutmix_tokenmix` supports `mixup`, `cutmix`, `tokenmix`, `mixtoken`, or `none`.
- **Optimisation**: Same SAM + Lookahead stack; `OneCycleLR` schedules LR across all epochs.

### 3. `pipeline/fine_tune_distill.py`
- **Goal**: Knowledge distillation from (optionally) pretrained teacher weights.
- **Loss components**:
  - Hard focal/evidential loss.
  - KL divergence on softened logits (temperature-scaled).
  - Feature-level `MSE` for representation alignment.
- **Teacher loading**: Provide `teacher_backbone_path` and `teacher_head_path` in config.

## Evaluation

`pipeline/evaluate.py` produces a comprehensive summary:
- **Metrics**: accuracy, configurable top-k, expected calibration error (ECE).
- **Artefacts**: JSON metrics, CSV reports, confusion matrix, optional energy histogram.
- **TTA**: Set `tta_runs` in `evaluation` config to apply flip/rotation test-time augmentation.
- **Ensembles**: Pass multiple `(model, head)` tuples to `Evaluator.evaluate` to average predictions.

## Inference

`pipeline/inference.py` exposes a CLI:
```bash
python -m pipeline.inference \
  --image_dir ./samples \
  --backbone_path snapshots/snapshot_epoch_30.pth \
  --head_path snapshots/head/snapshot_epoch_30.pth \
  --gradcam_dir ./gradcam_outputs
```

GradCAM++ overlays (if requested) are saved per image for qualitative inspection.

## Configuration

All knobs are stored in `configs/config.yaml`. Each section maps to dataclass constructors in the respective training modules, so adding new parameters is as simple as extending the dataclass and referencing the key in the YAML.

## Extending the Pipeline

- **Backbones**: Modify `pipeline/backbone.py` to add new timm models, ToMe token merging, or adapters (e.g., LoRA).
- **Losses**: Extend `pipeline/losses.py` with new metric-learning objectives or distillation strategies.
- **Schedulers/Loggers**: Hook your favourite scheduler or logging backends inside the trainer classes—AMP, EMA, and SAM calls are already centralised for convenience.
- **Datasets**: Replace the CIFAR-100 helpers (`create_*_loader`) with your dataset factory. Ensure transforms return tensors in `[0,1]` and update configs accordingly.

## Testing

Add regression/unit tests under `tests/`. Suggested coverage:
- Dataloader sanity (shapes, augmentation pipelines).
- Forward pass smoke tests for `HybridBackbone` on CPU.
- Metric computations (top-k, ECE) using synthetic logits.

## Support

For enhancements or bug reports, open an issue or submit a pull request. Contributions that add new SOTA techniques, datasets, or evaluation protocols are encouraged.
