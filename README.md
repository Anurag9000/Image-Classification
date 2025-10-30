# Image Classification Pipeline

End-to-end, research-grade image classification pipeline combining contrastive pretraining, metric-learning fine-tuning, and distillation. The project is built to be modular, reproducible, and extensible so that new state-of-the-art ideas can be slotted in with minimal friction.

## Highlights
- **Hybrid backbone factory** mixing ConvNeXt-V2, SwinV2, DINOv2, and EVA models with CBAM, token merging (ToMe), token learners, MixStyle, and optional LoRA / IA3 adapters plus drop-path controls.
- **Three-stage curriculum**: SupCon pretraining -> ArcFace / AdaFace metric tuning -> EMA-assisted distillation (teacher checkpoints or EMA teachers).
- **Optimisation toolkit**: SAM + Lookahead + Gradient Centralisation, AMP, `torch.compile`, gradient clipping, manifold mixup, and comprehensive scheduler support.
- **Augmentation lab**: configurable RandAugment / TrivialAugment, AutoAugment, MixUp / CutMix / TokenMix / MixToken, colour jitter, random erasing, and flexible TTA.
- **Evaluation & OOD**: top-k metrics, calibration (ECE), energy scores, Mahalanobis and ViM-style statistics, detailed CSV / JSON artefacts, and confusion matrices.
- **Observability**: Weights & Biases logging, snapshot ensembling, Optuna sweeps, GradCAM++ explainability, and rich CSV history.
- **Deployment ready**: ONNX export and dynamic quantisation helpers, Optuna-driven sweeps, and inference script with GradCAM overlays.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Datasets default to CIFAR-100 and download automatically on first run (`./data`). Update the config or dataloader helpers for custom datasets.

## Repository Layout
- `pipeline/` - core training, evaluation, and inference modules:
  - `train_supcon.py`, `train_arcface.py`, `fine_tune_distill.py` - stage-specific trainers.
  - `backbone.py` - modular hybrid backbone with adapters.
  - `run_pipeline.py` - CLI orchestrator consuming `configs/config.yaml`.
  - `optimizers/` - SAM, Lookahead, EMA, gradient centralisation helpers.
  - `evaluate.py`, `inference.py`, `gradcam.py`, `augmentations.py`, `losses.py` - evaluation & support utilities.
- `configs/config.yaml` - pipeline configuration samples.
- `scripts/` - helper shell scripts.
- `docs/` - supplementary documentation.
- `tests/` - starter tests (extend as needed).

## Running the Pipeline

```bash
python -m pipeline.run_pipeline --config configs/config.yaml --phases supcon arcface distill evaluate
```

Select specific stages if desired, e.g. `python -m pipeline.run_pipeline --phases evaluate`.

### Phase Outputs
- **Snapshots** under `./snapshots/` (backbone, head, EMA variants).
- **Logs** in `./logs/` (CSV metrics and consolidated logger output).
- **Evaluation artefacts**: `metrics.json`, `classification_report.csv`, `predictions.csv`, confusion matrix image, and optional OOD histograms in `./eval_results/`.

## Customising Training

Every phase reads arguments from `configs/config.yaml`. Common knobs:

```yaml
logging:
  file: ./logs/full_pipeline.log
  wandb:
    enabled: true
    project: image-classification-pipeline
    run_name: cifar100-experiment
supcon:
  enabled: true
  batch_size: 16
  image_size: 224
  steps: 200
  augmentations:
    randaugment: {num_ops: 2, magnitude: 7}
    trivialaugment: true
  backbone:
    vit_model: dinov2_base14
    token_merging_ratio: 0.3
    lora_rank: 4
arcface:
  lr: 1e-4
  mix_method: cutmix  # mixup | cutmix | tokenmix | mixtoken | none
  use_curricularface: true
  use_manifold_mixup: true
  ema_decay: 0.9995
  augmentations:
    random_resized_crop: true
    randaugment: {num_ops: 2, magnitude: 9}
    random_erasing: {p: 0.25}
  backbone:
    vit_model: eva02_base_patch14_224
    token_merging_ratio: 0.5
    token_learner_tokens: 8
    lora_rank: 8
    use_ia3: true
distill:
  teacher_backbone_path: ./snapshots/snapshot_epoch_30.pth
  teacher_head_path: ./snapshots/head/snapshot_epoch_30.pth
  use_ema_teacher: true
  use_manifold_mixup: true
  augmentations:
    randaugment: {num_ops: 2, magnitude: 6}
  backbone:
    vit_model: dinov2_large14
    token_merging_ratio: 0.4
```

## Evaluation & Inference
- Run `python -m pipeline.run_pipeline --phases evaluate` after training to generate accuracy, calibration, energy, Mahalanobis, and ViM-style metrics (set `compute_mahalanobis`, `compute_vim`, `tta_runs`, etc.).
- For ad-hoc inference with optional GradCAM overlays:

```bash
python -m pipeline.inference \ 
  --image_dir path/to/images \ 
  --backbone_path snapshots/snapshot_epoch_30.pth \ 
  --head_path snapshots/head/snapshot_epoch_30.pth \ 
  --gradcam_dir outputs/gradcam
```

- Export backbone embeddings to ONNX or quantise dynamically:

```bash
python -m pipeline.export export-onnx --weights snapshots/snapshot_epoch_30.pth --output backbone.onnx
python -m pipeline.export quantize-dynamic --weights snapshots/snapshot_epoch_30.pth --output backbone_int8.pth
```

- Run Optuna sweeps for ArcFace hyper-parameters: `python -m pipeline.optuna_search` (logs in `./optuna_logs/`, snapshots in `./optuna_snapshots/`).

## Adding More SOTA Tricks
- Extend `BackboneConfig` with additional timm / HuggingFace foundation models or stack adapters (LoRA, IA3, adapters of your choice).
- Add augmentation recipes inside `pipeline/augmentations.py` (FMix, Cutout, MixStyle variants, diffusion replay).
- Experiment with new distillation losses or teacher strategies in `fine_tune_distill.py`.
- Plug in additional uncertainty estimators or ensemble strategies via `pipeline/evaluate.py`.

## TODO / Ideas
- Add regression tests around dataloaders and gradient sanity.
- Support alternate datasets (ImageNet, custom folders) via dataset factories.
- Optional Lightning / Hydra variants for users who prefer those ecosystems.

## License

MIT License - see `LICENSE` for details. Contributions welcome! Submit issues or PRs with ideas and improvements.
