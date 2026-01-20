from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from .backbone import BackboneConfig, HybridBackbone

LOGGER = logging.getLogger(__name__)


def _load_backbone(cfg: Optional[dict], weights_path: Optional[str], device: torch.device) -> HybridBackbone:
    backbone_cfg = BackboneConfig(**(cfg or {}))
    model = HybridBackbone(backbone_cfg).to(device)
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        # Check for wrapped state dicts
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                # Some legacy or other tools might use this key
                state = state["state_dict"]
            # If "backbone" key exists (from snapshotting just backbone?), check that too.
            # But usually snapshot saves model.state_dict().
            if "backbone" in state: # Check for nested 'backbone' key if saved from a larger system
                 state = state["backbone"]
                 
        model.load_state_dict(state, strict=False)
        LOGGER.info("Loaded backbone weights from %s", weights_path)
    model.eval()
    return model


def export_backbone_onnx(
    weights_path: str,
    output_path: str,
    backbone_cfg: Optional[dict] = None,
    image_size: int = 224,
    opset: int = 17,
):
    device = torch.device("cpu")
    model = _load_backbone(backbone_cfg, weights_path, device)

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    output_path = str(Path(output_path).resolve())
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["embeddings"],
        opset_version=opset,
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
    )
    LOGGER.info("Exported backbone to ONNX at %s", output_path)


def quantize_backbone_dynamic(
    weights_path: str,
    output_path: str,
    backbone_cfg: Optional[dict] = None,
):
    device = torch.device("cpu")
    model = _load_backbone(backbone_cfg, weights_path, device)
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), output_path)
    LOGGER.info("Saved dynamically quantized backbone state dict to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or quantize backbone models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    onnx_parser = subparsers.add_parser("export-onnx", help="Export backbone to ONNX format.")
    onnx_parser.add_argument("--weights", required=True, help="Path to backbone weights (.pth).")
    onnx_parser.add_argument("--output", required=True, help="Destination ONNX file.")
    onnx_parser.add_argument("--image-size", type=int, default=224, help="Input image resolution.")
    onnx_parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    onnx_parser.add_argument(
        "--backbone-cfg",
        help="Optional JSON string or file containing BackboneConfig parameters.",
    )

    quant_parser = subparsers.add_parser("quantize-dynamic", help="Perform dynamic quantization on backbone.")
    quant_parser.add_argument("--weights", required=True, help="Path to backbone weights (.pth).")
    quant_parser.add_argument("--output", required=True, help="Destination path for quantized state dict.")
    quant_parser.add_argument(
        "--backbone-cfg",
        help="Optional JSON string or file containing BackboneConfig parameters.",
    )

    return parser.parse_args()


def _parse_cfg(cfg: Optional[str]) -> Optional[dict]:
    if not cfg:
        return None
    cfg_path = Path(cfg)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(cfg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    backbone_cfg = _parse_cfg(getattr(args, "backbone_cfg", None))

    if args.command == "export-onnx":
        export_backbone_onnx(
            weights_path=args.weights,
            output_path=args.output,
            backbone_cfg=backbone_cfg,
            image_size=args.image_size,
            opset=args.opset,
        )
    elif args.command == "quantize-dynamic":
        quantize_backbone_dynamic(
            weights_path=args.weights,
            output_path=args.output,
            backbone_cfg=backbone_cfg,
        )


if __name__ == "__main__":
    main()

