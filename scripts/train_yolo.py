"""
YOLO fine-tuning script for Task B simulation objects.

Trains YOLOv8n on simulation-captured data to detect:
  - sugar_box (class 0)
  - mustard_bottle (class 1)
  - banana (class 2)

Usage:
    python scripts/train_yolo.py --data data/yolo_dataset/dataset.yaml --epochs 100 --output demo/yolo_detector.pt
"""

import argparse
import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

parser = argparse.ArgumentParser(description="Train YOLO for Task B object detection.")
parser.add_argument("--data", type=str, default="data/yolo_dataset/dataset.yaml",
                    help="Path to YOLO dataset.yaml.")
parser.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Base YOLO model (yolov8n/s/m/l/x or custom .pt).")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
parser.add_argument("--batch", type=int, default=16, help="Batch size.")
parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, cpu, 0,1.")
parser.add_argument("--output", type=str, default="demo/yolo_detector.pt",
                    help="Output model path for deployment.")
parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer.")
parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate.")
parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")


def main():
    args = parser.parse_args()

    data_yaml = os.path.join(_PROJ_ROOT, args.data)
    if not os.path.isfile(data_yaml):
        print(f"[ERROR] Dataset config not found: {data_yaml}")
        print("[ERROR] Run collect_yolo_data.py first to generate the dataset.")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"[Train] Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"[Train] Starting training...")
    print(f"  data:     {data_yaml}")
    print(f"  epochs:   {args.epochs}")
    print(f"  batch:    {args.batch}")
    print(f"  imgsz:    {args.imgsz}")
    print(f"  device:   {args.device}")

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        # Validation every epoch
        val=True,
        # Save best model
        save=True,
        save_period=10,
        # Project name
        project="runs",
        name="train",
        exist_ok=True,
        # Use amp for speed
        amp=True,
        # Workers
        workers=4,
        # Augmentation (mild — simulation images are already varied)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        erasing=0.0,
    )

    # Export best model — use results.save_dir for the actual path
    import glob as _glob
    best_candidates = _glob.glob(
        os.path.join(_PROJ_ROOT, "runs", "**", "yolo_taskb", "train", "weights", "best.pt"),
        recursive=True,
    )
    if not best_candidates:
        # fallback: search anywhere under runs/
        best_candidates = _glob.glob(
            os.path.join(_PROJ_ROOT, "runs", "**", "best.pt"), recursive=True
        )
    best_pt = best_candidates[0] if best_candidates else None
    output_pt = os.path.join(_PROJ_ROOT, args.output)

    if best_pt and os.path.isfile(best_pt):
        import shutil
        os.makedirs(os.path.dirname(output_pt), exist_ok=True)
        shutil.copy2(best_pt, output_pt)
        print(f"[Train] Best model saved to: {output_pt}")
    else:
        print("[Train] Warning: best.pt not found. Training may have failed.")

    # Print metrics
    if results and hasattr(results, "results_dict"):
        metrics = results.results_dict
        print(f"\n[Train] Final metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
