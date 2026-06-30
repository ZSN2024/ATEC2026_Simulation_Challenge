"""
Verify YOLO auto-labels by drawing bounding boxes on images.

Usage:
    python scripts/verify_labels.py --data data/yolo_dataset --split train --num 20
"""

import argparse
import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

parser = argparse.ArgumentParser(description="Verify YOLO label quality by visualizing bounding boxes.")
parser.add_argument("--data", type=str, default="data/yolo_dataset", help="Dataset directory.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                    help="Dataset split to verify.")
parser.add_argument("--num", type=int, default=20, help="Number of samples to visualize.")
parser.add_argument("--output", type=str, default=None, help="Output directory for annotated images. If not set, display inline (requires GUI).")

CLASS_NAMES = {0: "sugar", 1: "mustard", 2: "banana"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}  # BGR


def main():
    args = parser.parse_args()

    import cv2
    import numpy as np
    from pathlib import Path

    data_dir = Path(_PROJ_ROOT) / args.data
    img_dir = data_dir / "images" / args.split
    lbl_dir = data_dir / "labels" / args.split

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[ERROR] Dataset directories not found. Check --data path.")
        print(f"  img_dir: {img_dir}  ({'✓' if img_dir.exists() else '✗'})")
        print(f"  lbl_dir: {lbl_dir}  ({'✓' if lbl_dir.exists() else '✗'})")
        return

    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        print(f"[ERROR] No images found in {img_dir}")
        return

    total_images = len(img_files)
    total_labels = len(list(lbl_dir.glob("*.txt")))
    print(f"[Verify] Found {total_images} images, {total_labels} label files in {args.split} split.")

    # Sample evenly
    step = max(1, total_images // args.num)
    selected = img_files[::step][:args.num]

    stats = {"total_boxes": 0, "by_class": {0: 0, 1: 0, 2: 0}, "empty": 0,
             "out_of_bounds": 0, "zero_area": 0}

    for img_path in selected:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]

        if not lbl_path.exists():
            stats["empty"] += 1
            print(f"  [WARN] No label for {img_path.name}")

        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:5])

                    # Check validity
                    if not (0 <= x_c <= 1 and 0 <= y_c <= 1):
                        stats["out_of_bounds"] += 1
                        continue
                    if bw <= 0 or bh <= 0 or bw > 1 or bh > 1:
                        stats["zero_area"] += 1
                        continue

                    stats["total_boxes"] += 1
                    stats["by_class"][cls_id] = stats["by_class"].get(cls_id, 0) + 1
                    boxes.append((cls_id, x_c, y_c, bw, bh))

        # Draw boxes
        for cls_id, x_c, y_c, bw, bh in boxes:
            x1 = int((x_c - bw / 2) * W)
            y1 = int((y_c - bh / 2) * H)
            x2 = int((x_c + bw / 2) * W)
            y2 = int((y_c + bh / 2) * H)
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(cls_id, str(cls_id))}"
            cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        n_boxes = len(boxes)
        cv2.putText(img, f"Objects: {n_boxes}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"verify_{img_path.name}"
            cv2.imwrite(str(out_path), img)
        else:
            cv2.imshow("Label Verification", img)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

    if not args.output:
        cv2.destroyAllWindows()

    # Print statistics
    print(f"\n[Verify] Label statistics ({args.split}):")
    print(f"  Total boxes:        {stats['total_boxes']}")
    print(f"  Sugar boxes (0):    {stats['by_class'].get(0, 0)}")
    print(f"  Mustard bottles (1):{stats['by_class'].get(1, 0)}")
    print(f"  Bananas (2):        {stats['by_class'].get(2, 0)}")
    print(f"  Images w/o labels:  {stats['empty']}")
    print(f"  Out-of-bounds bbox: {stats['out_of_bounds']}")
    print(f"  Zero-area bbox:     {stats['zero_area']}")

    if stats["out_of_bounds"] > 0 or stats["zero_area"] > 0:
        print("\n  ⚠️  Found invalid labels! Check bbox projection logic in collect_yolo_data.py")
    elif stats["total_boxes"] == 0:
        print("\n  ⚠️  No valid boxes found. Data collection may have issues.")
    else:
        print("\n  ✓ Labels look valid.")

    if args.output:
        print(f"  Annotated images saved to: {args.output}")


if __name__ == "__main__":
    main()
