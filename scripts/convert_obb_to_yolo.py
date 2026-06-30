"""
Convert OBB (rotated bbox) labels to YOLO axis-aligned format.

Input:  project-2-at-2026-06-30-19-13-916a3177/
Output: data/yolo_dataset_from_obb/

Label format conversion:
    OBB:   class_id x1 y1 x2 y2 x3 y3 x4 y4
    YOLO:  class_id x_center y_center width height

Class remapping (project → YOLO):
    0 (banana)         → 2
    1 (bin)             → skip
    2 (mustard_bottle)  → 1
    3 (sugar_box)       → 0
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="project-2-at-2026-06-30-19-13-916a3177")
parser.add_argument("--output", type=str, default="data/yolo_dataset_from_obb")
parser.add_argument("--val_split", type=float, default=0.1)
args = parser.parse_args()

# Class mapping: original → YOLO (None = skip)
CLASS_MAP = {
    0: 2,   # banana
    1: None,  # bin → skip
    2: 1,   # mustard_bottle
    3: 0,   # sugar_box
    4: 0,   # sugar_box (if there's a 5th class)
}

YOLO_CLASS_NAMES = ["sugar_box", "mustard_bottle", "banana"]


def convert_obb_to_yolo(line: str):
    """Convert OBB line to YOLO format line, or None if bin/skip."""
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    cls_orig = int(float(parts[0]))
    cls_new = CLASS_MAP.get(cls_orig)
    if cls_new is None:
        return None

    coords = [float(x) for x in parts[1:]]
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    return f"{cls_new} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def main():
    input_dir = Path(args.input)
    output_dir = Path(_PROJ_ROOT) / args.output

    img_src = input_dir / "images"
    lbl_src = input_dir / "labels"
    if not img_src.exists() or not lbl_src.exists():
        print(f"[ERROR] Input directories not found: {img_src}, {lbl_src}")
        sys.exit(1)

    # Collect all image files
    img_files = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
    print(f"[Convert] Found {len(img_files)} images, {len(list(lbl_src.glob('*.txt')))} labels")

    # Build pairs
    pairs = []
    skipped_bin = 0
    for img_path in img_files:
        lbl_path = lbl_src / (img_path.stem + ".txt")
        if not lbl_path.exists():
            pairs.append((img_path, None))
            continue
        with open(lbl_path) as f:
            lines = f.readlines()
        converted = []
        for line in lines:
            yolo_line = convert_obb_to_yolo(line)
            if yolo_line is None:
                # Check if it was bin
                parts = line.strip().split()
                if len(parts) > 0 and int(float(parts[0])) == 1:
                    skipped_bin += 1
                continue
            converted.append(yolo_line)
        if not converted:
            continue  # skip images with only bin or no valid labels
        pairs.append((img_path, converted))

    print(f"[Convert] Valid pairs: {len(pairs)} (skipped {skipped_bin} bin labels)")

    if len(pairs) == 0:
        print("[ERROR] No valid image-label pairs after conversion.")
        sys.exit(1)

    # Split train/val
    import random
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, labels in split_pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_name = img_path.stem + ".txt"
            with open(lbl_out / lbl_name, "w") as f:
                f.write("\n".join(labels))

    # Write dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""# YOLO dataset config
path: {output_dir}
train: images/train
val: images/val
nc: 3
names:
  0: sugar_box
  1: mustard_bottle
  2: banana
""")

    # Stats
    train_labels = sum(len(l) for _, l in train_pairs)
    val_labels = sum(len(l) for _, l in val_pairs)
    print(f"[Convert] Done: train={len(train_pairs)} images ({train_labels} objects), "
          f"val={len(val_pairs)} images ({val_labels} objects)")
    print(f"[Convert] Output: {output_dir}")
    print(f"[Convert] Dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
