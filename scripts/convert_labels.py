"""
Remap YOLO labels for Task B: skip bin, reorder classes.

Input classes:  0=banana, 1=bin, 2=mustard, 3=sugar_box
Output classes: 0=sugar_box, 1=mustard_bottle, 2=banana
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
parser.add_argument("--input", type=str, default="project-2-at-2026-06-30-19-20-916a3177")
parser.add_argument("--output", type=str, default="data/yolo_dataset")
args = parser.parse_args()

CLASS_MAP = {3: 0, 2: 1, 0: 2}  # orig → new (1=bin skipped)


def main():
    input_dir = Path(args.input)
    output_dir = Path(_PROJ_ROOT) / args.output

    # Reset output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (output_dir / sub / split).mkdir(parents=True)

    pairs = []
    for img_path in sorted((input_dir / "images").glob("*.jpg")):
        lbl_path = input_dir / "labels" / (img_path.stem + ".txt")
        pairs.append((img_path, lbl_path))

    import random
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    splits = {"train": pairs[n_val:], "val": pairs[:n_val]}

    stats = {"total": 0, "bin_skip": 0}

    for split, split_pairs in splits.items():
        for img_path, lbl_path in split_pairs:
            new_lines = []
            if lbl_path.exists():
                for line in open(lbl_path):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_orig = int(float(parts[0]))
                    cls_new = CLASS_MAP.get(cls_orig)
                    if cls_new is None:
                        stats["bin_skip"] += 1
                        continue
                    parts[0] = str(cls_new)
                    new_lines.append(" ".join(parts))
                    stats["total"] += 1

            if not new_lines:
                continue

            shutil.copy2(img_path, output_dir / "images" / split / img_path.name)
            lbl_out = output_dir / "labels" / split / img_path.stem
            with open(lbl_out.with_suffix(".txt"), "w") as f:
                f.write("\n".join(new_lines))

    # dataset.yaml
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(f"""path: {output_dir}
train: images/train
val: images/val
nc: 3
names:
  0: sugar_box
  1: mustard_bottle
  2: banana
""")

    used = len(list((output_dir / "images" / "train").glob("*.jpg"))) + len(list((output_dir / "images" / "val").glob("*.jpg")))
    print(f"[Remap] {used} images, {stats['total']} objects (skipped {stats['bin_skip']} bin)")
    print(f"[Remap] → {output_dir}")


if __name__ == "__main__":
    main()
