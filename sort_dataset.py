"""
Sorts the raw Food-11 dataset (as packaged by Kaggle/EPFL) into the folder
structure this project expects.

The raw dataset has images directly inside `training/`, `validation/`, and
`evaluation/` folders, with filenames like "0_101.jpg" where the number
before the underscore is the class ID (0-10). This script reads that
prefix and copies each image into data/train/<ClassName>/ or
data/validation/<ClassName>/.

Usage:
    python sort_dataset.py --raw_training "C:\\path\\to\\training" --raw_validation "C:\\path\\to\\validation"

Adjust the paths above to wherever you extracted the Kaggle download.
"""

import argparse
import os
import shutil

CLASS_MAP = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def sort_folder(raw_folder, split_name):
    """split_name is 'train' or 'validation'."""
    out_base = os.path.join(DATA_DIR, split_name)
    os.makedirs(out_base, exist_ok=True)

    for class_name in CLASS_MAP.values():
        os.makedirs(os.path.join(out_base, class_name), exist_ok=True)

    count = 0
    skipped = 0
    for fname in os.listdir(raw_folder):
        src_path = os.path.join(raw_folder, fname)
        if not os.path.isfile(src_path):
            continue

        try:
            class_id = int(fname.split("_")[0])
        except (ValueError, IndexError):
            skipped += 1
            continue

        class_name = CLASS_MAP.get(class_id)
        if class_name is None:
            skipped += 1
            continue

        dst_path = os.path.join(out_base, class_name, fname)
        shutil.copy2(src_path, dst_path)
        count += 1

    print(f"[{split_name}] Copied {count} images, skipped {skipped} unrecognized files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_training", required=True, help="Path to the raw 'training' folder from Kaggle"
    )
    parser.add_argument(
        "--raw_validation", required=True, help="Path to the raw 'validation' folder from Kaggle"
    )
    args = parser.parse_args()

    sort_folder(args.raw_training, "train")
    sort_folder(args.raw_validation, "validation")

    print("\nDone. Your data/ folder should now have train/ and validation/")
    print("each containing the 11 class subfolders, ready for train_model.py.")


if __name__ == "__main__":
    main()
