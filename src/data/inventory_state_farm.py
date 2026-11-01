"""
Inventory of the State Farm Distracted Driver Detection dataset.

This script is READ-ONLY. It never changes anything inside data/raw.

What it checks:
  1. driver_imgs_list.csv is complete (no blanks, no repeated image names)
  2. The CSV matches the image folders (no missing, extra, or mislabeled files)
  3. Every training image opens and fully decodes (finds corrupt files)
  4. Image sizes and color modes
  5. Exact duplicate files (identical bytes)
  6. Images per class and per driver

Output: data/manifests/state_farm_inventory.csv (one row per training image)
"""
import hashlib
import io
from pathlib import Path

import pandas as pd
from PIL import Image

# Paths are built from this file's location, so the script works no matter
# which folder you run it from. parents[2] = the repo root (RBG-SQUAD).
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "state_farm"
TRAIN_DIR = RAW_DIR / "imgs" / "train"
CSV_PATH = RAW_DIR / "driver_imgs_list.csv"
OUT_PATH = ROOT / "data" / "manifests" / "state_farm_inventory.csv"

# Class meanings from the Kaggle competition page.
CLASS_NAMES = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "talking on the phone - right",
    "c3": "texting - left",
    "c4": "talking on the phone - left",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}


def section(title):
    print(f"\n=== {title} ===")


def main():
    # ---- 1. Load the metadata CSV ----
    meta = pd.read_csv(CSV_PATH)
    section("1. Metadata CSV")
    print(f"Rows: {len(meta)}   Columns: {list(meta.columns)}")
    print("Blank values per column:")
    print(meta.isna().sum().to_string())
    print(f"Repeated image names: {meta.duplicated(subset='img').sum()}")

    # ---- 2. Compare the CSV to the files actually on disk ----
    on_disk = pd.DataFrame(
        [{"img": p.name, "folder_class": p.parent.name}
         for p in sorted(TRAIN_DIR.glob("c*/*")) if p.is_file()]
    )
    merged = meta.merge(on_disk, on="img", how="outer", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]   # in CSV, not on disk
    extra = merged[merged["_merge"] == "right_only"]    # on disk, not in CSV
    both = merged[merged["_merge"] == "both"].drop(columns="_merge")
    mislabeled = both[both["classname"] != both["folder_class"]]

    section("2. CSV vs image folders")
    print(f"Images on disk:              {len(on_disk)}")
    print(f"In CSV but missing on disk:  {len(missing)}")
    print(f"On disk but not in CSV:      {len(extra)}")
    print(f"CSV class != folder class:   {len(mislabeled)}")

    # ---- 3. Open every image ----
    section("3. Opening every image (takes a few minutes)")
    records = []
    for i, row in enumerate(both.itertuples(index=False), start=1):
        path = TRAIN_DIR / row.folder_class / row.img
        rec = {
            "path": path.relative_to(ROOT).as_posix(),
            "img": row.img,
            "subject": row.subject,
            "classname": row.classname,
            "class_description": CLASS_NAMES.get(row.classname, "UNKNOWN"),
        }
        try:
            data = path.read_bytes()
            rec["file_bytes"] = len(data)
            rec["md5"] = hashlib.md5(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                im.load()  # forces a full decode, which catches truncated files
                rec.update(width=im.width, height=im.height, mode=im.mode,
                           readable=True, error="")
        except Exception as e:
            rec.update(readable=False, error=str(e))
        records.append(rec)
        if i % 2000 == 0:
            print(f"  checked {i} / {len(both)}")

    df = pd.DataFrame(records)
    # Whole numbers, with blanks allowed for unreadable images
    df[["width", "height"]] = df[["width", "height"]].astype("Int64")
    bad = df[~df["readable"]]
    print(f"Unreadable images: {len(bad)}")
    if len(bad):
        print(bad[["path", "error"]].head(10).to_string(index=False))

    # ---- 4. Sizes and color modes ----
    good = df[df["readable"]]
    section("4. Image sizes (width x height) and color modes")
    print(good.groupby(["width", "height"]).size().to_string())
    print(good["mode"].value_counts().to_string())

    # ---- 5. Exact duplicates ----
    dup = df[df["md5"].duplicated(keep=False)]
    section("5. Exact duplicate files")
    print(f"Images that have an identical copy: {len(dup)}")
    if len(dup):
        groups = dup.groupby("md5")
        print(f"Duplicate groups: {groups.ngroups}")
        print(f"Groups spanning more than one driver: {(groups['subject'].nunique() > 1).sum()}")
        print(f"Groups spanning more than one class:  {(groups['classname'].nunique() > 1).sum()}")

    # ---- 6. Class and driver counts ----
    section("6. Images per class")
    counts = df["classname"].value_counts().sort_index()
    for c, n in counts.items():
        print(f"  {c} ({CLASS_NAMES.get(c, '?')}): {n}")

    section("7. Drivers")
    per_driver = df["subject"].value_counts()
    print(f"Number of drivers: {per_driver.size}")
    print(f"Images per driver: min {per_driver.min()}, "
          f"median {int(per_driver.median())}, max {per_driver.max()}")
    print("\nImages per driver and class (rows = drivers, columns = classes):")
    print(pd.crosstab(df["subject"], df["classname"]).to_string())

    # ---- Save ----
    df.to_csv(OUT_PATH, index=False)
    section("Saved")
    print(OUT_PATH.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
