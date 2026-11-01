"""
Inventory of the FL3D (Frame-Level Driver Drowsiness Detection) dataset.

This script is READ-ONLY. It never changes anything inside data/raw.

What it checks:
  1. annotations_all.json: labels, and which official split each frame is in
  2. The per-video annotations_final.json files agree with annotations_all.json
  3. The JSON matches the image files on disk (no missing or extra frames)
  4. Every frame opens and fully decodes (finds corrupt files)
  5. Frame sizes and color modes (FL3D frames are face crops of varying size)
  6. Exact duplicate files (identical bytes)
  7. Frames and labels per video

Note: FL3D has NO driver ID. video_id is the only grouping in the data.
Driver groups are added later by hand (see make_fl3d_video_sheet.py).

Output: data/manifests/fl3d_inventory.csv (one row per annotated frame)
"""
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "fl3d"
FRAMES_DIR = RAW_DIR / "classification_frames"
OUT_PATH = ROOT / "data" / "manifests" / "fl3d_inventory.csv"


def section(title):
    print(f"\n=== {title} ===")


def load_json(name):
    with open(FRAMES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def clean_key(key):
    """'./classification_frames/P1042751_720/frame10.jpg' -> 'P1042751_720/frame10.jpg'"""
    return key.replace("./classification_frames/", "", 1)


def main():
    # ---- 1. Main annotation file + official splits ----
    all_ann = {clean_key(k): v for k, v in load_json("annotations_all.json").items()}
    official = {}
    for split in ["train", "val", "test"]:
        for k in load_json(f"annotations_{split}.json"):
            official[clean_key(k)] = split

    df = pd.DataFrame(
        [{"rel": k, "label": v["driver_state"]} for k, v in all_ann.items()]
    )
    df["video_id"] = df["rel"].str.split("/").str[0]
    df["frame"] = df["rel"].str.split("/").str[1]
    df["frame_number"] = df["frame"].str.extract(r"(\d+)").astype(int)
    df["official_split"] = df["rel"].map(official).fillna("none")

    section("1. annotations_all.json")
    print(f"Annotated frames: {len(df)}   Videos: {df['video_id'].nunique()}")
    print(df["label"].value_counts().to_string())
    print("\nOfficial split sizes (frames):")
    print(df["official_split"].value_counts().to_string())

    # ---- 2. Per-video annotation files vs the main file ----
    section("2. Per-video annotations_final.json vs annotations_all.json")
    mismatched = 0
    only_in_video_file = 0
    for vid_dir in sorted(p for p in FRAMES_DIR.iterdir() if p.is_dir()):
        per_video_file = vid_dir / "annotations_final.json"
        if not per_video_file.exists():
            print(f"  {vid_dir.name}: no annotations_final.json")
            continue
        with open(per_video_file, encoding="utf-8") as f:
            per_video = json.load(f)
        for frame, info in per_video.items():
            key = f"{vid_dir.name}/{frame}"
            if key not in all_ann:
                only_in_video_file += 1
            elif all_ann[key]["driver_state"] != info["driver_state"]:
                mismatched += 1
    print(f"Frames only in a per-video file: {only_in_video_file}")
    print(f"Frames whose label differs:      {mismatched}")

    # ---- 3. JSON vs files on disk ----
    on_disk = {f"{p.parent.name}/{p.name}" for p in FRAMES_DIR.glob("*/*.jpg")}
    annotated = set(df["rel"])
    section("3. JSON vs image files")
    print(f"Image files on disk:            {len(on_disk)}")
    print(f"Annotated but missing on disk:  {len(annotated - on_disk)}")
    print(f"On disk but not annotated:      {len(on_disk - annotated)}")

    # ---- 4. Open every annotated frame ----
    section("4. Opening every frame (takes a few minutes)")
    records = []
    for i, rel in enumerate(df["rel"], start=1):
        path = FRAMES_DIR / rel
        rec = {"rel": rel}
        try:
            data = path.read_bytes()
            rec["file_bytes"] = len(data)
            rec["md5"] = hashlib.md5(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                rec.update(width=im.width, height=im.height, mode=im.mode,
                           readable=True, error="")
        except Exception as e:
            rec.update(readable=False, error=str(e))
        records.append(rec)
        if i % 5000 == 0:
            print(f"  checked {i} / {len(df)}")

    df = df.merge(pd.DataFrame(records), on="rel", how="left")
    df[["width", "height"]] = df[["width", "height"]].astype("Int64")
    df["path"] = (FRAMES_DIR.relative_to(ROOT).as_posix() + "/") + df["rel"]
    bad = df[~df["readable"]]
    print(f"Unreadable frames: {len(bad)}")
    if len(bad):
        print(bad[["path", "error"]].head(10).to_string(index=False))

    # ---- 5. Sizes and modes ----
    good = df[df["readable"]]
    section("5. Frame sizes and color modes")
    print(good[["width", "height"]].describe().round(1).to_string())
    print(good["mode"].value_counts().to_string())

    # ---- 6. Exact duplicates ----
    dup = df[df["md5"].duplicated(keep=False) & df["md5"].notna()]
    section("6. Exact duplicate files")
    print(f"Frames that have an identical copy: {len(dup)}")
    if len(dup):
        groups = dup.groupby("md5")
        print(f"Duplicate groups: {groups.ngroups}")
        print(f"Groups spanning more than one video: {(groups['video_id'].nunique() > 1).sum()}")
        print(f"Groups spanning more than one label: {(groups['label'].nunique() > 1).sum()}")

    # ---- 7. Per video ----
    section("7. Frames per video and label")
    table = pd.crosstab(df["video_id"], df["label"])
    table["total"] = table.sum(axis=1)
    table["official_split"] = df.groupby("video_id")["official_split"].agg(
        lambda s: "+".join(sorted(s.unique())))
    print(table.to_string())

    # ---- Save ----
    cols = ["path", "video_id", "frame", "frame_number", "label", "official_split",
            "width", "height", "mode", "file_bytes", "md5", "readable", "error"]
    df.sort_values(["video_id", "frame_number"])[cols].to_csv(OUT_PATH, index=False)
    section("Saved")
    print(OUT_PATH.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
