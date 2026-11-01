"""
Make a contact sheet with one face per FL3D video, so the team can mark
which videos show the same driver. FL3D does not include driver IDs.

Run inventory_fl3d.py first. This script is READ-ONLY for data/raw.

Outputs:
  outputs/figures/fl3d_video_faces.png      one labeled face per video
  data/manifests/fl3d_video_driver_map.csv  blank template for the team to fill
                                            (never overwritten if it exists)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifests" / "fl3d_inventory.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "fl3d_video_faces.png"
MAP_PATH = ROOT / "data" / "manifests" / "fl3d_video_driver_map.csv"

COLS = 6
THUMB = (180, 250)  # every face resized to the same size, for display only


def pick_frame(rows):
    """Middle 'alert' frame of the video: eyes open, face clearly visible."""
    alert = rows[rows["label"] == "alert"].sort_values("frame_number")
    pool = alert if len(alert) else rows.sort_values("frame_number")
    return pool.iloc[len(pool) // 2]


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["readable"]]
    videos = sorted(df["video_id"].unique())

    n_rows = -(-len(videos) // COLS)  # ceiling division
    fig, axes = plt.subplots(n_rows, COLS, figsize=(COLS * 2.2, n_rows * 3.2))
    for ax in axes.flat:
        ax.axis("off")

    for ax, vid in zip(axes.flat, videos):
        rows = df[df["video_id"] == vid]
        item = pick_frame(rows)
        with Image.open(ROOT / item["path"]) as im:
            ax.imshow(im.convert("RGB").resize(THUMB))
        kinds = sorted(set(rows["label"]) - {"alert"})
        ax.set_title(f"{vid.replace('_720', '')}\n{', '.join(kinds) or 'alert only'}",
                     fontsize=8)

    fig.suptitle("FL3D: one face per video (fill in fl3d_video_driver_map.csv)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=110)
    print(f"Saved {FIG_PATH.relative_to(ROOT).as_posix()}")

    if MAP_PATH.exists():
        print(f"Kept existing {MAP_PATH.relative_to(ROOT).as_posix()} (not overwritten)")
    else:
        pd.DataFrame({"video_id": videos, "driver_id": "", "notes": ""}).to_csv(
            MAP_PATH, index=False)
        print(f"Created blank {MAP_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
