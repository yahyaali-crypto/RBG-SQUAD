"""
Save a grid of example State Farm images: a few random images per class.

This script is READ-ONLY for data/raw. It reads the inventory manifest made by
inventory_state_farm.py, so run that script first.

Output: outputs/figures/state_farm_samples.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to a file without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifests" / "state_farm_inventory.csv"
OUT_PATH = ROOT / "outputs" / "figures" / "state_farm_samples.png"

PER_CLASS = 4   # images shown per class (one row per class)
SEED = 42       # fixed seed, so everyone gets the same sample images


def main():
    df = pd.read_csv(MANIFEST)
    df = df[df["readable"]]

    classes = sorted(df["classname"].unique())
    fig, axes = plt.subplots(len(classes), PER_CLASS,
                             figsize=(PER_CLASS * 3, len(classes) * 2.4))

    for r, cls in enumerate(classes):
        rows = df[df["classname"] == cls]
        sample = rows.sample(n=min(PER_CLASS, len(rows)), random_state=SEED)
        for c in range(PER_CLASS):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(sample):
                continue
            item = sample.iloc[c]
            with Image.open(ROOT / item["path"]) as im:
                ax.imshow(im)
            ax.set_title(f"driver {item['subject']}", fontsize=8)
        # Class label on the left of each row
        axes[r, 0].text(-0.05, 0.5, f"{cls}\n{rows.iloc[0]['class_description']}",
                        transform=axes[r, 0].transAxes, ha="right", va="center",
                        fontsize=9)

    fig.suptitle("State Farm: random examples per class (seed 42)", fontsize=12)
    fig.tight_layout(rect=(0.12, 0, 1, 0.98))
    fig.savefig(OUT_PATH, dpi=110)
    print(f"Saved {OUT_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
