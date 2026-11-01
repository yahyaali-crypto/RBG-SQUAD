"""
Phase 1 verification: confirm the prepared data is ready for model training.

This script is READ-ONLY for data/raw. It checks:
  1. The config and both split manifests load
  2. Every image path in the split manifests exists on disk
  3. Splits are driver-disjoint, labels match the config, excluded frames are gone
  4. A sample of images from every split loads through a TensorFlow pipeline
     exactly as training will load them (decode, pad to square, resize, scale)
  5. Raw data is unchanged since the inventory (file fingerprints still match)
     and is not tracked by Git

Outputs:
  outputs/reports/phase1_verification.txt   everything printed below
  outputs/figures/preprocessed_samples.png  what the model will actually see
"""
import hashlib
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "preprocessing.yaml"
REPORT_PATH = ROOT / "outputs" / "reports" / "phase1_verification.txt"
FIG_PATH = ROOT / "outputs" / "figures" / "preprocessed_samples.png"
SAMPLE_PER_SPLIT = 32    # images per split pushed through the TF pipeline
HASH_SAMPLE = 500        # raw files re-fingerprinted per dataset

lines = []      # everything we print also goes into the report file
failures = []   # any failed check is recorded here


def log(text=""):
    print(text)
    lines.append(str(text))


def check(ok, message):
    log(f"  [{'PASS' if ok else 'FAIL'}] {message}")
    if not ok:
        failures.append(message)


def load_and_preprocess(path, size, method):
    """The exact loading steps training will use. Imported lazily (TF is slow to load)."""
    import tensorflow as tf
    data = tf.io.read_file(path)
    img = tf.io.decode_jpeg(data, channels=3)
    if method == "pad_to_square":
        img = tf.image.resize_with_pad(img, size[0], size[1])
    else:
        img = tf.image.resize(img, size)
    return tf.cast(img, tf.float32) / 255.0


def tf_pipeline_check(name, df, cfg, rng_seed):
    import tensorflow as tf
    size = cfg["shared"]["image_size"]
    method = cfg["shared"]["resize_method"]
    n_classes = len(cfg[name]["labels"])
    samples = []
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        sample = part.sample(n=min(SAMPLE_PER_SPLIT, len(part)), random_state=rng_seed)
        paths = [str(ROOT / p) for p in sample["path"]]
        ds = tf.data.Dataset.from_tensor_slices((paths, sample["label_index"].values))
        ds = ds.map(lambda p, y: (load_and_preprocess(p, size, method), y)).batch(SAMPLE_PER_SPLIT)
        images, labels = next(iter(ds))
        check(tuple(images.shape) == (len(sample), size[0], size[1], 3),
              f"{name} {split}: batch shape {tuple(images.shape)}")
        lo, hi = float(tf.reduce_min(images)), float(tf.reduce_max(images))
        check(0.0 <= lo and hi <= 1.0 and hi > 0.1,
              f"{name} {split}: pixel range {lo:.2f} to {hi:.2f} (expected within 0 to 1)")
        ok_labels = bool(tf.reduce_all((labels >= 0) & (labels < n_classes)))
        check(ok_labels, f"{name} {split}: labels within 0 to {n_classes - 1}")
        samples.append((split, images.numpy()[:4], sample.iloc[:4]))
    return samples


def raw_unchanged_check(name, inventory_path, rng_seed):
    inv = pd.read_csv(ROOT / inventory_path)
    inv = inv[inv["readable"]]
    sample = inv.sample(n=min(HASH_SAMPLE, len(inv)), random_state=rng_seed)
    mismatched = 0
    for path, expected in zip(sample["path"], sample["md5"]):
        actual = hashlib.md5((ROOT / path).read_bytes()).hexdigest()
        mismatched += actual != expected
    check(mismatched == 0,
          f"{name}: {len(sample)} random raw files re-fingerprinted, {mismatched} changed")


def main():
    log("PHASE 1 DATA VERIFICATION")

    # ---- 1. Load config and split manifests ----
    log("\n=== 1. Config and manifests ===")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    splits = {}
    for name in ["state_farm", "fl3d"]:
        path = ROOT / cfg[name]["split_manifest"]
        check(path.exists(), f"{name}: {path.relative_to(ROOT).as_posix()} exists")
        splits[name] = pd.read_csv(path)
        log(f"  {name}: {len(splits[name])} rows")

    # ---- 2. Every path exists ----
    log("\n=== 2. Every image path exists on disk ===")
    for name, df in splits.items():
        missing = [p for p in df["path"] if not (ROOT / p).exists()]
        check(len(missing) == 0, f"{name}: {len(df) - len(missing)} of {len(df)} paths found")
        for p in missing[:5]:
            log(f"    missing: {p}")

    # ---- 3. Split integrity ----
    log("\n=== 3. Split integrity ===")
    for name, df in splits.items():
        group = cfg[name]["group_column"]
        label_col = cfg[name]["label_column"]
        per_group = df.groupby(group)["split"].nunique()
        check((per_group == 1).all(), f"{name}: every {group} is in exactly one split")
        check(set(df["split"]) == {"train", "val", "test"}, f"{name}: train, val and test all present")
        expected = {str(v): int(k) for k, v in cfg[name]["labels"].items()}
        mapped_ok = (df[label_col].astype(str).map(expected) == df["label_index"]).all()
        check(mapped_ok, f"{name}: label_index matches the config label mapping")
        for split in ["train", "val", "test"]:
            present = set(df.loc[df["split"] == split, label_col].astype(str))
            check(present == set(expected), f"{name} {split}: all {len(expected)} classes present")
    fl = splits["fl3d"]
    rel = fl["video_id"] + "/" + fl["path"].str.split("/").str[-1]
    leftover = rel.isin(set(cfg["fl3d"]["excluded"]["frames"])).sum()
    check(leftover == 0, f"fl3d: excluded conflicting frames absent ({leftover} found)")
    sf_paths = splits["state_farm"]["path"]
    check(not sf_paths.str.contains("/imgs/test/").any(), "state_farm: no unlabeled Kaggle test images used")

    # ---- 4. TensorFlow pipeline ----
    log("\n=== 4. TensorFlow loading pipeline (loading TensorFlow, may take a moment) ===")
    log(f"  size {cfg['shared']['image_size']}, method {cfg['shared']['resize_method']}, "
        f"scaling {cfg['shared']['scaling']}")
    figure_rows = {}
    for name, df in splits.items():
        figure_rows[name] = tf_pipeline_check(name, df, cfg, seed)

    # ---- 5. Raw data unchanged ----
    log("\n=== 5. Raw data unchanged ===")
    raw_unchanged_check("state_farm", cfg["state_farm"]["inventory"], seed)
    raw_unchanged_check("fl3d", cfg["fl3d"]["inventory"], seed)
    try:
        tracked = subprocess.run(["git", "ls-files", "data/raw"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
        check(tracked == "", "data/raw is not tracked by Git")
    except Exception as e:
        log(f"  [SKIP] could not run git ({e})")

    # ---- Figure: what the model sees ----
    fig, axes = plt.subplots(2, 6, figsize=(14, 6.4))
    for r, name in enumerate(["state_farm", "fl3d"]):
        label_col = cfg[name]["label_column"]
        items = [(s, img, row) for s, imgs, rows in figure_rows[name]
                 for img, (_, row) in zip(imgs[:2], rows.iloc[:2].iterrows())]
        for c, (split, img, row) in enumerate(items[:6]):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_title(f"{name} | {split}\n{row[label_col]}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Preprocessed inputs (224x224, padded to square, scaled 0-1)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    fig.savefig(FIG_PATH, dpi=100)

    # ---- Summary ----
    log("\n=== Summary ===")
    if failures:
        log(f"{len(failures)} CHECK(S) FAILED:")
        for f_ in failures:
            log(f"  - {f_}")
    else:
        log("ALL CHECKS PASSED. Data is ready for model training.")
    log(f"Saved {REPORT_PATH.relative_to(ROOT).as_posix()} and {FIG_PATH.relative_to(ROOT).as_posix()}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
