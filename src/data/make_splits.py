"""
Create driver-disjoint train / val / test splits for both datasets.

Reads:   configs/preprocessing.yaml and the inventory manifests
Writes:  data/manifests/state_farm_splits.csv
         data/manifests/fl3d_splits.csv

How the split works:
  Every driver goes entirely into ONE split, so the model is always tested on
  people it never saw in training. With only 26 (State Farm) and 19 (FL3D)
  drivers, a single random shuffle can give lopsided splits, so we try many
  shuffles (all from the fixed seed) and keep the one whose image counts and
  class mix are closest to the targets. Same seed = same split for everyone.

This script is READ-ONLY for data/raw.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "preprocessing.yaml"
N_TRIES = 5000          # number of random driver shuffles to compare
MIN_GROUPS_PER_SPLIT = 3  # val and test each need at least this many drivers


def section(title):
    print(f"\n=== {title} ===")


def score_assignment(class_counts, assignment, targets):
    """Lower is better. Measures how far a split is from the targets.

    class_counts: table with one row per driver and one column per class,
    holding how many images that driver has of each class.
    """
    total = class_counts.values.sum()
    overall = class_counts.sum() / total
    size_error = 0.0
    class_error = 0.0
    for name, target in targets.items():
        members = [g for g, s in assignment.items() if s == name]
        part = class_counts.loc[members].sum()
        if (part == 0).any():
            return float("inf")  # a class is missing from this split: not allowed
        size_error += abs(part.sum() / total - target)
        class_error += (part / part.sum() - overall).abs().sum()
    return size_error + class_error


def find_best_split(df, group_col, label_col, targets, seed):
    rng = np.random.default_rng(seed)
    class_counts = pd.crosstab(df[group_col], df[label_col])  # drivers x classes
    groups = np.array(sorted(class_counts.index))
    sizes = class_counts.sum(axis=1)
    total = sizes.sum()
    best, best_score = None, float("inf")
    for _ in range(N_TRIES):
        order = rng.permutation(groups)
        # Fill val, then test, up to their target share. The rest is train.
        assignment, filled = {}, {"val": 0, "test": 0}
        for g in order:
            for name in ["val", "test"]:
                if filled[name] < targets[name] * total:
                    assignment[g] = name
                    filled[name] += sizes[g]
                    break
            else:
                assignment[g] = "train"
        counts = pd.Series(assignment).value_counts()
        if counts.get("val", 0) < MIN_GROUPS_PER_SPLIT or counts.get("test", 0) < MIN_GROUPS_PER_SPLIT:
            continue
        s = score_assignment(class_counts, assignment, targets)
        if s < best_score:
            best, best_score = assignment, s
    return best, best_score


def report(df, group_col, label_col):
    order = ["train", "val", "test"]
    section("Images per split")
    counts = df["split"].value_counts().reindex(order)
    print(pd.DataFrame({"images": counts, "share": (counts / len(df)).round(3)}).to_string())
    section("Drivers per split")
    for name in order:
        drivers = sorted(df.loc[df["split"] == name, group_col].unique())
        print(f"{name:5s} ({len(drivers):2d}): {', '.join(drivers)}")
    section("Class share per split (each column sums to 1)")
    table = pd.crosstab(df[label_col], df["split"], normalize="columns")[order]
    print(table.round(3).to_string())


def verify(df, group_col):
    """Hard checks. The script stops with an error if any fail."""
    per_group = df.groupby(group_col)["split"].nunique()
    assert (per_group == 1).all(), "A driver appears in more than one split"
    assert df["split"].isin(["train", "val", "test"]).all(), "Some rows have no split"
    assert df["path"].is_unique, "Duplicate image paths"
    print("Checks passed: every driver in exactly one split, every row assigned, no duplicate paths")


def split_state_farm(cfg, seed, targets):
    sf = cfg["state_farm"]
    df = pd.read_csv(ROOT / sf["inventory"])
    df = df[df["readable"]].copy()
    label_to_index = {code: idx for idx, code in sf["labels"].items()}
    df["label_index"] = df[sf["label_column"]].map(label_to_index)

    assignment, score = find_best_split(df, sf["group_column"], sf["label_column"], targets, seed)
    df["split"] = df[sf["group_column"]].map(assignment)

    print("\n##########  STATE FARM  ##########")
    print(f"Best of {N_TRIES} shuffles, score {score:.3f} (lower is better)")
    report(df, sf["group_column"], sf["label_column"])
    verify(df, sf["group_column"])

    cols = ["path", "subject", "classname", "label_index", "split"]
    out = ROOT / sf["split_manifest"]
    df[cols].sort_values(["split", "subject", "path"]).to_csv(out, index=False)
    print(f"Saved {out.relative_to(ROOT).as_posix()}")


def split_fl3d(cfg, seed, targets):
    fl = cfg["fl3d"]
    df = pd.read_csv(ROOT / fl["inventory"])
    df = df[df["readable"]].copy()

    # Drop the frames whose labels conflict between the two annotation files
    excluded = set(fl["excluded"]["frames"])
    df["rel"] = df["video_id"] + "/" + df["frame"]
    before = len(df)
    df = df[~df["rel"].isin(excluded)].copy()
    print("\n##########  FL3D  ##########")
    print(f"Excluded {before - len(df)} conflicting frames (expected {len(excluded)})")

    # Attach the hand-made driver IDs
    driver_map = pd.read_csv(ROOT / fl["driver_map"])
    assert driver_map["driver_id"].notna().all(), "driver_map has blank driver_id values"
    df = df.merge(driver_map[["video_id", "driver_id"]], on="video_id", how="left")
    assert df["driver_id"].notna().all(), "Some videos are missing from driver_map"

    label_to_index = {name: idx for idx, name in fl["labels"].items()}
    df["label_index"] = df[fl["label_column"]].map(label_to_index)

    assignment, score = find_best_split(df, fl["group_column"], fl["label_column"], targets, seed)
    df["split"] = df[fl["group_column"]].map(assignment)

    print(f"Best of {N_TRIES} shuffles, score {score:.3f} (lower is better)")
    report(df, fl["group_column"], fl["label_column"])
    verify(df, fl["group_column"])

    cols = ["path", "video_id", "driver_id", "frame_number", "label", "label_index", "split"]
    out = ROOT / fl["split_manifest"]
    df[cols].sort_values(["split", "driver_id", "video_id", "frame_number"]).to_csv(out, index=False)
    print(f"Saved {out.relative_to(ROOT).as_posix()}")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    targets = cfg["shared"]["split_ratios"]
    split_state_farm(cfg, seed, targets)
    split_fl3d(cfg, seed, targets)


if __name__ == "__main__":
    main()
