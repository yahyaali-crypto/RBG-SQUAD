"""
Data loaders: turn a split manifest into batches Keras can train on.
Every setting comes from configs/preprocessing.yaml.

Self-test:  python -m src.data.loaders
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "preprocessing.yaml"
AUTOTUNE = tf.data.AUTOTUNE


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(cfg, dataset, split):
    """Rows of one split (train, val, or test) from the split manifest."""
    df = pd.read_csv(ROOT / cfg[dataset]["split_manifest"])
    return df[df["split"] == split].reset_index(drop=True)


def class_names(cfg, dataset):
    labels = cfg[dataset]["labels"]
    return [str(labels[i]) for i in sorted(labels)]


def load_image(path, cfg):
    """Read one image: decode, pad to square, resize, scale to 0-1.
    The app (UC-02) must prepare uploaded images with these same steps."""
    height, width = cfg["shared"]["image_size"]
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    if cfg["shared"]["resize_method"] == "pad_to_square":
        img = tf.image.resize_with_pad(img, height, width)
    else:
        img = tf.image.resize(img, (height, width))
    return tf.cast(img, tf.float32) / 255.0


def build_augmenter(cfg, dataset):
    """Random changes applied to TRAINING images only, as set in the config."""
    aug = cfg[dataset]["augmentation"]
    seed = cfg["seed"]
    layers = []
    if aug.get("horizontal_flip"):
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    if aug.get("rotation_degrees"):
        layers.append(tf.keras.layers.RandomRotation(aug["rotation_degrees"] / 360, seed=seed))
    if aug.get("zoom"):
        layers.append(tf.keras.layers.RandomZoom(aug["zoom"], seed=seed))
    if aug.get("brightness"):
        layers.append(tf.keras.layers.RandomBrightness(aug["brightness"], value_range=(0, 1), seed=seed))
    if aug.get("contrast"):
        layers.append(tf.keras.layers.RandomContrast(aug["contrast"], seed=seed))
    return tf.keras.Sequential(layers, name=f"{dataset}_augmentation")


def make_dataset(cfg, dataset, split, batch_size=32, limit=None):
    """Batches of (images, labels). Train is shuffled and augmented. Val/test are not."""
    df = load_split(cfg, dataset, split)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=cfg["seed"]).reset_index(drop=True)
    paths = [str(ROOT / p) for p in df["path"]]
    labels = df["label_index"].to_numpy(dtype="int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    is_train = split in cfg["shared"]["augmentation_applies_to"]
    if is_train:
        ds = ds.shuffle(len(df), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, cfg), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if is_train:
        augmenter = build_augmenter(cfg, dataset)
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def class_weights(cfg, dataset):
    """Makes rare classes count more in training. Used for FL3D only."""
    df = load_split(cfg, dataset, "train")
    counts = df["label_index"].value_counts().sort_index()
    total, n = counts.sum(), len(counts)
    return {int(k): float(total / (n * v)) for k, v in counts.items()}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    names = class_names(cfg, "state_farm")
    images, labels = next(iter(make_dataset(cfg, "state_farm", "train", batch_size=16)))
    print("Batch shape:", tuple(images.shape))
    print("Pixel range:", float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))
    print("Labels:", [names[y] for y in labels.numpy()])

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for ax, img, y in zip(axes.flat, images.numpy(), labels.numpy()):
        ax.imshow(img)
        ax.set_title(names[y], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "loader_check_state_farm.png"
    fig.savefig(out, dpi=90)
    print("Saved", out.relative_to(ROOT).as_posix())