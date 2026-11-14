"""
Timing test: estimate how long one training epoch takes on THIS computer.
No real training happens. Run:  python -m src.models.timing_test
"""
import time

import tensorflow as tf

from src.data.loaders import ROOT, class_names, load_config, load_split, make_dataset
from src.models.build import build_baseline_cnn, build_mobilenetv2

BATCH_SIZE = 32
WARMUP_BATCHES = 2   # first batches are slow (setup), so they are not timed
TIMED_BATCHES = 10


def images_per_second(step_fn, ds):
    batches = iter(ds)
    for _ in range(WARMUP_BATCHES):
        step_fn(next(batches))
    start = time.perf_counter()
    for _ in range(TIMED_BATCHES):
        step_fn(next(batches))
    return TIMED_BATCHES * BATCH_SIZE / (time.perf_counter() - start)


def main():
    cfg = load_config()
    num_classes = len(class_names(cfg, "state_farm"))
    n_train = len(load_split(cfg, "state_farm", "train"))
    n_val = len(load_split(cfg, "state_farm", "val"))
    sample = (WARMUP_BATCHES + TIMED_BATCHES) * BATCH_SIZE
    ds = make_dataset(cfg, "state_farm", "train", batch_size=BATCH_SIZE, limit=sample).repeat()

    print("1/3 Loading speed ...")
    load_rate = images_per_second(lambda b: b, ds)

    print("2/3 Baseline CNN ...")
    baseline = build_baseline_cnn(num_classes)
    baseline.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    base_rate = images_per_second(lambda b: baseline.train_on_batch(b[0], b[1]), ds)

    print("3/3 MobileNetV2 (downloads about 9 MB of pretrained weights the first time) ...")
    mobilenet, _ = build_mobilenetv2(num_classes)
    mobilenet.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    mn_rate = images_per_second(lambda b: mobilenet.train_on_batch(b[0], b[1]), ds)

    lines = [f"Loading only:       {load_rate:7.1f} images/sec",
             f"Baseline CNN:       {base_rate:7.1f} images/sec",
             f"MobileNetV2 frozen: {mn_rate:7.1f} images/sec", "",
             f"Estimated time per epoch ({n_train} train + {n_val} val images):"]
    for name, rate in [("Baseline CNN", base_rate), ("MobileNetV2 frozen", mn_rate)]:
        minutes = (n_train / rate + n_val / (rate * 3)) / 60  # validation is about 3x faster
        lines.append(f"  {name:20s} about {minutes:5.1f} min/epoch, {minutes * 10 / 60:4.1f} hours for 10 epochs")

    print("\n" + "\n".join(lines))
    report = ROOT / "outputs" / "reports" / "timing_test.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSaved", report.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()