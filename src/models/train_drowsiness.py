"""
Train a UC-04 drowsiness model on FL3D. Examples:
  python -m src.models.train_drowsiness --model baseline --epochs 1 --limit 256   (quick smoke test)
  python -m src.models.train_drowsiness --model baseline --epochs 10
  python -m src.models.train_drowsiness --model mobilenetv2 --epochs 10 --fine_tune 5

Same steps as src/models/train.py (UC-03), but on FL3D, with class weights
because 'alert' is about 73% of the frames. Uses ONLY the train and val splits.
The test split is saved for the final evaluation (src/evaluation/evaluate_uc04.py).
"""
import argparse
import json
import time
from datetime import datetime

import tensorflow as tf

from src.data.loaders import ROOT, class_names, class_weights, load_config, make_dataset
from src.models.build import build_baseline_cnn, build_mobilenetv2

DATASET = "fl3d"
OUT_DIR = ROOT / "outputs" / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "mobilenetv2"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_tune", type=int, default=0,
                        help="extra epochs with the top of MobileNetV2 unfrozen")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only N training images (for quick tests)")
    args = parser.parse_args()

    cfg = load_config()
    tf.keras.utils.set_random_seed(cfg["seed"])
    names = class_names(cfg, DATASET)
    train_ds = make_dataset(cfg, DATASET, "train", args.batch_size, limit=args.limit)
    val_limit = max(args.limit // 4, args.batch_size) if args.limit else None
    val_ds = make_dataset(cfg, DATASET, "val", args.batch_size, limit=val_limit)
    weights = class_weights(cfg, DATASET) if cfg[DATASET]["class_balance"] == "class_weights" else None
    print("Class weights:", weights)

    if args.model == "baseline":
        model, base = build_baseline_cnn(len(names)), None
    else:
        model, base = build_mobilenetv2(len(names))

    run_name = "drowsiness_" + args.model + ("_finetuned" if args.fine_tune else "") + ("_smoketest" if args.limit else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / f"{run_name}.keras"
    history_path = OUT_DIR / f"{run_name}_history.csv"

    # Stage 1: train (for MobileNetV2, only the new head learns)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(history_path),
    ]
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks, class_weight=weights)
    epochs_done = len(history.history["loss"])

    # Stage 2 (optional): unfreeze the last 30 MobileNetV2 layers and fine-tune slowly
    if args.fine_tune and base is not None:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True,
                                               initial_value_threshold=min(history.history["val_loss"])),
            tf.keras.callbacks.CSVLogger(history_path, append=True),
        ]
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_done + args.fine_tune,
                             initial_epoch=epochs_done, callbacks=callbacks, class_weight=weights)
        epochs_done += len(history2.history["loss"])

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    minutes = (time.time() - start) / 60
    summary = {
        "run": run_name, "model": args.model, "dataset": DATASET,
        "date": datetime.now().isoformat(timespec="seconds"),
        "epochs_run": epochs_done, "fine_tune_epochs": args.fine_tune,
        "batch_size": args.batch_size, "train_limit": args.limit,
        "best_val_loss": round(val_loss, 4), "best_val_accuracy": round(val_acc, 4),
        "training_minutes": round(minutes, 1), "seed": cfg["seed"],
        "image_size": cfg["shared"]["image_size"], "classes": names,
        "class_weights": weights,
    }
    (OUT_DIR / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {minutes:.1f} min. Best val accuracy {val_acc:.3f}, val loss {val_loss:.3f}")
    print(f"Saved {model_path.relative_to(ROOT).as_posix()} and its history and summary")


if __name__ == "__main__":
    main()
