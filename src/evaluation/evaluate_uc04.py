"""
UC-04 final evaluation (drowsiness, FL3D).
  1. Picks the final model and the confidence threshold on the VALIDATION split
  2. Evaluates the final model ONCE on the TEST split
Run:  python -m src.evaluation.evaluate_uc04

The final model is chosen by validation macro F1 only, so the test split never
influences which model is used. Accuracy alone is misleading here because
'alert' is about 73% of the frames, so macro F1 and per-class recall are reported.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score

from src.data.loaders import ROOT, class_names, load_config, load_split, make_dataset
from src.models.predict_drowsiness import STATES, decide

MODEL_FILES = {"baseline": "drowsiness_baseline.keras",
               "mobilenetv2": "drowsiness_mobilenetv2_finetuned.keras"}
TARGET_ACCURACY = 0.97
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def rate(mask, values):
    return f"{values[mask].mean():.1%}" if mask.any() else "n/a"


def main():
    cfg = load_config()
    names = class_names(cfg, "fl3d")
    val_df = load_split(cfg, "fl3d", "val")
    test_df = load_split(cfg, "fl3d", "test")
    val_y, test_y = val_df["label_index"].to_numpy(), test_df["label_index"].to_numpy()

    # ---- 1a. Pick the final model on VALIDATION macro F1 ----
    log("=== 1. Model choice (VALIDATION macro F1) ===")
    models, val_probs = {}, {}
    for name, file in MODEL_FILES.items():
        path = ROOT / "outputs" / "models" / file
        if not path.exists():
            log(f"{name:12s} not trained yet ({file} missing)")
            continue
        models[name] = tf.keras.models.load_model(path)
        val_probs[name] = models[name].predict(make_dataset(cfg, "fl3d", "val", batch_size=64), verbose=0)
        f1 = f1_score(val_y, val_probs[name].argmax(axis=1), average="macro", zero_division=0)
        log(f"{name:12s} val macro F1 {f1:.3f}")
        val_probs[name] = (val_probs[name], f1)
    if not models:
        raise SystemExit("No trained drowsiness model found. Run src.models.train_drowsiness first.")
    final_name = max(val_probs, key=lambda n: val_probs[n][1])
    final, (v_probs, _) = models[final_name], val_probs[final_name]
    log(f"Final model (chosen on validation): {final_name}")

    # ---- 1b. Threshold, chosen on validation ----
    log("\n=== Confidence threshold (chosen on VALIDATION) ===")
    conf = v_probs.max(axis=1)
    correct = v_probs.argmax(axis=1) == val_y
    log("threshold   kept   accuracy of kept")
    chosen = None
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        keep = conf >= t
        acc = correct[keep].mean() if keep.any() else 0.0
        log(f"{t:9.2f} {keep.mean():6.1%} {acc:12.1%}")
        if chosen is None and acc >= TARGET_ACCURACY:
            chosen = float(t)
    threshold = chosen if chosen is not None else 0.95
    log(f"Chosen threshold: {threshold}")
    (REPORTS / "uc04_decision.json").write_text(json.dumps(
        {"model": MODEL_FILES[final_name], "threshold": threshold,
         "target_accuracy": TARGET_ACCURACY, "chosen_on": "validation"}, indent=2))

    # ---- 2. Test set, run once ----
    log("\n=== 2. TEST set results (final model only) ===")
    probs = final.predict(make_dataset(cfg, "fl3d", "test", batch_size=64), verbose=0)
    pred = probs.argmax(axis=1)
    rep = classification_report(test_y, pred, output_dict=True, zero_division=0)
    log(f"{final_name}: accuracy {rep['accuracy']:.1%}   macro F1 {rep['macro avg']['f1-score']:.3f}")
    log("\nPer-class results:")
    log(classification_report(test_y, pred, zero_division=0,
                              target_names=[f"{n} ({STATES[i]})" for i, n in enumerate(names)]))

    log("Accuracy per test driver:")
    for driver, idx in test_df.groupby("driver_id").indices.items():
        log(f"  {driver}: {(pred[idx] == test_y[idx]).mean():.1%} on {len(idx)} frames")

    cm = confusion_matrix(test_y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"UC-04 {final_name}, test set (rows = true class)")
    fig.tight_layout()
    fig.savefig(FIGURES / "uc04_confusion_matrix.png", dpi=110)

    # ---- 3. The confidence rule on test ----
    kinds = np.array([decide(p, threshold)[0] for p in probs])
    spec, gen, unk = kinds == "specific", kinds == "drowsy_general", kinds == "unable"
    log(f"\n=== 3. Confidence rule on TEST (threshold {threshold}) ===")
    log(f"Specific state reported:  {spec.mean():.1%} of frames, correct {rate(spec, pred == test_y)}")
    log(f"General 'drowsy' only:    {gen.mean():.1%} of frames, truly drowsy {rate(gen, test_y != 0)}")
    log(f"Unable to determine:      {unk.mean():.1%} of frames")

    results = {"final_model": final_name, "threshold": threshold,
               "accuracy": round(rep["accuracy"], 4), "macro_f1": round(rep["macro avg"]["f1-score"], 4),
               "recall": {n: round(rep[str(i)]["recall"], 4) for i, n in enumerate(names)},               "rule": {"specific_share": round(float(spec.mean()), 4),
                        "general_share": round(float(gen.mean()), 4),
                        "unable_share": round(float(unk.mean()), 4)}}
    (REPORTS / "uc04_results.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "uc04_test_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\nSaved outputs/reports/uc04_results.json, uc04_test_report.txt, uc04_decision.json")
    log("Saved outputs/figures/uc04_confusion_matrix.png")


if __name__ == "__main__":
    main()
