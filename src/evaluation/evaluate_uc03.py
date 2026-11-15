"""
UC-03 final evaluation.
  1. Picks the confidence threshold on the VALIDATION split
  2. Evaluates both models ONCE on the TEST split
Run:  python -m src.evaluation.evaluate_uc03
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from src.data.loaders import ROOT, class_names, load_config, load_split, make_dataset
from src.models.predict import BEHAVIORS, decide

MODELS = {"baseline": "baseline.keras", "mobilenetv2": "mobilenetv2_finetuned.keras"}
FINAL_MODEL = "baseline"
TARGET_ACCURACY = 0.85  # kept predictions should be right at least this often
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
    names = class_names(cfg, "state_farm")
    val_y = load_split(cfg, "state_farm", "val")["label_index"].to_numpy()
    test_y = load_split(cfg, "state_farm", "test")["label_index"].to_numpy()
    final = tf.keras.models.load_model(ROOT / "outputs" / "models" / MODELS[FINAL_MODEL])

    # ---- 1. Threshold, chosen on validation ----
    log("=== 1. Confidence threshold (chosen on VALIDATION) ===")
    val_probs = final.predict(make_dataset(cfg, "state_farm", "val", batch_size=64), verbose=0)
    conf = val_probs.max(axis=1)
    correct = val_probs.argmax(axis=1) == val_y
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
    (REPORTS / "uc03_decision.json").write_text(json.dumps(
        {"model": MODELS[FINAL_MODEL], "threshold": threshold,
         "target_accuracy": TARGET_ACCURACY, "chosen_on": "validation"}, indent=2))

    # ---- 2. Test set, run once ----
    log("\n=== 2. TEST set results ===")
    results = {"threshold": threshold}
    for name, file in MODELS.items():
        model = final if name == FINAL_MODEL else tf.keras.models.load_model(ROOT / "outputs" / "models" / file)
        probs = model.predict(make_dataset(cfg, "state_farm", "test", batch_size=64), verbose=0)
        pred = probs.argmax(axis=1)
        rep = classification_report(test_y, pred, output_dict=True, zero_division=0)
        results[name] = {"accuracy": round(rep["accuracy"], 4), "macro_f1": round(rep["macro avg"]["f1-score"], 4)}
        log(f"{name:12s} accuracy {rep['accuracy']:.1%}   macro F1 {rep['macro avg']['f1-score']:.3f}")
        if name == FINAL_MODEL:
            final_probs, final_pred = probs, pred

    log("\nPer-class results (final model):")
    log(classification_report(test_y, final_pred, zero_division=0,
                              target_names=[f"{c} {BEHAVIORS[c]}" for c in names]))

    cm = confusion_matrix(test_y, final_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"UC-03 {FINAL_MODEL}, test set (rows = true class)")
    fig.tight_layout()
    fig.savefig(FIGURES / "uc03_confusion_matrix.png", dpi=110)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], names[i], names[j]) for i in range(10) for j in range(10)), reverse=True)[:5]
    log("Most common mix-ups (true -> predicted, share of the true class):")
    for share, a, b in pairs:
        log(f"  {BEHAVIORS[a]} -> {BEHAVIORS[b]}: {share:.0%}")

    # ---- 3. The confidence rule on test ----
    kinds = np.array([decide(p, threshold)[0] for p in final_probs])
    spec, gen, unk = kinds == "specific", kinds == "distracted_general", kinds == "unable"
    log(f"\n=== 3. Confidence rule on TEST (threshold {threshold}) ===")
    log(f"Specific behavior reported:  {spec.mean():.1%} of images, correct {rate(spec, final_pred == test_y)}")
    log(f"General 'distracted' only:   {gen.mean():.1%} of images, truly distracted {rate(gen, test_y != 0)}")
    log(f"Unable to determine:         {unk.mean():.1%} of images")
    results["rule"] = {"specific_share": round(float(spec.mean()), 4),
                       "general_share": round(float(gen.mean()), 4),
                       "unable_share": round(float(unk.mean()), 4)}

    (REPORTS / "uc03_results.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "uc03_test_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\nSaved outputs/reports/uc03_results.json, uc03_test_report.txt, uc03_decision.json")
    log("Saved outputs/figures/uc03_confusion_matrix.png")


if __name__ == "__main__":
    main()