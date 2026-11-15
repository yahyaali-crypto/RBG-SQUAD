# UC-03 Notes: Identify the Potential Distraction

Owner: Abbas Ankouni. Last updated November 15, 2026.

## Current status

UC-03 works from start to finish with the **baseline CNN**.
`predict_distraction(image_path)` in `src/models/predict.py` takes one image and returns
the category, behavior, confidence, and all 10 class probabilities.

## Current model: baseline CNN (trained from scratch)

- File: `outputs/models/baseline.keras` (1.2 MB, committed to GitHub)
- Trained 10 epochs, about 58 minutes on CPU. Validation loss was still dropping at the end.
- Validation accuracy 58.9%. Test accuracy 58.1%. Test macro F1 0.533.
- Accuracy per test driver: p014 70%, p042 63%, p047 56%, p056 56%, p072 30%.

## Confidence rule (BR-01 and BR-03)

Threshold **0.75**, chosen on validation (target: kept predictions at least 85% correct).

| Result on test | Share of images | How often it was right |
|---|---|---|
| Specific behavior | 20.3% | 88.4% correct (validation promised 85.9%) |
| "Distracted, behavior unclear" | 67.3% | 90.3% truly distracted |
| "Unable to determine" | 12.3% | |

The rule held up on new drivers, which MobileNetV2's did not.

## Strengths and weaknesses (test set)

- Strong: operating the radio (95% F1), texting right (78%), phone right (77%), drinking (77%)
- Weak: safe driving (36%), hair and makeup (7%), talking to passenger (0%, never correct)
- Biggest mix-up: 49% of safe driving was predicted as texting (left hand). This means false alarms.
- Driver p072 is hard for every model (30%). That driver has very few radio and reaching-behind images.

## Why not MobileNetV2 (yet)

- MobileNetV2 with fine-tuning: 67.8% validation, but only **44.5% test**.
- Training accuracy reached 96%, so it overfit to the 17 training drivers.
- On all 5 test drivers it over-predicted "hair and makeup" (26% to 85% of each driver's images).
- Its confidence was unreliable: the 0.85 threshold promised 85% accuracy, delivered 65% on test.

**Must disclose in the report:** validation selected MobileNetV2. The switch to the
baseline was made AFTER seeing test results.

---

## TO DO NEXT TIME

### 1. Tell the team how to use UC-03
- [ ] Send Aaron the function call and what it returns:
      `from src.models.predict import predict_distraction`
      `result = predict_distraction("uploaded_image.jpg")`
      Returns `category`, `behavior`, `confidence`, `status`, `probabilities`.
- [ ] Agree with Aaron on inputs: file path or raw bytes? JPEG only, or PNG too?
      (`load_image` currently only reads JPEG.)
- [ ] Tell Aaron his app does NOT need to copy the preprocessing. `predict_distraction` handles it.

### 2. Fix MobileNetV2 overfitting (Option B)
- [ ] Stronger augmentation and more dropout
- [ ] Unfreeze fewer layers, train the frozen head longer
- [ ] Choose settings with driver-level cross-validation, not one 4-driver validation set
- [ ] Use only validation results (never test) when deciding changes
- [ ] Final test run once, and disclose that test was seen before
- [ ] If it beats the baseline: change `MODEL_PATH` in `predict.py`, set `FINAL_MODEL`
      in `evaluate_uc03.py`, rerun the evaluation, update these notes

### 3. Optional: stronger baseline
- [ ] Baseline was still improving at epoch 10. Retrain longer (for example 20 epochs)
      and compare on validation.

### 4. UC-01 with Ali
- [ ] Combine `predict_distraction` with Ali's drowsiness model into one result:
      safe and alert, distracted, drowsy, or unable to determine
- [ ] Decide what happens when both models fire (distracted AND drowsy)

### 5. Report
- [ ] Baseline vs MobileNetV2 comparison and why driver-based splits matter
- [ ] Confusion matrix: `outputs/figures/uc03_confusion_matrix.png`
- [ ] Limitations: small test set (5 drivers), safe driving false alarms,
      talking to passenger never recognized, CPU-only training