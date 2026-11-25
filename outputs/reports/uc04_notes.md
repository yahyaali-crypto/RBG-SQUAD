# UC-04 Notes: Detect Drowsiness

Owner: Ali Yahya. Started November 24, 2026.

## Current status (November 25, 2026)

UC-04 is trained and evaluated on FL3D. Final model: MobileNetV2 with the last 30 layers
fine-tuned (`outputs/models/drowsiness_mobilenetv2_finetuned.keras`, not in Git, see below).
The model was chosen on validation macro F1 (baseline CNN 0.864, MobileNetV2 0.921).

Test results (3 held-out drivers, 8158 frames, test set used once):

- Accuracy 91.4%, macro F1 0.879.
- Recall: alert 0.97, microsleep 0.67, yawning 0.98. Microsleep is the weak class.
- Per driver accuracy: d02 94.7%, d06 88.9%, d17 89.1%.
- Confidence rule at threshold 0.85 (chosen on validation, target 97% correct on kept frames):
  specific state on 90.1% of frames (95.2% correct), general drowsy on 0.6% (97.9% truly drowsy),
  unable to determine on 9.3%.

Saved in `outputs/reports/`: `uc04_decision.json`, `uc04_results.json`, `uc04_test_report.txt`,
and `outputs/figures/uc04_confusion_matrix.png`.

Model files are ignored by Git (`outputs/models/`). Share the `.keras` file with Abbas outside
Git (Drive or the team chat), or he can rerun training with the commands below.

| Piece | File | State |
|---|---|---|
| Training | `src/models/train_drowsiness.py` | Run, both models |
| Final evaluation | `src/evaluation/evaluate_uc04.py` | Run |
| Prediction (what UC-01 calls) | `src/models/predict_drowsiness.py` | Ready, needs the model file and `uc04_decision.json` |
| Rule tests | `src/evaluation/test_uc04_rules.py` | Passes |

## How UC-04 follows UC-03 (Abbas)

Same structure on purpose: same loaders, same model builders, same threshold idea, same
output shape, so UC-01 can combine them.

- Classes: alert, microsleep, yawning (index 0, 1, 2 in `configs/preprocessing.yaml`).
- FL3D is imbalanced (alert about 73%), so training uses the class weights from `loaders.class_weights`.
- Confidence rule: top state at or above the threshold gives that state. Otherwise, if
  yawning plus microsleep together reach the threshold, the result is "drowsy" with no
  specific sign. Otherwise "unable to determine".
- Microsleep is always worded "Possible microsleep" (BR-02).
- `evaluate_uc04.py` picks the final model by **validation macro F1** and picks the
  threshold on validation. The test split is used once. This avoids the test-peeking that
  UC-03 had to disclose.

## To run (needs the FL3D data downloaded)

```
python -m src.models.train_drowsiness --model baseline --epochs 1 --limit 256   # smoke test first
python -m src.models.train_drowsiness --model baseline --epochs 10
python -m src.models.train_drowsiness --model mobilenetv2 --epochs 10 --fine_tune 5
python -m src.evaluation.evaluate_uc04
python -m src.evaluation.test_uc04_rules
```

Training time is a guess until the smoke test runs. UC-03's baseline took about 58 minutes
for 10 epochs on CPU with 15,606 images. FL3D train has 37,144 frames, so expect longer.

## Open issues to decide with Abbas (UC-01)

1. **Face crops vs cabin photos.** FL3D images are close face crops. State Farm images
   are whole-cabin photos. `classify_driver` currently sends the same image to both
   models, so the drowsiness model would see a cabin photo it was never trained on. Options:
   - Detect and crop the face first (OpenCV), and treat "no face found" as drowsiness *not checked*.
   - Or only run UC-04 when the uploaded image is a face crop, and say so in the UI.
2. **"Unable to determine" can overrule a good safe result.** In `combine`, a safe
   distraction result plus a drowsiness result of "unable to determine" gives an overall
   "unable to determine". If the face is not visible, that is what would happen for any
   cabin photo. Consider treating "face not visible" as "not checked" so a clear safe result survives.
3. **PNG uploads.** UC-02 accepts JPG and PNG, but `load_image` uses `decode_jpeg`, which
   only reads JPEG (Abbas noted this). Switching to `tf.io.decode_image` would cover both.
4. **Night vs day.** FL3D is night footage and State Farm is daytime. Report this as a limit.

## Report notes for UC-04

- Report macro F1 and per-class recall, not just accuracy, because of the 73% alert share.
- Report accuracy per test driver. There are only 3 FL3D test drivers.
- Driver IDs for FL3D were assigned by the team from face review, so grouping errors are possible.
- Disclose that the final model and threshold were chosen on validation only.
