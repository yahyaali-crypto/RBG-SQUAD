"""
UC-04 prediction: one driver face image in, drowsiness state + confidence out.

UC-01 (src/models/classify.py) calls predict_drowsiness() and expects this shape:
  {"category": "alert" | "drowsy" | "unable to determine",
   "behavior": "Yawning" | "Possible microsleep" | None,
   "confidence": number from 0 to 1}

In code:       from src.models.predict_drowsiness import predict_drowsiness
               predict_drowsiness("path/to/face_image.jpg")
Command line:  python -m src.models.predict_drowsiness path/to/face_image.jpg

The model was trained on close face crops (FL3D). Cabin photos like State Farm's
are outside what it was trained on. See outputs/reports/uc04_notes.md.
"""
import json
import sys
from functools import lru_cache

import numpy as np

# Same order as configs/preprocessing.yaml, fl3d labels
ALERT, MICROSLEEP, YAWNING = 0, 1, 2
STATES = {ALERT: "Alert", MICROSLEEP: "Possible microsleep", YAWNING: "Yawning"}  # BR-02: microsleep is always 'possible'


def decide(probs, threshold):
    """The confidence rule for UC-04 (BR-03 and the 'confidence too low' flow).

    - Top state confident enough  -> report that state
    - Not, but 'some fatigue sign' (yawning + microsleep added up) is
      confident enough            -> report 'drowsy', which sign is unclear
    - Otherwise                   -> 'unable to determine'
    """
    top = int(np.argmax(probs))
    if probs[top] >= threshold:
        return "specific", top
    if 1.0 - probs[ALERT] >= threshold:
        return "drowsy_general", None
    return "unable", None


def build_result(probs, threshold):
    """Turn model probabilities into the result UC-01 expects. No model needed here."""
    kind, idx = decide(probs, threshold)
    top = int(np.argmax(probs))
    result = {"status": kind,
              "probabilities": {STATES[i]: round(float(p), 3) for i, p in enumerate(probs)}}
    if kind == "specific" and idx == ALERT:
        result.update(category="alert", behavior=None, confidence=round(float(probs[ALERT]), 3))
    elif kind == "specific":
        result.update(category="drowsy", behavior=STATES[idx], confidence=round(float(probs[idx]), 3))
    elif kind == "drowsy_general":
        result.update(category="drowsy", behavior=None, confidence=round(float(1.0 - probs[ALERT]), 3))
    else:
        result.update(category="unable to determine", behavior=None, confidence=round(float(probs[top]), 3))
    return result


@lru_cache(maxsize=1)
def _load():
    """Load the model and threshold once, then reuse them for every call."""
    import tensorflow as tf  # imported here so the rule above can be tested without TensorFlow
    from src.data.loaders import ROOT, load_config

    decision = json.loads((ROOT / "outputs" / "reports" / "uc04_decision.json").read_text())
    model = tf.keras.models.load_model(ROOT / "outputs" / "models" / decision["model"])
    return model, decision["threshold"], load_config()


def predict_drowsiness(image_path):
    from src.data.loaders import load_image

    model, threshold, cfg = _load()
    img = load_image(str(image_path), cfg)  # same preparation as training
    probs = model.predict(img[None, ...], verbose=0)[0]
    return build_result(probs, threshold)


if __name__ == "__main__":
    print(json.dumps(predict_drowsiness(sys.argv[1]), indent=2))
