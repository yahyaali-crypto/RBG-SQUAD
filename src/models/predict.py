"""
UC-03 prediction: one driver image in, behavior + confidence out.

In code:       from src.models.predict import predict_distraction
               predict_distraction("path/to/image.jpg")
Command line:  python -m src.models.predict path/to/image.jpg
"""
import json
import sys
from functools import lru_cache

import numpy as np
import tensorflow as tf

from src.data.loaders import ROOT, load_config, load_image

MODEL_PATH = ROOT / "outputs" / "models" / "baseline.keras"
DECISION_PATH = ROOT / "outputs" / "reports" / "uc03_decision.json"
CODES = [f"c{i}" for i in range(10)]
BEHAVIORS = {
    "c0": "Safe driving",
    "c1": "Texting (right hand)",
    "c2": "Talking on the phone (right hand)",
    "c3": "Texting (left hand)",
    "c4": "Talking on the phone (left hand)",
    "c5": "Operating the radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair and makeup",
    "c9": "Talking to passenger",
}


def decide(probs, threshold):
    """The confidence rule (BR-01 and BR-03).

    - Top class confident enough   -> report that specific behavior
    - Not, but 'some distraction' (all classes except c0 added up) is
      confident enough             -> report 'distracted', behavior unclear
    - Otherwise                    -> 'unable to determine'
    """
    top = int(np.argmax(probs))
    if probs[top] >= threshold:
        return "specific", top
    if 1.0 - probs[0] >= threshold:
        return "distracted_general", None
    return "unable", None


@lru_cache(maxsize=1)
def _load():
    """Load the model and threshold once, then reuse them for every call."""
    model = tf.keras.models.load_model(MODEL_PATH)
    threshold = json.loads(DECISION_PATH.read_text())["threshold"]
    return model, threshold, load_config()


def predict_distraction(image_path):
    model, threshold, cfg = _load()
    img = load_image(str(image_path), cfg)  # same preparation as training
    probs = model.predict(img[None, ...], verbose=0)[0]
    kind, idx = decide(probs, threshold)
    top = int(np.argmax(probs))

    result = {"status": kind,
              "probabilities": {c: round(float(p), 3) for c, p in zip(CODES, probs)}}
    if kind == "specific":
        result.update(category="safe" if top == 0 else "distracted",
                      behavior=BEHAVIORS[CODES[top]], confidence=round(float(probs[top]), 3))
    elif kind == "distracted_general":
        result.update(category="distracted", behavior="Distracted (specific behavior unclear)",
                      confidence=round(float(1.0 - probs[0]), 3))
    else:
        result.update(category="unable to determine", behavior=None,
                      confidence=round(float(probs[top]), 3))
    return result


if __name__ == "__main__":
    print(json.dumps(predict_distraction(sys.argv[1]), indent=2))