"""
UC-01 Classify the Driver's Behavior.

Combines UC-03 (distraction) and UC-04 (drowsiness) into ONE overall result:
safe and alert, distracted, drowsy, or unable to determine (BR-01).

In code:       from src.models.classify import classify_driver
               classify_driver("path/to/image.jpg")
Command line:  python -m src.models.classify path/to/image.jpg
"""
import json
import sys
from pathlib import Path

from src.models.predict import predict_distraction

ESTIMATE_NOTE = "This is an estimate, not a guaranteed result."  # BR-05


def get_drowsiness(image_path):
    """UC-04 result from Ali's model, or None if it is not available yet.

    Ali's file src/models/predict_drowsiness.py must provide
    predict_drowsiness(image_path) returning:
      {"category": "alert" | "drowsy" | "unable to determine",
       "behavior": "Yawning" | "Possible microsleep" | None,
       "confidence": number from 0 to 1}
    """
    try:
        from src.models.predict_drowsiness import predict_drowsiness
    except ImportError:
        return None
    return predict_drowsiness(image_path)


def _result(overall, source):
    behavior = source.get("behavior")
    if source.get("status") == "distracted_general":
        behavior = None  # BR-03: general category only, no guessing
    return {"overall": overall, "behavior": behavior, "confidence": source["confidence"]}


def _message(r):
    """Plain-language result for a non-technical user (NFR-02)."""
    pct = f"{r['confidence']:.0%} confidence"
    if r["overall"] == "unable to determine":
        text = "Unable to determine the driver's behavior from this image. Try a clearer image of the driver."
    elif r["overall"] == "safe and alert":
        text = f"The driver appears safe and alert ({pct})."
    elif r["behavior"]:
        text = f"{r['behavior']} detected ({pct})."
    else:
        text = f"The driver appears {r['overall']}, but the specific behavior is unclear ({pct})."
    if r.get("also_detected"):
        text += f" The driver may also be {r['also_detected']}."
    if not r["drowsiness_checked"]:
        text += " Drowsiness was not checked."
    return f"{text} {ESTIMATE_NOTE}"


def combine(distraction, drowsiness):
    """The UC-01 decision rules. Pure logic with no model, so it is easy to test."""
    d_cat = distraction["category"]
    z_cat = drowsiness["category"] if drowsiness else "not checked"
    is_distracted = d_cat == "distracted"
    is_drowsy = z_cat == "drowsy"

    if is_distracted and is_drowsy:
        # Both found: report the more confident one, mention the other
        main, other = (distraction, "distracted"), (drowsiness, "drowsy")
        if drowsiness["confidence"] > distraction["confidence"]:
            main, other = other, main
        result = _result(main[1], main[0])
        result["also_detected"] = other[1]
    elif is_drowsy:
        result = _result("drowsy", drowsiness)
    elif is_distracted:
        result = _result("distracted", distraction)
    elif d_cat == "safe" and z_cat in ("alert", "not checked"):
        conf = distraction["confidence"]
        if drowsiness:
            conf = min(conf, drowsiness["confidence"])  # only as sure as the less sure model
        result = {"overall": "safe and alert", "behavior": None, "confidence": round(conf, 3)}
    else:
        result = {"overall": "unable to determine", "behavior": None,
                  "confidence": distraction["confidence"]}

    result["drowsiness_checked"] = drowsiness is not None
    result["message"] = _message(result)
    return result


def classify_driver(image_path):
    """Full UC-01 flow, including the exception flows 6.1 and 6.2."""
    path = Path(image_path)
    if not path.is_file():  # 6.1 No image is available
        return {"overall": None, "error": "no_image",
                "message": "No image found. Please upload a driver image first."}
    try:
        distraction = predict_distraction(path)
        drowsiness = get_drowsiness(path)
    except Exception as e:  # 6.2 AI model cannot be used
        return {"overall": None, "error": "model_unavailable",
                "message": "The analysis cannot be done right now. Please try again later.",
                "details": str(e)}
    result = combine(distraction, drowsiness)
    result["details"] = {"distraction": distraction, "drowsiness": drowsiness}
    return result


if __name__ == "__main__":
    print(json.dumps(classify_driver(sys.argv[1]), indent=2))