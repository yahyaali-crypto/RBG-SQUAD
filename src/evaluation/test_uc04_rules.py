"""Checks the UC-04 confidence rule with made-up probabilities. No model, no TensorFlow needed.
Run:  python -m src.evaluation.test_uc04_rules
Order of probabilities: [alert, microsleep, yawning]
"""
import numpy as np

from src.models.predict_drowsiness import build_result

T = 0.75
cases = [
    ("clearly alert", [0.90, 0.05, 0.05], ("alert", None, "specific")),
    ("clear yawn", [0.05, 0.10, 0.85], ("drowsy", "Yawning", "specific")),
    ("clear microsleep is worded 'possible' (BR-02)", [0.10, 0.80, 0.10], ("drowsy", "Possible microsleep", "specific")),
    ("fatigue likely, sign unclear", [0.20, 0.40, 0.40], ("drowsy", None, "drowsy_general")),
    ("too unsure to say", [0.50, 0.30, 0.20], ("unable to determine", None, "unable")),
]

failures = 0
for name, probs, (category, behavior, status) in cases:
    r = build_result(np.array(probs), T)
    ok = (r["category"], r["behavior"], r["status"]) == (category, behavior, status) and 0 <= r["confidence"] <= 1
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {r['category']}, {r['behavior']}, confidence {r['confidence']}")

# The shape UC-01 (classify.py) expects from UC-04
r = build_result(np.array([0.05, 0.10, 0.85]), T)
ok = {"category", "behavior", "confidence"} <= set(r)
failures += not ok
print(f"[{'PASS' if ok else 'FAIL'}] result has category, behavior, confidence for UC-01")

# Try the full UC-01 combination too, if TensorFlow is installed (classify.py imports UC-03)
try:
    from src.models.classify import combine
except ImportError:
    print("[SKIP] UC-01 combine check (TensorFlow not installed here)")
else:
    d = {"category": "safe", "confidence": 0.9, "behavior": "Safe driving", "status": "specific"}
    out = combine(d, build_result(np.array([0.05, 0.10, 0.85]), T))
    ok = out["overall"] == "drowsy" and out["behavior"] == "Yawning"
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] UC-01 combine: yawning is reported as drowsy -> {out['message']}")

print("\nAll UC-04 rules passed." if failures == 0 else f"\n{failures} rule(s) FAILED.")
