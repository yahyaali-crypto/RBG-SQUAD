"""Checks the UC-01 decision rules with made-up model outputs. No model needed.
Run:  python -m src.evaluation.test_uc01_rules
"""
from src.models.classify import combine


def D(category, conf, behavior=None, status="specific"):
    return {"category": category, "confidence": conf, "behavior": behavior, "status": status}


def Z(category, conf, behavior=None):
    return {"category": category, "confidence": conf, "behavior": behavior}


cases = [
    ("safe, drowsiness not checked", D("safe", 0.9, "Safe driving"), None, "safe and alert"),
    ("safe and alert", D("safe", 0.9, "Safe driving"), Z("alert", 0.8), "safe and alert"),
    ("texting only", D("distracted", 0.91, "Texting (right hand)"), Z("alert", 0.9), "distracted"),
    ("distracted, behavior unclear",
     D("distracted", 0.8, "Distracted (specific behavior unclear)", status="distracted_general"),
     None, "distracted"),
    ("yawning only", D("safe", 0.9, "Safe driving"), Z("drowsy", 0.84, "Yawning"), "drowsy"),
    ("both, drowsy more confident", D("distracted", 0.7, "Drinking"), Z("drowsy", 0.9, "Yawning"), "drowsy"),
    ("both, distracted more confident", D("distracted", 0.95, "Drinking"), Z("drowsy", 0.8, "Yawning"), "distracted"),
    ("distraction model unsure", D("unable to determine", 0.5, status="unable"), Z("alert", 0.9),
     "unable to determine"),
    ("safe but face not visible", D("safe", 0.9, "Safe driving"), Z("unable to determine", 0.4),
     "unable to determine"),
]

failures = 0
for name, d, z, expected in cases:
    r = combine(d, z)
    ok = r["overall"] == expected and r["confidence"] is not None and "estimate" in r["message"]
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {r['overall']}\n         {r['message']}")

general = combine(cases[3][1], None)
ok = general["behavior"] is None
failures += not ok
print(f"[{'PASS' if ok else 'FAIL'}] BR-03: an unclear behavior is not reported as specific")

print("\nAll UC-01 rules passed." if failures == 0 else f"\n{failures} rule(s) FAILED.")