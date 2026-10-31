"""Quick check that every required library imports and reports its version."""
import sys
from importlib.metadata import version

print(f"Python: {sys.version.split()[0]}")
print(f"Running from: {sys.executable}")

packages = ["tensorflow", "numpy", "pandas", "matplotlib",
            "pillow", "scikit-learn", "kaggle", "pyyaml"]
for pkg in packages:
    try:
        print(f"{pkg}: {version(pkg)}")
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED ({e})")

import tensorflow as tf
print("TensorFlow loads OK. GPUs found:", tf.config.list_physical_devices("GPU"))
