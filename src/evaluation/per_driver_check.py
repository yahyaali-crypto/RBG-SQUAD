"""Diagnosis only: test accuracy per driver for both models."""
import tensorflow as tf

from src.data.loaders import ROOT, load_config, load_split, make_dataset

cfg = load_config()
df = load_split(cfg, "state_farm", "test")
for name in ["baseline", "mobilenetv2_finetuned"]:
    model = tf.keras.models.load_model(ROOT / "outputs" / "models" / f"{name}.keras")
    probs = model.predict(make_dataset(cfg, "state_farm", "test", batch_size=64), verbose=0)
    df[name] = probs.argmax(axis=1) == df["label_index"]
    if name == "mobilenetv2_finetuned":
        df["mobilenet_says_c8"] = probs.argmax(axis=1) == 8

cols = ["baseline", "mobilenetv2_finetuned", "mobilenet_says_c8"]
print(df.groupby("subject")[cols].mean().round(2))
print("\nImages per driver:")
print(df["subject"].value_counts().sort_index())