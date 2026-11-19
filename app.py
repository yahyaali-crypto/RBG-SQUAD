from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from PIL import Image, UnidentifiedImageError

# set up titles
st.set_page_config(page_title="Driver Behavior Classifier")
st.title("Driver Behavior Classifier")
st.write("Upload a driver image to estimate their behavior.")

# initialize file image uploader
uploaded = st.file_uploader(
  "Choose an image",
  # formats that the user can upload
  type=["jpg","jpeg","png"]
)

if uploaded is not None:
  # upload image
  try:
    # convert image to RGB
    image = Image.open(uploaded).convert("RGB")
    st.image(image,caption="Uploaded image", width=450)
  except (UnidentifiedImageError, OSError):
    st.error("This file couldn't be opened as an image")
    st.stop()
  
  # trigger a button to analyze and classify the image once uploaded
  if st.button("Analyze image", type="primary"):
    # obtain the model path
    model_path = (
      Path(__file__).resolve().parent
      # this is the correct model path with the file
      / "outputs/models/baseline.keras"
    )

    # check if model path doesn't exist else start classifying it
    if not model_path.exists():
      st.error(
        "Model missing. Place baseline.keras "
        "in outputs/models/ before analyzing."
      )
      st.stop()
    try:
      # import the predict_distraction model
      from src.models.predict import predict_distraction

      with st.spinner("Analyzing image..."):
        with TemporaryDirectory() as folder:
          image_path = Path(folder) / "upload.jpg"
          image.save(image_path, format="JPEG")
          result = predict_distraction(image_path)

        if result["status"] == "unable":
          st.warning("Unable to determine the driver's behavior.")
        else:
          st.subheader(result["behavior"])
          st.metric(
            "Confidence",
            f'{result["confidence"]:.1%}'
          )
        with st.expander("View prediction details"):
          st.json(result)
    except Exception:
      st.error(
        "Prediction failed. Check your model, dependencies, "
        "and configuration."
      )

st.caption("Predictions are estimates, not guaranteed conclusions")