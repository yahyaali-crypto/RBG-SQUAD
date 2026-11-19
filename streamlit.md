# Sample Input/Output
sample input:
```zsh
python -m src.models.predict data/raw/state_farm/imgs/test/img_1.jpg
```

output:
```
{
  "status": "specific",
  "probabilities": {
    "c0": 0.001,
    "c1": 0.0,
    "c2": 0.0,
    "c3": 0.025,
    "c4": 0.006,
    "c5": 0.966,
    "c6": 0.0,
    "c7": 0.0,
    "c8": 0.0,
    "c9": 0.002
  },
  "category": "distracted",
  "behavior": "Operating the radio",
  "confidence": 0.966
}
```

# Install streamlit
1. create a virtual environment
2. enter the virtual environemnt
3. run *pip install -r requirements.txt* again since streamlit is added in the txt file

# Run streamlit
**make sure you're in the virtual environment**

how to run the app:
```zsh
python -m streamlit run app.py
```
you should see something like this:
```
2026-11-19 12:19:06.468 Uvicorn server started on :::8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://141.215.196.155:8501

  Help agents write better Streamlit apps?
  Install the official Streamlit skills by running streamlit skills in your terminal.

  For better performance, install the Watchdog module:

  $ xcode-select --install
  $ pip install watchdog
```

## Interacting with the app
1. enter the local URL
2. upload an image as jpg, jpeg, or png
3. click on Analyze Image and wait