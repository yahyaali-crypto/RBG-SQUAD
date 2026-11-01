# RBG-SQUAD: Driver Behavior Classification System

CIS 479 Introduction to Artificial Intelligence, University of Michigan-Dearborn.

A CNN-based system that looks at a driver image and estimates whether the driver
is safe and alert, distracted (State Farm dataset), or drowsy (FL3D dataset).
Results are estimates, not guaranteed conclusions.

## Setup (Windows)

Requires **Python 3.12** (TensorFlow does not support 3.14 yet) and Git.

```powershell
git clone https://github.com/yahyaali-crypto/RBG-SQUAD
cd RBG-SQUAD
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\check_env.py
```

Keep the project outside OneDrive (for example `C:\dev\RBG-SQUAD`).

## Get the data

Datasets are not in Git. Each person downloads their own copy.

1. Create a Kaggle API token at kaggle.com/settings and save it to
   `%USERPROFILE%\.kaggle\access_token`. Never commit it.
2. Accept the rules on the State Farm competition page (Data tab).
3. Download and extract:

```powershell
kaggle competitions download -c state-farm-distracted-driver-detection -p data\raw\state_farm
tar -xf data\raw\state_farm\state-farm-distracted-driver-detection.zip -C data\raw\state_farm

kaggle datasets metadata matjazmuc/frame-level-driver-drowsiness-detection-fl3d -p data\raw\fl3d
kaggle datasets download matjazmuc/frame-level-driver-drowsiness-detection-fl3d -p data\raw\fl3d
tar -xf data\raw\fl3d\frame-level-driver-drowsiness-detection-fl3d.zip -C data\raw\fl3d
```

## Reproduce the data preparation

The manifests and splits are already committed, so you normally only need the last command.

```powershell
python src\data\inventory_state_farm.py   # checks State Farm, writes its inventory
python src\data\inventory_fl3d.py         # checks FL3D, writes its inventory
python src\data\make_splits.py            # driver-disjoint train/val/test splits
python src\data\verify_data.py            # should end with ALL CHECKS PASSED
```

## Project layout

```
configs/preprocessing.yaml   every data preparation setting (size, labels, splits, exclusions)
data/raw/                    original datasets, never modified, not in Git
data/manifests/              inventories, FL3D driver map, split CSVs (in Git)
src/data/                    data inspection, splitting, and verification scripts
src/models/                  model code (not started)
outputs/figures/             sample grids and review sheets
outputs/reports/             data_decisions.md and verification results
```

Read `outputs/reports/data_decisions.md` for why each data decision was made.

## Team and use cases

| Use case | Owner |
|---|---|
| UC-01 Classify the Driver's Behavior | Ali Yahya, Abbas Ankouni |
| UC-02 Upload Driver Image | Aaron Amano |
| UC-03 Identify the Potential Distraction | Abbas Ankouni |
| UC-04 Detect Drowsiness | Ali Yahya |
| UC-05 Display the Prediction and Confidence Score | Aaron Amano |

## Data sources and licenses

- State Farm Distracted Driver Detection (Kaggle competition rules)
- FL3D on Kaggle (CC BY-SA 4.0), built from NITYMED. Cite both.
