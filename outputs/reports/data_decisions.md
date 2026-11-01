# Phase 1 Data Decisions

Driver Behavior Classification System (DBCS), RBG-SQUAD, CIS 479
Prepared by Abbas Ankouni (data preparation). Last updated October 31, 2026.

This note records what we found in each dataset, how we decided to use them
together, and every preprocessing choice. All settings live in
`configs/preprocessing.yaml`. All checks are reproducible with the scripts in `src/data/`.

## 1. Datasets as found

| | State Farm Distracted Driver Detection | FL3D (Frame-Level Driver Drowsiness Detection) |
|---|---|---|
| Use case | UC-03 Identify the Potential Distraction | UC-04 Detect Drowsiness |
| Labeled images used | 22,424 | 53,316 (53,331 minus 15 excluded) |
| Labels | 10 behaviors, c0 to c9 | 3 states: alert, microsleep, yawning |
| Image content | Whole cabin, side view from the passenger side | Close face crop |
| Image size | All 640x480 | Varies, about 107-304 wide and 156-411 tall |
| Lighting | Daytime | Night |
| People | 26 drivers, ID given (`subject`) | 44 videos from NITYMED. No driver ID given |
| Data quality | 0 corrupt, 0 exact duplicates, CSV matches folders | 0 corrupt, 0 exact duplicates, JSON matches files |
| License | Kaggle competition rules | CC BY-SA 4.0. Cite FL3D and NITYMED |

Class balance: State Farm is mildly imbalanced (largest class about 1.3x the
smallest). FL3D is imbalanced: alert 73%, microsleep 17%, yawning 10%.

## 2. How the two datasets are used together

**Decision: two separate pipelines and two separate models. Results are combined
only at the output stage (UC-01).**

We did not merge them into one dataset because:

- No image has both kinds of label, and a driver can be texting and drowsy at the same time.
- The images differ in content (cabin vs face), size, and lighting. A merged model
  would likely learn "night face crop vs day cabin photo" instead of driver behavior.
- Separate models are simpler to validate and explain.

**Limitation to state in the report:** the data does not support running both
models on one image. The drowsiness model expects a close face crop, which the
State Farm side camera does not produce. UC-01 can combine the two outputs, but
each model is only reliable on images like the ones it was trained on.

## 3. Leakage prevention

Images from the same person are near-duplicates (consecutive frames), so every
split is **driver-disjoint**: all images of one driver are in exactly one split.

- **State Farm:** split on the `subject` column.
- **FL3D:** the dataset has no driver ID. The official splits are not usable:
  official val and test come from the same 5 videos, and the author states 2 of the
  5 holdout drivers also appear in training. The team assigned driver IDs by
  reviewing faces (one-frame sheet, then a 6-frame sheet for uncertain pairs).
  Result: 44 videos map to 19 drivers (`data/manifests/fl3d_video_driver_map.csv`).
  Cross-checks: 19 is within NITYMED's 21 drivers, and the grouping matches the
  author's description of which holdout drivers are new.

## 4. Splits (seed 42, target 70/15/15 of images)

The split script tries 5,000 driver assignments and keeps the one closest to the
target sizes and class mix. Every class is present in every split.

| | Train | Val | Test |
|---|---|---|---|
| State Farm images | 15,606 (69.6%) | 3,376 (15.1%) | 3,442 (15.3%) |
| State Farm drivers | 17 | 4 | 5 |
| FL3D frames | 37,144 (69.7%) | 8,014 (15.0%) | 8,158 (15.3%) |
| FL3D drivers | 11 | 5 | 3 |

State Farm class shares stay within about 1 to 2 percentage points across splits.
FL3D varies more (yawning is 7.8% of val and 12.0% of test) because there are only
19 drivers and some contribute thousands of frames.

## 5. Preprocessing

| Setting | Value | Reason |
|---|---|---|
| Input size | 224x224 | Standard for pretrained Keras CNNs |
| Resize | Pad to square, then resize | Keeps shapes true. No stretching |
| Color | RGB, 3 channels | Both datasets are RGB |
| Scaling | Pixel / 255 | Simple default. Revisit if the chosen pretrained model needs its own scaling |
| Storage | Resize on the fly from raw files | No duplicate images. Split CSVs are the processed dataset |
| Augmentation | Training split only | Val and test stay untouched |
| State Farm flip | **Never** | Flipping swaps hands and changes the class (c1 and c3, c2 and c4) |
| FL3D flip | Allowed | The label does not depend on side |
| Class imbalance | State Farm: none. FL3D: class weights | Applied at training time |

Verified with `src/data/verify_data.py`: all paths exist, splits are disjoint,
images load through TensorFlow at 224x224 with values 0 to 1, and raw files are
unchanged (39 of 39 checks passed).

## 6. Excluded data

- **State Farm `imgs/test` (79,726 images):** Kaggle competition images with no labels. Not usable for training or evaluation.
- **15 FL3D frames:** the label differs between `annotations_all.json` and the
  per-video `annotations_final.json` (10 frames in P1042762, 2 in P1043078 where one
  file has no label, 2 in P1043081, 1 in P1043106). The correct label cannot be
  determined, so they are excluded. The list is in the config.

## 7. Open questions and notes for the team

- **Aaron (UC-02):** the app must prepare uploaded images exactly as above
  (pad to square, 224x224, scale 0 to 1). Otherwise predictions will be off.
- **Ali (UC-04):** FL3D splits, driver map, and config are ready.
- **UC-01:** "unable to determine" (BR-01) and "general category only" (BR-03) have no
  training class. They will come from a confidence threshold, chosen after training.
- **Small test sets:** with 5 State Farm and 3 FL3D test drivers, results can depend
  on which drivers are in test. Report this as a limitation.
- **FL3D driver IDs are human judgments** from face review. They are documented
  in the `notes` column of the driver map.
- **Normalization** may change once a pretrained model is chosen.

## 8. Sources

- State Farm Distracted Driver Detection, Kaggle competition.
- FL3D, Kaggle dataset by matjazmuc (description and license read from `data/raw/fl3d/dataset-metadata.json`).
- NITYMED, IEEE DataPort: https://ieee-dataport.org/documents/nitymed
- N. Petrellis et al., "High Speed Implementation of the Deformable Shape Tracking Face Alignment Algorithm", Euromicro DSD 2021 (cited by FL3D as the NITYMED source).
