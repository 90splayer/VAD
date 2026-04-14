✅ STEP 1: Get the datasets
UCF-Crime
Download from official site or Kaggle
Contains long videos (normal + anomaly)
XD-Violence
Same idea (violence detection videos)

🔥 STEP 2 (CRITICAL): Use pre-extracted features
Option A (BEST — recommended)

Use:

I3D features
C3D features
ResNet features

👉 Many GitHub repos already provide .npy features

Your goal format:

Each video becomes: (features.shape) = (num_frames, 1024)


🚀 STEP 3: Organise your dataset

In VS Code:

data/
├── normal/
│   ├── video1.npy
│   ├── video2.npy
│
├── anomaly/
│   ├── video3.npy
│   ├── video4.npy

main.py = orchestration (controls everything)
Load datasets ✅
Combine datasets ✅
Split data ✅
Call training ✅
train.py = model logic only
Training loop
Evaluation


model.py
AttentionMIL
loss.py
unbiased_mil_loss
utils.py
get_batches


train.py
✅ Training
Uses Unbiased MIL loss
Handles normal vs abnormal batches
Tracks loss per epoch
✅ Validation
Computes AUC (important for your report)
Saves best model
✅ Testing
Loads best model
Plots anomaly scores
✅ Visualisation
Shows:
Normal video → flat curve
Anomaly video → spikes

| File         | Responsibility            |
| ------------ | ------------------------- |
| `main.py`    | Load + combine datasets ✅ |
| `dataset.py` | Define `load_dataset()`   |
| `train.py`   | Train + evaluate          |
| `model.py`   | MIL model                 |
| `loss.py`    | Unbiased MIL loss         |


vad_project/
│
├── data/
│   ├── normal/
│   ├── anomaly/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── utils.py
│
├── requirements.txt
└── main.py
