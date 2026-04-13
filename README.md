Using Unbiased MIL with Top-K ranking loss


We utilise the UCF-Crime and XD-Violence datasets, converting each video into a bag of temporal segments. Instead of relying on frame-level annotations, we adopt a Multiple Instance Learning (MIL) framework. To reduce label noise inherent in weak supervision, we implement an Unbiased MIL approach that focuses on top-k anomalous segments, improving robustness and detection performance.



Raw Videos
   ↓
Feature Extraction (I3D / ResNet)
   ↓
Segment into T=32
   ↓
MIL (bag of segments)
   ↓
Unbiased MIL training


python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

cd src
python main.py



pip install torch numpy scikit-learn matplotlib

pip freeze > requirements.txt