# train.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model import AttentionMIL
from loss import unbiased_mil_loss
from utils import get_batches


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train_model(train_data, val_data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionMIL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    best_val_auc = 0

    print("🚀 Training started...\n")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        loop = tqdm(get_batches(train_data), leave=True)

        for normal_batch, abnormal_batch in loop:

            normal_batch = normal_batch.to(device).float()
            abnormal_batch = abnormal_batch.to(device).float()

            # Forward pass
            normal_scores, _ = model(normal_batch)
            abnormal_scores, _ = model(abnormal_batch)

            # Compute loss
            loss = unbiased_mil_loss(normal_scores, abnormal_scores)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_data)

        print(f"\nEpoch {epoch+1} Loss: {avg_loss:.4f}")

        # -------------------------------
        # VALIDATION
        # -------------------------------
        val_auc = evaluate_model(model, val_data, plot=False)

        print(f"Validation AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved!\n")

    print("🎯 Training complete!")
    return model


# -------------------------------
# EVALUATION FUNCTION
# -------------------------------
def evaluate_model(model, data, plot=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for i, (segments, label) in enumerate(data):

            segments = torch.tensor(segments).unsqueeze(0).to(device).float()

            scores, _ = model(segments)

            video_score = torch.max(scores)

            preds.append(video_score.item())
            labels.append(label)

            # -------------------------------
            # VISUALISATION (IMPORTANT)
            # -------------------------------
            if plot and i < 2:  # plot first 2 videos only
                scores_np = scores.squeeze().cpu().numpy()

                plt.figure()
                plt.plot(scores_np)
                plt.title(f"Video {i} - Label: {label}")
                plt.xlabel("Segments")
                plt.ylabel("Anomaly Score")
                plt.grid()
                plt.show()

    # Compute AUC
    auc = roc_auc_score(labels, preds)

    return auc


# -------------------------------
# TEST FUNCTION (OPTIONAL)
# -------------------------------
def test_model(test_data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionMIL().to(device)
    model.load_state_dict(torch.load("best_model.pth"))

    print("\n🚀 Testing best model...\n")

    test_auc = evaluate_model(model, test_data, plot=True)

    print(f"🎯 Test AUC: {test_auc:.4f}")