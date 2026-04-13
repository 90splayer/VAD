# main.py

from dataset import load_dataset
from train import train_model, test_model
from sklearn.model_selection import train_test_split


def main():
    print("🚀 Loading datasets...")

    # ✅ LOAD BOTH DATASETS
    ucf_data = load_dataset("data/ucf/")
    xd_data = load_dataset("data/xd/")

    # ✅ COMBINE THEM
    data = ucf_data + xd_data

    print(f"Total samples: {len(data)}")

    # -------------------------------
    # SPLIT DATA
    # -------------------------------
    train_data, temp = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------
    model = train_model(train_data, val_data)

    # -------------------------------
    # TEST MODEL
    # -------------------------------
    test_model(test_data)


if __name__ == "__main__":
    main()