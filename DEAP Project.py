# DEAP Project.py  (compact version using Kaggle data_preprocessed_python)

import argparse
import os
import random
import time
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------- Utils ---------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- Data loading ---------------------- #

def load_deap_python(data_dir: str, target: str = "valence"):
    """
    Load DEAP from Kaggle's data_preprocessed_python/*.dat.

    target: "valence", "arousal", "dominance", or "liking"
    Binary label: target > 5 -> 1 (high), else 0 (low)
    """
    target_map = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}
    col = target_map[target]

    all_X = []
    all_y = []

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".dat"))
    if not files:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    print(f"Found {len(files)} subject files in {data_dir}")

    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path, "rb") as f:
            sample = pickle.load(f, encoding="latin1")

        data = sample["data"]       # (40, 40, 8064)
        labels = sample["labels"]   # (40, 4)

        eeg = data[:, :32, :]       # keep first 32 EEG channels
        target_scores = labels[:, col]
        y_bin = (target_scores > 5).astype(int)  # 1 = high, 0 = low

        all_X.append(eeg)
        all_y.append(y_bin)

    X = np.vstack(all_X)            # (32*40, 32, 8064)
    y = np.concatenate(all_y)       # (32*40,)

    print("Combined X shape:", X.shape)
    print("Combined y shape:", y.shape)
    print(f"Class balance: {np.bincount(y)} (0 = low, 1 = high)")

    return X, y


class DEAPEEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i]               # (C, T)
        y = self.y[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ---------------------- CNN model ---------------------- #

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 4 == 0
        branch_channels = out_channels // 4

        self.branch1 = nn.Conv1d(in_channels, branch_channels, 3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, branch_channels, 5, padding=2)
        self.branch3 = nn.Conv1d(in_channels, branch_channels, 7, padding=3)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1),
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bn(out)
        return self.act(out)


class EEGLightNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.inception1 = InceptionBlock1D(32, 64)
        self.se1 = SEBlock(64, reduction=8)
        self.inception2 = InceptionBlock1D(64, 128)
        self.se2 = SEBlock(128, reduction=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception1(x)
        x = self.se1(x)
        x = self.inception2(x)
        x = self.se2(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return acc, f1, report


# ---------------------- SVM baseline ---------------------- #

def train_evaluate_svm(X, y, idx_train, idx_val, idx_test, n_components=100):
    N, C, T = X.shape
    X_flat = X.reshape(N, C * T)

    max_components = min(X_flat.shape[1], X_flat.shape[0] - 1)
    n_components = min(n_components, max_components)

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")

    X_train = X_flat[idx_train]
    y_train = y[idx_train]
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_scaled)
    svm.fit(X_train_pca, y_train)

    results = {}

    def eval_split(name, idx_split):
        X_split = X_flat[idx_split]
        y_split = y[idx_split]
        X_split_scaled = scaler.transform(X_split)
        X_split_pca = pca.transform(X_split_scaled)
        preds = svm.predict(X_split_pca)
        acc = accuracy_score(y_split, preds)
        f1 = f1_score(y_split, preds)
        print(f"[SVM] {name} Acc: {acc:.4f}, F1: {f1:.4f}")
        results[f"{name.lower()}_acc"] = acc
        results[f"{name.lower()}_f1"] = f1

    print("\nTraining SVM baseline...")
    eval_split("Train", idx_train)
    eval_split("Val", idx_val)
    eval_split("Test", idx_test)

    return results


# ---------------------- Main ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="EEG Emotion Recognition on DEAP with CNN + SVM (compact version)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="deap-dataset/data_preprocessed_python",
        help="Folder with DEAP .dat files (data_preprocessed_python).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="valence",
        choices=["valence", "arousal", "dominance", "liking"],
        help="Which DEAP label to binarize (column).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--svm_pca_components",
        type=int,
        default=100,
        help="Number of PCA components for SVM.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    print(f"Loading DEAP from: {args.data_dir}")
    X, y = load_deap_python(args.data_dir, target=args.target)
    n_trials, n_channels, n_samples = X.shape

    # Train / val / test split
    test_size = args.test_size
    val_size = args.val_size / (1.0 - test_size)

    idx = np.arange(n_trials)
    idx_train_val, idx_test, y_train_val, y_test = train_test_split(
        idx, y, test_size=test_size, stratify=y, random_state=args.seed
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train_val,
        y_train_val,
        test_size=val_size,
        stratify=y_train_val,
        random_state=args.seed,
    )

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    train_ds = DEAPEEGDataset(X, y, idx_train)
    val_ds = DEAPEEGDataset(X, y, idx_val)
    test_ds = DEAPEEGDataset(X, y, idx_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_time = time.perf_counter()

    # ----- Method 1: CNN ----- #
    print("\n=== Training Method 1: CNN (EEGLightNet) ===")
    model = EEGLightNet(n_channels=n_channels, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, _ = evaluate_model(model, val_loader, device)
        print(
            f"[CNN] Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating CNN on TEST set ...")
    cnn_test_acc, cnn_test_f1, cnn_report = evaluate_model(model, test_loader, device)
    print(f"[CNN] Test Accuracy: {cnn_test_acc:.4f}")
    print(f"[CNN] Test F1-score: {cnn_test_f1:.4f}")
    print("\n[CNN] Classification report:")
    for label, metrics in cnn_report.items():
        if label in ["0", "1"]:
            print(
                f"  Class {label}: "
                f"precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, "
                f"f1={metrics['f1-score']:.3f}, support={metrics['support']}"
            )

    # ----- Method 2: SVM baseline ----- #
    print("\n=== Training Method 2: SVM (RBF) baseline ===")
    svm_results = train_evaluate_svm(
        X, y, idx_train, idx_val, idx_test, n_components=args.svm_pca_components
    )

    end_time = time.perf_counter()
    total_seconds = end_time - start_time

    print("\n========== RUNTIME SUMMARY ==========")
    print(f"Total runtime (CNN + SVM): {total_seconds:.2f} seconds "
          f"({total_seconds / 60.0:.2f} minutes).")
    print("=====================================")

    print("\n========== METHOD COMPARISON (TEST) ==========")
    print(f"[CNN] Test Acc: {cnn_test_acc:.4f}, F1: {cnn_test_f1:.4f}")
    print(f"[SVM] Test Acc: {svm_results.get('test_acc', float('nan')):.4f}, "
          f"F1: {svm_results.get('test_f1', float('nan')):.4f}")
    print("==============================================")
    print("\nDone.")


if __name__ == "__main__":
    main()
