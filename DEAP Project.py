"""
Lightweight EEG emotion recognition on DEAP (data_preprocessed_python).

- Subject-wise split (no leakage): 22 train / 4 val / 6 test
- Preprocessing: remove 3s baseline, per-trial/channel z-score, optional noisy-channel removal
- Windowing after split: 256-sample windows, stride 128
- Class balancing on train windows (undersample majority)
- CNN: small 1D Conv + SE blocks, <2 MB, outputs logits for 2 classes
- Loss: CrossEntropy with class weights
- Training: ReduceLROnPlateau on val F1, early stopping, threshold tuning on val
- Baseline: SVM with PCA on trial-level flattened signals

Run:
    python DEAP-Project.py --data_dir data_preprocessed_python
"""
import logging
import platform
import argparse
import os
import pickle
import random
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------- Optional: CodeCarbon for total energy/CO2 estimate ----------
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Fully silence CodeCarbon's own logging (we'll print our own summary)
if CODECARBON_AVAILABLE:
    cc_logger = logging.getLogger("codecarbon")
    cc_logger.setLevel(logging.CRITICAL)
    cc_logger.propagate = False
    cc_logger.disabled = True


# ---------------------- Repro ---------------------- #


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- Preprocessing ---------------------- #


def load_deap_python(
    data_dir: str,
    target: str = "valence",
    remove_noisy: bool = False,
    noise_factor: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DEAP trials (no windowing) with per-trial/channel z-score after removing 3s baseline.

    Returns:
        X: (N_trials, C, 7680)
        y: (N_trials,)
        subject_ids: (N_trials,)
        kept_channels: indices of kept channels
    """
    target_map = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}
    if target not in target_map:
        raise ValueError(f"Invalid target {target}")
    col = target_map[target]

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".dat"))
    if not files:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_subjects: List[np.ndarray] = []

    print(f"Found {len(files)} subject files in {data_dir}")
    for subj_idx, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        with open(path, "rb") as f:
            sample = pickle.load(f, encoding="latin1")

        data = sample["data"]  # (40, 40, 8064)
        labels = sample["labels"]  # (40, 4)

        eeg = data[:, :32, :]  # (40, 32, 8064)
        eeg = eeg[..., 384 : 384 + 60 * 128]  # remove 3s baseline -> (40, 32, 7680)

        mean = eeg.mean(axis=-1, keepdims=True)
        std = eeg.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-6, 1e-6, std)
        eeg = (eeg - mean) / std

        target_scores = labels[:, col]
        y_bin = (target_scores > 5).astype(np.int64)

        all_X.append(eeg.astype(np.float32))
        all_y.append(y_bin)
        all_subjects.append(np.full(eeg.shape[0], subj_idx, dtype=np.int64))

    X = np.vstack(all_X)  # (N_trials, C, 7680)
    y = np.concatenate(all_y)
    subject_ids = np.concatenate(all_subjects)

    kept_channels = np.arange(X.shape[1])
    if remove_noisy:
        channel_std = X.std(axis=(0, 2))
        median_std = np.median(channel_std)
        keep = np.where(channel_std <= median_std * noise_factor)[0]
        if len(keep) == 0:
            keep = np.arange(X.shape[1])
        X = X[:, keep, :]
        kept_channels = keep
        print(f"Removed noisy channels, kept {len(kept_channels)} / 32")

    print(f"Final trials shape: {X.shape}")
    print(f"Class balance (trials): {np.bincount(y)} (0=low, 1=high)")
    print(f"Channels kept: {len(kept_channels)}")
    return X, y, subject_ids, kept_channels


# ---------------------- Splits ---------------------- #


def split_by_subject(subject_ids: np.ndarray, seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(subject_ids)
    rng.shuffle(unique_ids)
    n_subj = len(unique_ids)
    n_train = int(np.floor(n_subj * train_ratio))
    n_val = int(np.floor(n_subj * val_ratio))
    n_test = n_subj - n_train - n_val

    train_subj = unique_ids[:n_train]
    val_subj = unique_ids[n_train : n_train + n_val]
    test_subj = unique_ids[n_train + n_val :]

    idx_train = np.nonzero(np.isin(subject_ids, train_subj))[0]
    idx_val = np.nonzero(np.isin(subject_ids, val_subj))[0]
    idx_test = np.nonzero(np.isin(subject_ids, test_subj))[0]

    print(f"Subjects -> Train: {len(train_subj)}, Val: {len(val_subj)}, Test: {len(test_subj)}")
    print(f"Trials  -> Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    return idx_train, idx_val, idx_test


# ---------------------- Windowing ---------------------- #


def create_windows(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    windows, labels = [], []
    for i in indices:
        trial = X[i]
        label = y[i]
        T = trial.shape[-1]
        for start in range(0, T - window_size + 1, stride):
            seg = trial[:, start : start + window_size]
            windows.append(seg)
            labels.append(label)
    return np.stack(windows), np.array(labels, dtype=np.int64)


def undersample_majority(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y
    n = min(len(idx0), len(idx1))
    idx0_sel = rng.choice(idx0, n, replace=False)
    idx1_sel = rng.choice(idx1, n, replace=False)
    idx = np.concatenate([idx0_sel, idx1_sel])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ---------------------- Dataset ---------------------- #


class DEAPEEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (C, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ---------------------- Model ---------------------- #


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


class EEGConvNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int = 2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )
        self.se1 = SEBlock(32, reduction=8)
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.se2 = SEBlock(64, reduction=8)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.se1(x)
        x = self.block2(x)
        x = self.se2(x)
        return self.head(x)


# ---------------------- Training helpers ---------------------- #


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / max(1, len(loader.dataset))


def evaluate(model, loader, criterion, device, return_preds: bool = False):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * Xb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_loss = total_loss / max(1, len(loader.dataset))
    if return_preds:
        return avg_loss, acc, f1, y_true, y_pred
    return avg_loss, acc, f1


def model_size_mb(model: nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    total_bytes = (total_params + total_buffers) * 4  # float32
    return total_bytes / (1024**2)


def measure_inference_time(model, loader, device) -> float:
    model.eval()
    n_samples = 0
    start = time.perf_counter()
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            _ = model(Xb)
            n_samples += Xb.size(0)
    elapsed = time.perf_counter() - start
    return (elapsed / n_samples) if n_samples > 0 else 0.0


# -------- NEW: very small Arduino-style simulation helper -------- #


def simulate_arduino(model: nn.Module, size_mb: float, infer_time_per_sample: float) -> None:
    """
    Tiny, rough Arduino-style deployment estimate.
    Hard-coded to something like an Arduino Uno:
      - 32 KB flash
      - 2 KB SRAM
      - 16 MHz clock
    """
    # Model size in KB (flash)
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    model_bytes = (total_params + total_buffers) * 4
    model_kb = model_bytes / 1024.0

    flash_kb = 32.0       # typical Uno flash
    sram_kb = 2.0         # typical Uno SRAM
    arduino_clock_mhz = 16.0
    host_clock_ghz = 2.5  # rough assumption for your laptop/PC

    # Very rough latency scaling by clock ratio (ignores memory, SIMD, etc.)
    clock_ratio = (host_clock_ghz * 1000.0) / arduino_clock_mhz
    arduino_ms = infer_time_per_sample * 1000.0 * clock_ratio

    # Single summary line (so your original prints remain basically unchanged)
    print(
        f"[Arduino sim] ~{model_kb:.0f} KB model vs {flash_kb:.0f} KB flash, "
        f"{sram_kb:.0f} KB SRAM, est. {arduino_ms:.0f} ms/sample (very rough)."
    )


# ---------------------- SVM baseline ---------------------- #


def train_evaluate_svm(
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    n_components: int = 120,
) -> dict:
    N, C, T = X.shape
    X_flat = X.reshape(N, C * T)

    max_components = min(n_components, min(X_flat.shape) - 1)
    scaler = StandardScaler()
    pca = PCA(n_components=max_components)
    svm = SVC(kernel="rbf", class_weight="balanced", gamma="scale")

    X_train = scaler.fit_transform(X_flat[idx_train])
    X_train = pca.fit_transform(X_train)
    svm.fit(X_train, y[idx_train])

    results = {}

    def _eval(name: str, idx: np.ndarray):
        X_split = scaler.transform(X_flat[idx])
        X_split = pca.transform(X_split)
        preds = svm.predict(X_split)
        acc = accuracy_score(y[idx], preds)
        f1 = f1_score(y[idx], preds, zero_division=0)
        print(f"[SVM] {name} Acc: {acc:.4f}, F1: {f1:.4f}")
        results[f"{name.lower()}_acc"] = acc
        results[f"{name.lower()}_f1"] = f1

    print("\nTraining SVM baseline...")
    _eval("Train", idx_train)
    _eval("Val", idx_val)
    _eval("Test", idx_test)
    return results


# ---------------------- Main ---------------------- #


def main():
    parser = argparse.ArgumentParser(description="DEAP EEG Emotion Recognition (CNN + SVM)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data_preprocessed_python")
    parser.add_argument(
        "--target",
        type=str,
        default="valence",
        choices=["valence", "arousal", "dominance", "liking"],
        help="Which DEAP label to binarize (col index)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs)")
    parser.add_argument("--svm_pca_components", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--remove_noisy",
        action="store_true",
        help="Optionally drop extremely noisy channels (std > median*factor)",
    )
    parser.add_argument("--noise_factor", type=float, default=5.0)
    parser.add_argument(
        "--track_emissions",
        action="store_true",
        help="If set and CodeCarbon is installed, estimate total energy/CO2 for the run.",
    )
    parser.add_argument(
        "--country_code",
        type=str,
        default="CAN",
        help="Country ISO code for CodeCarbon (e.g. CAN, USA).",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    # --------- start global timer + optional energy tracker ----------
    overall_start = time.perf_counter()

    tracker = None
    if args.track_emissions and CODECARBON_AVAILABLE:
        tracker = OfflineEmissionsTracker(country_iso_code=args.country_code)
        tracker.start()
        print(f"[CodeCarbon] Emissions tracking started ({args.country_code}).")
    elif args.track_emissions and not CODECARBON_AVAILABLE:
        print("[CodeCarbon] --track_emissions enabled but package not installed; skipping energy estimate.")

    print(f"Loading DEAP from: {args.data_dir}")
    X_trials, y_trials, subject_ids, kept_channels = load_deap_python(
        args.data_dir, target=args.target, remove_noisy=args.remove_noisy, noise_factor=args.noise_factor
    )

    # Subject-wise trial split
    idx_train_trials, idx_val_trials, idx_test_trials = split_by_subject(subject_ids, seed=args.seed)

    # Windowing per split
    X_train_win, y_train_win = create_windows(X_trials, y_trials, idx_train_trials, window_size=256, stride=128)
    X_val_win, y_val_win = create_windows(X_trials, y_trials, idx_val_trials, window_size=256, stride=128)
    X_test_win, y_test_win = create_windows(X_trials, y_trials, idx_test_trials, window_size=256, stride=128)

    print(f"Windows -> Train: {len(X_train_win)}, Val: {len(X_val_win)}, Test: {len(X_test_win)}")
    print(f"Class balance (train windows) before balance: {np.bincount(y_train_win, minlength=2)}")

    # Balance training windows (undersample majority)
    X_train_bal, y_train_bal = undersample_majority(X_train_win, y_train_win, seed=args.seed)
    print(f"Class balance (train windows) after balance: {np.bincount(y_train_bal, minlength=2)}")

    # DataLoaders
    train_ds = DEAPEEGDataset(X_train_bal, y_train_bal)
    val_ds = DEAPEEGDataset(X_val_win, y_val_win)
    test_ds = DEAPEEGDataset(X_test_win, y_test_win)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Class weights for imbalance
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train_bal)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CNN
    model = EEGConvNet(n_channels=X_train_bal.shape[1], n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    best_state = None
    best_f1 = -1.0
    epochs_no_improve = 0

    print("\n=== Training CNN ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold tuning on validation windows (keep default threshold here)
    val_loss_full, val_acc_full, val_f1_full, y_true_val, y_pred_val = evaluate(
        model, val_loader, criterion, device, return_preds=True
    )
    best_thr = 0.5
    print(f"Using default decision threshold: {best_thr}")

    # Test evaluation
    test_loss, test_acc, test_f1, y_true_test, y_pred_test = evaluate(
        model, test_loader, criterion, device, return_preds=True
    )
    cm = confusion_matrix(y_true_test, y_pred_test)
    report = classification_report(y_true_test, y_pred_test, digits=4)
    infer_time = measure_inference_time(model, test_loader, device)
    size_mb = model_size_mb(model)

    print("\n=== CNN TEST METRICS ===")
    print(f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Inference time per sample: {infer_time*1000:.3f} ms")

    # ---- NEW: single-line Arduino simulation call (comment out if not needed) ----
    simulate_arduino(model, size_mb, infer_time)

    # SVM baseline on trial-level flattened signals (train/val/test trials)
    svm_results = train_evaluate_svm(
        X_trials, y_trials, idx_train_trials, idx_val_trials, idx_test_trials, n_components=args.svm_pca_components
    )

    print("\n========== METHOD COMPARISON (TEST) ==========")
    print(
        f"CNN -> Acc: {test_acc:.4f}, F1: {test_f1:.4f} | "
        f"SVM -> Acc: {svm_results.get('test_acc', float('nan')):.4f}, "
        f"F1: {svm_results.get('test_f1', float('nan')):.4f}"
    )
    print("============================================")

    # --------- stop energy tracker + print runtime summary ----------
    emissions_kg = None
    if tracker is not None:
        emissions_kg = tracker.stop()

    overall_end = time.perf_counter()
    total_seconds = overall_end - overall_start

    # Simple system summary
    system_str = f"{platform.system()} {platform.release()}, CPU threads: {os.cpu_count()}, device: {device}"

    print("\n========== RUNTIME & ENERGY SUMMARY ==========")
    print(f"System: {system_str}")
    print(f"Total runtime (CNN + SVM): {total_seconds:.2f} seconds "
          f"({total_seconds/60.0:.2f} minutes).")
    if emissions_kg is not None:
        print(f"Estimated carbon emissions: {emissions_kg:.6f} kg CO2eq (CodeCarbon estimate).")
    print("============================================")


if __name__ == "__main__":
    main()
