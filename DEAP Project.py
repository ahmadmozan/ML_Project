"""
Lightweight EEG emotion recognition on DEAP (data_preprocessed_python).

- Subject-wise split (no leakage): 22 train / 4 val / 6 test
- Preprocessing: remove 3s baseline, per-trial/channel z-score, optional noisy-channel removal
- Windowing after split: 256-sample windows, stride 128
- Class balancing on train windows (undersample majority)
- CNN: small 1D Conv + SE blocks, <2 MB, outputs logits for 2 classes
- Loss: CrossEntropy with class weights
- Training: ReduceLROnPlateau on val F1, early stopping
- Threshold: fixed at 0.5 (no tuning)
- Baseline: SVM with PCA on trial-level flattened signals

Run:
    python DEAP-Project.py --data_dir data_preprocessed_python --track_emissions --country_code CAN
Optional:
    python DEAP-Project.py --data_dir data_preprocessed_python --n_eeg_channels 32
    python DEAP-Project.py --data_dir data_preprocessed_python --channel_indices 0,1,2,...,31
"""

import logging
import platform
import argparse
import os
import pickle
import random
import time
from typing import List, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------- Optional: CodeCarbon ----------
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Fully silence CodeCarbon's own logging
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

def _parse_channel_indices(s: Optional[str]) -> Optional[np.ndarray]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    return np.array([int(p) for p in parts], dtype=np.int64)


def load_deap_python(
    data_dir: str,
    target: str = "valence",
    remove_noisy: bool = False,
    noise_factor: float = 5.0,
    n_eeg_channels: int = 32,                 # <-- back to 32
    channel_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X: (N_trials, C, 7680)
        y: (N_trials,)
        subject_ids: (N_trials,)
        kept_channels: indices (relative to original 32 EEG channels)
    """
    target_map = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}
    if target not in target_map:
        raise ValueError(f"Invalid target {target}")
    col = target_map[target]

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".dat"))
    if not files:
        raise RuntimeError(f"No .dat files found in {data_dir}")

    # Choose EEG channels out of 32
    if channel_indices is not None:
        if np.any(channel_indices < 0) or np.any(channel_indices > 31):
            raise ValueError("channel_indices must be between 0 and 31.")
        kept_channels = channel_indices
    else:
        if not (1 <= n_eeg_channels <= 32):
            raise ValueError("--n_eeg_channels must be in [1, 32].")
        kept_channels = np.arange(n_eeg_channels, dtype=np.int64)

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_subjects: List[np.ndarray] = []

    print(f"Found {len(files)} subject files in {data_dir}")
    print(f"Using EEG channels: {kept_channels.tolist()} (count={len(kept_channels)})")

    for subj_idx, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        with open(path, "rb") as f:
            sample = pickle.load(f, encoding="latin1")

        data = sample["data"]      # (40, 40, 8064)
        labels = sample["labels"]  # (40, 4)

        eeg_full = data[:, :32, :]                     # (40, 32, 8064)
        eeg_full = eeg_full[..., 384:384 + 60 * 128]   # remove 3s baseline -> (40, 32, 7680)

        eeg = eeg_full[:, kept_channels, :]            # (40, C, 7680)

        # Per-trial/channel z-score
        mean = eeg.mean(axis=-1, keepdims=True)
        std = eeg.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-6, 1e-6, std)
        eeg = (eeg - mean) / std

        y_bin = (labels[:, col] > 5).astype(np.int64)

        all_X.append(eeg.astype(np.float32))
        all_y.append(y_bin)
        all_subjects.append(np.full(eeg.shape[0], subj_idx, dtype=np.int64))

    X = np.vstack(all_X)  # (N_trials, C, 7680)
    y = np.concatenate(all_y)
    subject_ids = np.concatenate(all_subjects)

    if remove_noisy:
        channel_std = X.std(axis=(0, 2))
        median_std = np.median(channel_std)
        keep_local = np.where(channel_std <= median_std * noise_factor)[0]
        if len(keep_local) == 0:
            keep_local = np.arange(X.shape[1])
        kept_channels = kept_channels[keep_local]
        X = X[:, keep_local, :]
        print(f"Removed noisy channels, kept {len(kept_channels)} channels after filtering.")

    print(f"Final trials shape: {X.shape}")
    print(f"Class balance (trials): {np.bincount(y)} (0=low, 1=high)")
    print(f"Channels kept (final): {kept_channels.tolist()} (count={len(kept_channels)})")
    return X, y, subject_ids, kept_channels


# ---------------------- Splits ---------------------- #

def split_by_subject(subject_ids: np.ndarray, seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(subject_ids)
    rng.shuffle(unique_ids)

    n_subj = len(unique_ids)
    n_train = int(np.floor(n_subj * train_ratio))
    n_val = int(np.floor(n_subj * val_ratio))

    train_subj = unique_ids[:n_train]
    val_subj = unique_ids[n_train:n_train + n_val]
    test_subj = unique_ids[n_train + n_val:]

    idx_train = np.nonzero(np.isin(subject_ids, train_subj))[0]
    idx_val = np.nonzero(np.isin(subject_ids, val_subj))[0]
    idx_test = np.nonzero(np.isin(subject_ids, test_subj))[0]

    print(f"Subjects -> Train: {len(train_subj)}, Val: {len(val_subj)}, Test: {len(test_subj)}")
    print(f"Trials  -> Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    return idx_train, idx_val, idx_test


# ---------------------- Windowing ---------------------- #

def create_windows(X: np.ndarray, y: np.ndarray, indices: np.ndarray, window_size: int = 256, stride: int = 128):
    windows, labels = [], []
    for i in indices:
        trial = X[i]
        label = y[i]
        T = trial.shape[-1]
        for start in range(0, T - window_size + 1, stride):
            windows.append(trial[:, start:start + window_size])
            labels.append(label)
    return np.stack(windows), np.array(labels, dtype=np.int64)


def undersample_majority(X: np.ndarray, y: np.ndarray, seed: int = 42):
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
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


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


def evaluate_argmax(model, loader, criterion, device):
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
    return avg_loss, acc, f1, y_true, y_pred


def evaluate_threshold_fixed(model, loader, criterion, device, thr: float = 0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= thr).long()

            total_loss += loss.item() * Xb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, acc, f1, y_true, y_pred


def model_size_mb(model: nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    total_bytes = (total_params + total_buffers) * 4
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


def simulate_arduino(model: nn.Module, infer_time_per_sample: float) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    model_bytes = (total_params + total_buffers) * 4
    model_kb = model_bytes / 1024.0

    flash_kb = 32.0
    sram_kb = 2.0
    arduino_clock_mhz = 16.0
    host_clock_ghz = 2.5

    clock_ratio = (host_clock_ghz * 1000.0) / arduino_clock_mhz
    arduino_ms = infer_time_per_sample * 1000.0 * clock_ratio

    print(
        f"[Arduino sim] ~{model_kb:.0f} KB model vs {flash_kb:.0f} KB flash, "
        f"{sram_kb:.0f} KB SRAM, est. {arduino_ms:.0f} ms/sample (very rough)."
    )


def train_evaluate_svm(X, y, idx_train, idx_val, idx_test, n_components=120) -> dict:
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

    def _eval(name, idx):
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


def main():
    parser = argparse.ArgumentParser(description="DEAP EEG Emotion Recognition (CNN + SVM)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target", type=str, default="valence", choices=["valence", "arousal", "dominance", "liking"])
    parser.add_argument("--n_eeg_channels", type=int, default=32)  # <-- back to 32
    parser.add_argument("--channel_indices", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--svm_pca_components", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remove_noisy", action="store_true")
    parser.add_argument("--noise_factor", type=float, default=5.0)
    parser.add_argument("--track_emissions", action="store_true")
    parser.add_argument("--country_code", type=str, default="CAN")
    args = parser.parse_args()

    set_seed(args.seed)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    channel_indices = _parse_channel_indices(args.channel_indices)

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
        args.data_dir,
        target=args.target,
        remove_noisy=args.remove_noisy,
        noise_factor=args.noise_factor,
        n_eeg_channels=args.n_eeg_channels,
        channel_indices=channel_indices,
    )

    idx_train_trials, idx_val_trials, idx_test_trials = split_by_subject(subject_ids, seed=args.seed)

    X_train_win, y_train_win = create_windows(X_trials, y_trials, idx_train_trials, window_size=256, stride=128)
    X_val_win, y_val_win = create_windows(X_trials, y_trials, idx_val_trials, window_size=256, stride=128)
    X_test_win, y_test_win = create_windows(X_trials, y_trials, idx_test_trials, window_size=256, stride=128)

    print(f"Windows -> Train: {len(X_train_win)}, Val: {len(X_val_win)}, Test: {len(X_test_win)}")
    print(f"Class balance (train windows) before balance: {np.bincount(y_train_win, minlength=2)}")

    X_train_bal, y_train_bal = undersample_majority(X_train_win, y_train_win, seed=args.seed)
    print(f"Class balance (train windows) after balance: {np.bincount(y_train_bal, minlength=2)}")

    train_loader = DataLoader(DEAPEEGDataset(X_train_bal, y_train_bal), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DEAPEEGDataset(X_val_win, y_val_win), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(DEAPEEGDataset(X_test_win, y_test_win), batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train_bal)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EEGConvNet(n_channels=X_train_bal.shape[1], n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)

    best_state = None
    best_f1 = -1.0
    epochs_no_improve = 0

    print("\n=== Training CNN ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc, val_f1, _, _ = evaluate_argmax(model, val_loader, criterion, device)
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

    fixed_thr = 0.5
    print(f"Using fixed decision threshold: {fixed_thr}")

    test_loss, test_acc, test_f1, y_true_test, y_pred_test = evaluate_argmax(model, test_loader, criterion, device)

    print("\n=== CNN TEST METRICS (ARGMAX) ===")
    print(f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true_test, y_pred_test))
    print("Classification report:\n", classification_report(y_true_test, y_pred_test, digits=4))

    infer_time = measure_inference_time(model, test_loader, device)
    size_mb = model_size_mb(model)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Inference time per sample: {infer_time*1000:.3f} ms")
    simulate_arduino(model, infer_time)

    test_loss_t, test_acc_t, test_f1_t, y_true_t, y_pred_t = evaluate_threshold_fixed(
        model, test_loader, criterion, device, thr=fixed_thr
    )

    print("\n=== CNN TEST METRICS (THRESHOLD=0.5) ===")
    print(f"Loss: {test_loss_t:.4f} | Acc: {test_acc_t:.4f} | F1: {test_f1_t:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true_t, y_pred_t))
    print("Classification report:\n", classification_report(y_true_t, y_pred_t, digits=4))

    svm_results = train_evaluate_svm(
        X_trials, y_trials, idx_train_trials, idx_val_trials, idx_test_trials, n_components=args.svm_pca_components
    )

    print("\n========== METHOD COMPARISON (TEST) ==========")
    print(
        f"CNN(argmax) -> Acc: {test_acc:.4f}, F1: {test_f1:.4f} | "
        f"CNN(thr=0.5) -> Acc: {test_acc_t:.4f}, F1: {test_f1_t:.4f} | "
        f"SVM -> Acc: {svm_results.get('test_acc', float('nan')):.4f}, "
        f"F1: {svm_results.get('test_f1', float('nan')):.4f}"
    )
    print("============================================")

    emissions_kg = None
    if tracker is not None:
        emissions_kg = tracker.stop()

    total_seconds = time.perf_counter() - overall_start
    system_str = f"{platform.system()} {platform.release()}, CPU threads: {os.cpu_count()}, device: {device}"

    print("\n========== RUNTIME & ENERGY SUMMARY ==========")
    print(f"System: {system_str}")
    print(f"Total runtime (CNN + SVM): {total_seconds:.2f} seconds ({total_seconds/60.0:.2f} minutes).")
    if emissions_kg is not None:
        print(f"Estimated carbon emissions: {emissions_kg:.6f} kg CO2eq (CodeCarbon estimate).")
    print("============================================")


if __name__ == "__main__":
    main()
