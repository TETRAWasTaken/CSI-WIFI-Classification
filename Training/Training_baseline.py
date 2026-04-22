import os
import sys
import time
import argparse
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

# Allow direct script execution from Training/ while importing project modules.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import Paths


def evaluate_split(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc


def _override_data_paths(path_cfg, ut_har_root, cache_root=None):
    path_cfg.data_path = os.path.join(ut_har_root, "data")
    path_cfg.labels_path = os.path.join(ut_har_root, "label")
    path_cfg.cache_dir = cache_root or os.path.join(ut_har_root, "cache")

    path_cfg.train_data_path = os.path.join(path_cfg.data_path, "X_train.csv")
    path_cfg.train_labels_path = os.path.join(path_cfg.labels_path, "y_train.csv")
    path_cfg.test_data_path = os.path.join(path_cfg.data_path, "X_test.csv")
    path_cfg.test_labels_path = os.path.join(path_cfg.labels_path, "y_test.csv")
    path_cfg.val_data_path = os.path.join(path_cfg.data_path, "X_val.csv")
    path_cfg.val_labels_path = os.path.join(path_cfg.labels_path, "y_val.csv")

    path_cfg.train_cache_path = os.path.join(path_cfg.cache_dir, "train.zarr")
    path_cfg.test_cache_path = os.path.join(path_cfg.cache_dir, "test.zarr")
    path_cfg.val_cache_path = os.path.join(path_cfg.cache_dir, "val.zarr")


def train_baseline(args):
    path_cfg = Paths()
    path_cfg.add_pipeline_module_paths()

    if args.ut_har_root:
        runtime_cache_root = args.cache_dir or os.path.join(tempfile.gettempdir(), "ut_har_cache")
        _override_data_paths(path_cfg, args.ut_har_root, cache_root=runtime_cache_root)
        os.makedirs(path_cfg.cache_dir, exist_ok=True)
        print("Azure/local override enabled:")
        print(f"  ut_har_root: {args.ut_har_root}")
        print(f"  cache_dir:   {path_cfg.cache_dir}")

    from DataLoad import DataLoader
    from Model.CSIBaseline import CSIBaseline

    checkpoint_path = args.checkpoint_path or path_cfg.get_best_baseline_model_path()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Baseline Training Run")
    print("=" * 80)
    print(f"Executing on device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_dataset = DataLoader.Dataset(
        data_path=path_cfg.train_data_path,
        labels_path=path_cfg.train_labels_path,
        zarr_cache=path_cfg.train_cache_path
    )
    
    test_dataset = DataLoader.Dataset(
        data_path=path_cfg.test_data_path,
        labels_path=path_cfg.test_labels_path,
        zarr_cache=path_cfg.test_cache_path
    )

    val_dataset = DataLoader.Dataset(
        data_path=path_cfg.val_data_path,
        labels_path=path_cfg.val_labels_path,
        zarr_cache=path_cfg.val_cache_path
    )

    print("Loaded datasets:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("DataLoaders configured:")
    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch:   {len(val_loader)}")
    print(f"  Test batches/epoch:  {len(test_loader)}")

    model = CSIBaseline(num_classes=7).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val_acc = float('-inf')

    epochs = args.epochs

    log_interval = max(1, len(train_loader) // 10)

    print("Training configuration:")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {train_loader.batch_size}")
    print(f"  Optimizer:    AdamW")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"  Num workers:  {args.num_workers}")
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  Log interval: every {log_interval} batch(es)")
    print("Starting training...")
    
    for epoch in range(epochs):
        wall_start = time.perf_counter()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = 100 * correct_train / total_train
                avg_batch_loss = running_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1:02d}/{epochs} | "
                    f"Batch {batch_idx + 1:03d}/{len(train_loader)} | "
                    f"Avg Train Loss: {avg_batch_loss:.4f} | "
                    f"Running Train Acc: {batch_acc:.2f}%"
                )

        train_acc = 100 * correct_train / total_train
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_split(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time_msg = f"{time.perf_counter() - wall_start:.2f}s"

        if val_acc > best_val_acc:
            prev_best = best_val_acc
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )
            if prev_best == float('-inf'):
                checkpoint_msg = f" | New best Val Acc: {best_val_acc:.2f}% | Saved: {checkpoint_path}"
            else:
                checkpoint_msg = (
                    f" | New best Val Acc: {best_val_acc:.2f}% "
                    f"(prev {prev_best:.2f}%) | Saved: {checkpoint_path}"
                )
        else:
            checkpoint_msg = f" | Best Val Acc so far: {best_val_acc:.2f}%"
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Epoch Time: {epoch_time_msg}"
              f"{checkpoint_msg}")

    print("Training complete.")
    print(f"Best validation accuracy achieved: {best_val_acc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline CSI model")
    parser.add_argument("--ut_har_root", type=str, default=None, help="Path to UT_HAR root containing data/, label/, and cache/")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training/eval batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional path to save best checkpoint")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional writable cache directory for zarr files")
    return parser.parse_args()


if __name__ == "__main__":
    train_baseline(parse_args())