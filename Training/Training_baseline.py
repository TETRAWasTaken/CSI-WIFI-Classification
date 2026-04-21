import os
import sys

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


def train_baseline():
    path_cfg = Paths()
    path_cfg.add_pipeline_module_paths()

    from DataLoad import DataLoader
    from Model.CSIBaseline import CSIBaseline

    checkpoint_path = path_cfg.get_best_baseline_model_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

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

    train_loader = TorchDataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = CSIBaseline(num_classes=7).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_val_acc = float('-inf')

    epochs = 20

    print("Starting training...")
    
    for epoch in range(epochs):
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

        train_acc = 100 * correct_train / total_train
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_split(model, test_loader, criterion, device)

        if val_acc > best_val_acc:
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
            checkpoint_msg = f" | Saved: {checkpoint_path}"
        else:
            checkpoint_msg = ""
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
              f"{checkpoint_msg}")

if __name__ == "__main__":
    train_baseline()