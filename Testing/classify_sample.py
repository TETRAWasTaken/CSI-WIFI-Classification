import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow direct script execution from Testing/ while importing project modules.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from paths import Paths


def _load_raw_array(file_path: str):
    _, ext = os.path.splitext(file_path)

    with open(file_path, "rb") as file_obj:
        header = file_obj.read(6)
    if header == b"\x93NUMPY":
        return np.load(file_path, mmap_mode="r")

    if ext.lower() == ".csv":
        return np.loadtxt(file_path, delimiter=",")

    return np.load(file_path, mmap_mode="r")


def _normalize_sample(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float32)
    mean = sample.mean()
    std = sample.std()
    return (sample - mean) / (std + 1e-8)


def load_sample_tensor(args, path_cfg):
    if args.sample_path:
        sample_array = _load_raw_array(args.sample_path)
        if sample_array.ndim == 3 and sample_array.shape[0] == 1:
            sample_array = sample_array[0]
        if sample_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D sample matrix, got shape {sample_array.shape} from {args.sample_path}"
            )
        sample_array = _normalize_sample(sample_array)
        sample_tensor = torch.tensor(sample_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        label = None
        return sample_tensor, label

    from DataLoad import DataLoader

    split = args.split.lower()
    if split == "train":
        data_path = path_cfg.train_data_path
        labels_path = path_cfg.train_labels_path
        cache_path = path_cfg.train_cache_path
    elif split == "val":
        data_path = path_cfg.val_data_path
        labels_path = path_cfg.val_labels_path
        cache_path = path_cfg.val_cache_path
    elif split == "test":
        data_path = path_cfg.test_data_path
        labels_path = path_cfg.test_labels_path
        cache_path = path_cfg.test_cache_path
    else:
        raise ValueError("split must be one of: train, val, test")

    dataset = DataLoader.Dataset(
        data_path=data_path,
        labels_path=labels_path,
        zarr_cache=cache_path,
    )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index {args.sample_index} is out of range for {split} split with {len(dataset)} samples"
        )

    sample_tensor, label_tensor = dataset[args.sample_index]
    return sample_tensor.unsqueeze(0), int(label_tensor.item())


def load_model(checkpoint_path: str, device: torch.device):
    from Model.CSIBaseline import CSIBaseline

    model = CSIBaseline(num_classes=7).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, sample_tensor, device, top_k: int):
    sample_tensor = sample_tensor.to(device)

    with torch.no_grad():
        logits = model(sample_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    top_k = min(top_k, probabilities.shape[1])
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    return {
        "predicted_class": int(predicted_class.item()),
        "confidence": float(confidence.item()),
        "top_classes": [int(index) for index in top_indices[0].tolist()],
        "top_probabilities": [float(prob) for prob in top_probs[0].tolist()],
    }


def print_sample_preview(sample_tensor: torch.Tensor):
    sample_2d = sample_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    rows = min(3, sample_2d.shape[0])
    cols = min(8, sample_2d.shape[1])

    print("Sample preview:")
    print(f"  Shape: {sample_2d.shape}")
    print(f"  Mean:  {sample_2d.mean():.4f}")
    print(f"  Std:   {sample_2d.std():.4f}")
    print(f"  Min:   {sample_2d.min():.4f}")
    print(f"  Max:   {sample_2d.max():.4f}")
    print(f"  First {rows} row(s) x {cols} column(s):")
    for row_idx in range(rows):
        preview_values = " ".join(f"{value: .4f}" for value in sample_2d[row_idx, :cols])
        print(f"    row {row_idx:02d}: {preview_values}")


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a sample with the baseline CSI model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a baseline checkpoint. Defaults to checkpoints/baseline_best.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to sample from when --sample_path is not provided",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of the sample to classify from the selected split",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default=None,
        help="Optional path to a raw sample file (NumPy binary or CSV-like file) to classify",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of highest-probability classes to display",
    )
    parser.add_argument(
        "--ut_har_root",
        type=str,
        default=None,
        help="Optional override path to a UT_HAR root containing data/, label/, and cache/",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional writable cache directory for zarr files when using --ut_har_root",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    path_cfg = Paths()
    path_cfg.add_pipeline_module_paths()

    if args.ut_har_root:
        runtime_cache_root = args.cache_dir or os.path.join(Path.home(), ".cache", "ut_har")
        path_cfg.data_path = os.path.join(args.ut_har_root, "data")
        path_cfg.labels_path = os.path.join(args.ut_har_root, "label")
        path_cfg.cache_dir = runtime_cache_root
        path_cfg.train_data_path = os.path.join(path_cfg.data_path, "X_train.csv")
        path_cfg.train_labels_path = os.path.join(path_cfg.labels_path, "y_train.csv")
        path_cfg.test_data_path = os.path.join(path_cfg.data_path, "X_test.csv")
        path_cfg.test_labels_path = os.path.join(path_cfg.labels_path, "y_test.csv")
        path_cfg.val_data_path = os.path.join(path_cfg.data_path, "X_val.csv")
        path_cfg.val_labels_path = os.path.join(path_cfg.labels_path, "y_val.csv")
        path_cfg.train_cache_path = os.path.join(path_cfg.cache_dir, "train.zarr")
        path_cfg.test_cache_path = os.path.join(path_cfg.cache_dir, "test.zarr")
        path_cfg.val_cache_path = os.path.join(path_cfg.cache_dir, "val.zarr")
        os.makedirs(path_cfg.cache_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path or path_cfg.get_best_baseline_model_path()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    sample_tensor, true_label = load_sample_tensor(args, path_cfg)

    result = predict(model, sample_tensor, device, args.top_k)

    print("Baseline CSI classification")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")
    if args.sample_path:
        print(f"  Sample:     {args.sample_path}")
    else:
        print(f"  Split:      {args.split}")
        print(f"  Index:      {args.sample_index}")
    if true_label is not None:
        print(f"  True label: {true_label}")
    print_sample_preview(sample_tensor)
    print(f"  Predicted:  {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print("  Top classes:")
    for class_index, probability in zip(result["top_classes"], result["top_probabilities"]):
        print(f"    {class_index}: {probability:.4f}")
    print("Output guide:")
    print("  Predicted is the model's most likely class index for this sample.")
    print("  Confidence is the softmax probability for that top class.")
    print("  Top classes lists the highest-probability class indices in descending order.")
    print("  The sample preview shows the normalized input matrix used by the model.")


if __name__ == "__main__":
    main()
