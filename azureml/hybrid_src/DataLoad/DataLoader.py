import os
import zarr
import torch 
import numpy as np
import shutil
from torch.utils.data import Dataset as TorchDataset
import dask.array as da


class Dataset(TorchDataset):
    def __init__(self, data_path: str, labels_path: str, zarr_cache: str, chunk_size: int = 1000):
        """
        Initialise the dataset.

        Args:
            data_path (string): Path to the raw data file.
            labels_path (string): Path to the labels file.
            zarr_cache (string): Path to the zarr cache file.
            chunk_size (int): Size of the chunks to load from the zarr file.
        """

        self.data = data_path
        self.labels = labels_path
        self.zarr_cache = zarr_cache
        self.chunk_size = chunk_size

        if not os.path.exists(self.zarr_cache):
            print(f"Zarr cache not found at {self.zarr_cache}.")
            print(f"Building zarr cache at {self.zarr_cache}...")
            self._build_zarr_cache()

        else:
            print(f"Zarr cache found at {self.zarr_cache}. Loading from cache...")
        
        try:
            self.data = zarr.open(self.zarr_cache, mode='r')
        except Exception as cache_error:
            print(f"Invalid zarr cache at {self.zarr_cache}: {cache_error}")
            print("Rebuilding zarr cache...")
            if os.path.isdir(self.zarr_cache):
                shutil.rmtree(self.zarr_cache, ignore_errors=True)
            elif os.path.exists(self.zarr_cache):
                os.remove(self.zarr_cache)
            self._build_zarr_cache()
            self.data = zarr.open(self.zarr_cache, mode='r')

        self.labels = self._load_array(self.labels)

        assert self.data.shape[0] == self.labels.shape[0], "Data and labels must have the same number of samples."

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.data.shape[0]
    
    def _build_zarr_cache(self):
        raw_data = self._load_array(self.data)
        dask_data = da.from_array(raw_data, chunks=(self.chunk_size, *raw_data.shape[1:]))

        means = dask_data.mean(axis=(1, 2)).compute()
        stds = dask_data.std(axis=(1, 2)).compute()
        normalized_data = (dask_data - means[:, None, None]) / (stds[:, None, None] + 1e-8)

        normalized_data.to_zarr(self.zarr_cache, overwrite=True)
        print(f"Zarr cache built at {self.zarr_cache}.")

    @staticmethod
    def _load_array(file_path: str):
        _, ext = os.path.splitext(file_path)

        # Some datasets are stored in NumPy binary format with a non-.npy extension.
        with open(file_path, 'rb') as file_obj:
            header = file_obj.read(6)
        if header == b'\x93NUMPY':
            return np.load(file_path, mmap_mode='r')

        if ext.lower() == '.csv':
            return np.loadtxt(file_path, delimiter=',')

        return np.load(file_path, mmap_mode='r')
    
    def __getitem__(self, idx):
        """
        Return the sample at the given index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            tuple: (data, label) where data is the input data and label is the corresponding label.
        """
        x = self.data[idx]
        y = self.labels[idx]
        
        x = np.expand_dims(x, axis=0)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor
