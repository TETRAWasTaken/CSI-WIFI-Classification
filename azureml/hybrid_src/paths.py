import os
import sys

class Paths:
    def __init__(self):
        # Use the current file location so path resolution is stable.
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.training_path = os.path.join(self.base_path, 'Training')
        self.dataloader_path = os.path.join(self.base_path, 'DataLoad')
        self.data_path = os.path.join(self.base_path, 'UT_HAR', 'data')
        self.labels_path = os.path.join(self.base_path, 'UT_HAR', 'label')
        self.models_path = os.path.join(self.base_path, 'Model')

        self.dataloader_module_path = os.path.join(self.dataloader_path, 'DataLoader.py')
        self.baseline_model_module_path = os.path.join(self.models_path, 'CSIBaseline.py')
        self.hybrid_model_module_path = os.path.join(self.models_path, 'CSIHybrid.py')

        # Train files
        self.train_data_path = os.path.join(self.data_path, 'X_train.csv')
        self.train_labels_path = os.path.join(self.labels_path, 'y_train.csv')

        # Test files
        self.test_data_path = os.path.join(self.data_path, 'X_test.csv')
        self.test_labels_path = os.path.join(self.labels_path, 'y_test.csv')

        # Validation files
        self.val_data_path = os.path.join(self.data_path, 'X_val.csv')
        self.val_labels_path = os.path.join(self.labels_path, 'y_val.csv')

        # Cache paths (for preprocessed zarr files)
        self.cache_dir = os.path.join(self.base_path, 'UT_HAR', 'cache')
        self.train_cache_path = os.path.join(self.cache_dir, 'train.zarr')
        self.test_cache_path = os.path.join(self.cache_dir, 'test.zarr')
        self.val_cache_path = os.path.join(self.cache_dir, 'val.zarr')

        # Checkpoint paths
        self.checkpoints_dir = os.path.join(self.base_path, 'checkpoints')
        self.best_baseline_model_path = os.path.join(self.checkpoints_dir, 'baseline_best.pt')

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def get_data_path(self):
        return self.data_path
    
    def get_labels_path(self):
        return self.labels_path

    def get_models_path(self):
        return self.models_path

    def get_dataloader_path(self):
        return self.dataloader_path

    def get_dataloader_module_path(self):
        return self.dataloader_module_path

    def get_baseline_model_module_path(self):
        return self.baseline_model_module_path

    def get_hybrid_model_module_path(self):
        return self.hybrid_model_module_path

    def add_pipeline_module_paths(self):
        # Ensure Training scripts can import DataLoad/Model modules when run directly.
        candidate_paths = [
            self.base_path,
            self.training_path,
            self.dataloader_path,
            self.models_path,
        ]
        for path in candidate_paths:
            if path not in sys.path:
                sys.path.insert(0, path)

    def get_pipeline_module_paths(self):
        return {
            'root': self.base_path,
            'training': self.training_path,
            'dataloader_dir': self.dataloader_path,
            'dataloader_module': self.dataloader_module_path,
            'models_dir': self.models_path,
            'baseline_model_module': self.baseline_model_module_path,
            'hybrid_model_module': self.hybrid_model_module_path,
        }

    def get_best_baseline_model_path(self):
        return self.best_baseline_model_path

    def get_train_paths(self):
        return {
            'data': self.train_data_path,
            'labels': self.train_labels_path,
            'cache': self.train_cache_path,
        }

    def get_test_paths(self):
        return {
            'data': self.test_data_path,
            'labels': self.test_labels_path,
            'cache': self.test_cache_path,
        }

    def get_val_paths(self):
        return {
            'data': self.val_data_path,
            'labels': self.val_labels_path,
            'cache': self.val_cache_path,
        }