from dataclasses import dataclass


@dataclass
class DataModuleConfig:
    train_dir: str = "raw_data/brain-tumor-mri-dataset/Training"
    test_dir: str = "raw_data/brain-tumor-mri-dataset/Testing"
    batch_size: int = 64
    num_workers: int = 4
    image_size: tuple = (64, 64)