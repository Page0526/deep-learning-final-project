from dataclasses import dataclass


@dataclass
class DataModuleConfig:
    train_dir: str
    test_dir: str
    batch_size: int = 64
    num_workers: int = 4
    image_size: tuple = (64, 64)