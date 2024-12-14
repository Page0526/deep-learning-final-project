import lightning as pl 
from dataset import BrainTumourDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from ..configs.data_config import DataModuleConfig


class BrainTumorDatamodule(pl.LightningDataModule): 

    def __init__(self, configs: DataModuleConfig):
        super().__init__()
        self.train_dir = configs.train_dir
        self.test_dir = configs.test_dir 
        self.transform =  transforms.Compose([transforms.ToTensor(),
                                     transforms.Grayscale(),
                                     transforms.Resize((configs.image_size))])
                                     
        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers                                

    def setup(self, stage = None): 
        self.train_dataset = BrainTumourDataset(self.train_dir, self.transform)
        self.val_dataset = BrainTumourDataset(self.test_dir, self.transform)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, shuffle=True, batch_size= self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self): 
        return DataLoader(self.val_dataset, shuffle=True, batch_size = self.batch_size, num_workers=self.num_workers)