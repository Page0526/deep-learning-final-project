import lightning as pl 
from dataset import BrainTumourDataset
from torch.utils.data import DataLoader
from torchvision import transforms


class BrainTumorDatamodule(pl.LightningDataModule): 

    def __init__(self, train_dir: str, test_dir: str, transform):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir 
        self.transform =  transforms.Compose([transforms.ToTensor(),
                                     transforms.Grayscale(),
                                     transforms.Resize((64,64))])

    def setup(self, stage = None): 
        self.train_dataset = BrainTumourDataset(self.train_dir, self.transform)
        self.val_dataset = BrainTumourDataset(self.test_dir, self.transform)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, shuffle=True, batch_size= 64, num_workers=4)

    def val_dataloader(self): 
        return DataLoader(self.val_dataset, shuffle=True, batch_size=64, num_workers=4)