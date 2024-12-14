import lightning as pl 
import torch.nn as nn 
from torchmetrics.classification import Accuracy, F1Score
import torch


class BrainTumorLightning(pl.LightningModule):
    def __init__(self, model, num_classes=4, lr=1e-3, criterion = None, is_capsnet = False):
        super(BrainTumorLightning, self).__init__()
        self.model = model
        self.lr = lr
        if criterion : self.criterion = criterion
        else: self.criterion = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(task="multiclass", num_classes=4)
        self.f1_score = F1Score(task="multiclass", num_classes=4, average='macro')
        self.is_capsnet = is_capsnet

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.is_capsnet:
            # targets = torch.eye(4, device = "cuda").index_select(dim=0, index=targets)
            outputs, reconstructions, y = self.forward(inputs)
            loss = self.criterion(outputs, targets, inputs, reconstructions)
        else: 
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, targets)
        
        acc = self.accuracy(outputs, targets)
        f1 = self.f1_score(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        if self.is_capsnet:
            # targets = torch.eye(4, device = "cuda").index_select(dim=0, index=targets)
            outputs, reconstructions, y = self.forward(inputs)
            loss = self.criterion(outputs, targets, inputs, reconstructions)
        else: 
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, targets)
    
        acc = self.accuracy(outputs, targets)
        f1 = self.f1_score(outputs, targets)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1_score', f1, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            }
        }