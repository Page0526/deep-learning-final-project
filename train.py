from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.trainer import Trainer


import wandb
import torch
from dotenv import load_dotenv
import os
from model.resnet.resnet_model import ResNetModel
from model.vgg16.vgg16_model import VGG16
from model.densenet.densnet_model import DenseNet121
from model.capsnet.capsnet_model import CapsuleNetwork
from model.classifier import BrainTumorLightning

from configs.model_config import ResNet_Config, DenseNet_Config, CapsNet_Config, VGG16_Config
from configs.data_config import DataModuleConfig
from data.datamodule import BrainTumorDatamodule

from dataclasses import asdict
import argparse


load_dotenv()
wandb.login(key= os.getenv("WANDB_API"))


def setup_model(model_name: str): 

    if model_name == 'resnet': 
        args = ResNet_Config()
        model = ResNetModel(**asdict(args))
        return model
    elif model_name == 'vgg16':
        args = VGG16_Config()
        model = VGG16(**asdict(args))
        return model
    elif model_name == 'densenet':
        args = DenseNet_Config()
        model = DenseNet121(**asdict(args))
        return model
    elif model_name == 'capsnet':
        args = CapsNet_Config()
        model = CapsuleNetwork(**asdict(args))
        return model
    

def train(model_name: str, max_epochs: int = 50):
    
    # model 
    model = setup_model(model_name)
    lightning_model = BrainTumorLightning(model)

    # datamodule 
    data_args = DataModuleConfig()
    datamodule = BrainTumorDatamodule(data_args)

    wandb_logger = WandbLogger(
        project="Brain Tumor", name = model_name,         
        save_dir="/kaggle/working/",  
    )

    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename=model_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
    )
    
    trainer = Trainer(
        max_epochs=50,
        logger=wandb_logger,        
        callbacks=[early_stopping, model_checkpoint], 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto"
    )

    trainer.fit(lightning_model, datamodule)


def save_model_as_onnx(lightning_model, file_path="model.onnx", input_example=None):
    if input_example is None:
        if file_path.split(".")[-1] == "vgg16" or file_path.split(".")[-1] == "resnet":
           
            input_example = torch.randn(1, 1, 64, 64)  
        else : 
            input_example = torch.randn(1, 3, 224, 224)

    lightning_model.eval()

    torch.onnx.export(
        lightning_model.model,                 
        input_example,                        
        file_path,                            
        export_params=True,                    
        opset_version=11,                    
        input_names=["input"],                
        output_names=["output"],               
        dynamic_axes={                       
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model saved to {file_path} in ONNX format.")


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Train a brain tumor classification model")
    parser.add_argument("-model_name", type=str, required=True, help="Name of the model to train (resnet, vgg16, densenet, capsnet)")
    parser.add_argument("-max_epochs", type=str, required=True, help="Number of epochs to train the model")
    args = parser.parse_args()

    try:
        train(args.model_name, int(args.max_epochs))
        save_model_as_onnx(BrainTumorLightning(setup_model(args.model_name)), file_path=f"{args.model_name}.onnx")
    except ValueError as e:
        print(e)