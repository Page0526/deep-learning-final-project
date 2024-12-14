from lightning.pytorch.loggers import WandbLogger
import wandb
from dotenv import load_dotenv
import os
from model.resnet.resnet_model import ResNetModel
from model.vgg16.vgg16_model import VGG16
from model.densenet.densnet_model import DenseNet121
from model.capsnet.capsnet_model import CapsuleNetwork



load_dotenv()
wandb.login(key= os.getenv("WANDB_API"))




def train(model_name: str):
    if model_name == "resnet": 
        pass 
    


    pass

def save_model(): 
    pass 




if __name__ == '__main__': 
    pass 