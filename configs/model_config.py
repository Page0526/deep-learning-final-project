from dataclasses import dataclass


@dataclass
class ResNet_Config: 
    feat_dim: int = 4
    dim_in: int = 256

@dataclass 
class VGG16_Config: 
    num_classes: int = 4 



@dataclass
class DenseNet_Config:
    growth_rate: int = 32
    block_config: list = [6, 12, 24, 16]
    num_init_features: int = 64
    bn_size: int = 4
    drop_rate: int = 0.5
    num_classes: int = 4
    memory_efficient: bool = False
    compression_factor: float = 0.5 
    grayscale=True 


@dataclass 
class CapsNet_Config: 
    
    pass 


