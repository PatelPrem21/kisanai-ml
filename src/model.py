import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

def get_model(num_classes):
    model = efficientnet_b0(weights="IMAGENET1K_V1")
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    
    return model
