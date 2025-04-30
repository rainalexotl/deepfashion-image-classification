import torch.nn as nn

from torchvision import models
from src.models.baseline import BaselineCNN

def get_model(config):
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    pretrained = config['model'].get('pretrained', False)

    if model_name == 'baseline':
        return BaselineCNN(num_classes)
    elif model_name == 'alexnet':
        AlexNetModel = models.alexnet(pretrained=pretrained)
        for param in AlexNetModel.parameters():
            param.requires_grad = False

        AlexNetModel.classifier = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        return AlexNetModel
    else:
        raise ValueError(f"Unknown model: {model_name}")