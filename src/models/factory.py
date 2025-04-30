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
        AlexNetModel = models.alexnet(weights=('DEFAULT' if pretrained else None))
        for param in AlexNetModel.parameters():
            param.requires_grad = False

        AlexNetModel.classifier = nn.Sequential(
            nn.Linear(9216, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

        for param in AlexNetModel.classifier.parameters():
            param.requires_grad = True

        return AlexNetModel
    elif model_name == 'resnet34':
        ResNet34Model = models.resnet34(weights=('DEFAULT' if pretrained else None))
        for param in ResNet34Model.parameters():
            param.requires_grad = False
        
        ResNet34Model.fc = nn.Linear(512, num_classes)

        for param in ResNet34Model.fc.parameters():
            param.requires_grad = True

        return ResNet34Model
    else:
        raise ValueError(f"Unknown model: {model_name}")