import torch.nn as nn
import torch
from torchvision.models import resnet34, inception_v3, alexnet


class ResNet34(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet34, self).__init__()

        self.model = resnet34(pretrained=pretrained)
        #changing the last layer to a fully connected one with output size 10 (number of classes)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        
        return self.model(x)

class InceptionV3(nn.Module):
    
    def __init__(self, pretrained=False):
        super(ResNet34, self).__init__()

        self.model = inception_v3(pretrained=pretrained)
        #changing the last layer to a fully connected one with output size 10 (number of classes)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        
        return self.model(x)
