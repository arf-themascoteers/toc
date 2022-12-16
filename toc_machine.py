import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class TocMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        number_input = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(number_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, x):
        x = self.resnet(x)
        return x

