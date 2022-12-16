import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class TocMachine(nn.Module):
    def __init__(self):
        super().__init__()
        #self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(16, 32, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256,1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

