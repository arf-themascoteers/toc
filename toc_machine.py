import torch.nn as nn
import torch.nn.functional as F


class TocMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.machine = nn.Sequential(
            nn.Conv2d(3, 16, (16,16)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8)),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.Flatten(),
            nn.Linear(784, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )


    def forward(self, x):
        x = self.machine(x)
        return x

