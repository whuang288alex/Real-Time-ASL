import torch.nn as nn
from torchvision import models

class ASLModel(nn.Module):
    def __init__(self, num_class, device):
        super(ASLModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_class)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return x