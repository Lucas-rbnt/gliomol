# Standard libraries
from typing import Union, Tuple

# Third-party libraries
import torch
import torch.nn as nn
from monai.networks.nets import resnet10

class ResNet10Wrapper(nn.Module):
    def __init__(self, in_channels=1) -> None:
        super().__init__()
        self.encoder = resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=1)
        self.encoder.fc = nn.Identity()
        
        self.classification = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
            
    def forward(self, x: torch.tensor) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        x = self.encoder(x)
        return self.classification(x)