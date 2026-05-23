# models/visual_model.py  -  VisioVox v4
# TemporalVisualEncoder: 25 lip frames -> 128-dim embedding
# Uses 3D Conv frontend for temporal motion + ResNet18 + LayerNorm
# LayerNorm on output keeps visual embedding on same scale as audio bottleneck

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

VISUAL_DIM = 128


class TemporalVisualEncoder(nn.Module):
    def __init__(self, embedding_dim=VISUAL_DIM):
        super().__init__()
        self.frontend3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        resnet       = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.norm    = nn.LayerNorm(embedding_dim)
        self.drop    = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: [B, T, C, H, W] = [B, 25, 1, 112, 112]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)   # [B, C, T, H, W]
        x = self.frontend3d(x)          # [B, 64, T', H', W']
        x = x.mean(dim=2)               # temporal mean pool -> [B, 64, H', W']
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return self.norm(x)             # [B, 128]


# Alias so any old import of VisualEncoder still works
VisualEncoder = TemporalVisualEncoder