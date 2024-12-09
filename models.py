import torch
import torch.nn as nn
import torch.nn.functional as F
from geometric_regularization import GeometricRegularization


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class BaseCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseCNN, self).__init__()
        self.layer1 = ResidualBlock(3, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        feature_maps = []
        x = self.layer1(x)
        feature_maps.append(x)
        x = self.layer2(x)
        feature_maps.append(x)
        x = self.layer3(x)
        feature_maps.append(x)
        x = self.pool(x).view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x), feature_maps


class GeometricCNN(BaseCNN):
    def __init__(self, num_classes=10, lambda_area=0.01, lambda_curv=0.1):
        super(GeometricCNN, self).__init__(num_classes)
        self.geo_reg = GeometricRegularization(lambda_area, lambda_curv)

    def forward(self, x: torch.Tensor):
        logits, feature_maps = super().forward(x)
        return logits, feature_maps
