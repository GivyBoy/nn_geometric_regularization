import torch
import torch.nn as nn
import torch.nn.functional as F
from geometric_regularization import GeometricRegularization
from typing import List, Tuple


class ResidualBlock(nn.Module):
    """
    Residual Block with two convolutional layers and a shortcut connection.
    Facilitates better gradient flow and allows for deeper networks.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out


class BaseCNN(nn.Module):
    """
    Residual CNN architecture serving as the baseline model.
    Includes residual connections for better gradient flow and feature reuse.
    """

    def __init__(self, num_classes: int = 10):
        super(BaseCNN, self).__init__()
        # Define layers using ResidualBlock
        self.layer1 = ResidualBlock(3, 64, stride=1)  # Output: 64 x 32 x 32
        self.layer2 = ResidualBlock(64, 128, stride=2)  # Output: 128 x 16 x 16
        self.layer3 = ResidualBlock(128, 256, stride=2)  # Output: 256 x 8 x 8

        # Pooling and dropout
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Output: 256 x 4 x 4
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the BaseCNN.
        Returns:
            - Output logits
            - List of feature maps from different layers
        """
        feature_maps = []

        x = self.layer1(x)  # 64 x 32 x 32
        feature_maps.append(x)

        x = self.layer2(x)  # 128 x 16 x 16
        feature_maps.append(x)

        x = self.layer3(x)  # 256 x 8 x 8
        feature_maps.append(x)

        x = self.pool(x)  # 256 x 4 x 4
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x, feature_maps


class GeometricCNN(BaseCNN):
    """
    CNN with geometric regularization.
    Extends BaseCNN to include geometric regularization on feature maps.
    """

    def __init__(
        self,
        num_classes: int = 10,
        lambda_area: float = 0.001,
        lambda_curv: float = 0.001,
    ):
        super(GeometricCNN, self).__init__(num_classes)
        self.geo_reg = GeometricRegularization(lambda_area, lambda_curv)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for GeometricCNN.
        Returns:
            - Output logits
            - List of feature maps from different layers for geometric regularization
        """
        feature_maps = []

        x = self.layer1(x)  # 64 x 32 x 32
        feature_maps.append(x)

        x = self.layer2(x)  # 128 x 16 x 16
        feature_maps.append(x)

        x = self.layer3(x)  # 256 x 8 x 8
        feature_maps.append(x)

        x = self.pool(x)  # 256 x 4 x 4
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x, feature_maps
