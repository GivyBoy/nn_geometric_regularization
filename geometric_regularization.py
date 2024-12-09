import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class GeometricRegularization(nn.Module):
    """
    Implements geometric regularization for CNN feature maps using differential geometry concepts.
    This version includes improved numerical stability and better loss scaling.
    """

    def __init__(self, lambda_area: float = 0.001, lambda_curv: float = 0.001):
        super().__init__()
        self.lambda_area = lambda_area
        self.lambda_curv = lambda_curv
        self.eps = 1e-6  # Increased epsilon for better numerical stability

        # Instance normalization to normalize feature maps before regularization
        self.instance_norm = nn.InstanceNorm2d(1, affine=False)

    def compute_derivatives(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute first derivatives using Sobel filters for better stability.
        Args:
            feature_map: Tensor of shape (B, C, H, W)
        Returns:
            du, dv: First derivatives in u and v directions
        """
        B, C, H, W = feature_map.shape

        # Sobel filters for better gradient computation
        du_kernel = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=feature_map.device)
            .view(1, 1, 3, 3)
            .repeat(C, 1, 1, 1)
        )
        dv_kernel = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=feature_map.device)
            .view(1, 1, 3, 3)
            .repeat(C, 1, 1, 1)
        )

        # Convert kernels to feature map's dtype
        du_kernel = du_kernel.to(feature_map.dtype)
        dv_kernel = dv_kernel.to(feature_map.dtype)

        # Compute gradients using convolution
        padded = F.pad(feature_map, (1, 1, 1, 1), mode="reflect")
        du = F.conv2d(padded, du_kernel, groups=C) / 8.0
        dv = F.conv2d(padded, dv_kernel, groups=C) / 8.0

        return du, dv

    def compute_second_derivatives(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute second derivatives with improved stability using central differences.
        Args:
            feature_map: Tensor of shape (B, C, H, W)
        Returns:
            duu, dvv, duv: Second derivatives
        """
        padded = F.pad(feature_map, (2, 2, 2, 2), mode="reflect")

        # Second derivatives using central differences
        duu = (padded[:, :, 2:-2, 4:] - 2 * padded[:, :, 2:-2, 2:-2] + padded[:, :, 2:-2, :-4]) / 4.0
        dvv = (padded[:, :, 4:, 2:-2] - 2 * padded[:, :, 2:-2, 2:-2] + padded[:, :, :-4, 2:-2]) / 4.0
        duv = (
            padded[:, :, 3:-1, 3:-1] - padded[:, :, 3:-1, 1:-3] - padded[:, :, 1:-3, 3:-1] + padded[:, :, 1:-3, 1:-3]
        ) / 4.0

        return duu, dvv, duv

    def compute_metric_tensor(
        self, du: torch.Tensor, dv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute components of the metric tensor (first fundamental form) with improved stability.
        Args:
            du, dv: First derivatives
        Returns:
            guu, gvv, guv: Metric tensor components
        """
        # Add small constant for stability
        guu = torch.sum(du * du, dim=1) + self.eps
        gvv = torch.sum(dv * dv, dim=1) + self.eps
        guv = torch.sum(du * dv, dim=1)

        return guu, gvv, guv

    def compute_mean_curvature(
        self,
        du: torch.Tensor,
        dv: torch.Tensor,
        duu: torch.Tensor,
        dvv: torch.Tensor,
        duv: torch.Tensor,
        guu: torch.Tensor,
        gvv: torch.Tensor,
        guv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean curvature with improved numerical stability.
        Uses the Laplacian formulation that works for feature maps of any dimensionality.
        """
        # Normalize derivatives for better numerical stability
        du_norm = torch.sqrt(torch.sum(du * du, dim=1) + self.eps)
        dv_norm = torch.sqrt(torch.sum(dv * dv, dim=1) + self.eps)

        du = du / (du_norm.unsqueeze(1) + self.eps)
        dv = dv / (dv_norm.unsqueeze(1) + self.eps)

        # Compute mean curvature using normalized derivatives
        det_g = guu * gvv - guv * guv + self.eps
        H = (
            gvv * torch.sum(duu * du, dim=1) + guu * torch.sum(dvv * dv, dim=1) - 2 * guv * torch.sum(duv * du, dim=1)
        ) / (2 * torch.sqrt(det_g))

        return H

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric regularization loss with improved stability and normalization.
        Args:
            feature_map: Tensor of shape (B, C, H, W)
        Returns:
            loss: Geometric regularization loss
        """
        # Normalize each feature map channel-wise
        # Assuming feature_map shape is (B, C, H, W)
        B, C, H, W = feature_map.shape
        # Reshape to (B*C, 1, H, W) for instance normalization
        feature_map = feature_map.view(B * C, 1, H, W)
        feature_map = self.instance_norm(feature_map)
        # Reshape back to (B, C, H, W)
        feature_map = feature_map.view(B, C, H, W)

        # Compute derivatives
        du, dv = self.compute_derivatives(feature_map)
        duu, dvv, duv = self.compute_second_derivatives(feature_map)

        # Compute metric tensor
        guu, gvv, guv = self.compute_metric_tensor(du, dv)

        # Compute area term with gradient clipping
        det_g = guu * gvv - guv * guv + self.eps
        area_loss = torch.sqrt(det_g).mean()
        area_loss = torch.clamp(area_loss, max=10.0)

        # Compute mean curvature with stability improvements
        H = self.compute_mean_curvature(du, dv, duu, dvv, duv, guu, gvv, guv)
        curvature_loss = torch.abs(H).mean()
        curvature_loss = torch.clamp(curvature_loss, max=10.0)

        # Combine losses with proper scaling
        total_loss = self.lambda_area * area_loss + self.lambda_curv * curvature_loss

        return total_loss
