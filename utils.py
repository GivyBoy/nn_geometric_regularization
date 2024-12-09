from torch import nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import Dict, List


def plot_training_comparison(base_metrics: Dict[str, List[float]], geo_metrics: Dict[str, List[float]]) -> None:
    """
    Create side-by-side comparison of base CNN and geometric CNN performance.

    Args:
        base_metrics: Training metrics from base CNN
        geo_metrics: Training metrics from geometric CNN
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Performance Comparison: Base CNN vs Geometric CNN", fontsize=16, y=1.05)

    epochs = range(1, len(base_metrics["train_loss"]) + 1)

    # Plot training loss comparison
    axes[0].plot(epochs, base_metrics["train_loss"], "b-", label="Base CNN")
    axes[0].plot(epochs, geo_metrics["train_loss"], "r-", label="Geometric CNN")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot validation accuracy comparison
    axes[1].plot(epochs, base_metrics["val_accuracy"], "b-", label="Base CNN")
    axes[1].plot(epochs, geo_metrics["val_accuracy"], "r-", label="Geometric CNN")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def compare_activation_maps(
    base_model: nn.Module, geo_model: nn.Module, test_loader: DataLoader, device: torch.device, num_maps: int = 5
):
    """
    Randomly selects a sample image from the test set and compares 5 activation maps from the first and second layers
    of BaseCNN and GeometricCNN.

    Args:
        base_model: Trained BaseCNN model
        geo_model: Trained GeometricCNN model
        test_loader: DataLoader for the test set
        device: Computation device
        num_maps: Number of activation maps to compare per layer
    """
    base_model.eval()
    geo_model.eval()

    # Get a random sample from the test set
    try:
        sample_inputs, _ = next(iter(test_loader))
    except StopIteration:
        print("Test loader is empty. Skipping activation maps comparison.")
        return

    sample_idx = random.randint(0, sample_inputs.size(0) - 1)
    sample_input = sample_inputs[sample_idx].unsqueeze(0).to(device)

    # Get the image before normalization for display
    inv_normalize = transforms.Normalize(
        mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010], std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
    )
    sample_img = inv_normalize(sample_input.squeeze(0)).cpu().numpy()
    sample_img = np.transpose(sample_img, (1, 2, 0))
    sample_img = np.clip(sample_img, 0, 1)

    # Get feature maps
    with torch.no_grad():
        _, base_feature_maps = base_model(sample_input)
        _, geo_feature_maps = geo_model(sample_input)

    # Layers to compare (first and second layers)
    layers = ["Layer 1", "Layer 2"]
    layer_indices = [0, 1]  # Corresponding to layer1 and layer2

    for layer_idx, layer_name in zip(layer_indices, layers):
        base_fmap = base_feature_maps[layer_idx]  # (1, C, H, W)
        geo_fmap = geo_feature_maps[layer_idx]  # (1, C, H, W)

        # Select 5 random channels
        num_channels = base_fmap.size(1)
        if num_channels < num_maps:
            selected_channels = list(range(num_channels))
        else:
            selected_channels = random.sample(range(num_channels), num_maps)

        fig, axes = plt.subplots(num_maps + 1, 3, figsize=(15, 5 * (num_maps + 1)))
        fig.suptitle(f"{layer_name} Activation Maps Comparison", fontsize=16, y=1.02)

        # Display Reference Image
        axes[0, 0].imshow(sample_img)
        axes[0, 0].axis("off")
        axes[0, 0].set_title("Reference Image")

        # Titles for Base and Geometric CNN
        axes[0, 1].imshow(np.zeros((10, 10)), cmap="gray")  # Placeholder
        axes[0, 1].axis("off")
        axes[0, 1].set_title("Base CNN")

        axes[0, 2].imshow(np.zeros((10, 10)), cmap="gray")  # Placeholder
        axes[0, 2].axis("off")
        axes[0, 2].set_title("Geometric CNN")

        for i, channel in enumerate(selected_channels, start=1):
            # BaseCNN activation map
            base_feat = base_fmap[0, channel].cpu().numpy()
            base_feat_norm = (base_feat - base_feat.min()) / (base_feat.max() - base_feat.min() + 1e-8)
            axes[i, 1].imshow(base_feat_norm, cmap="viridis")
            axes[i, 1].axis("off")
            axes[i, 1].set_title(f"Base CNN - Channel {channel}")

            # GeometricCNN activation map
            geo_feat = geo_fmap[0, channel].cpu().numpy()
            geo_feat_norm = (geo_feat - geo_feat.min()) / (geo_feat.max() - geo_feat.min() + 1e-8)
            axes[i, 2].imshow(geo_feat_norm, cmap="viridis")
            axes[i, 2].axis("off")
            axes[i, 2].set_title(f"Geometric CNN - Channel {channel}")

        # Hide the reference image's other columns
        for i in range(1, num_maps + 1):
            axes[i, 0].axis("off")

        plt.tight_layout()
        plt.show()
