import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from collections import defaultdict
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

from models import BaseCNN, GeometricCNN


class Trainer:
    """
    Handles model training and evaluation with improved stability and monitoring.
    Includes learning rate scheduling, gradient clipping, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        base_model: nn.Module = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler setup based on model type
        base_lr = 0.001
        if isinstance(model, GeometricCNN):
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=base_lr * 0.5,
                weight_decay=1e-4,
                amsgrad=True,
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=15, eta_min=1e-6)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=base_lr)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

        # Training state
        self.metrics = defaultdict(list)
        self.best_accuracy = 0.0
        self.patience = 0
        self.max_patience = 15
        self.clip_value = 1.0

        # Loss scaling for geometric regularization
        self.geo_weight_scheduler = lambda epoch: min(1.0, epoch / 10.0)

        self.base_model = base_model.to(device) if base_model else None

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with gradient clipping and loss scaling"""
        self.model.train()
        running_task_loss = 0.0
        running_geo_loss = 0.0
        current_epoch = len(self.metrics["epoch"])

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if isinstance(self.model, GeometricCNN):
                # Forward pass with geometric regularization
                outputs, feature_maps = self.model(inputs)
                task_loss = self.criterion(outputs, labels)

                # Compute geometric loss with progressive scaling
                geo_weight = self.geo_weight_scheduler(current_epoch)
                geo_loss = sum(self.model.geo_reg(fm) for fm in feature_maps)
                loss = task_loss + geo_weight * geo_loss

                running_task_loss += task_loss.item()
                running_geo_loss += geo_loss.item()
            else:
                # Forward pass for BaseCNN
                outputs, feature_maps = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_task_loss += loss.item()

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()

        # Compute average losses
        avg_task_loss = running_task_loss / len(self.train_loader)
        avg_geo_loss = running_geo_loss / len(self.train_loader) if isinstance(self.model, GeometricCNN) else 0.0

        return avg_task_loss, avg_geo_loss

    def validate(self) -> Tuple[float, float]:
        """Compute validation accuracy and loss"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs, _ = self.model(inputs)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate final metrics
        accuracy = 100.0 * correct / total
        avg_val_loss = val_loss / len(self.val_loader)

        # Update best accuracy and patience for early stopping
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.patience = 0
        else:
            self.patience += 1

        return accuracy, avg_val_loss

    def train(self, epochs: int = 15) -> Dict[str, List[float]]:
        """
        Complete training process with comprehensive monitoring and visualization
        Args:
            epochs: Number of training epochs
        Returns:
            Dictionary containing training metrics
        """
        for epoch in range(epochs):
            # Training phase with loss computation
            train_loss, geo_loss = self.train_epoch()
            accuracy, val_loss = self.validate()

            # Update learning rate scheduler
            self.scheduler.step()

            # Store all metrics for plotting
            self.metrics["epoch"].append(epoch)
            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_accuracy"].append(accuracy)
            if isinstance(self.model, GeometricCNN):
                self.metrics["geo_loss"].append(geo_loss)

            # Print detailed training progress
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"Training Loss: {train_loss:.3f}")
            if isinstance(self.model, GeometricCNN):
                print(f"Geometric Loss: {geo_loss:.3f}")
            print(f"Validation Loss: {val_loss:.3f}")
            print(f"Validation Accuracy: {accuracy:.2f}%")
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Check for early stopping
            if self.patience >= self.max_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {self.best_accuracy:.2f}%")
                break

        # Create final visualizations
        self.plot_training_metrics()

        return self.metrics

    def plot_training_metrics(self):
        """
        Create comprehensive visualization of training metrics.
        Includes loss curves, accuracy progression, and geometric metrics if applicable.
        """
        # Determine number of subplots based on model type
        n_plots = 3 if isinstance(self.model, GeometricCNN) else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        fig.suptitle("Training Progress", fontsize=16, y=1.05)

        epochs = self.metrics["epoch"]

        # Plot training and validation loss
        axes[0].plot(epochs, self.metrics["train_loss"], "b-", label="Training Loss")
        axes[0].plot(epochs, self.metrics["val_loss"], "r-", label="Validation Loss")
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Plot validation accuracy
        axes[1].plot(epochs, self.metrics["val_accuracy"], "g-")
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(True)

        # Plot geometric loss if applicable
        if isinstance(self.model, GeometricCNN):
            axes[2].plot(epochs, self.metrics["geo_loss"], "m-")
            axes[2].set_title("Geometric Regularization Loss")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Loss")
            axes[2].grid(True)

        plt.tight_layout()
        plt.show()
