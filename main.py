import torch
from trainer import Trainer
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
from models import BaseCNN, GeometricCNN
from utils import plot_training_comparison, compare_activation_maps

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# Train Base CNN
print("\nTraining Base CNN...")
base_cnn = BaseCNN()
base_trainer = Trainer(base_cnn, trainloader, testloader, device)
base_metrics = base_trainer.train(epochs=15)

# Train Geometric CNN
print("\nTraining Geometric CNN...")
geo_cnn = GeometricCNN(lambda_area=0.01, lambda_curv=0.1)
geo_trainer = Trainer(geo_cnn, trainloader, testloader, device, base_model=base_cnn)
geo_metrics = geo_trainer.train(epochs=15)

# Plot performance comparison
plot_training_comparison(base_metrics, geo_metrics)

# Compare Activation Maps
compare_activation_maps(base_cnn, geo_cnn, testloader, device, num_maps=5)

# Final Results
print("\nFinal Results:")
print(f"Base CNN - Best Validation Accuracy: {base_trainer.best_accuracy:.2f}%")
print(f"Geometric CNN - Best Validation Accuracy: {geo_trainer.best_accuracy:.2f}%")
