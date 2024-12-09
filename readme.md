#### **Geometric Regularization in CNNs**

---

#### **Overview**

This project introduces **Geometric Regularization** to improve the generalization and robustness of Convolutional Neural Networks (CNNs) by imposing curvature and area constraints on feature maps. Inspired by **Differential Geometry**, it applies mathematical principles to enforce smoothness and structure in learned representations. 

Key Features:
- **Residual CNN Architecture** with geometric constraints.
- **Geometric Regularization Module** using area minimization and curvature control.
- Training and evaluation on the **CIFAR-10** dataset.

---

#### **Files**

| File                       | Description                                         |
|----------------------------|-----------------------------------------------------|
| `main.py`                  | Runs the training and evaluation pipeline.          |
| `cnn_models.py`            | Contains all CNN-related classes and architectures.|
| `geometric_regularization.py` | Implements geometric regularization functionality.|
| `utils.py`                 | Data loaders, trainers, and visualization utilities.|
| `requirements.txt`         | Python dependencies for the project.               |
| `mth551_final_project.pdf`         | Research Paper               |
---

#### **Theoretical Background**

This project utilizes principles from **Differential Geometry**:
1. **Feature Maps as Manifolds**: Viewing activation maps as surfaces in high-dimensional space.
2. **Geometric Regularization**:
   - **Area Term**: Penalizes unnecessary expansion in feature maps.
   - **Curvature Term**: Controls abrupt changes in feature representation.

---

#### **Implemented Models**

1. **Base CNN**:
   - Built with residual blocks.
   - Benchmark for comparison.

2. **Geometric CNN**:
   - Extends Base CNN with a geometric regularization module.

---

#### **How to Run**

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Run Training**:
   ```
   python main.py
   ```

3. **View Results**:
   - Training metrics and feature map comparisons are visualized.

---

#### **Experiment Details**

1. **Dataset**: CIFAR-10
   - Training: 50,000 images
   - Validation: 10,000 images
2. **Hyperparameters**:
   - Optimizer: Adam
   - Learning Rate: 0.001
   - Regularization Coefficients: λ_area=0.01, λ_curv=0.1
   - Training Epochs: 15

---

#### **Results**

| Model             | Validation Accuracy | Overfitting Reduction |
|--------------------|---------------------|------------------------|
| **Base CNN**       | 87.51%             | Low                   |
| **Geometric CNN**  | 87.10%             | Moderate              |

---

#### **Key Features**

1. **Geometric Regularization**:
   - Enforces smooth and efficient representations by minimizing curvature and area.
   - Promotes generalization and reduces overfitting.

2. **Residual Connections**:
   - Simplify gradient flow for training deeper architectures.

---

#### **Future Directions**

1. Apply geometric constraints to specialized tasks (e.g., medical imaging).
2. Explore adaptive and architecture-specific regularization strategies.
3. Scale experiments to larger architectures and datasets.

---

#### **References**

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- *Theoretical Foundations of Geometric Regularization in CNNs* (Anthony Givans)
---
