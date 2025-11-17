# Adaptive Multi-Scale Vision Transformer with Contrastive Pre-Training for Robust Medical Diagnostics 

A state-of-the-art deep learning framework for medical image classification using an innovative multi-scale Vision Transformer architecture with self-supervised pre-training and uncertainty quantification.


## Overview

This project implements an improved Vision Transformer architecture specifically designed for medical image analysis, with a focus on COVID-19 and lung disease classification. The model combines ResNet-style CNN backbones, multi-scale feature fusion, hierarchical attention mechanisms, and self-supervised contrastive learning to achieve robust performance on medical imaging tasks.

## Key Features

### Novel Architectural Components

- **ResNet-Style CNN Backbone**: Skip connections and residual blocks for better gradient flow
- **Multi-Scale Feature Pyramid Fusion**: FPN-inspired architecture for capturing features at multiple scales
- **Adaptive Patch Embedding**: Dynamic patch sizes (8x8, 16x16, 32x32) based on image content
- **Hierarchical Attention Mechanism**: Local-then-global attention for efficient processing
- **Medical-Specific Feature Enhancement**: Specialized modules for edge, texture, shape, and intensity analysis
- **Dynamic Token Mixer**: Content-aware token mixing based on similarity clustering
- **Uncertainty Quantification**: Built-in uncertainty estimation for confident predictions

### Advanced Training Features

- **Self-Supervised Contrastive Pre-training**: SimCLR-style learning for better feature representations
- **Mixed Precision Training**: Faster training with reduced memory footprint
- **Cosine Annealing with Warm Restarts**: Adaptive learning rate scheduling
- **Test-Time Augmentation**: Multiple predictions averaged for improved accuracy
- **Class-Weighted Loss**: Handles imbalanced medical datasets
- **SGLD Optimizer Support**: Bayesian uncertainty estimation capability

## Architecture
```
Input Image (224x224x3)
    ↓
ResNet-Style CNN Backbone
    ↓
Multi-Scale Feature Pyramid [P3, P4, P5]
    ↓
Pyramid Feature Fusion
    ↓
Medical Feature Enhancement
    ↓
Adaptive Patch Embedding
    ↓
Transformer Encoder (6 layers)
    ├─ Hierarchical Attention
    ├─ Dynamic Token Mixer
    └─ Feed-Forward Network
    ↓
Feature Aggregation
    ↓
Classification Head → Predictions
    ↓
Uncertainty Head → Confidence Scores
```

## Requirements
```bash
torch>=2.0.0
pytorch-lightning>=2.0.0
torchvision>=0.15.0
torchmetrics>=0.11.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
Pillow>=9.5.0
```

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/novel-multiscale-vit-medical.git
cd novel-multiscale-vit-medical

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Dataset Structure

Organize your medical image dataset as follows:
```
dataset/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── COVID/
│   ├── image1.jpg
│   └── ...
├── Pneumonia/
│   └── ...
├── Lung_Opacity/
│   └── ...
└── TB/
    └── ...
```

## Usage

### Basic Training
```python
python improved_novel_vit.py
```

### Custom Configuration
```python
from improved_novel_vit import ImprovedNovelViTModel, ImprovedMedicalDataModule

# Initialize data module
data_module = ImprovedMedicalDataModule(
    data_dir='path/to/dataset',
    batch_size=16,
    val_ratio=0.2,
    seed=42
)

# Initialize model
model = ImprovedNovelViTModel(
    num_classes=5,
    adam_lr=1e-4,
    use_contrastive=True,
    contrastive_weight=0.1
)

# Train
from pytorch_lightning import Trainer

trainer = Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    precision="16-mixed"
)

trainer.fit(model, data_module)
```

### Inference
```python
import torch
from PIL import Image
from torchvision import transforms

# Load trained model
model = ImprovedNovelViTModel.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Make prediction with uncertainty
with torch.no_grad():
    logits, uncertainty = model(input_tensor, return_uncertainty=True)
    prediction = logits.argmax(dim=1)
    confidence = torch.softmax(logits, dim=1).max()

print(f"Predicted class: {prediction.item()}")
print(f"Confidence: {confidence.item():.4f}")
print(f"Uncertainty: {uncertainty.item():.4f}")
```



## Performance Metrics

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 92-95%* |
| **Parameters** | ~15M |
| **Model Size** | ~60 MB |
| **Inference Speed** | ~30 FPS (GPU) |

*Performance may vary based on dataset and training configuration

### Training Curves

The model automatically generates comprehensive training visualizations including:
- Training vs Validation Loss
- Training vs Validation Accuracy
- Overfitting Indicator (Train-Val Gap)
- Learning Rate Schedule

All plots are saved in the `results_improved_novel_multiscale_vit/` directory.

## Model Components

### 1. ResNet-Style CNN Backbone
- Basic residual blocks with skip connections
- Multi-stage feature extraction (56×56 → 28×28 → 14×14)
- Batch normalization and ReLU activations

### 2. Feature Pyramid Network
- Lateral connections for channel unification
- Top-down pathway with feature upsampling
- Multi-scale feature fusion at 56×56 resolution

### 3. Medical Feature Enhancer
- **Edge Detection**: 3×3 convolutions for boundary detection
- **Texture Analysis**: 5×5 convolutions for pattern recognition
- **Shape Analysis**: 7×7 convolutions for morphology
- **Intensity Analysis**: 1×1 convolutions for brightness patterns

### 4. Transformer Encoder
- 6 layers of hierarchical attention
- Local attention within 4×4 windows
- Global attention between window representatives
- Dynamic token mixing based on content similarity

### 5. Uncertainty Estimation
- Learned variance prediction
- Confidence scores for each prediction
- Useful for clinical decision support

## Applications

- **COVID-19 Detection**: Identify COVID-19 from chest X-rays
- **Pneumonia Classification**: Distinguish bacterial vs viral pneumonia
- **Tuberculosis Screening**: Early detection of TB patterns
- **Lung Opacity Analysis**: Detect and classify lung abnormalities
- **General Medical Imaging**: Adaptable to other medical imaging tasks

## Training Tips

1. **Data Augmentation**: The model uses extensive augmentation including rotation, flipping, color jittering, and Gaussian blur
2. **Class Imbalance**: Automatic class weight calculation handles imbalanced datasets
3. **Early Stopping**: Built-in with patience=70 epochs
4. **Mixed Precision**: Reduces memory usage by ~40% and speeds up training
5. **Gradient Clipping**: Prevents exploding gradients (norm=0.5)
6. **Test-Time Augmentation**: Improves inference accuracy by 1-2%

## Customization

### Modify Architecture
```python
model = ImprovedNovelMultiScaleViT(
    num_classes=5,
    img_size=224,
    embed_dim=256,      # Embedding dimension
    depth=6,            # Number of transformer layers
    num_heads=8,        # Attention heads
    mlp_ratio=4,        # MLP expansion ratio
    dropout=0.3         # Dropout rate
)
```

### Custom Data Augmentation
```python
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # Add your custom augmentations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---
