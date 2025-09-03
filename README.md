# 🎭 DeepFake Detection using EfficientNet + BiLSTM

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</div>

<p align="center">
  <strong>A sophisticated deep learning approach to detect AI-generated fake videos using spatial-temporal analysis</strong>
</p>

---

## 🌟 Overview

This project implements a state-of-the-art deepfake detection system that combines the power of **EfficientNet-B0** for spatial feature extraction with **bidirectional LSTMs** for temporal analysis. The model is specifically designed to identify manipulated videos with high accuracy while maintaining computational efficiency.

### 🎯 Key Features

- **Hybrid Architecture**: EfficientNet + BiLSTM for comprehensive spatial-temporal analysis
- **Attention Mechanism**: Temporal attention for focusing on critical video segments  
- **Advanced Training**: Mixed precision training with gradient scaling
- **Robust Data Handling**: Comprehensive error handling and data validation
- **Class Imbalance**: Weighted sampling and Focal Loss for imbalanced datasets
- **Freezing Strategy**: Selective layer freezing for transfer learning optimization

## 🏗️ Architecture

```
Video Input (20 frames)
        ↓
   Frame Extraction
        ↓
   EfficientNet-B0 (Feature Extraction)
        ↓
   Spatial Dropout
        ↓
   Bidirectional LSTM (2 layers)
        ↓
   Temporal Attention
        ↓
   Classification Head
        ↓
   Binary Output (Real/Fake)
```

### 📊 Model Components

| Component | Description | Purpose |
|-----------|-------------|---------|
| **EfficientNet-B0** | Pre-trained CNN backbone | Spatial feature extraction from frames |
| **BiLSTM** | 2-layer bidirectional LSTM | Temporal sequence modeling |
| **Attention** | Custom temporal attention | Focus on discriminative time steps |
| **Dropout** | Spatial & temporal regularization | Prevent overfitting |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
Pillow>=8.3.0
```

## 📁 Dataset Structure

The model expects the **Celeb-DF v2** dataset structure:

```
dataset/
├── Celeb-real/           # Real celebrity videos
├── YouTube-real/         # Real YouTube videos  
├── Celeb-synthesis/      # Deepfake videos
└── List_of_testing_videos.txt  # Test set definition
```

## 🚀 Usage

### Quick Start

```python
from model import DeepfakeDetector
import torch

# Load pre-trained model
model = DeepfakeDetector(config)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict on video
prediction = model.predict_video('path/to/video.mp4')
print(f"Prediction: {'Real' if prediction > 0.5 else 'Fake'}")
```

### Training from Scratch

```bash
# Ensure your dataset path is set correctly in the config
python main.py
```

### Configuration

Key configuration parameters in `CONFIG`:

```python
CONFIG = {
    'data_root': '/path/to/celeb-df-v2',
    'batch_size': 4,
    'num_frames': 20,
    'frame_size': 224,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'use_attention': True,
    'use_focal_loss': True,
    'layers_to_freeze': 3,  # EfficientNet layers to freeze
}
```

## 📈 Performance

### Training Features

- **Mixed Precision Training**: Faster training with FP16
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive LR reduction
- **Comprehensive Metrics**: Accuracy, Precision, Recall, AUC-ROC
- **Confusion Matrix Visualization**: Per-epoch analysis

### Monitoring

The training process generates:
- Real-time progress bars
- Confusion matrices per epoch
- Training curves and metrics plots
- Model checkpoints every 5 epochs

## 🔧 Advanced Features

### 🧠 Transfer Learning Strategy
- **🔒 Selective Layer Freezing**: Strategic EfficientNet layer control
- **🎯 Fine-tuning**: Deepfake-specific feature adaptation
- **🏛️ Pre-trained Knowledge**: Preserved ImageNet representations

### 👁️ Attention Visualization
```python
# Extract attention weights for analysis
context, attention_weights = model.attention(lstm_output)
visualize_attention(attention_weights, frame_indices)
```

<div align="center">
  
  **Attention Heatmap Demo:**
  
  ![Attention](https://img.shields.io/badge/Temporal%20Attention-Visualized-purple?style=for-the-badge&logo=eye)
  
</div>

### 🔄 Data Augmentation
- **🔁 Random Horizontal Flipping**: Spatial variation
- **🎨 Color Jittering**: Brightness, contrast, saturation enhancement  
- **🛡️ Robust Frame Extraction**: Comprehensive error handling

## 📊 Results Visualization

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="500">
</div>

The training process automatically generates:

1. **📈 Training Curves**: Loss and accuracy evolution over epochs
2. **🎭 Confusion Matrices**: Per-epoch classification performance  
3. **📊 Metric Plots**: Precision, recall, and AUC trends over time
4. **🔥 Attention Maps**: Temporal attention weight visualization

<div align="center">
  
  **Visualization Gallery:**
  
  ![Plots](https://img.shields.io/badge/Training%20Plots-Generated-blue?style=flat-square)
  ![Matrix](https://img.shields.io/badge/Confusion%20Matrix-Updated-green?style=flat-square)
  ![Attention](https://img.shields.io/badge/Attention%20Maps-Visualized-purple?style=flat-square)
  
</div>

## 🤝 Contributing

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">
</div>

We welcome contributions! Join our mission to combat deepfakes 🚀

### 🛠️ Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting  
black .
flake8 .
```

<div align="center">
  
  **Contribution Stats:**
  
  ![Contributors](https://img.shields.io/github/contributors/saim-glitch/Deepfake_detection?style=for-the-badge)
  ![Issues](https://img.shields.io/github/issues/saim-glitch/Deepfake_detection?style=for-the-badge)
  ![PRs](https://img.shields.io/github/issues-pr/saim-glitch/Deepfake_detection?style=for-the-badge)
  
</div>

## 📄 License

<div align="center">
  
  ![License](https://img.shields.io/github/license/saim-glitch/Deepfake_detection?style=for-the-badge&color=brightgreen)
  
</div>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257449-6eda921d-6d4f-4b69-b00b-5e1c3d60b5b5.gif" width="300">
</div>

- 📖 EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- 📖 Celeb-DF Dataset: [Li et al., 2020](https://arxiv.org/abs/1909.12962)  
- 📖 Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)

## 🙏 Acknowledgments

- 🎭 The Celeb-DF v2 dataset creators
- 🔥 PyTorch and torchvision teams
- 🧠 EfficientNet architecture developers
- 🌟 Open source community

## 📬 Contact

<div align="center">
  
  **Let's Connect!**
  
  [![GitHub](https://img.shields.io/badge/GitHub-saim--glitch-black?style=for-the-badge&logo=github)](https://github.com/saim-glitch)
  [![Email](https://img.shields.io/badge/Email-Contact%20Me-red?style=for-the-badge&logo=gmail)](mailto:your.email@domain.com)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourprofile)
  
</div>

For questions or support:
- 🐛 Open an issue on GitHub  
- 📧 Email: mohammadsaim78622@gmail.com
- 🐦 Twitter: @yourusername

---

<!-- Footer Animation -->
<div align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6&height=100&section=footer&text=⭐%20Star%20this%20repo%20if%20you%20find%20it%20helpful!&fontSize=24&fontColor=white&animation=twinkling" />
</div>

<div align="center">
  
  **Made with ❤️ and lots of ☕**
  
  ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=saim-glitch.Deepfake_detection)
  ![Stars](https://img.shields.io/github/stars/saim-glitch/Deepfake_detection?style=social)
  ![Forks](https://img.shields.io/github/forks/saim-glitch/Deepfake_detection?style=social)
  
</div>
