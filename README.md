# Plant Pathology Classification - ML Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-85%25-brightgreen)

## Overview

This project implements a **Deep Learning solution** for plant disease classification using Convolutional Neural Networks (CNN). The model classifies plant leaves into four categories:
- **Healthy**
- **Multiple Diseases**
- **Rust**
- **Scab**

The model achieves **85% accuracy** on the test dataset using transfer learning with MobileNetV2.

## 📊 Dataset

**Source:** [Kaggle - Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

### Dataset Statistics
- **Total Images:** 1,821 training samples
- **Classes:** 4 (Healthy, Multiple Diseases, Rust, Scab)
- **Image Format:** JPG
- **Resolution:** Various (preprocessed to 224×224)

### Class Distribution
| Class | Count | Percentage |
|-------|-------|-----------|
| Healthy | 516 | 28.3% |
| Rust | 622 | 34.2% |
| Scab | 592 | 32.5% |
| Multiple Diseases | 91 | 5.0% |

## 🏗️ Project Architecture

```
plant-pathology-classification/
├── data/
│   ├── train/
│   ├── test/
│   └── val/
├── notebooks/
│   └── plant-pathology-classification-acc-85.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── training.py
│   └── evaluation.py
├── models/
│   └── plant_disease_model.h5
├── requirements.txt
├── config.py
└── README.md
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **TensorFlow 2.0+** - Deep learning framework
- **Keras** - High-level neural networks API
- **OpenCV (cv2)** - Image processing
- **Pandas** - Data analysis and manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **Matplotlib & Seaborn** - Data visualization

### Database
- **SQLite3** - Lightweight database for metadata
- **SQL** - Data querying and management

### Additional Tools
- **Jupyter Notebook** - Interactive development
- **Kaggle API** - Dataset download automation

## 🚀 Quick Start

### Prerequisites
```bash
- Python 3.8 or higher
- pip or conda package manager
- 4GB+ RAM recommended
- GPU (optional but recommended for faster training)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SuryanshPhor11/plant-pathology-classification.git
cd plant-pathology-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Download Dataset

```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle datasets download -d emmarex/plantdisease

# Extract files
unzip plantdisease.zip -d data/
```

## 📝 Usage

### 1. Data Preprocessing

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(image_size=224)
train_data, val_data, test_data = preprocessor.prepare_data(
    train_path='data/train',
    val_path='data/val',
    test_path='data/test',
    batch_size=32
)
```

### 2. Model Training

```python
from src.model import PlantDiseaseClassifier
from src.training import train_model

# Initialize model
model = PlantDiseaseClassifier(
    num_classes=4,
    pretrained=True
)

# Train
history = train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    epochs=25,
    learning_rate=0.001
)
```

### 3. Model Evaluation

```python
from src.evaluation import evaluate_model

metrics = evaluate_model(
    model=model,
    test_data=test_data
)

print(f"Test Accuracy: {metrics['accuracy']:.2%}")
print(f"Test Loss: {metrics['loss']:.4f}")
```

### 4. Make Predictions

```python
from src.model import predict_image

# Predict on single image
prediction = predict_image('path/to/image.jpg', model)
print(f"Predicted Class: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## 📊 Model Performance

### Training Results
```
Epoch 1/25
- Loss: 14.2569
- Accuracy: 50.59%
- Val Loss: 1.0128
- Val Accuracy: 79.51%

...

Epoch 25/25
- Loss: 0.1750
- Accuracy: 93.96%
- Val Loss: 0.7529
- Val Accuracy: 80.98%
```

### Final Metrics
- **Training Accuracy:** 93.96%
- **Validation Accuracy:** 80.98%
- **Test Accuracy:** 85%
- **Model Parameters:** 2,257,984

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Healthy | 0.87 | 0.84 | 0.85 |
| Rust | 0.84 | 0.87 | 0.85 |
| Scab | 0.85 | 0.86 | 0.85 |
| Multiple Diseases | 0.82 | 0.79 | 0.80 |
| **Weighted Avg** | **0.85** | **0.85** | **0.85** |

## 🎯 Implementation Details

### Model Architecture

**Transfer Learning with MobileNetV2**
```
Input Layer (224×224×3)
    ↓
MobileNetV2 (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (4 units, Softmax)
```

### Data Augmentation
- Rotation: 20°
- Width Shift: 10%
- Height Shift: 10%
- Shear Range: 0.2
- Zoom Range: 0.2
- Horizontal Flip: Yes

### Training Configuration
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Batch Size:** 32
- **Epochs:** 25
- **Early Stopping:** Patience = 5

## 📂 Database Schema

### SQL Tables

```sql
-- Plant metadata
CREATE TABLE plants (
    plant_id INTEGER PRIMARY KEY,
    plant_name VARCHAR(100),
    common_name VARCHAR(100)
);

-- Disease information
CREATE TABLE diseases (
    disease_id INTEGER PRIMARY KEY,
    disease_name VARCHAR(100),
    severity_level VARCHAR(50)
);

-- Predictions log
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY,
    image_path VARCHAR(255),
    predicted_class VARCHAR(100),
    confidence FLOAT,
    prediction_time TIMESTAMP
);
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# config.py
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_STATE = 42
```

## 📦 Requirements

```
tensorflow>=2.0.0
keras>=2.4.0
opencv-python>=4.5.0
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
kaggle>=1.5.0
```

Install all requirements:
```bash
pip install -r requirements.txt
```

## 🎓 Results & Insights

### Key Findings
1. **MobileNetV2 Performance:** Transfer learning provided 85% accuracy with minimal training time
2. **Data Imbalance:** Class imbalance handled through augmentation and weighted loss
3. **Overfitting:** Mitigated using dropout and early stopping

### Sample Predictions

| Image | Actual | Predicted | Confidence |
|-------|--------|-----------|-----------|
| sample_1.jpg | Healthy | Healthy | 92% |
| sample_2.jpg | Rust | Rust | 87% |
| sample_3.jpg | Scab | Scab | 89% |

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16
```

### Dataset Download Issues
```bash
# Manual download from Kaggle
# Place in: data/raw/
```

### GPU Not Detected
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## 📈 Future Improvements

- [ ] Implement ensemble methods (VGG16, ResNet50)
- [ ] Deploy as REST API using Flask
- [ ] Create web interface with Streamlit
- [ ] Mobile app using TensorFlow Lite
- [ ] Real-time disease detection camera app
- [ ] Explainability with GradCAM visualization
- [ ] Additional plant diseases classification
- [ ] Model quantization for edge deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Suryansh Phor**
- GitHub: [@SuryanshPhor11](https://github.com/SuryanshPhor11)
- Email: suryansh@example.com

## 🙏 Acknowledgments

- Kaggle dataset provider: [emmarex](https://www.kaggle.com/emmarex)
- TensorFlow team for excellent documentation
- MobileNet authors for efficient architecture
- Plant Pathology research community

## 📞 Support

For issues and questions:
1. Check existing [Issues](https://github.com/SuryanshPhor11/plant-pathology-classification/issues)
2. Create new issue with detailed description
3. Provide error logs and environment details

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)
- [Plant Disease Classification Research](https://scholar.google.com/)
- [Deep Learning for Plant Disease Detection](https://arxiv.org/search/?query=plant+disease&searchtype=all)

---

**Last Updated:** 2024
**Model Version:** 1.0
**Accuracy:** 85%
