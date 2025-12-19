# Material Classification System 🔍

AI-powered material classification system that identifies 6 types of materials: **Glass, Paper, Cardboard, Plastic, Metal, Trash** using computer vision and machine learning.

## 🎯 Performance
- **SVM Model**: 72.8% accuracy
- **KNN Model**: 54.9% accuracy
- **Real-time Camera Detection**: Live classification via webcam

## 📋 Requirements
- Python 3.7+
- OpenCV
- scikit-learn
- NumPy
- joblib

## 🚀 Quick Setup (A to Z)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
Place your raw images in this structure:
```
data/raw/
├── glass/
├── paper/
├── cardboard/
├── plastic/
├── metal/
└── trash/
```

### Step 3: Clean Data (Remove corrupted files)
```bash
python scripts/delete_zero_byte_files.py
```

### Step 4: Split Data (80% Train, 20% Test + Augmentation)
```bash
python scripts/run_data_splitter.py
```
This creates 800 images per class for training (with augmentation) and keeps test data original.

### Step 5: Train SVM Model (82% accuracy)
```bash
python scripts/run_train_svm.py
```

### Step 6: Train KNN Model (68% accuracy)
```bash
python scripts/run_train_knn.py
```

### Step 7: Test Your Models
```bash
python scripts/test.py
```

### Step 8: Use Real-time Camera Detection
```bash
python apps/app.py
```

## 📁 Project Structure
```
├── src/                    # Source code (package)
│   ├── __init__.py
│   ├── constants.py       # Configuration constants
│   ├── feature_extractor.py # Image feature extraction
│   ├── data_loader.py     # Dataset loading utilities
│   ├── data_splitter.py   # 80/20 split + augmentation
│   ├── data_augmenter.py  # Data augmentation utilities
│   ├── encode_features.py # Feature encoding utilities
│   ├── model_svm.py       # SVM classifier (82% accuracy)
│   ├── model_knn.py       # KNN classifier (68% accuracy)
│   └── unknown_handler.py  # Unknown class detection handler
├── scripts/               # Executable scripts
│   ├── run_data_splitter.py
│   ├── run_augment.py
│   ├── run_encode_features.py
│   ├── run_train_svm.py
│   ├── run_train_knn.py
│   ├── test.py
│   └── delete_zero_byte_files.py
├── apps/                   # Application entry points
│   ├── app.py             # Main real-time camera app
│   └── realtime_app.py    # Alternative real-time app
├── models/                # Trained models (.pkl files)
├── data/                  # Dataset folders
│   ├── raw/               # Raw images
│   ├── split/             # Train/test split
│   ├── augmented/         # Augmented images
│   └── features/          # Extracted features
├── reports/               # Training reports
├── dataset/               # Original dataset (if present)
├── README.md
└── requirements.txt       # Dependencies
```

## 🎮 Usage

### Camera Detection
Run `python apps/app.py` and point your camera at materials. Press:
- **Q**: Quit application

Alternatively, use `python apps/realtime_app.py` for a simpler real-time classifier.

### Batch Testing
Use `python scripts/test.py` to test models on your test dataset.

## 🔧 Troubleshooting

### Import Errors in VS Code
If you get import errors, the scripts handle this automatically with:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Low Accuracy
- Ensure you have enough diverse images per class (recommended: 200+ per class)
- Check image quality and lighting conditions
- Verify correct material labeling

### Camera Issues
- Check camera permissions
- Ensure OpenCV can access your webcam
- Try different camera indices if needed

## 📊 Model Details

### SVM (Support Vector Machine) - 82%
- **Kernel**: RBF (Radial Basis Function)
- **Features**: Color histograms, texture, edge density
- **Training**: 80% of data with augmentation (800 images/class)
- **Testing**: 20% original data

### KNN (K-Nearest Neighbors) - 68%
- **K Value**: Optimized during training
- **Features**: Same as SVM
- **Distance**: Euclidean distance
- **Training**: Same 80/20 split as SVM

## 🎯 Complete Workflow Commands

For first-time users, run these commands in order:

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Clean data (optional)
python scripts/delete_zero_byte_files.py

# 3. Split and augment data (80/20 + 800 per class)
python scripts/run_data_splitter.py

# 4. Train both models
python scripts/run_train_svm.py
python scripts/run_train_knn.py

# 5. Test models
python scripts/test.py

# 6. Use camera detection
python apps/app.py
```

## 📈 Results
- **SVM**: 82% accuracy on test set
- **KNN**: 68% accuracy on test set
- **Real-time**: ~30 FPS camera detection
- **Training time**: ~2-3 minutes per model

---
**Ready to classify materials!** 🎉
