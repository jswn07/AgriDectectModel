# 🌾 Agricultural Disease Detection

A deep learning project that detects **Blackgram and Corn leaf diseases** from leaf images using **Transfer Learning with MobileNetV2**.  
This project focuses on accuracy, clean evaluation, and production-ready training practices.

---

## 📌 What This Project Does

- Classifies leaf images into **9 disease classes**
- Uses a **two-phase training strategy** (feature extraction + fine-tuning)
- Handles **class imbalance** effectively
- Prevents overfitting using augmentation and regularization
- Generates clear evaluation metrics and visual outputs

---

## 🦠 Disease Classes

**Blackgram**
- Anthracnose  
- Healthy  
- Leaf Crinkle  
- Powdery Mildew  
- Yellow Mosaic  

**Corn**
- Blight  
- Common Rust  
- Gray Leaf Spot  
- Healthy  

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- MobileNetV2 (ImageNet pretrained)  
- NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  

---

## 📂 Dataset Structure

dataset/
├── train/
│ ├── class_name/
│ └── ...
├── val/
│ └── ...
└── test/
└── ...


✔ A built-in check ensures **no data leakage** between training, validation, and test sets.

---

## ⚙️ Training Overview

### Phase 1 – Base Training
- MobileNetV2 frozen
- Strong data augmentation
- Loss: Categorical Crossentropy (with label smoothing)

### Phase 2 – Fine Tuning
- Top layers of MobileNetV2 unfrozen
- Batch Normalization layers frozen
- Loss: Focal Loss (to handle class imbalance)

**Optimizer:** Adam  
**Metrics:** Accuracy, Precision, Recall, AUC  

---

## 📊 Outputs Generated

model/
├── agri_model_best.keras # Best model based on validation loss
├── agri_disease_detector_final.keras # Final fine-tuned model
├── model_summary.txt # Full model architecture
training_history.csv # Training metrics per epoch
confusion_matrix.png # Normalized confusion matrix

---

## ▶️ How to Run

### 1. Install Dependencies
```
pip install tensorflow numpy matplotlib seaborn scikit-learn
```
### 2. Train the Model
```
python train.py
```
