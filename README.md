# EE658 – Machine Learning (hmannem)

A collection of machine learning assignments and a production-level project notebook for the EE658 course, covering foundational algorithms through hands-on implementation and evaluation on real-world datasets.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `Assignment-1.ipynb` | Linear Regression – gradient descent from scratch |
| `Assignment-2.ipynb` | Logistic Regression – sklearn and custom implementation |
| `Assignment-3.ipynb` | Neural Networks – from scratch and with sklearn |
| `Assignment-4.ipynb` | PCA + Classification on MNIST |
| `Project - Dimensionality Reduction on MNIST.ipynb` | Production-level project: SVD & PCA for dimensionality reduction |
| `app.py` | Production Streamlit app for interactive ML model prediction |

---

## 📓 Assignments

### Assignment 1 – Linear Regression on Insurance Expenses
**Dataset:** Medical insurance expenses (1,338 records)

Implements and compares multiple linear regression approaches to predict insurance expenses based on age, BMI, smoking status, and other features.

**Key steps:**
- Data preprocessing: missing value removal, binary encoding, Min-Max normalization
- Custom gradient descent implementation (constant and exponential decay learning rates)
- Learning rate analysis across multiple values (0.05, 0.1, 0.5)
- Scikit-learn `LinearRegression` and normal equation for comparison

**Outcomes:**
| Model | MAE | MSE |
|-------|-----|-----|
| Gradient Descent (constant LR) | ~4,530 | ~43,026,754 |
| Scikit-learn LinearRegression | ~4,587 | ~43,164,059 |
| Normal Equation | ~4,587 | ~43,164,059 |

> The custom gradient descent achieves performance on par with scikit-learn's solver, validating the from-scratch implementation.

---

### Assignment 2 – Logistic Regression on 2D Classification Data

Explores binary classification with logistic regression using both scikit-learn and a hand-crafted implementation.

**Key steps:**
- Part A: Scikit-learn `LogisticRegression` as baseline
- Part B: Custom logistic regression from scratch using sigmoid + gradient descent
- Part C: Polynomial feature engineering (adding Feature1², Feature2²)

**Outcomes:**
| Approach | Accuracy |
|----------|----------|
| Scikit-learn Logistic Regression | **90%** |
| Custom Logistic Regression (scratch) | **90%** |
| Feature-Engineered (polynomial) | **82.5%** |

> The from-scratch logistic regression matches scikit-learn's accuracy, confirming correct implementation of the sigmoid function and gradient update rules.

---

### Assignment 3 – Neural Networks for Customer Segmentation
**Dataset:** Customer segmentation (train/test CSVs, 4-class classification)

Implements and compares neural network models for customer segment prediction (A, B, C, D).

**Key steps:**
- Custom 2-layer neural networks with Sigmoid and ReLU activations (from scratch)
- Scikit-learn `MLPClassifier` with 2-layer (16, 8) and 3-layer (32, 16, 8) architectures
- Logistic regression as baseline for comparison

**Outcomes:**
| Model | Accuracy |
|-------|----------|
| Custom NN – Sigmoid (from scratch) | ~54.07% |
| Custom NN – ReLU (from scratch) | **~54.62%** |
| MLPClassifier (2-layer: 16, 8) | ~53.49% |
| MLPClassifier (3-layer: 32, 16, 8) | ~52.24% |
| Logistic Regression (baseline) | ~49.67% |

> Custom neural networks outperform both the MLPClassifier configurations and the logistic regression baseline on this multi-class segmentation task.

---

### Assignment 4 – PCA & Classification on MNIST
**Dataset:** MNIST handwritten digits (70,000 samples, 784 features)

Applies PCA for dimensionality reduction and compares Logistic Regression with Neural Network classifiers.

**Key steps:**
- PCA with 50 components vs. 95% variance retention (327 components)
- Logistic Regression on original and PCA-reduced data
- Neural networks (TensorFlow/Keras) on original and PCA-reduced data

**Outcomes:**
| Model | Data | Accuracy |
|-------|------|----------|
| Logistic Regression | Original (784-D) | 91.62% |
| Logistic Regression | PCA (50 components) | 90.52% |
| Logistic Regression | PCA (327 components, 95% var) | 92.15% |
| Neural Network (64→10) | PCA (327 components) | 96.51% |
| Neural Network (64→10) | Original (784-D) | 96.67% |
| Neural Network (128→64→10) | Original (784-D) | **97.31%** |
| Neural Network (128→64→10) | PCA (327 components) | 97.23% |

> PCA with 95% variance retention slightly improves logistic regression accuracy while significantly reducing training time. Neural networks achieve ~97% accuracy on MNIST with faster training on PCA-reduced data.

---

## 🚀 Production-Level Project

### Project – Dimensionality Reduction on MNIST (SVD vs PCA)
**Dataset:** MNIST handwritten digits (70,000 samples)

A production-level notebook that systematically compares **Singular Value Decomposition (SVD)** and **PCA** for dimensionality reduction, evaluating both image quality and downstream classification performance.

**Key steps:**
1. **Data Preparation:** Load, normalize, and split MNIST (80/20 train/test)
2. **SVD Analysis:** Custom SVD pipeline with explained variance computation and cumulative variance plots
3. **Dimensionality Reduction:** Select top-k components retaining 90% and 95% variance
4. **Image Reconstruction:** Visual comparison with compression ratio and PSNR metrics
5. **PCA Pipeline:** Apply sklearn PCA and compare against SVD components
6. **Classification:** Train Logistic Regression on all variants and benchmark accuracy vs. training time

**SVD Compression Results:**

| Variance Retained | Components (k) | Compression Ratio | PSNR |
|-------------------|----------------|-------------------|------|
| 90% | 238 | 3.26× | 10.40 dB |
| 95% | 332 | 2.34× | 13.39 dB |

**Classification Results (Logistic Regression):**

| Dataset | Accuracy | Training Time |
|---------|----------|---------------|
| Original Data (784-D) | 91.62% | 21.90 s |
| SVD-Reduced (90%, k=238) | 92.14% | 9.94 s |
| SVD-Reduced (95%, k=332) | 92.18% | 11.81 s |
| PCA-Reduced (90%, k=238) | 92.03% | 9.83 s |
| PCA-Reduced (95%, k=332) | **92.26%** | 12.08 s |

> **Key insight:** Both SVD and PCA dimensionality reduction improve classification accuracy over the original high-dimensional data while **cutting training time by more than half** (from ~22 s to ~10 s). PCA with 95% variance retention achieves the best accuracy (92.26%), while SVD with 90% is the fastest (9.94 s) with only marginal accuracy trade-off.

---

## 🌐 Production App – Interactive ML Model Prediction (`app.py`)

A **Streamlit web application** that lets users interactively select a dataset and a classifier, input feature values or upload an image, and get real-time predictions with model accuracy.

**Features:**
- **Datasets supported:** IRIS (flower classification), Digits (handwritten digit recognition)
- **Classifiers supported:**
  - Logistic Regression
  - Neural Network (MLPClassifier with layers 128→64)
  - Naive Bayes
- **IRIS mode:** Dynamic numeric input fields for all 4 features → predicted class label
- **Digits mode:** Upload a handwritten digit image (JPG/PNG) → predicted digit (0–9)
- Displays model accuracy on the test set after every prediction

**How to run:**
```bash
pip install streamlit scikit-learn pandas numpy pillow
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Python 3** – Core language
- **NumPy / Pandas** – Data manipulation
- **Scikit-learn** – ML models, preprocessing, and evaluation
- **TensorFlow / Keras** – Deep learning (Assignments 4 & Project)
- **Matplotlib** – Visualization
- **Streamlit** – Production web app
- **Pillow** – Image processing (app.py)