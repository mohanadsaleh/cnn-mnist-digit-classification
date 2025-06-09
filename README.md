# 🧠 CNN for Handwritten Digit Recognition – MNIST Dataset

This project implements a **Convolutional Neural Network (CNN)** from scratch using **TensorFlow** to classify handwritten digits from the **MNIST dataset**.

---

## 📊 Dataset

**Source:** [MNIST Handwritten Digit Database](http://yann.lecun.com/exdb/mnist/)  
**Description:** The dataset contains 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels.

- **Training samples:** 60,000  
- **Test samples:** 10,000  
- **Image size:** 28x28  
- **Classes:** 10 (digits from 0 to 9)

---

## 🛠️ Features & Workflow

1. Load and preprocess the MNIST dataset  
2. Normalize images and apply one-hot encoding to labels  
3. Build a deep CNN using Keras Sequential API  
4. Train the model with dropout regularization  
5. Visualize training history (accuracy and loss)  
6. Evaluate model using a confusion matrix

---

## 🧠 Model Architecture

- `Conv2D (32 filters, 5x5, ReLU)`  
- `Conv2D (32 filters, 5x5, ReLU)`  
- `MaxPooling2D`  
- `Dropout (25%)`  
- `Conv2D (64 filters, 3x3, ReLU)`  
- `Conv2D (64 filters, 3x3, ReLU)`  
- `MaxPooling2D`  
- `Dropout (25%)`  
- `Flatten`  
- `Dense (128 neurons, ReLU)`  
- `Dropout (50%)`  
- `Dense (10 neurons, Softmax)`

---

## ✅ Results

The model achieves high accuracy on both the training and validation sets. A confusion matrix is plotted to visually inspect the performance on each digit class.

---

## ▶️ How to Run

1. Install the required libraries:
   ```bash
   pip install tensorflow matplotlib seaborn
