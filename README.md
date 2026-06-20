# Neural Network from Scratch 🧠

A fully connected neural network built using **only NumPy and math** — no TensorFlow, no PyTorch. Trained on the MNIST dataset to classify handwritten digits (0–9) with ~85-90% accuracy.

> Built following [Samson Zhang's tutorial](https://youtu.be/w8yWXqWQYmU) on YouTube.

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]
(https://www.kaggle.com/code/bluesky827/mnist-digit-recognizer)

---

## 📌 What is this?

This project implements a neural network completely from scratch, covering:
- Forward propagation
- Activation functions (ReLU, Softmax)
- Backpropagation using the chain rule
- Gradient descent parameter updates

No deep learning libraries — just NumPy and a solid understanding of the math.

---

## 🏗️ Architecture

```
Input Layer    →  784 neurons  (28×28 pixels flattened)
Hidden Layer   →  128 neurons  + ReLU activation
Output Layer   →  10 neurons   + Softmax activation (digits 0–9)
```

---

## 🧮 The Math

### Forward Pass
Each layer computes a linear combination followed by a non-linear activation:

```
Z1 = W1 · X + b1       # Linear combination
A1 = ReLU(Z1)          # Non-linear activation

Z2 = W2 · A1 + b2      # Linear combination
A2 = Softmax(Z2)       # Output probabilities
```

### Loss
We measure error using Mean Absolute Error on the output gradient:
```
Loss = mean(|A2 - Y|)
```

### Backpropagation (Chain Rule)
Working backwards from the output to compute gradients:

```
dZ2 = A2 - Y                        # Output layer error
dW2 = dZ2 · A1ᵀ / m                # Weight gradient
db2 = sum(dZ2) / m                  # Bias gradient

dA1 = W2ᵀ · dZ2                    # Propagate error back
dZ1 = dA1 * ReLU'(Z1)              # Apply ReLU derivative
dW1 = dZ1 · Xᵀ / m                 # Weight gradient
db1 = sum(dZ1) / m                  # Bias gradient
```

Where `ReLU'(Z) = 1 if Z > 0 else 0`

### Gradient Descent
```
W := W - α · dW
b := b - α · db
```
Where `α` is the learning rate.

---

## 📂 Project Structure

```
neural-network-from-scratch/
│
├── neural_network.py     # Complete implementation
└── README.md             # You are here
```

---

## 🚀 How to Run

### Option 1: Run on Kaggle (Recommended — no setup needed)
1. Go to the [Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer)
2. Create a new notebook and add the competition dataset
3. Paste the code from `neural_network.py`
4. Run all cells

### Option 2: Run Locally
**Requirements:**
```bash
pip install numpy
```

**Download MNIST dataset:**
- Download `train.csv` from [Kaggle's Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)
- Place it in the same folder as `neural_network.py`

**Run:**
```bash
python neural_network.py
```

---

## 📊 Training Output

```
Epoch    0 | Loss: 2.6409 | Val Acc: 16.61%
Epoch   50 | Loss: 0.6763 | Val Acc: 83.43%
Epoch  100 | Loss: 0.4865 | Val Acc: 86.98%
Epoch  150 | Loss: 0.4174 | Val Acc: 88.46%
Epoch  200 | Loss: 0.3793 | Val Acc: 89.20%
Epoch  250 | Loss: 0.3541 | Val Acc: 89.72%
Epoch  300 | Loss: 0.3354 | Val Acc: 90.16%
Epoch  350 | Loss: 0.3206 | Val Acc: 90.60%
Epoch  400 | Loss: 0.3082 | Val Acc: 90.92%
Epoch  450 | Loss: 0.2976 | Val Acc: 91.16%

Final Training Accuracy: 91.82%

Test Accuracy: 92.29%

```

---

## 🔑 Key Concepts

| Concept | Description |
|---|---|
| **Weights & Biases** | Learnable parameters — linear combination of inputs |
| **ReLU** | `max(0, z)` — adds non-linearity, prevents vanishing gradients |
| **Softmax** | Converts scores to probabilities summing to 1 |
| **Backpropagation** | Chain rule to assign blame to each weight |
| **Gradient Descent** | Iteratively nudge weights to reduce loss |
| **Epochs** | Number of full passes through the training dataset |
| **Learning Rate (α)** | Step size for each weight update |

---

## 💡 Why 128 Hidden Neurons?

128 is a hyperparameter — a design choice. Too few neurons (e.g. 4) causes underfitting — the network can't capture enough patterns. Too many (e.g. 10,000) wastes computation and risks overfitting. 128 is a sweet spot for MNIST.

---

## 📖 References

- [Building a Neural Network FROM SCRATCH by Samson Zhang](https://youtu.be/w8yWXqWQYmU)
- [MNIST Dataset — Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
