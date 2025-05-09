import numpy as np
import matplotlib.pyplot as plt

# Data (X: input, y: target)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# Initialize parameters
w, b = 0.0, 0.0
learning_rate = 0.01
epochs = 1000

def f(x, w, b ):
    return w *x +b
for _ in range (epochs):
    pred_y = f(X, w, b)
    loss = pred_y - y
    
