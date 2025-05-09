import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    Objective function: f(x) = x^2
    """
    return x ** 4.0 + 3 * (x **3) + 5*x

def derivative(W):
    """
    Compute the derivative of f(x) = x^2 with respect to W.
    Arguments:
    W -- np.array: Input weights
    Returns:
    dW -- np.array: Gradients of the objective function
    """
    return 2 * W

def sgd(W, lr):
    """
    Perform Stochastic Gradient Descent to update W.
    Arguments:
    W -- np.array: Current weights
    lr -- float: Learning rate
    Returns:
    W -- np.array: Updated weights
    """
    dW = derivative(W)
    W = W - lr * dW
    return W

def train_p1(W, optimizer, lr, epochs):
    """
    Perform optimization to find the minimum of the function.
    Arguments:
    W -- np.array: Initial weights
    optimizer -- function: Optimization function
    lr -- float: Learning rate
    epochs -- int: Number of iterations
    Returns:
    weights -- list: List of weights after each epoch
    """
    weights = [W]
    for epoch in range(epochs):
        W = optimizer(W, lr)
        weights.append(W)
        print(f"Epoch {epoch + 1}: W = {W}")
    return weights

# Example: Initialize weights and perform optimization
W = np.array([-4.0, -4.0], dtype=np.float32)  # Initial weights
weights = train_p1(W, sgd, lr=0.1, epochs=30)

# Plotting the results
inputs = np.linspace(-4.0, 4.0, 100)
results = f(inputs)

plt.plot(inputs, results, label="Objective Function: f(x)")
for i, w in enumerate(weights):
    plt.scatter(w[0], f(w[0]), color='red', s=10, label=f"Epoch {i}" if i == 0 else "")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
