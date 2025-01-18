import numpy as np


def rms_prop(W, lr, S, gamma):
    """
    Perform RMSprop optimization to update W
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2]
    lr -- float: learning rate
    S -- np.array: [s1, s2], running average of squared gradients
    gamma -- float: decay rate for S
    Returns:
    W -- np.array: updated weights [w1, w2]
    S -- np.array: updated running average of squared gradients
    """
    dW = [(0.2, 4.0)] * W
    epsilon = 1e-6
    S = gamma * S + (1 - gamma) * dW**2  # Update running average of squared gradients
    adapt_lr = lr / np.sqrt(S + epsilon)  # Adaptive learning rate
    W = W - adapt_lr * dW  # Update weights
    return W, S

def train_p1(W, optimizer, lr, epochs, S, gamma):
    """
    Train the optimization process to minimize the function
    Arguments:
    W -- np.array: initial weights [w1, w2]
    optimizer -- function: optimization algorithm
    lr -- float: learning rate
    epochs -- int: number of iterations
    S -- np.array: initial running average of squared gradients
    gamma -- float: decay rate for S
    Returns:
    results -- list: list of weights after each epoch
    """
    print(f'Initial W = {W}, Initial S = {S}')
    results = [W]  # Store initial weights
    for epoch in range(epochs):
        W, S = optimizer(W, lr, S, gamma)  # Update weights and S
        results.append(W)  # Store updated weights
        print(f"Epoch {epoch + 1}: W = {W}")
    return results

# Example usage
# Initial weights
W = np.array([-5, -2], dtype=np.float32)

# Train with RMSprop
train_p1(
    W,
    rms_prop,
    lr=0.3,
    epochs=30,
    S=np.array([0, 0], dtype=np.float32),
    gamma=0.9
)
