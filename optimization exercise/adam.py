import numpy as np


def Adam(W, lr, V, S, beta1, beta2, t):
    """
    Perform Adam optimization to update W.
    Arguments:
    W -- np.array [w1, w2], weights to update
    lr -- float: learning rate
    V -- np.array [v1, v2], first moment vector
    S -- np.array [s1, s2], second moment vector
    beta1 -- float: decay rate for the first moment
    beta2 -- float: decay rate for the second moment
    t -- int: iteration number (time step)
    Returns:
    dW -- np.array [dw1, dw2], gradients of the loss with respect to W
    W -- np.array: updated weights
    V -- np.array: updated first moment vector
    S -- np.array: updated second moment vector
    """
    dW = [(0.2, 4.0)] * W
    epsilon = 1e-6
    V = beta1 * V + (1 - beta1) * dW
    S = beta2 * S + (1 - beta2) * (dW ** 2)
    V_corrected = V / (1 - beta1 ** t)
    S_corrected = S / (1 - beta2 ** t)
    W = W - lr * V_corrected / (np.sqrt(S_corrected) + epsilon)
    return W, V_corrected, S_corrected

def train_p1(W, optimizer, V, S, lr, epochs, beta1, beta2):
    """
    Train the model using the specified optimizer.
    Arguments:
    optimizer -- function that performs optimization updates
    lr -- float: learning rate
    epochs -- int: number of iterations
    Returns:
    results -- list of np.array: list of [w1, w2] after each epoch
    """
    results = [W]
    for t in range(1, epochs + 1):
        W, V, S = optimizer(W, lr, V, S, beta1, beta2, t+1)
        results.append(W)
    return results
# Example Usage
W = np.array([-5, -2], dtype=np.float32)
print(train_p1(W, Adam, V=np.array([0, 0], dtype=np.float32), S=np.array([0, 0], dtype=np.float32), lr=0.2, epochs=30, beta1=0.9, beta2=0.999))

