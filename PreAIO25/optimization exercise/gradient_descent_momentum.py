import numpy as np

def sgd_momentum(W, lr, V, beta):
    """
    Perform Stochastic Gradient Descent to update w1 and w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2]
    lr -- float: learning rate
    V -- np.array: velocity (momentum vector)
    beta -- float: momentum factor
    Returns:
      W -- np.array: [w1, w2] after update
    V -- np.array: updated velocity vector
    """
    dW = [(0.2, 4.0)] * W
    V = beta * V + (1 - beta) * dW
    W = W - lr * V

    return W, V

def train_p1(W, optimizer, lr, epochs, V, beta):
    """
    Perform optimization to find the minimum of the function
    Arguments:
    W: np.array: [w1, w2]
    optimizer : function that performs optimization updates
    lr -- float: learning rate
    epochs -- int: number of iterations
    V -- np.array: initial velocity (momentum vector)
    beta -- float: momentum factor
    Returns:
    results -- list: list of [w1, w2] after each epoch
    """

    print(f'Initial V = {V}')
    # list of results
    results = [W]
    for epoch in range(epochs):
        W, V = optimizer(W,lr, V, beta)
        results.append(W)
        print(f"Epoch {epoch + 1}: W = {W}")
    return results

# For example
# Initial weights
W = np.array([-5, -2], dtype=np.float32)
train_p1(W, sgd_momentum, lr=0.6, epochs=30, V=np.array([0, 0], dtype=np.float32), beta=0.5)