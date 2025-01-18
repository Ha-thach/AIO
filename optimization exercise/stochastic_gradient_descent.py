import numpy as np
def df_w(W):
    """
    𝑓(𝑤1,𝑤2)=0.1(𝑤1)^2+0.2(𝑤2)^2
    Compute derivative with respect to w1 and w2
    Arguments:
    W -- np.array [w1, w2]
    Returns:
    dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    """

    dW = [(0.2, 0.4)] * W

    return dW

def sgd(W, dW, lr):
    """
    Perform Stochastic Gradient Descent to update w1 and w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2]
    lr -- float: learning rate
    Returns:
    W -- np.array: [w1, w2] after update
    """
    W = W - lr * dW

    return W

def train_p1(W, optimizer, lr, epochs):
    """
    Perform optimization to find the minimum of the function
    Arguments:
    W: np.array: [w1, w2]
    optimizer : function that performs optimization updates
    lr -- float: learning rate
    epochs -- int: number of iterations
    Returns:
    results -- list: list of [w1, w2] after each epoch
    """
    # list of results
    results = [W]
    for epoch in range(epochs):
        dw = df_w(W)
        W = optimizer(W,dw,lr)
        results.append(W)
        print(f"Epoch {epoch + 1}: W = {W}, Gradient = {dw}")
    return results

# For example
# Initial weights
W = np.array([-5, -2], dtype=np.float32)
train_p1(W, sgd, lr=0.4, epochs=30)