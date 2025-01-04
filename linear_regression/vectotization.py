import numpy as np
import matplotlib.pyplot as plt

"""
Assume that we have a model of linear regression to predict price of house from area input.
Data: Areas and prices. Two column. First column is input x, second colum for input y. 
- get_column: read x areas and y prices from list (generated from list from np.genfromtxt)
Linear regression 
Initial w weights and b bias and learning rate
Get sample ( one sample per an computation) => predict y^ from random x and w (bias) 
=> calculate loss : can be MSR Mean Square Error or MAE Mean Absolute Error or Huber Loss
=> calculate gradient: partial derivative of loss (careful with code here)
=> update weight => repeat
"""

def get_column(data, index):
    return [row[index] for row in data]


def predict(w, x, b):
    y_pred = w * x + b
    return y_pred


def calculate_MSE_loss(y_pred, y):
    loss = (y_pred - y) ** 2
    return loss

def calculate_MAE_loss(y_pred, y):
    loss= abs(y_pred - y)
    return loss
def calculate_Huber_loss(y_pred, y, delta=5):
    if abs(y_pred - y) < delta:
        loss = (y_pred- y) * (y_pred - y)
    else:
        loss = delta * abs(y_pred - y) - 0.5 * delta * delta
    return loss

def gradient_Huber(y_pred, y, x, delta=5):
    if abs(y_pred - y) < delta:
        dw, db = gradient_from_MSE_loss(y_pred, y, x)
    else:
        dw, db = gradient_from_MAE_loss(y_pred, y, x) * delta
    return dw, db
def gradient_from_MSE_loss(x, y_pred, y):
    dw = 2 * x * (y_pred - y)
    db = x * (y_pred - y)
    return dw, db
def gradient_from_MAE_loss(x, y_pred, y):
    dw = x * (y_pred - y) / abs(y_pred - y)
    db = (y_pred - y) / abs(y_pred - y)
    return dw, db

def update_weight(w, b, lr, dw, db):
    w = w - lr * dw
    b = b - lr * db
    return w, b


def main():
    # Load and validate data
    data = np.genfromtxt("/Users/thachha/PycharmProjects/AIO24/linear_regression/data.csv", delimiter=',').tolist()
    x_data = get_column(data, 0)
    y_data = get_column(data, 1)
    print(x_data)
    print(y_data)
    size = len(x_data)
    print(size)
    # Visualization data
    plt.scatter(x_data, y_data)
    plt.xlabel('area')
    plt.ylabel('price')
    plt.title("Price versus area")
    plt.show()

    # Initialize parameters w, b, lr
    w = 0.2
    b = 1.5
    lr = 0.01
    epoch = 50
    losses = []

    for i in range(epoch):
        epoch_loss = 0
        for idx in range(size):
            x = x_data[idx]
            y = y_data[idx]

            y_pred = predict(w, x, b)
            loss = calculate_MSE_loss(y_pred, y)
            dw, db = gradient_from_MSE_loss(w, y_pred, y)
            w, b = update_weight(w, b, lr, dw, db)
            epoch_loss +=loss
        losses.append(epoch_loss/size)
    print(losses)
    print(f'Final weight w={w}, bias b= {b}')
    # Loss visualisation
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.show()


main()
